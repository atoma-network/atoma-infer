use candle_core::{DType, Device, INdexOp, IndexOp, Result, Tensor};
use csrc::{copy_blocks, swap_blocks};

/// `FlashAttentionDecodingMetadata` - Structure wrapping the metadata
/// required for running flash and paged attention kernels for decoding
/// inference
pub struct FlashAttentionDecodingMetadata {
    /// The block tables, used for mapping each sequence id
    /// to the list of physical blocks that have been currently
    /// allocated for it.
    ///
    /// E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    /// in the KV cache. Each block can contain up to `block_size` tokens.
    ///
    /// It is of shape `[batch_size, max_blocks_per_sequence]`
    pub block_tables: Tensor,
    /// The maximum decoding sequence length for the current batch.
    /// It is `0` if there are only prefill sequences.
    pub max_decoding_sequence_length: usize,
    /// The sequence length per sequence, as a tensor.
    /// Sequence length means the computed
    /// tokens + new tokens `None` if it is a decoding,
    /// of shape `[batch_size,]`
    pub sequence_lengths: Option<Tensor>,
}

/// `FlashAttentionPrefillMetadata` - Structure wrapping the metadata
/// required for running flash and paged attention kernels for prefill
/// inference
pub struct FlashAttentionPrefillMetadata {
    /// The block tables, used for mapping each sequence id
    /// to the list of physical blocks that have been currently
    /// allocated for it.
    ///
    /// E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    /// in the KV cache. Each block can contain up to `block_size` tokens.
    ///
    /// It is of shape `[batch_size, max_blocks_per_sequence]`
    pub block_tables: Tensor,
    /// The maximum query length for the current batch
    /// sequences. It is `None` if it is a decoding
    pub max_query_length: Option<usize>,
    /// The maximum prefill sequence length for the current batch.
    /// It is `0` if there are only decoding sequences.
    pub max_prefill_sequence_length: usize,
    /// The cumulative sub-query lengths of the sequences in
    /// the batch, used to index into subquery. E.g., if the sub-query length
    /// is [4, 6], it is [0, 4, 10]. It is of shape `[batch_size + 1,]`
    pub query_start_locations: Option<Tensor>,
    /// The cumulative sequence lengths of the sequences in
    /// the batch, used to index into sequence. E.g., if the sequence length is
    /// [4, 6], it is [0, 4, 10]. It is of shape `[batch_size + 1,]`
    pub sequence_start_locations: Option<Tensor>,
        /// The sequence length per sequence, as a tensor.
    /// Sequence length means the computed
    /// tokens + new tokens `None` if it is a decoding,
    /// of shape `[batch_size,]`
    pub sequence_lengths: Option<Tensor>,
}

/// `FlashAttentionMetadata` - Structure wrapping the metadata
/// required for running flash and paged attention kernels
pub struct FlashAttentionMetadata {
    /// A tensor of context lengths (tokens that are computed
    /// so far). Of shape `[batch_size,]`
    pub context_lengths: Option<Tensor>,
    /// Slot mapping, maps each token (or element in the input sequence) to a specific slot
    /// or segment in the cached memory. This allows for efficient access and organization
    /// of attention computations over large sequences.
    pub slot_mapping: Tensor,
    /// Flash attention decoding metadata
    pub decoding_metadata: Option<FlashAttentionDecodingMetadata>,
    /// Flash attention prefill metadata
    pub prefill_metadata: Option<FlashAttentionPrefillMetadata>,
}

/// Flash attention
///
/// It encapsulates the flash attention algorithm for fast attention computation.
/// It is further compatible with the paged attention algorithm, including
/// cache and memory management, using a block pagination method.
///
/// It supports both prefill and decode generation.
///
/// If the input tensors contain prompt tokens, the layout is as follows:
///
/// |<--------------- num_prefill_tokens ----------------->|
/// |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|
///
/// Otherwise, the layout is as follows:
///
/// |<----------------- num_decode_tokens ------------------>|
/// |<--decode_0-->|<--decode_1-->|.........|<--decode_M-1-->|
///
/// The prompts might have different lengths, while the generation tokens
/// always have length 1.
///
/// Moreover, it is possible
/// to use it with chunked prefill, where the prefill tokens and decode tokens
/// are batched together in a flattened 1D query, with layout as follows:
///
/// |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
/// |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|
pub struct FlashAttention {
    /// Total number of heads
    pub num_heads: usize,
    /// Number of Key/Value cache heads
    pub num_kv_heads: usize,
    /// Number of queries per KV cache
    pub num_queries_per_kv: usize,
    /// Each head's dimension
    pub head_dim: usize,
    /// Softmax scale
    pub softmax_scale: f32,
    /// Alibi slopes,
    pub alibi_slopes: Option<Tensor>,
    /// Sliding window, for local attention
    /// with both left and right sliding
    /// local window size
    pub sliding_window: (i64, i64),
    /// Key and value cache dtype
    pub kv_cache_dtype: DType,
    /// Device, in most cases it should be
    /// a `CudaDevice`
    pub device: Device,
}

impl FlashAttention {
    /// Constructor
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        softmax_scale: f32,
        alibi_slopes: Option<Tensor>,
        sliding_window: Option<usize>,
        kv_cache_dtype: DType,
        device: Device,
    ) -> Result<Self> {
        if num_heads % num_kv_heads != 0 {
            candle_core::bail!(
                "number of heads {num_heads} must divide number of kv heads {num_kv_heads}"
            )
        }
        if !Self::supported_head_sizes().contains(&head_dim) {
            candle_core::bail!("head_dim {head_dim} is not supported")
        }
        Ok(Self {
            num_heads,
            num_kv_heads,
            num_queries_per_kv: num_heads / num_kv_heads,
            head_dim,
            softmax_scale,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            device,
        })
    }

    /// Available supported head sizes
    pub fn supported_head_sizes() -> Vec<u32> {
        vec![64, 80, 96, 112, 128, 192, 256]
    }

    /// Returns the KV cache shape for the given model
    /// configurations.
    pub fn get_kv_cache_shape(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_size: usize,
    ) -> Vec<usize> {
        vec![2, num_blocks, block_size * num_kv_heads * head_size]
    }

    /// Splits the KV cache
    pub fn split_kv_cache(
        &self,
        kv_cache: &Tensor,
        num_kv_heads: usize,
        head_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        if kv_cache.dims().len() != 5 {
            candle_core::bail!("KV cache must have rank 5 (got {})", kv_cache.shape().len())
        }

        let [cache_size, num_blocks, block_size, num_kv_heads, head_size] = kv_cache.dims()?;
        if cache_size != 2 {
            candle_core::bail!("KV cache must have cache_size 2 (got {cache_size})")
        }
        if num_kv_heads != self.num_kv_heads {
            candle_core::bail!(
                "KV cache must have num_heads {} (got {num_kv_heads})",
                self.num_kv_heads
            )
        }
        if head_size != self.head_dim {
            candle_core::bail!(
                "KV cache must have head_size {} (got {head_size})",
                self.head_dim
            )
        }

        let key_cache = kv_cache.i(0)?.unsqueeze(0)?;
        let value_cache = kv_cache.i(1)?.unsqueeze(0)?;

        Ok((key_cache, value_cache))
    }

    /// Initiates a swap blocks operation on the current CUDA device
    pub fn swap_blocks(
        &self,
        src: &Tensor,
        dst: &mut Tensor,
        block_mapping: HashMap<i64, i64>,
    ) -> Result<()> {
        let (src_key, src_value) =
            self.split_kv_cache(kv_cache, self.num_kv_heads, self.head_dim)?;
        let (dst_key, dst_value) = self.split_kv_cache(dst, self.num_kv_heads, self.head_dim)?;
        swap_blocks(src, dst, block_mapping)
    }

    /// Initiates a copy blocks operation on the current CUDA device
    pub fn copy_blocks(
        kv_caches: &mut Vec<Tensor>,
        block_mapping: Tensor,
    ) -> Result<(), CandleError> {
        let key_caches = kv_caches
            .iter_mut()
            .map(|kv_cache| kv_cache.i(0)?.unsqueeze(0))
            .collect::<Result<Vec<_>>>()?
            .iter_mut()
            .collect();
        let value_caches = kv_caches
            .iter_mut()
            .map(|kv_cache| kv_cache.i(1)?.unsqueeze(0))
            .collect::<Result<Vec<_>>>()?
            .iter_mut()
            .collect();
        unsafe { copy_blocks(key_caches, value_caches, block_mapping) }
    }

    fn forward(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        kv_cache: &Tensor,
        attention_metadata: FlashAttentionMetadata,
    ) -> Result<Tensor> {
    }
}
