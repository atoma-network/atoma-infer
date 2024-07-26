use candle_core::{DType, Device, IndexOp, Result, Tensor};
use csrc::{
    copy_blocks, flash_attn_kv_cache_full, flash_attn_varlen_with_block_table,
    reshape_and_cache_flash, swap_blocks,
};
use std::collections::HashMap;

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
    pub block_tables: Option<Tensor>,
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
    pub block_tables: Option<Tensor>,
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
    /// of attention computations over large sequences. Of shape `[num_prefill_tokens + num_decoding_tokens,]`
    pub slot_mapping: Tensor,
    /// Flash attention decoding metadata
    pub decoding_metadata: Option<FlashAttentionDecodingMetadata>,
    /// Flash attention prefill metadata
    pub prefill_metadata: Option<FlashAttentionPrefillMetadata>,
    /// Number of prefill tokens
    pub num_prefill_tokens: usize,
    /// Number of decoding tokens
    pub num_decoding_tokens: usize,
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
    /// Sliding window, for local attention,
    /// only supports causal sliding window
    pub sliding_window: Option<usize>,
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
        if !Self::supported_head_sizes().contains(&(head_dim as u32)) {
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
    pub fn split_kv_cache(&self, kv_cache: &Tensor) -> Result<(Tensor, Tensor)> {
        if kv_cache.dims().len() != 5 {
            candle_core::bail!(
                "KV cache must have rank 5 (got {})",
                kv_cache.shape().dims().len()
            )
        }

        let (cache_size, _num_blocks, _block_size, num_kv_heads, head_dim) = kv_cache.dims5()?;
        if cache_size != 2 {
            candle_core::bail!("KV cache must have cache_size 2 (got {cache_size})")
        }
        if num_kv_heads != self.num_kv_heads {
            candle_core::bail!(
                "KV cache must have num_heads {} (got {num_kv_heads})",
                self.num_kv_heads
            )
        }
        if head_dim != self.head_dim {
            candle_core::bail!(
                "KV cache must have head_dim {} (got {head_dim})",
                self.head_dim
            )
        }

        let key_cache = kv_cache.i(0)?.squeeze(0)?;
        let value_cache = kv_cache.i(1)?.squeeze(0)?;

        Ok((key_cache, value_cache))
    }

    /// Initiates a swap blocks operation on the current CUDA device
    pub fn swap_blocks(
        &self,
        src: &Tensor,
        dst: &mut Tensor,
        block_mapping: &HashMap<i64, i64>,
    ) -> Result<()> {
        let (src_key, src_value) = self.split_kv_cache(src)?;
        let (mut dst_key, mut dst_value) = self.split_kv_cache(dst)?;
        swap_blocks(&src_key, &mut dst_key, block_mapping)?;
        swap_blocks(&src_value, &mut dst_value, block_mapping)
    }

    /// Initiates a copy blocks operation on the current CUDA device
    pub fn copy_blocks(kv_caches: &mut Vec<Tensor>, block_mapping: Tensor) -> Result<()> {
        let mut key_caches = kv_caches
            .iter_mut()
            .map(|kv_cache| kv_cache.i(0)?.squeeze(0))
            .collect::<Result<Vec<_>>>()?;
        let key_caches = key_caches.iter_mut().collect();
        let mut value_caches = kv_caches
            .iter_mut()
            .map(|kv_cache| kv_cache.i(1)?.squeeze(0))
            .collect::<Result<Vec<_>>>()?;
        let value_caches = value_caches.iter_mut().collect();
        unsafe { copy_blocks(&key_caches, &value_caches, block_mapping) }
    }

    /// Flash attention forward pass
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor with shape `[num_tokens, num_heads * head_size]`
    /// * `k` - Key tensor with shape `[num_tokens, num_kv_heads * head_size]`
    /// * `v` - Value tensor with shape `[num_tokens, num_kv_heads * head_size]`
    /// * `kv_cache` - KV cache tensor with shape `[2, num_blocks, block_size, num_kv_heads, head_size]`
    /// * `attention_metadata` - Metadata for flash attention
    ///
    /// # Returns
    ///
    /// * `shape` - [num_tokens, num_heads * head_size]
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        kv_cache: &Tensor,
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let (q_num_tokens, q_num_heads, q_hidden_dim) = q.dims3()?;
        let (k_num_tokens, k_num_heads, k_hidden_dim) = k.dims3()?;
        let (v_num_tokens, v_num_heads, v_hidden_dim) = v.dims3()?;

        if q_num_tokens != k_num_tokens || q_num_tokens != v_num_tokens {
            candle_core::bail!(
                "query, key, and value must have the same number of tokens (got {q_num_tokens}, {k_num_tokens}, {v_num_tokens})"
            )
        }
        if (k_num_tokens, k_num_heads, k_hidden_dim) != (v_num_tokens, v_num_heads, v_hidden_dim) {
            candle_core::bail!(
                "key and value must have the same shape (got [{k_num_tokens}, {k_num_heads}, {k_hidden_dim}], [{v_num_tokens}, {v_num_heads}, {v_hidden_dim}])"
            )
        }
        if (q_num_heads, q_hidden_dim) != (self.num_heads, self.head_dim) {
            candle_core::bail!(
                "query must have shape [{q_num_heads}, {q_hidden_dim}] (got [{self.num_heads}, {self.head_dim}])"
            )
        }
        if (k_num_heads, k_hidden_dim) != (self.num_kv_heads, self.head_dim) {
            candle_core::bail!(
                "key must have hidden size {} (got {k_hidden_size})",
                self.num_kv_heads * self.head_dim
            )
        }

        // Reshape the input keys and values and store them in the cache.
        let (k_cache, v_cache) = self.split_kv_cache(kv_cache)?;
        reshape_and_cache_flash(&k, &v, &k_cache, &v_cache, &attention_metadata.slot_mapping)?;

        let num_prefill_tokens = attention_metadata.num_prefill_tokens;
        let num_decoding_tokens = attention_metadata.num_decoding_tokens;

        if k_num_tokens != num_prefill_tokens + num_decoding_tokens {
            candle_core::bail!(
                "query must have number of tokens {} (got {})",
                num_prefill_tokens + num_decoding_tokens,
                q_num_tokens
            )
        }

        let output = Tensor::zeros(q.shape(), q.dtype(), &self.device)?;

        // Query for decode
        // KV is not needed because it is already cached
        let decode_q = q.i(num_prefill_tokens..)?;
        // QKV for prefill
        let q = q.i(..num_prefill_tokens)?;
        let k = k.i(..num_prefill_tokens)?;
        let v = v.i(..num_prefill_tokens)?;

        if let Some(prefill_metadata) = &attention_metadata.prefill_metadata {
            if prefill_metadata
                .block_tables
                .as_ref()
                .map(|bt| bt.elem_count() == 0)
                .unwrap_or(true)
            {
                // This is the case in which we have new incoming prompts,
                // to which there is no previous query, key and cache to reuse.
                let sequence_start_locations = prefill_metadata
                    .sequence_start_locations
                    .as_ref()
                    .ok_or(candle_core::Error::Msg(
                    "Missing sequence start locations tensor for prefill inference".into(),
                ))?;
                let out = flash_attn_varlen_with_block_table(
                    &q,
                    &k,
                    &v,
                    self.alibi_slopes.as_ref(),
                    sequence_start_locations,
                    sequence_start_locations,
                    prefill_metadata.max_prefill_sequence_length,
                    prefill_metadata.max_prefill_sequence_length,
                    self.softmax_scale,
                    self.sliding_window,
                    None,
                    None,
                )?;
                output.slice_assign(
                    &[..num_prefill_tokens, ..output.dims()[1], ..output.dims()[2]],
                    &out,
                )?;
            } else {
                // We support prefix enabled attention, in which a block table is provided.
                let sequence_lengths = if let Some(sequence_lengths) =
                    prefill_metadata.sequence_lengths.as_ref()
                {
                    sequence_lengths
                } else {
                    candle_core::bail!("Missing sequence lengths tensor for prefill inference, with prefix enabled attention")
                };
                let max_sequence_length_k = sequence_lengths.max(0)?.to_scalar::<i64>()? as usize;
                let query_start_locations = if let Some(query_start_locations) =
                    prefill_metadata.query_start_locations.as_ref()
                {
                    query_start_locations
                } else {
                    candle_core::bail!("Missing query start locations tensor for prefill inference, with prefix enabled attention")
                };
                let sequence_start_locations = if let Some(sequence_start_locations) =
                    prefill_metadata.sequence_start_locations.as_ref()
                {
                    sequence_start_locations
                } else {
                    candle_core::bail!("Missing sequence start locations tensor for prefill inference, with prefix enabled attention")
                };
                let out = flash_attn_varlen_with_block_table(
                    &q,
                    &k_cache,
                    &v_cache,
                    self.alibi_slopes.as_ref(),
                    query_start_locations,
                    sequence_start_locations,
                    prefill_metadata.max_prefill_sequence_length,
                    max_sequence_length_k,
                    self.softmax_scale,
                    self.sliding_window,
                    None,
                    prefill_metadata.block_tables.as_ref(),
                )?;
                output.slice_assign(
                    &[..num_prefill_tokens, ..output.dims()[1], ..output.dims()[2]],
                    &out,
                )?;
            }
        }

        if let Some(decoding_metadata) = &attention_metadata.decoding_metadata {
            // Decoding inference forward pass
            let out = flash_attn_kv_cache_full(
                &decode_q.unsqueeze(1)?, // in decoding phase, each batch sequence has length 1
                &k_cache,
                &v_cache,
                self.alibi_slopes.as_ref(),
                self.softmax_scale,
                self.sliding_window,
                None,
                decoding_metadata.block_tables.as_ref(),
                decoding_metadata.sequence_lengths.as_ref(),
                None,
            )?;
            output.slice_assign(&[num_prefill_tokens.., 0.., 0..], &out.squeeze(1)?)?;
        }

        output.reshape((q_num_tokens, self.num_heads * self.head_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_new() {
        let device = Device::new_cuda(0).unwrap();
        let result = FlashAttention::new(8, 4, 64, 1.0, None, None, DType::F32, device);

        assert!(result.is_ok());

        let flash_attention = result.unwrap();

        assert_eq!(flash_attention.num_heads, 8);
        assert_eq!(flash_attention.num_kv_heads, 4);
        assert_eq!(flash_attention.num_queries_per_kv, 2);
        assert_eq!(flash_attention.head_dim, 64);
        assert_eq!(flash_attention.softmax_scale, 1.0);
        assert_eq!(flash_attention.kv_cache_dtype, DType::F32);
    }

    #[test]
    fn test_new_invalid_heads() {
        let device = Device::Cpu;
        let result = FlashAttention::new(7, 4, 64, 1.0, None, None, DType::F32, device);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_invalid_head_dim() {
        let device = Device::Cpu;
        let result = FlashAttention::new(8, 4, 65, 1.0, None, None, DType::F32, device);
        assert!(result.is_err());
    }

    #[test]
    fn test_supported_head_sizes() {
        let sizes = FlashAttention::supported_head_sizes();
        assert_eq!(sizes, vec![64, 80, 96, 112, 128, 192, 256]);
    }

    #[test]
    fn test_get_kv_cache_shape() {
        let shape = FlashAttention::get_kv_cache_shape(10, 32, 4, 64);
        assert_eq!(shape, vec![2, 10, 8192]);
    }

    #[test]
    fn test_split_kv_cache() {
        let device = Device::Cpu;
        let flash_attention =
            FlashAttention::new(8, 4, 64, 1.0, None, None, DType::F32, device.clone()).unwrap();

        let kv_cache = Tensor::zeros((2, 10, 32, 4, 64), DType::F32, &device).unwrap();
        let result = flash_attention.split_kv_cache(&kv_cache);

        assert!(result.is_ok());

        let (key_cache, value_cache) = result.unwrap();

        assert_eq!(key_cache.shape().dims(), &[10, 32, 4, 64]);
        assert_eq!(value_cache.shape().dims(), &[10, 32, 4, 64]);
    }

    #[test]
    fn test_split_kv_cache_invalid_shape() {
        let device = Device::Cpu;
        let flash_attention =
            FlashAttention::new(8, 4, 64, 1.0, None, None, DType::F32, device.clone()).unwrap();

        let kv_cache = Tensor::zeros((2, 10, 32, 4), DType::F32, &device).unwrap();
        let result = flash_attention.split_kv_cache(&kv_cache);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward() {
        let device = Device::new_cuda(0).unwrap();
        let flash_attention = FlashAttention {
            num_heads: 8,
            num_kv_heads: 4,
            num_queries_per_kv: 2,
            head_dim: 64,
            softmax_scale: 1.0,
            alibi_slopes: None,
            sliding_window: None,
            kv_cache_dtype: DType::BF16,
            device: device.clone(),
        };

        let q = Tensor::rand(1.0, 10.0, (15, 512), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::rand(1.0, 10.0, (15, 256), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = Tensor::rand(1.0, 10.0, (15, 256), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let kv_cache = Tensor::rand(1.0, 10.0, (2, 10, 32, 4, 64), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let attention_metadata = FlashAttentionMetadata {
            context_lengths: None,
            slot_mapping: Tensor::arange(0i64, 15, &device).unwrap(),
            prefill_metadata: Some(FlashAttentionPrefillMetadata {
                block_tables: Some(
                    Tensor::arange(0i64, 2, &device)
                        .unwrap()
                        .reshape((2, 1))
                        .unwrap(),
                ),
                max_query_length: Some(3),
                max_prefill_sequence_length: 3,
                query_start_locations: Some(
                    Tensor::from_vec(vec![0u32, 5, 10], (3,), &device).unwrap(),
                ),
                sequence_start_locations: Some(
                    Tensor::from_vec(vec![0u32, 5, 10], (3,), &device).unwrap(),
                ),
                sequence_lengths: Some(Tensor::from_vec(vec![5i64, 5], (2,), &device).unwrap()),
            }),
            decoding_metadata: Some(FlashAttentionDecodingMetadata {
                block_tables: Some(
                    Tensor::arange(2i64, 7, &device)
                        .unwrap()
                        .reshape((5, 1))
                        .unwrap(),
                ),
                max_decoding_sequence_length: 3,
                sequence_lengths: Some(Tensor::from_vec(vec![3u32; 5], (5,), &device).unwrap()),
            }),
            num_prefill_tokens: 10,
            num_decoding_tokens: 5,
        };

        let result = flash_attention.forward(&q, &k, &v, &kv_cache, &attention_metadata);

        assert!(result.is_ok());

        let output = result.unwrap();

        assert_eq!(output.shape().dims(), &[15, 512]);
        assert!(!output
            .eq(0.)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap()
            .iter()
            .any(|&x| x == 0));
    }
}
