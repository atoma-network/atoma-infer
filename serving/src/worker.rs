use std::{collections::HashMap, sync::Arc};

use crate::{
    model_executor::{ModelExecutor, ModelExecutorError, ModelLoaderError},
    sequence::{ExecuteModelRequest, SequenceGroupMetadata, SequenceGroupOutput},
};
use atoma_paged_attention::flash_attention::{FlashAttention, FlashAttentionMetadata};
use candle_core::{DType, DTypeParseError, Device, Error as CandleError, Tensor};
use thiserror::Error;
use tracing::{error, info, info_span, instrument, warn, Span};

const PAD_SLOT_ID: i64 = -1;

/// `ModelInput` - Input for LLM model
/// forward pass
pub struct ModelInput {
    /// Input tokens tensor
    input_tokens_tensor: Tensor,
    /// Input positions tensor
    input_positions: Tensor,
    /// Attention Metadata
    attention_metadata: FlashAttentionMetadata,
    /// Number of decoded tokens
    #[allow(dead_code)]
    num_decode_tokens: usize,
    /// Number of prefills
    #[allow(dead_code)]
    num_prefills: usize,
    /// Cumulative query lengths, of size `batch_size + 1`
    cu_query_lengths: Tensor,
}

/// `ModelWorker` - Responsible for running a LLM model
/// instance (or a partition of it).
///
/// Each worker is associated with a single GPU. The worker is responsible for
/// maintaining the KV cache and executing the model on the GPU. In case of
/// distributed inference, each worker is assigned a partition of the model.
pub struct ModelWorker<M: ModelExecutor> {
    /// Cache engine
    cache_engine: CacheEngine,
    /// Device,
    device: Device,
    /// Enable chunked prefill (boolean)
    enable_chunked_prefill: bool,
    /// Model runner instance
    model: M,
    /// Initial GPU available memory
    #[allow(dead_code)]
    initial_gpu_memory: usize,
    /// Tracing Span
    span: Span,
}

impl<M> ModelWorker<M>
where
    M: ModelExecutor,
{
    /// Constructor
    #[instrument(skip_all)]
    pub fn new(
        block_size: usize,
        device: Device,
        dtype: DType,
        model: M,
        num_cpu_blocks: usize,
        num_gpu_blocks: usize,
        enable_chunked_prefill: bool,
    ) -> Result<Self, ModelWorkerError> {
        let span = info_span!("model-worker");
        let _span = span.clone();
        let _enter = _span.enter();

        info!("Starting a new `ModelWorker` instance");
        let cache_engine = CacheEngine::new(
            block_size,
            device.clone(),
            dtype,
            model.alibi_slopes(),
            model.hidden_size() / model.num_attention_heads(),
            model.num_attention_heads(),
            model.num_attention_heads(),
            model.num_kv_heads(),
            num_cpu_blocks,
            num_gpu_blocks,
            model.softmax_scale(),
            model.sliding_window(),
        )?;

        // TODO:
        // 1. Check cuda is available (error otherwise);
        // 2. Access initial GPU memory (using cudarc)
        Ok(Self {
            cache_engine,
            device,
            enable_chunked_prefill,
            model,
            initial_gpu_memory: 0, // TODO 2.
            span,
        })
    }

    /// Determines the number of available GPU blocks
    pub fn num_available_gpu_blocks(&self) -> usize {
        todo!()
    }

    /// Executes model's forward pass
    #[instrument(skip_all)]
    pub fn execute_model(
        &mut self,
        request: ExecuteModelRequest,
    ) -> Result<Vec<SequenceGroupOutput>, ModelWorkerError> {
        info!("Executing model on new request..");

        let span = self.span.clone();
        let _enter = span.enter();

        let ExecuteModelRequest {
            sequence_groups_metadata,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            ..
        } = request;

        let num_sequence_groups = sequence_groups_metadata.len();

        // `blocks_to_copy` is a GPU tensor. The source and target of
        // blocks to copy are in the same device, and `blocks_to_copy`
        // can be used directly within cuda kernels.
        let blocks_to_copy = blocks_to_copy
            .into_iter()
            .flat_map(|(i, j)| [i, j])
            .collect::<Vec<_>>();
        let blocks_to_copy = if blocks_to_copy.is_empty() {
            None
        } else {
            let num_block_pairs_to_copy = blocks_to_copy.len() / 2;
            Some(Tensor::from_vec(
                blocks_to_copy,
                (num_block_pairs_to_copy, 2),
                &self.device,
            )?)
        };

        // At this point we need to perform cache swap operations
        self.cache_swap(&blocks_to_swap_in, &blocks_to_swap_out, blocks_to_copy)?;

        // NOTE: Number of sequence groups should not be zero,
        // as we don't schedule empty sequences, for now.
        if num_sequence_groups == 0 {
            warn!("Number of sequence groups to run model on should not be empty");
            return Ok(vec![]);
        }

        let ModelInput {
            input_tokens_tensor,
            input_positions,
            attention_metadata,
            cu_query_lengths,
            ..
        } = self.prepare_input_tensors(&sequence_groups_metadata)?;

        let selected_token_indices = utils::compute_selected_token_indices(&cu_query_lengths)?;

        let kv_cache = self.cache_engine.gpu_cache.iter_mut().collect();
        let logits = self.model.forward(
            &input_tokens_tensor,
            &input_positions,
            &selected_token_indices,
            kv_cache,
            attention_metadata,
        )?;

        let sampled_outputs = self
            .model
            .sample(&logits.squeeze(0)?, &sequence_groups_metadata)?;

        Ok(sampled_outputs)
    }

    /// Swaps cached blocks
    #[instrument(skip_all)]
    pub fn cache_swap(
        &mut self,
        blocks_to_swap_in: &HashMap<u32, u32>,
        blocks_to_swap_out: &HashMap<u32, u32>,
        blocks_to_copy: Option<Tensor>,
    ) -> Result<(), ModelWorkerError> {
        if !blocks_to_swap_in.is_empty() {
            self.cache_engine.swap_in(blocks_to_swap_in)?
        }
        if !blocks_to_swap_out.is_empty() {
            self.cache_engine.swap_out(blocks_to_swap_out)?
        }
        if let Some(bs) = blocks_to_copy {
            self.cache_engine.copy_blocks(bs)?
        }
        Ok(())
    }

    /// Prepares input tensors for model forward run, based
    /// on availabe sequence groups metadata.
    ///
    /// The API assumes seq_group_metadata_list is sorted by prefill -> decode.
    ///
    /// The result tensors and data structure also batches input in prefill
    /// -> decode order. For example,
    ///
    /// - input_tokens.i(..num_prefill_tokens) contains prefill tokens.
    /// - input_tokens.i(num_prefill_tokens..) contains decode tokens.
    ///
    #[instrument(skip_all)]
    pub fn prepare_input_tensors(
        &self,
        sequence_groups_metadata: &[Arc<SequenceGroupMetadata>],
    ) -> Result<ModelInput, ModelWorkerError> {
        let _enter = self.span.enter();
        info!("Preparing input tensors for new inference request..");

        let mut input_tokens = Vec::<u32>::new();
        let mut input_positions = Vec::new();
        let mut slot_mapping = Vec::new();
        let mut sequence_lengths = Vec::new();
        let mut prefill_sequence_lengths = Vec::new();
        let mut decode_sequence_lengths = Vec::new();
        let mut context_lengths = Vec::new();
        let mut query_lengths = Vec::new();
        let mut block_tables = Vec::new();

        let mut num_prefills = 0;
        let mut num_prefill_tokens = 0;
        let mut num_decode_tokens = 0;

        for sequence_group_metadata in sequence_groups_metadata.iter() {
            let is_prompt = sequence_group_metadata.is_prompt;

            for (sequence_id, sequence_data) in sequence_group_metadata.sequence_data.iter() {
                // 1. Context length
                let context_length = if is_prompt {
                    sequence_data.get_num_computed_tokens()
                } else {
                    // NOTE: If we ever want to introduce speculative
                    // decoding in the future, this is invalid
                    // for it, so one needs to introduce additional
                    // logic.
                    sequence_data.length() - 1
                };

                // 2. Sequence length
                let sequence_length = sequence_data
                    .length()
                    .min(context_length + sequence_group_metadata.token_chunk_size);

                // 3. Tokens
                let tokens = if is_prompt {
                    &sequence_data.get_token_ids()[context_length..sequence_length]
                } else {
                    if sequence_data.get_last_token_id().is_none() {
                        error!("Empty prompts should not be received in `ModelWorker`");
                        return Err(ModelWorkerError::EmptyPrompt(
                            "Empty prompts should not be received in `ModelWorker`".into(),
                        ));
                    }
                    // DON'T PANIC: we should not receive empty prompts
                    let last_token_id = sequence_data.get_last_token_id().unwrap();
                    &[last_token_id]
                };

                // 4. Query length
                let query_length = if is_prompt {
                    sequence_length - context_length
                } else {
                    1
                };

                // 5. Update previous values if sliding window is used

                // These are seq_len/context_len capped to the sliding window.
                // They are passed to decode kernel.
                // We still need original seq_len/context_len to compute slot
                // mapping (and input position) below.
                let mut sliding_sequence_length = sequence_length;
                let sliding_context_length = context_length;

                // This is a hack to make sliding window work with
                // Paged Attention. We can remove it if we make paged attn kernel
                // to properly handle sliding window attention.
                if self.model.sliding_window().is_some() && !is_prompt {
                    // DON'T PANIC: by the branch check
                    sliding_sequence_length =
                        self.model.sliding_window().unwrap().min(sequence_length);
                }

                // 6. Get block table for the current sequence
                let block_table = if self.enable_chunked_prefill || !is_prompt {
                    // DON'T PANIC: Unwrap is safe here because block_tables
                    // should have allocated a block table for this sequence
                    let mut block_table = sequence_group_metadata
                        .block_tables
                        .get(sequence_id)
                        .expect("Block table should be allocated for sequence on decoding phase")
                        .clone();

                    // 7. If sliding window is used, we need to trim the block table
                    if let Some(sliding_window) = self.model.sliding_window() {
                        let sw_block_num = (sliding_window + self.cache_engine.block_size - 1)
                            / self.cache_engine.block_size;
                        let start = block_table.len().saturating_sub(sw_block_num);
                        block_table = block_table[start..].to_vec();
                    }

                    block_table
                } else {
                    // Prefill without chunked prefill
                    vec![]
                };

                // 8. Update intermediate states
                block_tables.push(block_table);
                sequence_lengths.push(sliding_sequence_length as u32);
                context_lengths.push(sliding_context_length as u32);

                query_lengths.push(query_length as u32);
                input_tokens.extend(tokens);
                input_positions.extend((context_length as i64)..(sequence_length as i64));

                // 9. Update intermediate states depending on the type of the sequence
                //    (prompt or decode)
                if is_prompt {
                    debug_assert_eq!(
                        sequence_group_metadata.sequence_data.len(),
                        1,
                        "Prompt should have only one sequence ID"
                    );
                    num_prefills += 1;
                    num_prefill_tokens += tokens.len();
                    prefill_sequence_lengths.push(sequence_length);
                } else {
                    debug_assert_eq!(
                        query_length, 1,
                        "Invalid query length: seq_len: {}, context_len: {}, query_len: {}",
                        sequence_length, context_length, query_length
                    );
                    num_decode_tokens += query_length;
                    decode_sequence_lengths.push(sliding_sequence_length);
                }

                if sequence_group_metadata.block_tables.is_empty() {
                    // During memory profiling, the block tables are not
                    // initialized yet. In this case, we just use a dummy
                    // slot mapping.
                    // In embeddings, the block tables are {seq_id: None}.
                    slot_mapping.extend(vec![PAD_SLOT_ID; sequence_length]);
                    continue;
                }

                // 10. Compute the slot mapping.
                let block_table = sequence_group_metadata
                    .block_tables
                    .get(sequence_id)
                    .expect("Block table should exist for a sequence on decoding phase");

                // Mask the [0, start_idx) tokens of the prompt with
                // _PAD_SLOT_ID, where start_idx is max(0, seq_len -
                // sliding_window). For example, if the prompt len is 10,
                // sliding window is 8, and block size is 4, the first two
                // tokens are masked and the slot mapping will be
                // [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
                let start_index = self
                    .model
                    .sliding_window()
                    .map(|sw| query_length.saturating_sub(sw))
                    .unwrap_or(0);

                slot_mapping.extend((context_length..sequence_length).map(|i| {
                    if i < start_index {
                        PAD_SLOT_ID
                    } else {
                        let block_number = block_table[i / self.cache_engine.block_size];
                        let block_offset = i % self.cache_engine.block_size;
                        ((block_number as usize) * self.cache_engine.block_size + block_offset)
                            as i64
                    }
                }));
            }
        }

        // 11. Build the required tensors for attention metadata
        let max_query_len = *query_lengths.iter().max().unwrap_or(&0) as usize;
        let max_prefill_seq_len = *prefill_sequence_lengths.iter().max().unwrap_or(&0);
        let max_decode_seq_len = *decode_sequence_lengths.iter().max().unwrap_or(&0);

        let max_block_table_len = block_tables.iter().map(|bt| bt.len()).max().unwrap();
        let block_tables_tensor =
            utils::make_tensor_with_pad(block_tables, max_block_table_len, 0u32, &self.device)?;

        let seq_lens_tensor = Tensor::new(sequence_lengths, &self.device)?;
        let mut seq_start_loc =
            Tensor::zeros(seq_lens_tensor.dims1()? + 1, DType::U32, &self.device)?;

        let cumsum_seq_lens = seq_lens_tensor
            .to_dtype(DType::F32)?
            .cumsum(0)?
            .to_dtype(DType::U32)?;
        seq_start_loc = seq_start_loc.slice_assign(&[1..], &cumsum_seq_lens)?;

        let input_tokens_tensor = Tensor::new(input_tokens, &self.device)?.unsqueeze(0)?;
        let input_positions_tensor = Tensor::new(input_positions, &self.device)?.unsqueeze(0)?;
        let slot_mapping_tensor = Tensor::new(slot_mapping, &self.device)?;

        let context_lens_tensor = Tensor::new(context_lengths, &self.device)?;
        let query_lens_tensor = Tensor::new(query_lengths, &self.device)?.to_dtype(DType::F32)?;
        let mut query_start_loc =
            Tensor::zeros(query_lens_tensor.dims1()? + 1, DType::U32, &self.device)?;

        let cumsum_query_lens = query_lens_tensor.cumsum(0)?.to_dtype(DType::U32)?;
        query_start_loc = query_start_loc.slice_assign(&[1..], &cumsum_query_lens)?;

        let attention_metadata = FlashAttentionMetadata::new(
            context_lens_tensor,
            slot_mapping_tensor,
            query_start_loc.clone(),
            num_prefill_tokens,
            num_decode_tokens,
            max_query_len,
            max_decode_seq_len,
            max_prefill_seq_len,
            num_prefills,
            seq_start_loc,
            seq_lens_tensor,
            block_tables_tensor,
            false, // TODO: this parameter should be configurable
        )?;

        Ok(ModelInput {
            input_tokens_tensor,
            input_positions: input_positions_tensor,
            num_decode_tokens,
            num_prefills,
            cu_query_lengths: query_start_loc,
            attention_metadata,
        })
    }
}

#[derive(Debug, Error)]
pub enum ModelWorkerError {
    #[error("Candle error: `{0}`")]
    CandleError(#[from] CandleError),
    #[error("Model loader error: `{0}`")]
    ModelLoader(#[from] ModelLoaderError),
    #[error("Cache engine error: `{0}`")]
    CacheEngineError(#[from] CacheEngineError),
    #[error("Prefill chunked error: `{0}`")]
    InvalidChunkedPrefill(String),
    #[error("Empty prompt error: `{0}`")]
    EmptyPrompt(String),
    #[error("Invalid number sequences error: `{0}`")]
    InvalidNumberSequences(String),
    #[error("Model executor error: `{0}`")]
    ModelExecutorError(#[from] ModelExecutorError),
}

/// `CacheEngine` - Manages the KV cache.
///
/// This class is responsible for initializing and managing the GPU and CPU KV
/// caches. It also provides methods for performing KV cache operations, such
/// as swapping and copying.
pub struct CacheEngine {
    /// Block size
    block_size: usize,
    /// Model's Cache dtype
    dtype: DType,
    /// Number of layers
    num_layers: usize,
    /// Number of CPU blocks
    num_cpu_blocks: usize,
    /// Number of GPU blocks
    num_gpu_blocks: usize,
    /// Flash attention backend,
    /// compatible with paged attention
    attention: FlashAttention,
    /// The CPU cache
    cpu_cache: Vec<Tensor>,
    /// The GPU cache
    gpu_cache: Vec<Tensor>,
    /// Tracing span
    span: Span,
}

impl CacheEngine {
    /// Constructor
    #[instrument(skip_all)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        block_size: usize,
        device: Device,
        dtype: DType,
        alibi_slopes: Option<&Tensor>,
        head_dim: usize,
        num_attention_heads: usize,
        num_layers: usize,
        num_kv_heads: usize,
        num_cpu_blocks: usize,
        num_gpu_blocks: usize,
        softmax_scale: f32,
        sliding_window: Option<usize>,
    ) -> Result<Self, CacheEngineError> {
        info!("Starting a new `CacheEngine` instance");
        let mut this = Self {
            block_size,
            dtype,
            num_layers,
            num_cpu_blocks,
            num_gpu_blocks,
            attention: FlashAttention::new(
                num_attention_heads,
                num_kv_heads,
                head_dim,
                softmax_scale,
                alibi_slopes.cloned(),
                sliding_window,
                dtype,
                device.clone(),
            )?,
            cpu_cache: vec![],
            gpu_cache: vec![],
            span: info_span!("cache-engine"),
        };

        this.cpu_cache = this.allocate_blocks(this.num_cpu_blocks, &Device::Cpu)?;
        this.gpu_cache = this.allocate_blocks(this.num_gpu_blocks, &device)?;

        Ok(this)
    }

    /// Allocates KV cache blocks, on the specified blocks
    #[instrument(skip_all)]
    fn allocate_blocks(
        &mut self,
        num_blocks: usize,
        device: &Device,
    ) -> Result<Vec<Tensor>, CacheEngineError> {
        let _enter = self.span.enter();
        let kv_cache_shape = FlashAttention::get_kv_cache_shape(
            num_blocks,
            self.block_size,
            self.attention.num_kv_heads,
            self.attention.head_dim,
        );
        let mut kv_caches = Vec::with_capacity(self.num_layers);
        for _ in 0..self.num_layers {
            kv_caches.push(Tensor::zeros(kv_cache_shape.clone(), self.dtype, device)?);
        }

        Ok(kv_caches)
    }

    /// Swaps CPU blocks into GPU physical blocks
    #[instrument(skip_all)]
    pub fn swap_in(
        &mut self,
        blocks_to_swap_in: &HashMap<u32, u32>,
    ) -> Result<(), CacheEngineError> {
        let _enter = self.span.enter();
        for i in 0..self.num_layers {
            self.attention.swap_blocks(
                &self.cpu_cache[i],
                &mut self.gpu_cache[i],
                blocks_to_swap_in,
            )?
        }
        Ok(())
    }

    /// Swaps GPU blocks out to CPU
    #[instrument(skip_all)]
    pub fn swap_out(
        &mut self,
        blocks_to_swap_out: &HashMap<u32, u32>,
    ) -> Result<(), CacheEngineError> {
        let _enter = self.span.enter();
        for i in 0..self.num_layers {
            self.attention.swap_blocks(
                &self.gpu_cache[i],
                &mut self.cpu_cache[i],
                blocks_to_swap_out,
            )?
        }
        Ok(())
    }

    /// Copy blocks
    #[instrument(skip_all)]
    pub fn copy_blocks(&mut self, blocks_to_copy: Tensor) -> Result<(), CacheEngineError> {
        let _enter = self.span.enter();
        Ok(FlashAttention::copy_blocks(
            &mut self.gpu_cache,
            blocks_to_copy,
        )?)
    }
}

#[derive(Debug, Error)]
pub enum CacheEngineError {
    #[error("Candle error: `{0}`")]
    CandleError(#[from] CandleError),
    #[error("DType parse error: `{0}`")]
    DTypeParseError(#[from] DTypeParseError),
}

pub(crate) mod utils {
    use candle_core::{Device, IndexOp, Tensor, WithDType};

    use super::ModelWorkerError;

    pub(crate) fn make_tensor_with_pad<D: WithDType>(
        x: Vec<Vec<D>>,
        max_length: usize,
        pad: D,
        device: &Device,
    ) -> Result<Tensor, ModelWorkerError> {
        let mut padded_output = Vec::new();
        for mut x_i in x {
            x_i.extend([pad].repeat(max_length - x_i.len()));
            let shape = (1, x_i.len());
            padded_output.push(Tensor::from_vec(x_i, shape, device)?);
        }
        Ok(Tensor::cat(&padded_output[..], 0)?)
    }

    /// Computes selected token indices, for each sequence in the batch.
    /// For a given sequence, the associated selected token index should
    /// correspond to the right end of the sequence, in the output tensor
    pub(crate) fn compute_selected_token_indices(
        cumulative_query_lengths: &Tensor,
    ) -> Result<Tensor, ModelWorkerError> {
        let length = cumulative_query_lengths.dims()[0] - 1;
        let ones = Tensor::ones(
            (length,),
            cumulative_query_lengths.dtype(),
            cumulative_query_lengths.device(),
        )?;
        Ok(cumulative_query_lengths.i(1..)?.sub(&ones)?)
    }
}
