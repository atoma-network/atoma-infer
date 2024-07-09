use candle_core::{
    cuda::cudarc::driver::DevicePtr,
    cuda_backend::{cudarc::driver::DeviceRepr, CudaDType},
    DType, Device, Error as CandleError, IndexOp, Storage, Tensor, WithDType, D,
};
use half::{bf16, f16};

//structure of the metadata
pub struct PagedAttentionMetadata {
    pub prompt_lengths: Vec<usize>,
    pub max_sequence_length: usize,
    pub block_tables: Tensor,
    pub sequence_lengths: Vec<usize>,
    pub sequence_lens_tensor: Tensor,
    pub slot_mapping: Tensor,     // The address to write the new KV to of each token
    pub is_prompt: bool,
    pub kv_cache_dtype: String,
    pub attention_bias: Vec<Option<Tensor>>,
}

impl PagedAttentionMetadata {
    pub fn new(
        prompt_lengths: Vec<usize>,
        max_sequence_length: usize,
        block_tables: Tensor,
        sequence_lengths: Vec<usize>,
        sequence_lens_tensor: Tensor,
        slot_mapping: Tensor,
        kv_cache_dtype: String,
    ) -> Self {
        let is_prompt = !prompt_lengths.is_empty();
        Self {
            prompt_lengths,
            max_sequence_length,
            block_tables,
            sequence_lengths,
            sequence_lens_tensor,
            slot_mapping,
            is_prompt,
            kv_cache_dtype,
            attention_bias: vec![],
        }
    }
}
