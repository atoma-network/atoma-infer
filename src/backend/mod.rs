use candle::CustomOp1;

/// `PagedAttention` - Backend to run
/// Paged Attention based attention cuda kernels
pub struct PagedAttention {
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    sequence_lengths: Tensor,
    max_sequence_length: i64,
    kv_cache_dtype: String,
    num_kv_heads: i64,
    scale: f32,
    alibi_slopes: Option<Tensor>,
    kv_scale: f32,
}

impl CustomOp1 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention"
    }   

    fn cpu_fwd(&self, storage: &candle::CpuStorage, layout: &candle::Layout) -> candle::Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("PagedAttention is not implemented for CPU");
    }

    fn cuda_fwd(&self, storage: &candle::CudaStorage, layout: &candle::Layout) -> candle::Result<(candle::CudaStorage, candle::Shape)> {
        
    }
}