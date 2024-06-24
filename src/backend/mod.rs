use candle::{
    backend::BackendStorage, cuda::{cudarc::driver::DeviceRepr, CudaDType}, CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape, Tensor
};
use half::{bf16, f16};


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

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("PagedAttention is not implemented for CPU");
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(storage, layout),
            DType::F16 => self.cuda_fwd_t::<f16>(storage, layout),
            DType::BF16 => self.cuda_fwd_t::<bf16>(storage, layout),
            dtype => candle::bail!("Unsupported dtype for paged attention: {}", dtype),
        }
    }
}

impl PagedAttention {
    fn cuda_fwd_t<T: CudaDType + DeviceRepr>(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dtype = storage.dtype();
        let internal_type = match dtype {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::BF16 => 2,
            _ => candle::bail!("Unsupported dtype for paged attention: {}", dtype),
        };

        let device = storage.device();
        let output_shape = layout.shape();
        todo!()
    }
}
