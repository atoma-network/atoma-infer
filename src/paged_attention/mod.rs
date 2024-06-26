use crate::kernels::ffi::{copy_blocks, reshape_and_cache, swap_blocks};
use candle_core::{cuda::cudarc::driver::CudaSlice, Device, Error as CandleError, IndexOp, Layout, Storage, Tensor};
use thiserror::Error;

/// `PagedAttention` - Structure wrapping the CUDA
/// kernels implementing the paged attention memory
/// management algorithm
pub struct PagedAttention {
    num_attention_heads: usize,
    head_dim: usize,
    num_kv_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    num_queries_per_kv: usize,
    alibi_slopes: Option<Tensor>,
}

impl PagedAttention {
    /// Constructor
    pub fn new(
        num_attention_heads: usize,
        head_dim: usize,
        scale: f32,
        num_kv_heads: Option<usize>,
        sliding_window: Option<usize>,
        device: &Device,
        alibi_slopes: Option<Vec<f64>>,
    ) -> Result<Self, CandleError> {
        let num_kv_heads = num_kv_heads.unwrap_or(num_attention_heads);
        let num_queries_per_kv = num_attention_heads / num_kv_heads;
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            Some(Tensor::new(alibi_slopes, device)?)
        } else {
            None
        };
        Ok(Self {
            num_attention_heads,
            head_dim,
            num_kv_heads,
            scale,
            sliding_window,
            num_queries_per_kv,
            alibi_slopes,
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
    pub fn split_kv_cache() {
        todo!()
    }

    /// Initiates a swap blocks operation on the current CUDA device
    pub fn swap_blocks(
        src_kv_cache: Tensor,
        dst_kv_cache: Tensor,
        src_to_dst: Tensor,
    ) -> Result<(), CandleError> {
        // 1. Handle block mapping tensor
        let (block_mapping, block_mapping_layour) = src_to_dst.storage_and_layout();
        let block_mapping = match block_mapping {
            Storage::Cuda(storage) => storage,
            _ => candle_core::bail!("Only CUDA storage is supported"),
        };

        // Get CUDA slices for block_mapping tensor
        let block_mapping_slice = block_mapping.as_cuda_slice()?;
        let block_mapping_view =
            block_mapping_slice.slice(block_mapping_layour.start_offset()..)?;

        // 2. Handle source and destination key_cache tensor
        let src_key_cache = src_kv_cache.i(0)?;
        let dst_key_cache = dst_kv_cache.i(0)?;

        let (src_key_cache_storage, src_key_cache_layout) = src_key_cache.storage_and_layout();
        let src_key_cache = match src_key_cache_storage {
            Storage::Cuda(storage) => storage,
            _ => candle_core::bail!("Only CUDA storage is supported"),
        };

        let (dst_key_cache_storage, dst_key_cache_layout) = dst_key_cache.storage_and_layout();
        let dst_key_cache = match dst_key_cache_storage {
            Storage::Cuda(storage) => storage,
            _ => candle_core::bail!("Only CUDA storage is supported"),
        };

        // Get CUDA slices for both source and destiny key_cache tensors
        let src_key_cache_slice = src_key_cache.as_cuda_slice()?;
        let dst_key_cache_slice = dst_key_cache.as_cuda_slice()?;

        // Get CUDA views for all tensors
        let src_key_cache_view =
            src_key_cache_slice.slice(src_key_cache_layout.start_offset()..)?;
        let dst_key_cache_view =
            dst_key_cache_slice.slice(dst_key_cache_layout.start_offset()..)?;

        unsafe {
            swap_blocks(
                src_key_cache_view as *const core::ffi::c_void,
                dst_key_cache_view as *const core::ffi::c_void,
                block_mapping_view as *const core::ffi::c_void,
            )
        };

        // 3. Handle source and destination value_cache tensor
        let src_value_cache = src_kv_cache.i(1)?;
        let dst_value_cache = dst_kv_cache.i(1)?;

        let (src_value_cache_storage, src_value_cache_layout) =
            src_value_cache.storage_and_layout();
        let src_value_cache = match src_value_cache_storage {
            Storage::Cuda(storage) => storage,
            _ => candle_core::bail!("Only CUDA storage is supported"),
        };

        let (dst_value_cache_storage, dst_value_cache_layout) =
            dst_value_cache.storage_and_layout();
        let dst_value_cache = match dst_value_cache_storage {
            Storage::Cuda(storage) => storage,
            _ => candle_core::bail!("Only CUDA storage is supported"),
        };

        // Get CUDA slices for both source and destiny value_cache tensors
        let src_value_cache_slice = src_value_cache_storage.as_cuda_slice()?;
        let dst_value_cache_slice = dst_value_cache_storage.as_cuda_slice()?;

        // Get CUDA views for all tensors
        let src_value_cache_view =
            src_value_cache_slice.slice(src_value_cache_layout.start_offset()..)?;
        let dst_value_cache_view =
            dst_value_cache_slice.slice(dst_value_cache_layout.start_offset()..)?;

        unsafe {
            swap_blocks(
                src_value_cache_view as *const core::ffi::c_void,
                dst_value_cache_view as *const core::ffi::c_void,
                block_mapping_view as *const core::ffi::c_void,
            )
        };

        Ok(())
    }

    pub fn copy_blocks(kv_caches: Vec<Tensor>, block_mapping: Tensor) -> Result<(), CandleError> {
        // 1. Handle block mapping tensor
        let (block_mapping, block_mapping_layout) = block_mapping.storage_and_layout();
        let block_mapping = match block_mapping {
            Storage::Cuda(storage) => storage,
            _ => candle_core::bail!("Only CUDA storage is supported"),
        };

        // Get CUDA slices for block_mapping tensor
        let block_mapping_slice = block_mapping.as_cuda_slice()?;
        let block_mapping_view =
            block_mapping_slice.slice(block_mapping_layout.start_offset()..)?;
        let key_caches = kv_caches
            .iter()
            .map(|t| t.i(0))
            .collect::<Result<Vec<_>, _>>()?;
        let value_caches = kv_caches
            .iter()
            .map(|t| t.i(1))
            .collect::<Result<Vec<_>, _>>()?;

        let key_caches_length = key_caches.len();
        let value_caches_length = value_caches.len();

        // 2. Handle key_caches and value_caches tensors
        let key_caches = key_caches
            .iter()
            .map(|t| t.storage_and_layout())
            .collect::<Result<Vec<_>, _>>()?;
        let value_caches = value_caches
            .iter()
            .map(|t| t.storage_and_layout())
            .collect::<Result<Vec<_>, _>>()?;

        // Get CUDA slices for all tensors
        let key_caches_slice = key_caches
            .iter()
            .map(|(storage, layout): &(Storage, layout)| match storage {
                Storage::Cuda(storage) => storage.as_cuda_slice().map(|s| (s, layout)),
                _ => candle_core::bail!("Only CUDA storage is supported"),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let value_caches_slice = value_caches
            .iter()
            .map(|(storage, layout): &(Storage, Layout)| 
            match storage {
                Storage::Cuda(storage) => storage.as_cuda_slice().map(|s| (s, layout)),
                _ => candle_core::bail!("Only CUDA storage is supported"),
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Get CUDA views for all tensors
        let key_caches_view = key_caches_slice
            .iter()
            .map(|(slice, layout): &(CudaSlice<_>, Layout)| slice.slice(layout.start_offset()..))
            .collect::<Result<Vec<_>, _>>()?;
        let value_caches_view = value_caches_slice
            .iter()
            .map(|(slice, layout): &(CudaSlice<_>, Layout)| slice.slice(layout.start_offset()..))
            .collect::<Result<Vec<_>, _>>()?;

        unsafe {
            copy_blocks(
                key_caches_view as *const *const core::ffi::c_void,
                key_caches_length,
                value_caches_view as *const *const core::ffi::c_void,
                value_caches_length,
                block_mapping_view as *const core::ffi::c_void,
            )
        }

        Ok(())
    }
}
