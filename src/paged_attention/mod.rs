use candle::{Error as CandleError, IndexOp, Tensor};
use thiserror::Error;

/// `PagedAttention` - Structure wrapping the CUDA
/// kernels implementing the paged attention memory
/// management algorithm
pub struct PagedAttention {}

impl PagedAttention {
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
        head_size: u32,
    ) -> Vec<usize> {
        vec![2, num_blocks, block_size * num_kv_heads * head_size]
    }

    /// Initiates a swap blocks operation on the CUDA device
    pub fn swap_blocks(
        src_kv_cache: Tensor,
        dst_kv_cache: Tensor,
        src_to_dst: Tensor,
    ) -> Result<(), PagedAttentionError> {
        let src_key_cache = src_kv_cache.i(0)?;
        let dst_key_cache = dst_kv_cache.i(0)?;
        unsafe { swap_blocks(src_key_cache, dst_key_cache) };

        let src_value_cache = src_kv_cache.i(1)?;
        let dst_value_cache = dst_kv_cache.i(1)?;
        unsafe { swap_blocks(src_value_cache, dst_value_cache) };

        Ok(())
    }

    pub fn copy_blocks(
        kv_caches: Vec<Tensor>,
        block_mapping: Tensor,
    ) -> Result<(), PagedAttentionError> {
        let key_caches = kv_caches
            .iter()
            .map(|t| t.i(0))
            .collect::<Result<Vec<_>, _>>()?;
        let value_caches = kv_caches
            .iter()
            .map(|t| t.i(1))
            .collect::<Result<Vec<_>, _>>()?;
        unsafe { copy_blocks(key_caches, value_caches, block_mapping) }
    }
}

#[derive(Debug, Error)]
pub enum PagedAttentionError {
    #[error("Candle error: `{0}`")]
    CandleError(#[from] CandleError),
}
