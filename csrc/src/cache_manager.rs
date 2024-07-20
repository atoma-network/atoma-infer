use candle_core::backend::BackendDevice;
use candle_core::cuda::cudarc::driver::{CudaSlice, DevicePtr};
use candle_core::cuda_backend::cudarc::driver::CudaStream;
use candle_core::{DType, Device, Result, Tensor};
use cuda_runtime_sys::cudaMemcpyKind;
use half::{bf16, f16};
use std::collections::HashMap;

/// Swaps blocks from `src` to `dst` tensors, through the block_mapping.
/// Both `src` and `dst` tensors must have the same dtype, and either be on
/// the same cuda device, or one in either cpu and the other in a cuda device.
/// Moreover, both `src` and `dst` have shape `[num_blocks, block_size, num_kv_heads, head_size]`,
/// where `num_blocks` is the total number of blocks available for the current device.
pub fn swap_blocks(src: &Tensor, dst: &mut Tensor, block_mapping: HashMap<i64, i64>) -> Result<()> {
    match (src.dtype(), dst.dtype()) {
        (DType::F16, DType::F16) => swap_blocks_t::<f16>(src, dst, block_mapping),
        (DType::BF16, DType::BF16) => swap_blocks_t::<bf16>(src, dst, block_mapping),
        _ => {
            candle_core::bail!("Only support f16/bf16 dtypes and src and dst must have same dtype")
        }
    }
}

/// Swaps blocks from `src` to `dst` tensors, through the block_mapping.
fn swap_blocks_t<
    T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    src: &Tensor,
    dst: &mut Tensor,
    block_mapping: HashMap<i64, i64>,
) -> Result<()> {
    let block_size_in_bytes = (src.dtype().size_in_bytes() * src.dims()[0]);
    let src_device = src.device();
    let dst_device = dst.device();
    match (src_device, dst_device) {
        (Device::Cuda(src_device), Device::Cuda(dst_device)) => {
            if src_device.ordinal() != dst_device.ordinal() {
                candle_core::bail!("Both src and dst tensors should be on the same device to swap")
            }

            let (src, src_l) = src.storage_and_layout();
            let (dst, dst_l) = dst.storage_and_layout();
            let (src_ptr, dst_ptr) = match (&*src, &*dst) {
                (candle_core::Storage::Cuda(src_c), candle_core::Storage::Cuda(dst_c)) => {
                    let src_c = src_c.as_cuda_slice::<T>()?;
                    let dst_c = dst_c.as_cuda_slice::<T>()?;
                    let src_c = src_c.slice(src_l.start_offset()..);
                    let dst_c = dst_c.slice(dst_l.start_offset()..);

                    (*src_c.device_ptr(), *dst_c.device_ptr())
                }
                _ => {
                    candle_core::bail!(
                        "Both src and dst tensors should be on the same cuda device to swap"
                    )
                }
            };

            for (src_block, dst_block) in block_mapping.iter() {
                let src_offset = (*src_block as u64) * (block_size_in_bytes as u64);
                let dst_offset = (*dst_block as u64) * (block_size_in_bytes as u64);
                let src_slice: CudaSlice<u8> = unsafe {
                    src_device.upgrade_device_ptr(src_ptr + src_offset, block_size_in_bytes)
                };
                let mut dst_slice: CudaSlice<u8> = unsafe {
                    dst_device.upgrade_device_ptr(dst_ptr + dst_offset, block_size_in_bytes)
                };
                src_device
                    .dtod_copy(&src_slice, &mut dst_slice)
                    .map_err(|e| candle_core::Error::Cuda(e.to_string().into()))?;
            }
        }
        (Device::Cpu, Device::Cuda(dst_device)) => {
            let (src, src_l) = src.storage_and_layout();
            let (dst, dst_l) = dst.storage_and_layout();
            let (src_slice, dst_ptr) = match (&*src, &*dst) {
                (candle_core::Storage::Cpu(src_c), candle_core::Storage::Cuda(dst_c)) => {
                    let src_c = src_c.as_slice::<u8>()?;
                    let dst_c = dst_c.as_cuda_slice::<T>()?;
                    let dst_c = dst_c.slice(dst_l.start_offset()..);

                    (src_c, *dst_c.device_ptr())
                }
                _ => {
                    candle_core::bail!("Invalid combination of src and dst tensors storage to swap")
                }
            };

            for (src_block, dst_block) in block_mapping.iter() {
                let src_offset = (*src_block as usize) * block_size_in_bytes;
                let dst_offset = (*dst_block as u64) * (block_size_in_bytes as u64);
                let mut dst_slice: CudaSlice<u8> = unsafe {
                    dst_device.upgrade_device_ptr(dst_ptr + dst_offset, block_size_in_bytes)
                };
                dst_device
                    .htod_sync_copy_into(
                        &src_slice[src_offset..src_offset + block_size_in_bytes],
                        &mut dst_slice,
                    )
                    .map_err(|e| candle_core::Error::Cuda(e.to_string().into()))?;
            }
        }
        (Device::Cuda(src_device), Device::Cpu) => {
            let (src, src_l) = src.storage_and_layout();
            let (dst, dst_l) = dst.storage_and_layout();
            let (src_ptr, dst_slice) = match (&*src, &*dst) {
                (candle_core::Storage::Cuda(src_c), candle_core::Storage::Cpu(dst_c)) => {
                    let src_c = src_c.as_cuda_slice::<T>()?;
                    let src_c = src_c.slice(src_l.start_offset()..);
                    let mut dst_c = dst_c.as_slice()?;

                    (*src_c.device_ptr(), dst_c)
                }
                _ => {
                    candle_core::bail!("Invalid combination of src and dst tensors storage to swap")
                }
            };

            for (src_block, dst_block) in block_mapping.iter() {
                let src_offset = (*src_block as u64) * (block_size_in_bytes as u64);
                let dst_offset = (*dst_block as usize) * block_size_in_bytes;
                let src_slice: CudaSlice<u8> = unsafe {
                    src_device.upgrade_device_ptr(src_ptr + src_offset, block_size_in_bytes)
                };
                src_device
                    .dtoh_sync_copy_into(
                        &src_slice,
                        &mut dst_slice.clone()[dst_offset..dst_offset + block_size_in_bytes],
                    )
                    .map_err(|e| candle_core::Error::Cuda(e.into()))?;
            }
        }
        _ => {
            candle_core::bail!("Either src and dst are on the same cuda device, or src and dst are on cpu and cuda devices, alternately")
        }
    }
    Ok(())
}

/// Launches the `copy_blocks_kernel` on the given `key_caches` and `value_caches`,
/// following the `block_mapping`, to copy the blocks on both `key_cache` and `value_cache`.
///
/// # Note:
///
///     For for each block_pair in `block_mapping`, `[src_block_index, dst_block_index]`,
///     both `src_block_index` blocks in the `key_cache` and `value_cache` are copied to the
///     `dst_block_index` blocks in the `key_cache` and `value_cache`.
///
/// # Arguments
///
/// * `key_caches` - A vector of `Tensor`s to copy the blocks to.
/// * `value_caches` - A vector of `Tensor`s to copy the blocks to.
/// * `block_mapping` - A `Tensor` of shape `[num_pairs, 2]` that maps the block indices
///    to be copied, where `num_pairs` is the number of block pairs to be copied.
pub unsafe fn copy_blocks_t<
    T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: Tensor,
) -> Result<()> {
    let num_layers = key_caches.len();
    if num_layers != value_caches.len() {
        candle_core::bail!("key_caches and value_caches must have the same length")
    }
    if num_layers == 0 {
        return Ok(());
    }

    let device = key_caches[0].device();
    if !device.is_cuda() {
        candle_core::bail!("device must be a cuda device")
    }
    if !value_caches[0].device().is_cuda() {
        candle_core::bail!("key_caches and value_caches must be on the same device")
    }
    if key_caches[0].dtype() != value_caches[0].dtype() {
        candle_core::bail!("key_caches and value_caches must have the same dtype")
    }

    let mut key_cache_ptrs = Vec::with_capacity(num_layers);
    let mut value_cache_ptrs = Vec::with_capacity(num_layers);
    for (key_cache, value_cache) in key_caches.iter().zip(value_caches.iter()) {
        let key_cache_storage_and_layout = key_cache.storage_and_layout();
        let value_cache_storage_and_layout = value_cache.storage_and_layout();
        let key_cache_ptr = match &*key_cache_storage_and_layout.0 {
            candle_core::Storage::Cuda(c) => {
                let cuda_slice = c.as_cuda_slice::<T>()?;
                let cuda_slice = cuda_slice.slice(key_cache_storage_and_layout.1.start_offset()..);
                *cuda_slice.device_ptr() as *const core::ffi::c_void
            }
            _ => candle_core::bail!("key_caches must be a cuda tensor"),
        };
        let value_cache_ptr = match &*value_cache_storage_and_layout.0 {
            candle_core::Storage::Cuda(c) => {
                let cuda_slice = c.as_cuda_slice::<T>()?;
                let cuda_slice =
                    cuda_slice.slice(value_cache_storage_and_layout.1.start_offset()..);
                *cuda_slice.device_ptr() as *const core::ffi::c_void
            }
            _ => candle_core::bail!("value_caches must be a cuda tensor"),
        };
        key_cache_ptrs.push(key_cache_ptr);
        value_cache_ptrs.push(value_cache_ptr);
    }

    let num_pairs = block_mapping.dims()[0];

    Ok(())
}
