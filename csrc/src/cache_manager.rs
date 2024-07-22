use crate::ffi;
use candle_core::{
    backend::{BackendDevice, BackendStorage},
    cuda::cudarc::driver::{result::stream, DeviceSlice},
    cuda_backend::cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut},
    DType, Device, IndexOp, Result, Tensor,
};
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
            candle_core::bail!(
                "Only support f16/bf16 dtypes and src and dst must have same dtype, src_dtype = {:?}, dst_dtype = {:?}",
                src.dtype(),
                dst.dtype()
            )
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
    let t_size_in_bytes = src.dtype().size_in_bytes();
    let block_size_in_bytes = src.dtype().size_in_bytes() * src.dims()[0];
    let src_device = src.device();
    let dst_device = dst.device();
    match (src_device, dst_device) {
        (Device::Cuda(src_device), Device::Cuda(dst_device)) => {
            if src_device.ordinal() != dst_device.ordinal() {
                candle_core::bail!("Both src and dst tensors should be on the same device to swap")
            }

            let (src, src_l) = src.storage_and_layout();
            let (dst, dst_l) = dst.storage_and_layout();
            let (src_ptr, mut dst_ptr) = match (&*src, &*dst) {
                (candle_core::Storage::Cuda(src_c), candle_core::Storage::Cuda(dst_c)) => {
                    let src_c = src_c.as_cuda_slice::<T>()?;
                    let dst_c = dst_c.as_cuda_slice::<T>()?;
                    let src_c = unsafe {
                        src_c.transmute::<u8>(src_c.num_bytes()).ok_or_else(|| {
                            candle_core::Error::Cuda("enable to transmute src_c".to_string().into())
                        })?
                    };
                    let mut dst_c = unsafe {
                        dst_c
                            .transmute_mut::<u8>(dst_c.num_bytes())
                            .ok_or_else(|| {
                                candle_core::Error::Cuda(
                                    "enable to transmute src_c".to_string().into(),
                                )
                            })?
                    };
                    // NOTE: We need to do the conversion here, as we cast the slice to u8,
                    // but the layout is still in the original dtype.
                    let src_c = src_c.slice(src_l.start_offset() * t_size_in_bytes..);
                    let dst_c = dst_c.slice_mut(dst_l.start_offset() * t_size_in_bytes..);

                    (src_c, dst_c)
                }
                _ => {
                    candle_core::bail!(
                        "Both src and dst tensors should be on the same cuda device to swap"
                    )
                }
            };

            for (src_block, dst_block) in block_mapping {
                let src_offset = (src_block as u64) * (block_size_in_bytes as u64);
                let dst_offset = (dst_block as u64) * (block_size_in_bytes as u64);
                let src_slice = src_ptr.slice(src_offset..src_offset + (block_size_in_bytes as u64));
                let mut dst_slice = dst_ptr.slice_mut(dst_offset..dst_offset + (block_size_in_bytes as u64));
                dst_device
                    .dtod_copy(
                        &src_slice,
                        &mut dst_slice,
                    )
                    .map_err(|e| candle_core::Error::Cuda(e.to_string().into()))?;
                // unsafe {
                //     let err = cuda_runtime_sys::cudaMemcpyAsync(
                //         (dst_ptr as u64 + dst_offset) as *mut core::ffi::c_void,
                //         (src_ptr as u64 + src_offset) as *const core::ffi::c_void,
                //         block_size_in_bytes,
                //         cuda_runtime_sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                //         stream,
                //     );
                //     if err != cuda_runtime_sys::cudaError::cudaSuccess {
                //         candle_core::bail!("cudaMemcpyAsync failed with error: {:?}", err);
                //     }
                // }
                // let src_slice: CudaSlice<u8> = unsafe {
                //     src_device.upgrade_device_ptr(src_ptr + src_offset, block_size_in_bytes)
                // };
                // let mut dst_slice: CudaSlice<u8> = unsafe {
                //     dst_device.upgrade_device_ptr(dst_ptr + dst_offset, block_size_in_bytes)
                // };
                // src_device
                //     .dtod_copy(&src_slice, &mut dst_slice)
                //     .map_err(|e| candle_core::Error::Cuda(e.to_string().into()))?;
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
                    let dst_slice = dst_c.as_slice::<u8>()?;
                    let dst_slice_len = dst_slice.len() * std::mem::size_of::<T>();
                    // SAFETY: this code should run synchronously, therefore we can have
                    // safe mutable access to the slice, once we are reading from it.
                    let dst_slice = unsafe {
                        std::slice::from_raw_parts_mut(dst_slice.as_ptr() as *mut u8, dst_slice_len)
                    };
                    (*src_c.device_ptr(), dst_slice)
                }
                _ => {
                    candle_core::bail!("Invalid combination of src and dst tensors storage to swap")
                }
            };

            for (src_block, dst_block) in block_mapping.iter() {
                let src_offset = (*src_block as u64) * (block_size_in_bytes as u64);
                let dst_offset =
                    (*dst_block as usize) * block_size_in_bytes / std::mem::size_of::<T>();
                let src_slice: CudaSlice<u8> = unsafe {
                    src_device.upgrade_device_ptr(src_ptr + src_offset, block_size_in_bytes)
                };
                let dst_block_slice = &mut dst_slice
                    [dst_offset..dst_offset + block_size_in_bytes / std::mem::size_of::<T>()];
                src_device
                    .dtoh_sync_copy_into(&src_slice, dst_block_slice)
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
pub unsafe fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: Tensor,
) -> Result<()> {
    match (key_caches[0].dtype(), value_caches[0].dtype()) {
        (DType::F16, DType::F16) => copy_blocks_t::<f16>(key_caches, value_caches, block_mapping),
        (DType::BF16, DType::BF16) => {
            copy_blocks_t::<bf16>(key_caches, value_caches, block_mapping)
        }
        _ => {
            candle_core::bail!("Only support f16/bf16 dtypes and src and dst must have same dtype")
        }
    }
}
/// Launches the `copy_blocks_kernel` on the given `key_caches` and `value_caches`,
/// following the `block_mapping`, to copy the blocks on both `key_cache` and `value_cache`.
unsafe fn copy_blocks_t<
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
    let device = if let Device::Cuda(device) = device {
        device
    } else {
        candle_core::bail!("device must be a cuda device")
    };
    if !value_caches[0].device().is_cuda() {
        candle_core::bail!("key_caches and value_caches must be on the same device")
    }
    if key_caches[0].dtype() != value_caches[0].dtype() {
        candle_core::bail!("key_caches and value_caches must have the same dtype")
    }

    let dtype = key_caches[0].dtype();

    let mut key_cache_ptrs = Vec::with_capacity(num_layers);
    let mut value_cache_ptrs = Vec::with_capacity(num_layers);
    for (key_cache, value_cache) in key_caches.iter().zip(value_caches.iter()) {
        let key_cache_storage_and_layout = key_cache.storage_and_layout();
        let value_cache_storage_and_layout = value_cache.storage_and_layout();
        let key_cache_ptr = match &*key_cache_storage_and_layout.0 {
            candle_core::Storage::Cuda(c) => {
                let cuda_slice = c.as_cuda_slice::<T>()?;
                let cuda_slice = cuda_slice.slice(key_cache_storage_and_layout.1.start_offset()..);
                *cuda_slice.device_ptr()
            }
            _ => candle_core::bail!("key_caches must be a cuda tensor"),
        };
        let value_cache_ptr = match &*value_cache_storage_and_layout.0 {
            candle_core::Storage::Cuda(c) => {
                let cuda_slice = c.as_cuda_slice::<T>()?;
                let cuda_slice =
                    cuda_slice.slice(value_cache_storage_and_layout.1.start_offset()..);
                *cuda_slice.device_ptr()
            }
            _ => candle_core::bail!("value_caches must be a cuda tensor"),
        };
        key_cache_ptrs.push(key_cache_ptr);
        value_cache_ptrs.push(value_cache_ptr);
    }

    let key_cache_ptrs = key_cache_ptrs.as_ptr() as *const i64;
    let value_cache_ptrs = value_cache_ptrs.as_ptr() as *const i64;
    let num_pairs = block_mapping.dims()[0];

    if &[num_pairs, 2] != block_mapping.shape().dims() {
        candle_core::bail!("block_mapping must have shape [num_pairs, 2]")
    }

    let (block_mapping_storage, block_mapping_layout) = block_mapping.storage_and_layout();
    let block_mapping_ptr = match &*block_mapping_storage {
        candle_core::Storage::Cuda(c) => {
            let cuda_slice = c.as_cuda_slice::<i64>()?;
            let cuda_slice = cuda_slice.slice(block_mapping_layout.start_offset()..);
            *cuda_slice.device_ptr() as *const i64
        }
        _ => candle_core::bail!("block_mapping must be a cuda tensor"),
    };

    let numel_per_block = key_caches[0]
        .i(0)?
        .shape()
        .dims()
        .iter()
        .product::<usize>()
        .try_into()
        .unwrap();

    // let mut stream: cudaStream_t = std::ptr::null_mut();
    // let cuda_result = cudaStreamCreate(&mut stream);
    // if cuda_result != cudaError::cudaSuccess {
    //     return Err(APIError::new("Failed to create CUDA stream"));
    // }

    let stream = device
        .fork_default_stream()
        .map_err(|e| candle_core::Error::Cuda(e.into()))?;

    match dtype {
        DType::F16 => unsafe {
            ffi::copy_blocks_f16(
                key_cache_ptrs,
                value_cache_ptrs,
                block_mapping_ptr,
                num_layers as i32,
                num_pairs as i32,
                numel_per_block,
                stream.stream as *mut std::ffi::c_void,
            );
        },
        DType::BF16 => unsafe {
            ffi::copy_blocks_bf16(
                key_cache_ptrs,
                value_cache_ptrs,
                block_mapping_ptr,
                num_layers as i32,
                num_pairs as i32,
                numel_per_block,
                stream.stream as *mut std::ffi::c_void,
            );
        },
        _ => {
            candle_core::bail!("Only support f16/bf16 dtypes and src and dst must have same dtype")
        }
    }

    Ok(())
}
