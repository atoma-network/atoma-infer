use crate::ffi;
use crate::ops::{SwapBlockCpuToGpuOp, SwapBlockGpuToCpuOp, SwapBlockOp};
use candle_core::CpuStorage;
use candle_core::{
    backend::BackendStorage,
    cuda::{cudarc::driver::DeviceSlice, CudaStorageSlice},
    cuda_backend::cudarc::driver::DevicePtr,
    DType, Device, IndexOp, Result, Tensor,
};
use half::{bf16, f16};
use std::collections::HashMap;

/// Swaps blocks from `src` to `dst` tensors, using the block_mapping.
/// Both `src` and `dst` tensors must have the same dtype, and either be on
/// the same cuda device, or one in either cpu and the other in a cuda device.
/// Moreover, both `src` and `dst` have shape `[num_blocks, block_size, num_kv_heads, head_size]`,
/// where `num_blocks` is the total number of blocks available for the current device.
pub fn swap_blocks(
    src: &Tensor,
    dst: &mut Tensor,
    block_mapping: &HashMap<u32, u32>,
) -> Result<()> {
    let t_size_in_bytes = src.dtype().size_in_bytes();
    // NOTE: the rhs of * should be equivalent to src.i(0)?.elem_count()
    // but in this way, we do not need to clone the underlying `Tensor`
    let block_size_in_bytes =
        src.dtype().size_in_bytes() * src.dims()[1..].iter().product::<usize>();
    let src_device = src.device();
    let dst_device = dst.device();
    match (src_device, dst_device) {
        (Device::Cuda(src_device), Device::Cuda(dst_device)) => {
            if src_device.ordinal() != dst_device.ordinal() {
                candle_core::bail!(
                    "swap_blocks: Both src and dst tensors should be on the same device to swap"
                )
            }

            for (src_block, dst_block) in block_mapping.iter() {
                let swap_op = SwapBlockOp {
                    block_size_in_bytes,
                    src_offset: (*src_block as usize) * block_size_in_bytes,
                    dst_offset: (*dst_block as usize) * block_size_in_bytes,
                };
                dst.inplace_op2(src, &swap_op)?;
            }
        }
        (Device::Cpu, Device::Cuda(_)) => {
            let (src, _src_l) = src.storage_and_layout();
            let src_slice = match &*src {
                candle_core::Storage::Cpu(CpuStorage::BF16(ref src_c)) => {
                    crate::ops::utils::cast_slice(src_c.as_slice())
                }
                candle_core::Storage::Cpu(CpuStorage::F16(ref src_c)) => {
                    crate::ops::utils::cast_slice(src_c.as_slice())
                }
                _ => {
                    candle_core::bail!(
                        "swap_blocks: Invalid combination of src and dst tensors storage to swap"
                    )
                }
            };

            for (src_block, dst_block) in block_mapping.iter() {
                let src_offset = (*src_block as usize) * block_size_in_bytes;
                let dst_offset = (*dst_block as usize) * block_size_in_bytes;
                let swap_block_cpu_to_gpu_op = SwapBlockCpuToGpuOp {
                    src_slice: &src_slice[src_offset..src_offset + block_size_in_bytes],
                    block_size_in_bytes,
                    src_offset,
                    dst_offset,
                };
                dst.inplace_op1(&swap_block_cpu_to_gpu_op)?;
            }
        }
        (Device::Cuda(src_device), Device::Cpu) => {
            let (src, src_l) = src.storage_and_layout();
            let src_slice = match &*src {
                candle_core::Storage::Cuda(src_c) => match &src_c.slice {
                    CudaStorageSlice::BF16(src_c) => unsafe {
                        src_c.transmute::<u8>(src_c.num_bytes()).ok_or_else(|| {
                            candle_core::Error::Cuda(
                                "swap_blocks: unable to transmute src_c".to_string().into(),
                            )
                        })?
                    },
                    CudaStorageSlice::F16(src_c) => unsafe {
                        src_c.transmute::<u8>(src_c.num_bytes()).ok_or_else(|| {
                            candle_core::Error::Cuda(
                                "swap_blocks: unable to transmute src_c".to_string().into(),
                            )
                        })?
                    },
                    _ => {
                        candle_core::bail!(
                            "swap_blocks:Invalid dtype for cuda src tensor, expected f16/bf16, got {:?}",
                            src_c.dtype()
                        )
                    }
                },
                _ => {
                    candle_core::bail!(
                        "swap_blocks: Invalid combination of src and dst tensors storage to swap"
                    )
                }
            };

            // NOTE: We need to do the conversion here, as we cast the slice to u8,
            // but the layout is still in the original dtype.
            let src_slice = src_slice.slice(src_l.start_offset() * t_size_in_bytes..);
            for (src_block, dst_block) in block_mapping.iter() {
                let src_offset = (*src_block as usize) * block_size_in_bytes;
                let dst_offset = (*dst_block as usize) * block_size_in_bytes;
                let swap_block_gpu_to_cpu_op = SwapBlockGpuToCpuOp {
                    src_slice: src_slice.slice(src_offset..src_offset + block_size_in_bytes),
                    cuda_device: src_device,
                    block_size_in_bytes,
                    dst_offset,
                };
                dst.inplace_op1(&swap_block_gpu_to_cpu_op)?;
            }
        }
        _ => {
            candle_core::bail!("swap_blocks: Either src and dst are on the same cuda device, or src and dst are on cpu and cuda devices, alternately")
        }
    }

    Ok(())
}

/// Launches the `copy_blocks_kernel` on the given `key_caches` and `value_caches`,
/// following the `block_mapping`, to copy the blocks on both `key_cache` and `value_cache`.
///
/// # Note: For each block_pair in `block_mapping`, `[src_block_index, dst_block_index]`,
/// both `src_block_index` blocks in the `key_cache` and `value_cache` are copied to the
/// `dst_block_index` blocks in the `key_cache` and `value_cache`.
///
/// # Arguments
///
///  * `key_caches` - A vector of `Tensor`s to copy the blocks to.
///  * `value_caches` - A vector of `Tensor`s to copy the blocks to.
///  * `block_mapping` - A `Tensor` of shape `[num_pairs, 2]` that maps the block indices
///  *  to be copied, where `num_pairs` is the number of block pairs to be copied.
///      The `block_mapping` tensor if of dtype `u32`.
///
/// # Safety
///
/// Unsafe due to dangling CUDA pointers
pub unsafe fn copy_blocks(
    key_caches: &[&mut Tensor],
    value_caches: &[&mut Tensor],
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
    key_caches: &[&mut Tensor],
    value_caches: &[&mut Tensor],
    block_mapping: Tensor,
) -> Result<()> {
    let device = key_caches[0].device();
    let cuda_device = if let Device::Cuda(device) = device {
        device
    } else {
        candle_core::bail!("device must be a cuda device")
    };
    let num_layers = key_caches.len();
    if num_layers != value_caches.len() {
        candle_core::bail!("key_caches and value_caches must have the same length")
    }
    if num_layers == 0 {
        return Ok(());
    }

    let device = key_caches[0].device();
    let cuda_device = if let Device::Cuda(device) = device {
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
        let (key_cache_storage, key_cache_layout) = key_cache.storage_and_layout();
        let (value_cache_storage, value_cache_layout) = value_cache.storage_and_layout();
        let key_cache_ptr = match &*key_cache_storage {
            candle_core::Storage::Cuda(c) => {
                let cuda_slice = c.as_cuda_slice::<T>()?;
                let cuda_slice = cuda_slice.slice(key_cache_layout.start_offset()..);
                *cuda_slice.device_ptr() as i64
            }
            _ => candle_core::bail!("key_caches must be a cuda tensor"),
        };
        let value_cache_ptr = match &*value_cache_storage {
            candle_core::Storage::Cuda(c) => {
                let cuda_slice = c.as_cuda_slice::<T>()?;
                let cuda_slice = cuda_slice.slice(value_cache_layout.start_offset()..);
                *cuda_slice.device_ptr() as i64
            }
            _ => candle_core::bail!("value_caches must be a cuda tensor"),
        };
        key_cache_ptrs.push(key_cache_ptr);
        value_cache_ptrs.push(value_cache_ptr);
    }

    let key_cache_ptrs = Tensor::from_vec(key_cache_ptrs, (num_layers,), device)?;
    let value_cache_ptrs = Tensor::from_vec(value_cache_ptrs, (num_layers,), device)?;
    let key_cache_ptrs = {
        let (key_cache_ptrs_s, key_cache_ptrs_l) = key_cache_ptrs.storage_and_layout();
        match &*key_cache_ptrs_s {
            candle_core::Storage::Cuda(c) => {
                let cuda_slice = c.as_cuda_slice::<i64>()?;
                let cuda_slice = cuda_slice.slice(key_cache_ptrs_l.start_offset()..);
                *cuda_slice.device_ptr() as *const core::ffi::c_void
            }
            _ => candle_core::bail!("key_caches must be a cuda tensor"),
        }
    };
    let value_cache_ptrs = {
        let (value_cache_ptrs_s, _value_cache_ptrs_l) = value_cache_ptrs.storage_and_layout();
        match &*value_cache_ptrs_s {
            candle_core::Storage::Cuda(c) => {
                let cuda_slice = c.as_cuda_slice::<i64>()?;
                let cuda_slice = cuda_slice.slice(value_cache_ptrs.layout().start_offset()..);
                *cuda_slice.device_ptr() as *const core::ffi::c_void
            }
            _ => candle_core::bail!("value_caches must be a cuda tensor"),
        }
    };
    let num_pairs = block_mapping.dims()[0];

    if [num_pairs, 2] != block_mapping.shape().dims() {
        candle_core::bail!("block_mapping must have shape [num_pairs, 2]")
    }

    let (block_mapping_storage, block_mapping_layout) = block_mapping.storage_and_layout();
    let block_mapping_ptr = match &*block_mapping_storage {
        candle_core::Storage::Cuda(c) => {
            let cuda_slice = c.as_cuda_slice::<i64>()?;
            let cuda_slice = cuda_slice.slice(block_mapping_layout.start_offset()..);
            *cuda_slice.device_ptr() as *const core::ffi::c_void
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

    let stream = cuda_device
        .fork_default_stream()
        .map_err(|e| candle_core::Error::Cuda(e.into()))?;

    match dtype {
        DType::F16 => unsafe {
            ffi::copy_blocks_f16(
                key_cache_ptrs,
                value_cache_ptrs,
                block_mapping_ptr,
                num_layers as i64,
                num_pairs as i64,
                numel_per_block,
                stream.stream as *mut std::ffi::c_void,
            );
        },
        DType::BF16 => unsafe {
            ffi::copy_blocks_bf16(
                key_cache_ptrs,
                value_cache_ptrs,
                block_mapping_ptr,
                num_layers as i64,
                num_pairs as i64,
                numel_per_block,
                stream.stream as *mut std::ffi::c_void,
            );
        },
        _ => {
            candle_core::bail!("Only support f16/bf16 dtypes and src and dst must have same dtype")
        }
    }

    cuda_device
        .synchronize()
        .map_err(|e| candle_core::Error::Cuda(e.into()))?;

    Ok(())
}

/// Launches the `reshape_and_cache_kernel_flash` on the given `key_caches` and `value_caches`,
/// respecting a slot mapping.
///
/// # Arguments
///
///  * `key` - A `Tensor` of shape `[num_tokens, num_heads, head_size]`.
///  * `value` - A `Tensor` of shape `[num_tokens, num_heads, head_size]`.
///  * `key_cache` - A `Tensor` of shape `[num_blocks, block_size, num_heads, head_size]`.
///  * `value_cache` - A `Tensor` of shape `[num_blocks, block_size, num_heads, head_size]`.
///  * `slot_mapping` - A `Tensor` of shape `[num_tokens]`.
pub fn reshape_and_cache_flash(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    if key.dtype() != value.dtype()
        || key.dtype() != key_cache.dtype()
        || key.dtype() != value_cache.dtype()
    {
        candle_core::bail!("Only support f16/bf16 dtypes and key, value, key_cache and value_cache must have same dtype")
    }

    match key.dtype() {
        DType::F16 => {
            reshape_and_cache_flash_t::<f16>(key, value, key_cache, value_cache, slot_mapping)?
        }
        DType::BF16 => {
            reshape_and_cache_flash_t::<bf16>(key, value, key_cache, value_cache, slot_mapping)?
        }
        _ => {
            candle_core::bail!("Only support f16/bf16 dtypes must have same dtype")
        }
    }
    Ok(())
}

/// Launches the `reshape_and_cache_kernel_flash` on the given `key_caches` and `value_caches`,
/// respecting a slot mapping.
fn reshape_and_cache_flash_t<
    T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let device = key.device();
    let cuda_device = if let Device::Cuda(device) = device {
        device
    } else {
        candle_core::bail!("device must be a cuda device")
    };

    match (value.device(), key_cache.device(), value_cache.device()) {
        (
            Device::Cuda(v_cuda_device),
            Device::Cuda(kc_cuda_device),
            Device::Cuda(vc_cuda_device),
        ) => {
            if v_cuda_device.ordinal() != kc_cuda_device.ordinal()
                || v_cuda_device.ordinal() != vc_cuda_device.ordinal()
            {
                candle_core::bail!("value, key_cache and value_cache must be on the same device")
            }
        }
        _ => candle_core::bail!("value, key_cache and value_cache must be on the same device"),
    }

    let dtype = match key.dtype() {
        DType::F16 => 0,
        DType::BF16 => 1,
        _ => {
            candle_core::bail!("Only support f16/bf16 dtypes and src and dst must have same dtype")
        }
    };
    let num_tokens = key.dims()[0];
    let num_heads = key.dims()[1];
    let head_size = key.dims()[2];
    let num_blocks = key_cache.dims()[0];
    let block_size = key_cache.dims()[1];

    let key_stride = key.stride()[0];
    let value_stride = value.stride()[0];
    let block_stride = key_cache.stride()[0];
    let k_rank = key.rank();
    let v_rank = value.rank();
    let kc_rank = key_cache.rank();
    let vc_rank = value_cache.rank();

    if block_stride != value_cache.stride()[0] {
        candle_core::bail!(
            "Only support block_stride == value_cache.stride[0] (got block_stride {} and value_cache.stride[0] {})", 
            block_stride,
            value_cache.stride()[0]
        )
    }
    if k_rank != 3 || v_rank != 3 {
        candle_core::bail!(
            "Only support key and value tensors with rank 3 (got {} and v_rank {})",
            k_rank,
            v_rank
        )
    }
    if kc_rank != 4 {
        candle_core::bail!(
            "Only support key_cache tensors with rank 4 (got {})",
            kc_rank
        )
    }
    if vc_rank != 4 {
        candle_core::bail!(
            "Only support value_cache tensors with rank 4 (got {})",
            vc_rank
        )
    }
    if [num_blocks, block_size, num_heads, head_size] != key_cache.dims() {
        candle_core::bail!(
            "Only support key_cache with shape [{num_blocks}, {block_size}, {num_heads}, {head_size}] (got {:?})",
            key_cache.dims()
        )
    }
    if [num_blocks, block_size, num_heads, head_size] != value_cache.dims() {
        candle_core::bail!(
            "Only support value_cache with shape [{num_blocks}, {block_size}, {num_heads}, {head_size}] (got {:?})",
            value_cache.dims()
        )
    }
    if [num_tokens, num_heads, head_size] != value.dims() {
        candle_core::bail!(
            "Only support value with shape [{num_tokens}, {num_heads}, {head_size}] (got {:?})",
            value.dims()
        )
    }
    if (num_tokens) != slot_mapping.dims1()? {
        candle_core::bail!(
            "Only support slot_mapping with shape [{num_tokens}] (got {:?})",
            slot_mapping.dims1()
        )
    }

    let (k, k_l) = key.storage_and_layout();
    let (v, v_l) = value.storage_and_layout();
    let (kc, kc_l) = key_cache.storage_and_layout();
    let (vc, vc_l) = value_cache.storage_and_layout();
    let (slot_mapping, slot_mapping_l) = slot_mapping.storage_and_layout();

    let k_ptr = match &*k {
        candle_core::Storage::Cuda(c) => {
            let k = c.as_cuda_slice::<T>()?;
            let k = k.slice(k_l.start_offset()..);
            *k.device_ptr() as *const core::ffi::c_void
        }
        _ => candle_core::bail!("key must be a cuda tensor"),
    };
    let v_ptr = match &*v {
        candle_core::Storage::Cuda(c) => {
            let v = c.as_cuda_slice::<T>()?;
            let v = v.slice(v_l.start_offset()..);
            *v.device_ptr() as *const core::ffi::c_void
        }
        _ => candle_core::bail!("value must be a cuda tensor"),
    };
    let kc_ptr = match &*kc {
        candle_core::Storage::Cuda(c) => {
            let kc = c.as_cuda_slice::<T>()?;
            let kc = kc.slice(kc_l.start_offset()..);
            *kc.device_ptr() as *const core::ffi::c_void
        }
        _ => candle_core::bail!("key_cache must be a cuda tensor"),
    };
    let vc_ptr = match &*vc {
        candle_core::Storage::Cuda(c) => {
            let vc = c.as_cuda_slice::<T>()?;
            let vc = vc.slice(vc_l.start_offset()..);
            *vc.device_ptr() as *const core::ffi::c_void
        }
        _ => candle_core::bail!("value_cache must be a cuda tensor"),
    };
    let slot_mapping_ptr = match &*slot_mapping {
        candle_core::Storage::Cuda(c) => {
            let slot_mapping = c.as_cuda_slice::<i64>()?;
            let slot_mapping = slot_mapping.slice(slot_mapping_l.start_offset()..);
            *slot_mapping.device_ptr() as *const i64
        }
        _ => candle_core::bail!("slot_mapping must be a cuda tensor"),
    };

    unsafe {
        ffi::reshape_and_cache_flash(
            k_ptr,
            v_ptr,
            kc_ptr,
            vc_ptr,
            slot_mapping_ptr,
            block_stride as i64,
            num_tokens as i64,
            num_heads as i64,
            head_size as i64,
            block_size as i64,
            key_stride as i64,
            value_stride as i64,
            dtype,
        )
    }

    cuda_device
        .synchronize()
        .map_err(|e| candle_core::Error::Cuda(e.into()))?;

    Ok(())
}
