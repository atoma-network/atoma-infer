use crate::ffi;
use candle_core::{
    backend::{BackendDevice, BackendStorage},
    cuda::{
        cudarc::driver::{result::stream, DeviceSlice},
        CudaStorageSlice,
    },
    cuda_backend::cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut},
    CudaDevice, CudaStorage, DType, Device, IndexOp, InplaceOp2, Layout, Result, Tensor,
};
use half::{bf16, f16};
use std::{collections::HashMap, sync::RwLockWriteGuard};

/// Swap block operation
/// for two tensors
pub struct SwapBlockOp {
    pub block_size_in_bytes: usize,
    pub src_offset: usize,
    pub dst_offset: usize,
}

impl InplaceOp2 for SwapBlockOp {
    fn name(&self) -> &'static str {
        "swap_block_op"
    }

    fn cpu_fwd(
        &self,
        _: &mut candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<()> {
        Ok(())
    }

    fn cuda_fwd(
        &self,
        dst_c: &mut CudaStorage,
        dst_l: &Layout,
        src_c: &CudaStorage,
        src_l: &Layout,
    ) -> Result<()> {
        let src_device = src_c.device();
        let dst_device = dst_c.device();
        let (src_c, dst_c) = match (src_c.slice, dst_c.slice) {
            (CudaStorageSlice::BF16(src_c), CudaStorageSlice::BF16(dst_c)) => {
                let src_c = unsafe {
                    src_c.transmute::<u8>(src_c.num_bytes()).ok_or_else(|| {
                        candle_core::Error::Cuda("enable to transmute src_c".to_string().into())
                    })?
                };
                let mut dst_c = unsafe {
                    dst_c
                        .transmute_mut::<u8>(dst_c.num_bytes())
                        .ok_or_else(|| {
                            candle_core::Error::Cuda("enable to transmute src_c".to_string().into())
                        })?
                };
                (src_c, dst_c)
            }
            (CudaStorageSlice::F16(src_c), CudaStorageSlice::F16(dst_c)) => {
                let src_c = unsafe {
                    src_c.transmute::<u8>(src_c.num_bytes()).ok_or_else(|| {
                        candle_core::Error::Cuda("enable to transmute src_c".to_string().into())
                    })?
                };
                let mut dst_c = unsafe {
                    dst_c
                        .transmute_mut::<u8>(dst_c.num_bytes())
                        .ok_or_else(|| {
                            candle_core::Error::Cuda("enable to transmute src_c".to_string().into())
                        })?
                };
                (src_c, dst_c)
            }
            _ => {
                candle_core::bail!(
                    "Only support f16/bf16 dtypes and src and dst must have same dtype"
                )
            }
        };

        // NOTE: We need to do the conversion here, as we cast the slice to u8,
        // but the layout is still in the original dtype.
        let src_c = src_c.slice(src_l.start_offset() * t_size_in_bytes..);
        let dst_c = dst_c.slice_mut(dst_l.start_offset() * t_size_in_bytes..);

        let src_c = src_c.slice(self.src_offset..self.src_offset + self.block_size_in_bytes);
        let mut dst_c = dst_c.slice_mut(self.dst_offset..dst_offset + self.block_size_in_bytes);
        dst_device
            .dtod_copy(&src_c, &mut dst_c)
            .map_err(|e| candle_core::Error::Cuda(e.to_string().into()))?;

        Ok((()))
    }
}

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

pub struct SwapBlockCpuToGpuOp<'a> {
    pub src_slice: &'a [u8],
    pub block_size_in_bytes: usize,
    pub src_offset: usize,
    pub dst_offset: usize,
}

impl<'a> InplaceOp1 for SwapBlockCpuToGpuOp<'a> {
    fn name(&self) -> &'static str {
        "swap_block_cpu_to_gpu_op"
    }

    fn cpu_fwd(&self, _: &mut candle_core::CpuStorage, _: &candle_core::Layout) -> Result<()> {
        Ok(())
    }

    fn cuda_fwd(&self, dst_c: &mut CudaStorage, dst_l: &Layout) -> Result<()> {
        let dst_device = dst_c.device();
        let dst_c = match dst_c.slice {
            CudaStorageSlice::BF16(dst_c) => {
                let mut dst_c = unsafe {
                    dst_c
                        .transmute_mut::<u8>(dst_c.num_bytes())
                        .ok_or_else(|| {
                            candle_core::Error::Cuda("enable to transmute src_c".to_string().into())
                        })?
                };
                dst_c
            }
            CudaStorageSlice::F16(dst_c) => {
                let mut dst_c = unsafe {
                    dst_c
                        .transmute_mut::<u8>(dst_c.num_bytes())
                        .ok_or_else(|| {
                            candle_core::Error::Cuda("enable to transmute src_c".to_string().into())
                        })?
                };
                dst_c
            }
            _ => {
                candle_core::bail!(
                    "Only support f16/bf16 dtypes and src and dst must have same dtype"
                )
            }
        };

        // NOTE: We need to do the conversion here, as we cast the slice to u8,
        // but the layout is still in the original dtype.
        let dst_c = dst_c.slice_mut(dst_l.start_offset() * t_size_in_bytes..);
        let mut dst_c = dst_c.slice_mut(self.dst_offset..dst_offset + self.block_size_in_bytes);

        dst_device
            .htod_sync_copy_into(self.src_slice, &mut dst_c)
            .map_err(|e| candle_core::Error::Cuda(e.to_string().into()))?;

        Ok(())
    }
}

pub struct SwapBlockGpuToCpuOp {
    pub src_slice: CudaSlice<u8>,
    pub cuda_device: CudaDevice,
    pub block_size_in_bytes: usize,
    pub src_offset: usize,
    pub dst_offset: usize,
}

impl InplaceOp1 for SwapBlockGpuToCpuOp {
    fn name(&self) -> &'static str {
        "swap_block_gpu_to_cpu_op"
    }

    fn cpu_fwd(&self, dst_s: &mut candle_core::CpuStorage, dst_l: &Layout) -> Result<()> {
        let src_device = dst_s.device();
        let dst_s = match dst_s {
            candle_core::CpuStorage::F16(dst_s) => {
                bytemuck::try_cast_mut::<[f16], [u8]>(dst_s.as_mut_slice())
                    .map_err(|e| candle_core::Error::Cuda(e.into()))?
            }
            candle_core::CpuStorage::BF16(dst_s) => {
                bytemuck::try_cast_mut::<[bf16], [u8]>(dst_s.as_mut_slice())
                    .map_err(|e| candle_core::Error::Cuda(e.into()))?
            }
            _ => {
                candle_core::bail!(
                    "Only support f16/bf16 dtypes and src and dst must have same dtype"
                )
            }
        };

        self.cuda_device
            .dtoh_sync_copy_into(
                &self.src_slice,
                &mut dst_s[self.src_offset..self.src_offset + self.block_size_in_bytes],
            )
            .map_err(|e| candle_core::Error::Cuda(e.into()))?;

        Ok(())
    }
}
