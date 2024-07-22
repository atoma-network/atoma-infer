use candle_core::{
    backend::BackendStorage,
    cuda::{
        cudarc::driver::{CudaView, DeviceSlice},
        CudaStorageSlice,
    },
    CudaDevice, CudaStorage, InplaceOp1, InplaceOp2, Layout, Result, Tensor,
};
use half::{bf16, f16};

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
        let t_size_in_bytes = src_c.dtype().size_in_bytes();
        let dst_device = dst_c.device().clone();

        // Use a closure to handle the different slice types
        let handle_slices = |src: &CudaStorageSlice, dst: &mut CudaStorageSlice| -> Result<()> {
            let (src_bytes, mut dst_bytes) = match (src, dst) {
                (CudaStorageSlice::BF16(src), CudaStorageSlice::BF16(dst)) => {
                    let src_bytes = unsafe {
                        src.transmute::<u8>(src.num_bytes()).ok_or_else(|| {
                            candle_core::Error::Cuda("unable to transmute src".to_string().into())
                        })?
                    };
                    let dst_bytes = unsafe {
                        dst.transmute_mut::<u8>(dst.num_bytes()).ok_or_else(|| {
                            candle_core::Error::Cuda("unable to transmute dst".to_string().into())
                        })?
                    };
                    (src_bytes, dst_bytes)
                }
                (CudaStorageSlice::F16(src), CudaStorageSlice::F16(dst)) => {
                    let src_bytes = unsafe {
                        src.transmute::<u8>(src.num_bytes()).ok_or_else(|| {
                            candle_core::Error::Cuda("unable to transmute src".to_string().into())
                        })?
                    };
                    let dst_bytes = unsafe {
                        dst.transmute_mut::<u8>(dst.num_bytes()).ok_or_else(|| {
                            candle_core::Error::Cuda("unable to transmute dst".to_string().into())
                        })?
                    };
                    (src_bytes, dst_bytes)
                }
                _ => {
                    candle_core::bail!(
                        "Only support f16/bf16 dtypes and src and dst must have same dtype"
                    )
                }
            };

            let src_bytes = src_bytes.slice(src_l.start_offset() * t_size_in_bytes..);
            let mut dst_bytes = dst_bytes.slice_mut(dst_l.start_offset() * t_size_in_bytes..);

            let src_block = src_bytes.slice(self.src_offset..self.src_offset + self.block_size_in_bytes);
            let mut dst_block = dst_bytes.slice_mut(self.dst_offset..self.dst_offset + self.block_size_in_bytes);

            dst_device
                .dtod_copy(&src_block, &mut dst_block)
                .map_err(|e| candle_core::Error::Cuda(e.to_string().into()))?;

            Ok(())
        };

        // Call the closure with references to the slices
        handle_slices(&src_c.slice, &mut dst_c.slice)
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
        let t_size_in_bytes = dst_c.dtype().size_in_bytes();
        let dst_device = dst_c.device().clone();
        let mut dst_c = match dst_c.slice {
            CudaStorageSlice::BF16(ref mut dst_c) => {
                let mut dst_c = unsafe {
                    dst_c
                        .transmute_mut::<u8>(dst_c.num_bytes())
                        .ok_or_else(|| {
                            candle_core::Error::Cuda("enable to transmute src_c".to_string().into())
                        })?
                };
                dst_c
            }
            CudaStorageSlice::F16(ref mut dst_c) => {
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
        let mut dst_c = dst_c.slice_mut(dst_l.start_offset() * t_size_in_bytes..);
        let mut dst_c =
            dst_c.slice_mut(self.dst_offset..self.dst_offset + self.block_size_in_bytes);

        dst_device
            .htod_sync_copy_into(self.src_slice, &mut dst_c)
            .map_err(|e| candle_core::Error::Cuda(e.to_string().into()))?;

        Ok(())
    }
}

pub struct SwapBlockGpuToCpuOp<'a> {
    pub src_slice: CudaView<'a, u8>,
    pub cuda_device: CudaDevice,
    pub block_size_in_bytes: usize,
    pub src_offset: usize,
    pub dst_offset: usize,
}

impl<'a> InplaceOp1 for SwapBlockGpuToCpuOp<'a> {
    fn name(&self) -> &'static str {
        "swap_block_gpu_to_cpu_op"
    }

    fn cpu_fwd(&self, dst_s: &mut candle_core::CpuStorage, _: &Layout) -> Result<()> {
        let src_device = dst_s.device();
        let dst_s = match dst_s {
            candle_core::CpuStorage::BF16(dst_s) => {
                utils::cast_slice_mut::<bf16>(dst_s.as_mut_slice())
            }
            candle_core::CpuStorage::F16(dst_s) => {
                utils::cast_slice_mut::<f16>(dst_s.as_mut_slice())
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

pub(crate) mod utils {
    pub(crate) fn cast_slice_mut<T>(slice: &mut [T]) -> &mut [u8] {
        let ptr = slice.as_mut_ptr() as *mut u8;
        let len = slice.len() * std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    pub(crate) fn cast_slice<T>(slice: &[T]) -> &[u8] {
        let ptr = slice.as_ptr() as *const u8;
        let len = slice.len() * std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}
