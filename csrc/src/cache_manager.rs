use candle_core::backend::BackendDevice;
use candle_core::cuda::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::cudarc::driver::CudaStream;
use candle_core::{DType, Device, Result, Tensor};
use cuda_runtime_sys::cudaMemcpyKind;

pub fn swap_blocks_t<
    T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    src: &Tensor,
    dst: &mut Tensor,
    block_mapping: HashMap<i64, i64>,
) -> Result<()> {
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
            let (src_ptr, dst_ptr) = match (src, dst) {
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
                let src_offset = (src_block * block_size_in_bytes) as u64;
                let dst_offset = (dst_block * block_size_in_bytes) as u64;
                let src_slice: CudaSlice<u8> = unsafe {
                    src_device.upgrade_device_ptr(src_ptr + src_offset, block_size_in_bytes)
                };
                let dst_slice: CudaSlice<u8> = unsafe {
                    dst_device.upgrade_device_ptr(dst_ptr + dst_offset, block_size_in_bytes)
                };
                src_device.dtod_copy(&src_slice, &mut dst_slice)?;
            }
        }
        (Device::Cpu, Device::Cuda(dst_device)) => {
            let (src, src_l) = src.storage_and_layout();
            let (dst, dst_l) = dst.storage_and_layout();
            let (src_ptr, dst_ptr) = match (src, dst) {
                (candle_core::Storage::Cpu(src_c), candle_core::Storage::Cuda(dst_c)) => {
                    let src_c = src_c.as_slice::<u8>()?;
                    let dst_c = dst_c.as_cuda_slice::<T>()?;
                    let dst_c = dst_c.slice(dst_l.start_offset()..);
                    
                    (src_c, *dst_c.device_ptr())
                }
                _ => {
                    candle_core::bail!(
                        "Invalid combination of src and dst tensors storage to swap"
                    )
                }
            };

            for (src_block, dst_block) in block_mapping.iter() {
                let src_offset = src_block * block_size_in_bytes;
                let dst_offset = (dst_block * block_size_in_bytes) as u64;
                let dst_slice: CudaSlice<u8> = unsafe {
                    dst_device.upgrade_device_ptr(dst_ptr + dst_offset, block_size_in_bytes)
                };
                dst_device.htod_sync_copy_into(&src_slice[src_offset..src_offset + block_size_in_bytes], &mut dst_slice)?;
            }
        }
        (Device::Cuda(src_device), Device::Cpu) => {}
        _ => {
            candle_core::bail!("Either src and dst are on the same cuda device, or src and dst are on cpu and cuda devices, alternately")
        }
    }
    Ok(())
}
