use candle_core::{DType, Device, IndexOp, Result, Tensor};
use std::collections::HashMap;

const NUM_BLOCKS: usize = 3;
const BLOCK_SIZE: usize = 16;

fn create_random_tensor(device: &Device, dtype: DType) -> Result<Tensor> {
    let tensor = Tensor::arange(0f32, (NUM_BLOCKS * BLOCK_SIZE) as f32, &device)?
        .reshape((NUM_BLOCKS, BLOCK_SIZE, 1, 1))?
        .to_dtype(dtype)?;
    Ok(tensor)
}

#[cfg(test)]
mod swap_blocks {
    use super::*;

    fn verify_swap<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + std::fmt::Debug
            + PartialEq
            + candle_core::WithDType,
    >(
        src: &Tensor,
        original_dst: &Tensor,
        swapped_dst: &Tensor,
        block_mapping: &HashMap<i64, i64>,
    ) -> Result<()> {
        panic!(
            "{:?}",
            original_dst.flatten_all()?.to_vec1::<T>()?
                == swapped_dst.flatten_all()?.to_vec1::<T>()?
        );
        for (src_block, dst_block) in block_mapping {
            let src_slice = src.i(*src_block as usize)?;
            let swapped_dst_slice = swapped_dst.i(*dst_block as usize)?;
            assert_eq!(
                src_slice.flatten_all()?.to_vec1::<T>()?,
                swapped_dst_slice.flatten_all()?.to_vec1::<T>()?,
                "Block {} from source was not correctly swapped to block {} in destination",
                src_block,
                dst_block
            );

            // Check that non-swapped blocks remain unchanged
            for i in 0..NUM_BLOCKS {
                if !block_mapping.values().any(|&v| v == i as i64) {
                    let original_dst_slice = original_dst.i(i)?;
                    let current_dst_slice = swapped_dst.i(i)?;
                    assert_eq!(
                        original_dst_slice.flatten_all()?.to_vec1::<T>()?,
                        current_dst_slice.flatten_all()?.to_vec1::<T>()?,
                        "Block {} in destination should not have changed",
                        i
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_swap_blocks_f16_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let src = create_random_tensor(&device, DType::F16)?;
        let mut dst = create_random_tensor(&device, DType::F16)?;

        let mut block_mapping = HashMap::new();
        block_mapping.insert(0, 2);
        block_mapping.insert(1, 0);

        let original_src = src.clone();
        let original_dst = dst.clone();

        csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

        // verify_swap::<half::f16>(&original_src, &original_dst, &dst, &block_mapping)?;

        Ok(())
    }

    #[test]
    fn test_swap_blocks_bf16_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let src = create_random_tensor(&device, DType::BF16)?;
        let mut dst = create_random_tensor(&device, DType::BF16)?;

        let mut block_mapping = HashMap::new();
        block_mapping.insert(0, 1);
        block_mapping.insert(2, 0);

        let original_src = src.clone();
        let original_dst = dst.clone();

        csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

        verify_swap::<half::bf16>(&original_src, &original_dst, &dst, &block_mapping)?;

        Ok(())
    }

    #[test]
    fn test_swap_blocks_cpu_to_cuda_f16() -> Result<()> {
        let cpu_device = Device::Cpu;
        let cuda_device = Device::new_cuda(0)?;
        let src = create_random_tensor(&cpu_device, DType::F16)?;
        let mut dst = create_random_tensor(&cuda_device, DType::F16)?;

        let mut block_mapping = HashMap::new();
        block_mapping.insert(0, 2);
        block_mapping.insert(1, 1);

        let original_src = src.clone();
        let original_dst = dst.clone();

        csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

        verify_swap::<half::f16>(&original_src, &original_dst, &dst, &block_mapping)?;

        Ok(())
    }

    #[test]
    fn test_swap_blocks_cuda_to_cpu_f16() -> Result<()> {
        let cpu_device = Device::Cpu;
        let cuda_device = Device::new_cuda(0)?;
        let src = create_random_tensor(&cuda_device, DType::F16)?;
        let mut dst = create_random_tensor(&cpu_device, DType::F16)?;

        let mut block_mapping = HashMap::new();
        block_mapping.insert(1, 0);
        block_mapping.insert(2, 2);

        let original_src = src.clone();
        let original_dst = dst.clone();

        csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

        verify_swap::<half::f16>(&original_src, &original_dst, &dst, &block_mapping)?;

        Ok(())
    }

    #[test]
    fn test_swap_blocks_cpu_to_cuda_bf16() -> Result<()> {
        let cpu_device = Device::Cpu;
        let cuda_device = Device::new_cuda(0)?;
        let src = create_random_tensor(&cpu_device, DType::BF16)?;
        let mut dst = create_random_tensor(&cuda_device, DType::BF16)?;

        let mut block_mapping = HashMap::new();
        block_mapping.insert(0, 2);
        block_mapping.insert(1, 1);

        let original_src = src.clone();
        let original_dst = dst.clone();

        csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

        verify_swap::<half::bf16>(&original_src, &original_dst, &dst, &block_mapping)?;

        Ok(())
    }

    #[test]
    fn test_swap_blocks_cuda_to_cpu_bf16() -> Result<()> {
        let cpu_device = Device::Cpu;
        let cuda_device = Device::new_cuda(0)?;
        let src = create_random_tensor(&cuda_device, DType::BF16)?;
        let mut dst = create_random_tensor(&cpu_device, DType::BF16)?;

        let mut block_mapping = HashMap::new();
        block_mapping.insert(1, 0);
        block_mapping.insert(2, 2);

        let original_src = src.clone();
        let original_dst = dst.clone();

        csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

        verify_swap::<half::bf16>(&original_src, &original_dst, &dst, &block_mapping)?;

        Ok(())
    }

    #[test]
    #[should_panic(expected = "Only support f16/bf16 dtypes and src and dst must have same dtype")]
    fn test_swap_blocks_invalid_dtype() {
        let device = Device::Cpu;
        let src = create_random_tensor(&device, DType::F32).unwrap();
        let mut dst = create_random_tensor(&device, DType::F32).unwrap();

        let mut block_mapping = HashMap::new();
        block_mapping.insert(0, 2);
        block_mapping.insert(1, 0);

        csrc::swap_blocks(&src, &mut dst, block_mapping).unwrap();
    }

    // #[test]
    // #[should_panic(expected = "Both src and dst tensors should be on the same device to swap")]
    // fn test_swap_blocks_different_cuda_devices() {
    //     let device1 = Device::new_cuda(0).unwrap();
    //     let device2 = Device::new_cuda(1).unwrap();
    //     let src = create_random_tensor(&device1, DType::F16).unwrap();
    //     let mut dst = create_random_tensor(&device2, DType::F16).unwrap();

    //     let mut block_mapping = HashMap::new();
    //     block_mapping.insert(0, 2);
    //     block_mapping.insert(1, 0);

    //     csrc::swap_blocks(&src, &mut dst, block_mapping).unwrap();
    // }
}

// #[test]
// fn test_copy_blocks() -> Result<()> {
//     // Create a CUDA device
//     let device = Device::new_cuda(0)?;

//     // Create sample tensors
//     let mut key_cache1 = Tensor::arange(0f32, 6f32, &device)?.reshape((2, 3, 1, 1))?;
//     let mut key_cache2 = Tensor::arange(6f32, 12f32, &device)?.reshape((2, 3, 1, 1))?;
//     let mut value_cache1 = Tensor::arange(12f32, 18f32, &device)?.reshape((2, 3, 1, 1))?;
//     let mut value_cache2 = Tensor::arange(18f32, 24f32, &device)?.reshape((2, 3, 1, 1))?;

//     // Create input vectors
//     let key_caches = vec![&mut key_cache1, &mut key_cache2];
//     let value_caches = vec![&mut value_cache1, &mut value_cache2];

//     // Create block mapping
//     let block_mapping = Tensor::from_vec(vec![0_i64, 1], (1, 2), &device)?; // Copy block 0 to block 1

//     // Call the copy_blocks function
//     unsafe { csrc::copy_blocks(key_caches, value_caches, block_mapping)? };

//     // Verify that the blocks have been copied correctly
//     assert_eq!(
//         key_cache1.to_vec2::<f32>()?,
//         vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]
//     );
//     assert_eq!(
//         key_cache2.to_vec2::<f32>()?,
//         vec![vec![0.0, 1.0, 2.0], vec![9.0, 10.0, 11.0]]
//     );
//     assert_eq!(
//         value_cache1.to_vec2::<f32>()?,
//         vec![vec![12.0, 13.0, 14.0], vec![15.0, 16.0, 17.0]]
//     );
//     assert_eq!(
//         value_cache2.to_vec2::<f32>()?,
//         vec![vec![12.0, 13.0, 14.0], vec![21.0, 22.0, 23.0]]
//     );

//     Ok(())
// }
