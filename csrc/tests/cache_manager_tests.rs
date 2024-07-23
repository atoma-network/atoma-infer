use candle_core::{DType, Device, IndexOp, Result, Tensor};
use std::collections::HashMap;

// #[cfg(test)]
// mod swap_blocks {
//     use super::*;

//     const NUM_BLOCKS: usize = 3;
//     const BLOCK_SIZE: usize = 16;
//     const NUM_HEADS: usize = 2;
//     const HEAD_SIZE: usize = 8;

//     fn create_random_tensor(device: &Device, dtype: DType) -> Result<Tensor> {
//         let tensor = Tensor::rand(
//             0f32,
//             10f32,
//             (NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE),
//             &device,
//         )?
//         .to_dtype(dtype)?;
//         Ok(tensor)
//     }

//     fn verify_swap<
//         T: candle_core::cuda_backend::CudaDType
//             + candle_core::cuda_backend::cudarc::driver::DeviceRepr
//             + std::fmt::Debug
//             + PartialEq
//             + candle_core::WithDType,
//     >(
//         src: &Tensor,
//         original_dst: &Tensor,
//         swapped_dst: &Tensor,
//         block_mapping: &HashMap<i64, i64>,
//     ) -> Result<()> {
//         for (src_block, dst_block) in block_mapping {
//             let src_slice = src.i(*src_block as usize)?;
//             let swapped_dst_slice = swapped_dst.i(*dst_block as usize)?;
//             assert_eq!(
//                 src_slice.flatten_all()?.to_vec1::<T>()?,
//                 swapped_dst_slice.flatten_all()?.to_vec1::<T>()?,
//                 "Block {} from source was not correctly swapped to block {} in destination",
//                 src_block,
//                 dst_block
//             );

//             // Check that non-swapped blocks remain unchanged
//             for i in 0..NUM_BLOCKS {
//                 if !block_mapping.values().any(|&v| v == i as i64) {
//                     let original_dst_slice = original_dst.i(i)?;
//                     let current_dst_slice = swapped_dst.i(i)?;
//                     assert_eq!(
//                         original_dst_slice.flatten_all()?.to_vec1::<T>()?,
//                         current_dst_slice.flatten_all()?.to_vec1::<T>()?,
//                         "Block {} in destination should not have changed",
//                         i
//                     );
//                 }
//             }
//         }
//         Ok(())
//     }

//     #[test]
//     fn test_swap_blocks_f16_cuda() -> Result<()> {
//         let device = Device::new_cuda(0)?;
//         let src = create_random_tensor(&device, DType::F16)?;
//         let mut dst = create_random_tensor(&device, DType::F16)?;

//         let mut block_mapping = HashMap::new();
//         block_mapping.insert(0, 2);
//         block_mapping.insert(1, 0);

//         let original_src = src.clone();
//         let original_dst = dst.clone();

//         csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

//         verify_swap::<half::f16>(&original_src, &original_dst, &dst, &block_mapping)?;

//         Ok(())
//     }

//     #[test]
//     fn test_swap_blocks_bf16_cuda() -> Result<()> {
//         let device = Device::new_cuda(0)?;
//         let src = create_random_tensor(&device, DType::BF16)?;
//         let mut dst = create_random_tensor(&device, DType::BF16)?;

//         let mut block_mapping = HashMap::new();
//         block_mapping.insert(0, 1);
//         block_mapping.insert(2, 0);

//         let original_src = src.clone();
//         let original_dst = dst.clone();

//         csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

//         verify_swap::<half::bf16>(&original_src, &original_dst, &dst, &block_mapping)?;

//         Ok(())
//     }

//     #[test]
//     fn test_swap_blocks_cpu_to_cuda_f16() -> Result<()> {
//         let cpu_device = Device::Cpu;
//         let cuda_device = Device::new_cuda(0)?;
//         let src = create_random_tensor(&cpu_device, DType::F16)?;
//         let mut dst = create_random_tensor(&cuda_device, DType::F16)?;

//         let mut block_mapping = HashMap::new();
//         block_mapping.insert(0, 2);
//         block_mapping.insert(1, 1);

//         let original_src = src.clone();
//         let original_dst = dst.clone();

//         csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

//         verify_swap::<half::f16>(&original_src, &original_dst, &dst, &block_mapping)?;

//         Ok(())
//     }

//     #[test]
//     fn test_swap_blocks_cuda_to_cpu_f16() -> Result<()> {
//         let cpu_device = Device::Cpu;
//         let cuda_device = Device::new_cuda(0)?;
//         let src = create_random_tensor(&cuda_device, DType::F16)?;
//         let mut dst = create_random_tensor(&cpu_device, DType::F16)?;

//         let mut block_mapping = HashMap::new();
//         block_mapping.insert(1, 0);
//         block_mapping.insert(2, 2);

//         let original_src = src.clone();
//         let original_dst = dst.clone();

//         csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

//         verify_swap::<half::f16>(&original_src, &original_dst, &dst, &block_mapping)?;

//         Ok(())
//     }

//     #[test]
//     fn test_swap_blocks_cpu_to_cuda_bf16() -> Result<()> {
//         let cpu_device = Device::Cpu;
//         let cuda_device = Device::new_cuda(0)?;
//         let src = create_random_tensor(&cpu_device, DType::BF16)?;
//         let mut dst = create_random_tensor(&cuda_device, DType::BF16)?;

//         let mut block_mapping = HashMap::new();
//         block_mapping.insert(0, 2);
//         block_mapping.insert(1, 1);

//         let original_src = src.clone();
//         let original_dst = dst.clone();

//         csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

//         verify_swap::<half::bf16>(&original_src, &original_dst, &dst, &block_mapping)?;

//         Ok(())
//     }

//     #[test]
//     fn test_swap_blocks_cuda_to_cpu_bf16() -> Result<()> {
//         let cpu_device = Device::Cpu;
//         let cuda_device = Device::new_cuda(0)?;
//         let src = create_random_tensor(&cuda_device, DType::BF16)?;
//         let mut dst = create_random_tensor(&cpu_device, DType::BF16)?;

//         let mut block_mapping = HashMap::new();
//         block_mapping.insert(1, 0);
//         block_mapping.insert(2, 2);

//         let original_src = src.clone();
//         let original_dst = dst.clone();

//         csrc::swap_blocks(&src, &mut dst, block_mapping.clone())?;

//         verify_swap::<half::bf16>(&original_src, &original_dst, &dst, &block_mapping)?;

//         Ok(())
//     }

//     #[test]
//     #[should_panic(
//         expected = "swap_blocks: Either src and dst are on the same cuda device, or src and dst are on cpu and cuda devices, alternately"
//     )]
//     fn test_swap_blocks_invalid_dtype() {
//         let device = Device::Cpu;
//         let src = create_random_tensor(&device, DType::F32).unwrap();
//         let mut dst = create_random_tensor(&device, DType::F32).unwrap();

//         let mut block_mapping = HashMap::new();
//         block_mapping.insert(0, 2);
//         block_mapping.insert(1, 0);

//         csrc::swap_blocks(&src, &mut dst, block_mapping).unwrap();
//     }
// }

#[cfg(test)]
mod copy_blocks {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    const NUM_LAYERS: usize = 2;
    const NUM_PAIRS: usize = 3;

    fn create_test_tensor(device: &Device, dtype: DType) -> Tensor {
        const NUM_BLOCKS: usize = 4;
        const BLOCK_SIZE: usize = 64;
        const NUM_HEADS: usize = 2;
        const HEAD_SIZE: usize = 8;

        Tensor::rand(
            0f32,
            1f32,
            &[NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE],
            device,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
    }

    fn compare_blocks<T: candle_core::WithDType>(
        tensor: &Tensor,
        src_block: usize,
        dst_block: usize,
        block_size: usize,
    ) -> Result<bool> {
        let src_data = tensor.i(src_block)?.flatten(0, 1)?.to_vec1::<T>()?;
        let dst_data = tensor.i(dst_block)?.flatten(0, 1)?.to_vec1::<T>()?;
        Ok(src_data[..block_size] == dst_data[..block_size])
    }

    #[test]
    fn test_copy_blocks_f16() {
        let device = Device::new_cuda(0).unwrap();

        let mut key_caches: Vec<_> = (0..NUM_LAYERS)
            .map(|_| create_test_tensor(&device, DType::F16))
            .collect();
        let mut value_caches: Vec<_> = (0..NUM_LAYERS)
            .map(|_| create_test_tensor(&device, DType::F16))
            .collect();

        let block_mapping =
            Tensor::from_slice(&[0i64, 2, 1, 3, 2, 0], (NUM_PAIRS, 2), &device).unwrap();

        let key_caches_refs: Vec<_> = key_caches.iter_mut().collect();
        let value_caches_refs: Vec<_> = value_caches.iter_mut().collect();

        unsafe {
            csrc::copy_blocks(key_caches_refs, value_caches_refs, block_mapping).unwrap();
        }

        // Check if blocks were correctly copied
        for layer in 0..num_layers {
            assert!(compare_blocks::<half::f16>(&key_caches[layer], 0, 2, block_size).unwrap());
            assert!(compare_blocks::<half::f16>(&key_caches[layer], 1, 3, block_size).unwrap());
            assert!(compare_blocks::<half::f16>(&key_caches[layer], 2, 0, block_size).unwrap());

            assert!(compare_blocks::<half::f16>(&value_caches[layer], 0, 2, block_size).unwrap());
            assert!(compare_blocks::<half::f16>(&value_caches[layer], 1, 3, block_size).unwrap());
            assert!(compare_blocks::<half::f16>(&value_caches[layer], 2, 0, block_size).unwrap());

            // Check that untouched blocks remain the same
            assert_eq!(
                key_caches[layer].i(1).unwrap().to_vec2::<half::f16>().unwrap(),
                original_key_caches[layer]
                    .i(1)
                    .unwrap()
                    .to_vec2::<half::f16>()
                    .unwrap()
            );
            assert_eq!(
                value_caches[layer].i(1).unwrap().to_vec2::<half::f16>().unwrap(),
                original_value_caches[layer]
                    .i(1)
                    .unwrap()
                    .to_vec2::<half::f16>()
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_copy_blocks_bf16() {
        let device = Device::new_cuda(0).unwrap();

        let mut key_caches: Vec<_> = (0..NUM_LAYERS)
            .map(|_| create_test_tensor(&device, DType::BF16))
            .collect();
        let mut value_caches: Vec<_> = (0..NUM_LAYERS)
            .map(|_| create_test_tensor(&device, DType::BF16))
            .collect();

        let block_mapping =
            Tensor::from_slice(&[0i64, 2, 1, 3, 2, 0], (NUM_PAIRS, 2), &device).unwrap();

        let key_caches_refs: Vec<_> = key_caches.iter_mut().collect();
        let value_caches_refs: Vec<_> = value_caches.iter_mut().collect();

        unsafe {
            csrc::copy_blocks(key_caches_refs, value_caches_refs, block_mapping).unwrap();
        }

        // Check if blocks were correctly copied
        for layer in 0..num_layers {
            assert!(compare_blocks::<half::bf16>(&key_caches[layer], 0, 2, block_size).unwrap());
            assert!(compare_blocks::<half::bf16>(&key_caches[layer], 1, 3, block_size).unwrap());
            assert!(compare_blocks::<half::bf16>(&key_caches[layer], 2, 0, block_size).unwrap());

            assert!(compare_blocks::<half::bf16>(&value_caches[layer], 0, 2, block_size).unwrap());
            assert!(compare_blocks::<half::bf16>(&value_caches[layer], 1, 3, block_size).unwrap());
            assert!(compare_blocks::<half::bf16>(&value_caches[layer], 2, 0, block_size).unwrap());

            // Check that untouched blocks remain the same
            assert_eq!(
                key_caches[layer].i(1).unwrap().to_vec2::<half::bf16>().unwrap(),
                original_key_caches[layer]
                    .i(1)
                    .unwrap()
                    .to_vec2::<half::bf16>()
                    .unwrap()
            );
            assert_eq!(
                value_caches[layer].i(1).unwrap().to_vec2::<half::bf16>().unwrap(),
                original_value_caches[layer]
                    .i(1)
                    .unwrap()
                    .to_vec2::<half::bf16>()
                    .unwrap()
            );
        }
    }

    #[test]
    #[should_panic(expected = "key_caches and value_caches must have the same length")]
    fn test_copy_blocks_unequal_lengths() {
        let device = Device::new_cuda(0).unwrap();
        let mut key_caches = vec![create_test_tensor(&device, DType::F16)];
        let mut value_caches = vec![
            create_test_tensor(&device, DType::F16),
            create_test_tensor(&device, DType::F16),
        ];
        let block_mapping = Tensor::from_slice(&[0i64, 1], (1, 2), &device).unwrap();

        let key_caches_refs: Vec<_> = key_caches.iter_mut().collect();
        let value_caches_refs: Vec<_> = value_caches.iter_mut().collect();

        unsafe {
            csrc::copy_blocks(key_caches_refs, value_caches_refs, block_mapping).unwrap();
        }
    }

    #[test]
    #[should_panic(expected = "device must be a cuda device")]
    fn test_copy_blocks_non_cuda_device() {
        let device = Device::Cpu;
        let mut key_caches = vec![create_test_tensor(&device, DType::F16)];
        let mut value_caches = vec![create_test_tensor(&device, DType::F16)];
        let block_mapping = Tensor::from_slice(&[0i64, 1], (1, 2), &device).unwrap();

        let key_caches_refs: Vec<_> = key_caches.iter_mut().collect();
        let value_caches_refs: Vec<_> = value_caches.iter_mut().collect();

        unsafe {
            csrc::copy_blocks(key_caches_refs, value_caches_refs, block_mapping).unwrap();
        }
    }

    #[test]
    #[should_panic(expected = "Only support f16/bf16 dtypes and src and dst must have same dtype")]
    fn test_copy_blocks_different_dtypes() {
        let device = Device::new_cuda(0).unwrap();
        let mut key_caches = vec![create_test_tensor(&device, DType::F16)];
        let mut value_caches = vec![create_test_tensor(&device, DType::BF16)];
        let block_mapping = Tensor::from_slice(&[0i64, 1], (1, 2), &device).unwrap();

        let key_caches_refs: Vec<_> = key_caches.iter_mut().collect();
        let value_caches_refs: Vec<_> = value_caches.iter_mut().collect();

        unsafe {
            csrc::copy_blocks(key_caches_refs, value_caches_refs, block_mapping).unwrap();
        }
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: ShapeMismatch { buffer_size: 3, shape: [1, 2] }"
    )]
    fn test_copy_blocks_invalid_block_mapping_shape() {
        let device = Device::new_cuda(0).unwrap();
        let mut key_caches = vec![create_test_tensor(&device, DType::F16)];
        let mut value_caches = vec![create_test_tensor(&device, DType::F16)];
        let block_mapping = Tensor::from_slice(&[0i64, 1, 2], (1, 2), &device).unwrap(); // Invalid shape

        let key_caches_refs: Vec<_> = key_caches.iter_mut().collect();
        let value_caches_refs: Vec<_> = value_caches.iter_mut().collect();

        unsafe {
            csrc::copy_blocks(key_caches_refs, value_caches_refs, block_mapping).unwrap();
        }
    }
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
