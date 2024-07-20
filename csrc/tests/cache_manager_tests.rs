use candle_core::{Result, Tensor};

#[test]
fn test_copy_blocks() -> Result<()> {
    // Create a CUDA device
    let device = Device::Cuda(0)?;

    // Create sample tensors
    let mut key_cache1 = Tensor::arange(0f32, 6f32, &device)?.reshape((2, 3, 1, 1))?;
    let mut key_cache2 = Tensor::arange(6f32, 12f32, &device)?.reshape((2, 3, 1, 1))?;
    let mut value_cache1 = Tensor::arange(12f32, 18f32, &device)?.reshape((2, 3, 1, 1))?;
    let mut value_cache2 = Tensor::arange(18f32, 24f32, &device)?.reshape((2, 3, 1, 1))?;

    // Create input vectors
    let key_caches = vec![&mut key_cache1, &mut key_cache2];
    let value_caches = vec![&mut value_cache1, &mut value_cache2];

    // Create block mapping
    let block_mapping = Tensor::from_vec(vec![0, 1], (1, 2), &device)?; // Copy block 0 to block 1

    // Call the copy_blocks function
    unsafe { csrc::copy_blocks(key_caches, value_caches, block_mapping)? };

    // Verify that the blocks have been copied correctly
    assert_eq!(
        key_cache1.to_vec2::<f32>()?,
        vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]
    );
    assert_eq!(
        key_cache2.to_vec2::<f32>()?,
        vec![vec![0.0, 1.0, 2.0], vec![9.0, 10.0, 11.0]]
    );
    assert_eq!(
        value_cache1.to_vec2::<f32>()?,
        vec![vec![12.0, 13.0, 14.0], vec![15.0, 16.0, 17.0]]
    );
    assert_eq!(
        value_cache2.to_vec2::<f32>()?,
        vec![vec![12.0, 13.0, 14.0], vec![21.0, 22.0, 23.0]]
    );

    Ok(())
}
