#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

namespace vllm {

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs,
                                   int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int64_t numel_per_block) {
    const int layer_idx = blockIdx.x;
    const int pair_idx = blockIdx.y;

    scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
    scalar_t* value_cache =
        reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
    int64_t src_block_number = block_mapping[2 * pair_idx];
    int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

    const int64_t src_block_offset = src_block_number * numel_per_block;
    const int64_t dst_block_offset = dst_block_number * numel_per_block;
    for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
        int64_t src_offset = src_block_offset + i;
        int64_t dst_offset = dst_block_offset + i;
        key_cache[dst_offset] = key_cache[src_offset];
    }
    for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
        int64_t src_offset = src_block_offset + i;
        int64_t dst_offset = dst_block_offset + i;
        value_cache[dst_offset] = value_cache[src_offset];
    }
}

}  // namespace vllm

// f16, bf16 are special cases: We use a 16-bit integer to simulate the bit width.
// SAFETY: This is technically UB due to aliasing, but it is OK because the width is compatible.
extern "C" {
void copy_blocks_f16(
    void* key_cache_ptrs,
    void* value_cache_ptrs,
    const void* block_mapping,
    int64_t num_layers,
    int64_t num_pairs,
    int64_t numel_per_block,
    cudaStream_t stream) {
    dim3 grid(num_layers, num_pairs);
    dim3 block(std::min(int64_t(1024), numel_per_block));

    vllm::copy_blocks_kernel<int16_t><<<grid, block, 0, stream>>>(
        (int64_t*)key_cache_ptrs,
        (int64_t*)value_cache_ptrs,
        (const int64_t*)block_mapping,
        numel_per_block);
}
}

extern "C" {
void copy_blocks_bf16(
    void* key_cache_ptrs,
    void* value_cache_ptrs,
    const void* block_mapping,
    int64_t num_layers,
    int64_t num_pairs,
    int64_t numel_per_block,
    cudaStream_t stream) {
    dim3 grid(num_layers, num_pairs);
    dim3 block(std::min(int64_t(1024), numel_per_block));

    vllm::copy_blocks_kernel<int16_t><<<grid, block, 0, stream>>>(
        (int64_t*)key_cache_ptrs,
        (int64_t*)value_cache_ptrs,
        (const int64_t*)block_mapping,
        numel_per_block);
}
}

namespace vllm {

template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,          // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,        // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key_cache,          // [num_blocks, num_heads, head_size/x, block_size, x]
    scalar_t* __restrict__ value_cache,        // [num_blocks, num_heads, head_size, block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x) {
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
        // Padding token that should be ignored.
        return;
    }

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    const int n = num_heads * head_size;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t src_key_idx = token_idx * key_stride + i;
        const int64_t src_value_idx = token_idx * value_stride + i;

        const int head_idx = i / head_size;
        const int head_offset = i % head_size;
        const int x_idx = head_offset / x;
        const int x_offset = head_offset % x;

        const int64_t tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x + head_idx * (head_size / x) * block_size * x + x_idx * block_size * x + block_offset * x + x_offset;
        const int64_t tgt_value_idx = block_idx * num_heads * head_size * block_size + head_idx * head_size * block_size + head_offset * block_size + block_offset;
        key_cache[tgt_key_idx] = key[src_key_idx];
        value_cache[tgt_value_idx] = value[src_value_idx];
    }
}

#define CALL_RESHAPE_AND_CACHE(T)                                  \
    vllm::reshape_and_cache_kernel<T><<<grid, block, 0, stream>>>( \
        reinterpret_cast<T*>(key),                                 \
        reinterpret_cast<T*>(value),                               \
        reinterpret_cast<T*>(key_cache),                           \
        reinterpret_cast<T*>(value_cache),                         \
        slot_mapping,                                              \
        key_stride,                                                \
        value_stride,                                              \
        num_heads,                                                 \
        head_size,                                                 \
        block_size,                                                \
        x);

template <typename scalar_t>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,          // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,        // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ k_cache,            // [num_blocks, block_size, num_heads,
                                               // head_size]
    scalar_t* __restrict__ v_cache,            // [num_blocks, block_size, num_heads,
                                               // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride, const int key_stride, const int value_stride,
    const int num_heads, const int head_size, const int block_size) {
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0) {
        return;
    }
    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;
    const int n = num_heads * head_size;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t src_key_idx = token_idx * key_stride + i;
        const int64_t src_value_idx = token_idx * value_stride + i;
        const int head_idx = i / head_size;
        const int head_offset = i % head_size;
        const int64_t tgt_value_idx = block_idx * block_stride +
                                      block_offset * num_heads * head_size +
                                      head_idx * head_size + head_offset;
        k_cache[tgt_value_idx] = key[src_key_idx];
        v_cache[tgt_value_idx] = value[src_value_idx];
    }
}

#define CALL_RESHAPE_AND_CACHE_FLASH(T)                                  \
    vllm::reshape_and_cache_flash_kernel<T><<<grid, block, 0, stream>>>( \
        reinterpret_cast<T*>(key),                                       \
        reinterpret_cast<T*>(value),                                     \
        reinterpret_cast<T*>(key_cache),                                 \
        reinterpret_cast<T*>(value_cache),                               \
        slot_mapping,                                                    \
        block_stride,                                                    \
        key_stride,                                                      \
        value_stride,                                                    \
        num_heads,                                                       \
        head_size,                                                       \
        block_size);
}  // namespace vllm

extern "C" void reshape_and_cache(
    void* key,              // [num_tokens, num_heads, head_size]
    void* value,            // [num_tokens, num_heads, head_size]
    void* key_cache,        // [num_blocks, num_heads, head_size/x, block_size, x]
    void* value_cache,      // [num_blocks, num_heads, head_size, block_size]
    int64_t* slot_mapping,  // [num_tokens]

    int32_t num_tokens,
    int32_t num_heads,
    int32_t head_size,
    int32_t block_size,
    int32_t x,
    int32_t key_stride,
    int32_t value_stride,

    uint32_t dtype  // 0 => f16; 1 => bf16
) {
    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_size, 512));
    const cudaStream_t stream = 0;

    if (dtype == 0) {
        CALL_RESHAPE_AND_CACHE(uint16_t);
    } else if (dtype == 1) {
        CALL_RESHAPE_AND_CACHE(__nv_bfloat16);
    }
}

extern "C" void reshape_and_cache_flash(
    void* key,              // [num_tokens, num_heads, head_size]
    void* value,            // [num_tokens, num_heads, head_size]
    void* key_cache,        // [num_blocks, num_heads, head_size, block_size]
    void* value_cache,      // [num_blocks, num_heads, head_size, block_size]
    int64_t* slot_mapping,  // [num_tokens]

    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_size,
    int64_t block_size,
    int64_t key_stride,
    int64_t value_stride,
    int64_t block_stride,

    uint32_t dtype  // 0 => f16; 1 => bf16
) {
    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_size, int64_t(512)));
    const cudaStream_t stream = 0;

    if (dtype == 0) {
        CALL_RESHAPE_AND_CACHE_FLASH(uint16_t);
    } else if (dtype == 1) {
        CALL_RESHAPE_AND_CACHE_FLASH(__nv_bfloat16);
    }
}
