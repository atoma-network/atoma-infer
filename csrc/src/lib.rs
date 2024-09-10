pub mod cache_manager;
mod ffi;
pub mod ops;
pub use cache_manager::{copy_blocks, reshape_and_cache_flash, swap_blocks};

use std::mem::MaybeUninit;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use half::{bf16, f16};
mod ffi;

pub struct FlashAttention {
    pub softmax_scale: f32,
    pub alibi_slopes: Option<Tensor>,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
}

fn round_multiple(x: usize, m: usize) -> usize {
    (x + m - 1) / m * m
}

impl FlashAttention {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
        is_bf16: bool,
    ) -> Result<(candle::CudaStorage, Shape)> {
        // https://github.com/Dao-AILab/flash-attention/blob/b252072409e69c25f2b9d473cc534e49b24decd2/csrc/flash_attn/flash_api.cpp#L187
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let out_l = Layout::contiguous(&out_shape);

        let q = q.as_cuda_slice::<T>()?;
        let k = k.as_cuda_slice::<T>()?;
        let v = v.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();
        let o_rank = o_stride.len();

        if q_rank != 4 || k_rank != 4 || v_rank != 4 {
            candle::bail!(
                "flash-attn expects input tensors of rank 4 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        }
        if q_stride[q_rank - 1] != 1 {
            candle::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let (b_sz, seqlen_q, num_heads, head_size_og) = q_l.shape().dims4()?;
        let (_b_sz, seqlen_k, num_heads_k, _head_size_og) = k_l.shape().dims4()?;
        let expected_kv = (b_sz, seqlen_k, num_heads_k, head_size_og);
        if expected_kv != k_l.shape().dims4()? {
            candle::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
        }
        if expected_kv != v_l.shape().dims4()? {
            candle::bail!("shape mismatch q {:?} and v {:?}", q_l.shape(), v_l.shape())
        }
        if head_size_og > 256 {
            candle::bail!("only supports head dimension at most 256 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size_og})")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let alibi_slopes_ptr = if let Some(alibi_slopes) = &self.alibi_slopes {
            if alibi_slopes.dtype() != DType::F32 {
                candle::bail!(
                    "DType mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes.dtype(),
                    DType::F32
                );
            }

            let (alibi_slopes, alibi_slopes_layout) = alibi_slopes.storage_and_layout();

            if num_heads != alibi_slopes_layout.shape().dims1()? {
                candle::bail!(
                    "shape mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes_layout.shape(),
                    (num_heads)
                );
            }

            let alibi_slopes = match &*alibi_slopes {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("alibi_slopes must be a cuda tensor"),
            };

            let alibi_slopes = alibi_slopes.slice(alibi_slopes_layout.start_offset()..);

            *alibi_slopes.device_ptr() as *const core::ffi::c_void
        } else {
            std::ptr::null()
        };

        // if window_size_left > self.max_seqlen_k or None => -1
        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        // if window_size_right > self.max_seqlen_k or None => -1
        let mut window_size_right = self
            .window_size_right
            .filter(|v| v <= &seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let head_size = round_multiple(head_size_og, 8);
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<T>(elem_count) }.w()?;
        let softmax_lse = dev
            .alloc_zeros::<f32>(b_sz * 128 * num_heads * seqlen_q)
            .w()?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };

        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        let is_causal = if window_size_left < 0 && window_size_right == 0 {
            1
        } else {
            0
        };
        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = seqlen_k as i32;
        }

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
            let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
            let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
            let softmax_lse_ptr = *softmax_lse.device_ptr() as *const core::ffi::c_void;
            ffi::run_mha(
                q_ptr,
                k_ptr,
                v_ptr,
                dst_ptr,
                softmax_lse_ptr,
                /* alibi_slopes_ptr */ alibi_slopes_ptr,
                /* cu_seqlens_q_ptr */ std::ptr::null(),
                /* cu_seqlens_k_ptr */ std::ptr::null(),
                /* q_batch_stride */ q_stride[0] as u32,
                /* k_batch_stride */ k_stride[0] as u32,
                /* v_batch_stride */ v_stride[0] as u32,
                /* o_batch_stride */ o_stride[0] as u32,
                /* alibi_slopes_batch_stride */ 0,
                /* q_row_stride   */ q_stride[q_rank - 3] as u32,
                /* k_row_stride   */ k_stride[k_rank - 3] as u32,
                /* v_row_stride   */ v_stride[v_rank - 3] as u32,
                /* o_row_stride   */ o_stride[o_rank - 3] as u32,
                /* q_head_stride  */ q_stride[q_rank - 2] as u32,
                /* k_head_stride  */ k_stride[k_rank - 2] as u32,
                /* v_head_stride  */ v_stride[v_rank - 2] as u32,
                /* o_head_stride  */ o_stride[o_rank - 2] as u32,
                /* b */ b_sz as u32,
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size as u32,
                /* d_rounded */ head_size_rounded as u32,
                /* softmax_scale*/ self.softmax_scale,
                /* seqlen_q */ seqlen_q as u32,
                /* seqlen_k */ seqlen_k as u32,
                /* seqlen_q_rounded */ seqlen_q_rounded as u32,
                /* seqlen_k_rounded */ seqlen_k_rounded as u32,
                /* is_bf16 */ is_bf16,
                /* is_causal */ is_causal,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for FlashAttention {
    fn name(&self) -> &'static str {
        "flash-attn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for flash-attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
            dt => candle::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttention {
        softmax_scale,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttention {
        softmax_scale,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttention {
        softmax_scale,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v2 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size)`.
pub fn flash_attn_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttention {
        softmax_scale,
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
    };
    q.apply_op3(k, v, op)
}

struct FlashAttentionVarLen {
    pub softmax_scale: f32,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub seqlens_q: Tensor,
    pub seqlens_k: Tensor,
    pub alibi_slopes: Option<Tensor>,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
}

impl FlashAttentionVarLen {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
        is_bf16: bool,
    ) -> Result<(candle::CudaStorage, Shape)> {
        // https://github.com/Dao-AILab/flash-attention/blob/184b992dcb2a0890adaa19eb9b541c3e4f9d2a08/csrc/flash_attn/flash_api.cpp#L327
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let out_l = Layout::contiguous(&out_shape);

        let (seqlens_q, seqlens_q_layout) = self.seqlens_q.storage_and_layout();
        let seqlens_q = match &*seqlens_q {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?, // Should be i32!
            _ => candle::bail!("seqlens_q must be a cuda tensor"),
        };
        let seqlens_q = match seqlens_q_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_q.slice(o1..o2),
            None => candle::bail!("seqlens_q has to be contiguous"),
        };

        let (seqlens_k, seqlens_k_layout) = self.seqlens_k.storage_and_layout();
        let seqlens_k = match &*seqlens_k {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?, // Should be i32!
            _ => candle::bail!("seqlens_k must be a cuda tensor"),
        };
        let seqlens_k = match seqlens_k_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_k.slice(o1..o2),
            None => candle::bail!("seqlens_k has to be contiguous"),
        };

        let q = q.as_cuda_slice::<f16>()?;
        let k = k.as_cuda_slice::<f16>()?;
        let v = v.as_cuda_slice::<f16>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();
        let o_rank = o_stride.len();

        if q_rank != 3 || k_rank != 3 || v_rank != 3 {
            candle::bail!(
                "flash-attn-varlen expects input tensors of rank 3 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        }
        if q_stride[q_rank - 1] != 1 {
            candle::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let (_total_q, num_heads, head_size_og) = q_l.shape().dims3()?;
        let (total_k, num_heads_k, _head_size_og) = k_l.shape().dims3()?;
        let expected_kv = (total_k, num_heads_k, head_size_og);
        if expected_kv != k_l.shape().dims3()? {
            candle::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
        }
        if expected_kv != v_l.shape().dims3()? {
            candle::bail!("shape mismatch q {:?} and v {:?}", q_l.shape(), v_l.shape())
        }
        if head_size_og > 256 {
            candle::bail!("only supports head dimension at most 256 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size_og})")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let nseqlens_q = seqlens_q_layout.shape().dims1()?;
        if nseqlens_q < 2 {
            candle::bail!("seqlens_q should have a len >= 2 {nseqlens_q}")
        }
        let nseqlens_k = seqlens_k_layout.shape().dims1()?;
        if nseqlens_k != nseqlens_q {
            candle::bail!("seqlens_q and seqlens_k should have the same number of elements {nseqlens_q} <> {nseqlens_k}")
        }

        let batch_size = nseqlens_q - 1;

        let alibi_slopes_ptr = if let Some(alibi_slopes) = &self.alibi_slopes {
            if alibi_slopes.dtype() != DType::F32 {
                candle::bail!(
                    "DType mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes.dtype(),
                    DType::F32
                );
            }

            let (alibi_slopes, alibi_slopes_layout) = alibi_slopes.storage_and_layout();

            if num_heads != alibi_slopes_layout.shape().dims1()? {
                candle::bail!(
                    "shape mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes_layout.shape(),
                    (num_heads)
                );
            }

            let alibi_slopes = match &*alibi_slopes {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("alibi_slopes must be a cuda tensor"),
            };

            let alibi_slopes = alibi_slopes.slice(alibi_slopes_layout.start_offset()..);

            *alibi_slopes.device_ptr() as *const core::ffi::c_void
        } else {
            std::ptr::null()
        };

        // if window_size_left > self.max_seqlen_k or None => -1
        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &self.max_seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        // if window_size_right > self.max_seqlen_k or None => -1
        let mut window_size_right = self
            .window_size_right
            .filter(|v| v <= &self.max_seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let head_size = round_multiple(head_size_og, 8);
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(self.max_seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(self.max_seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
        let softmax_lse = dev
            .alloc_zeros::<f32>(batch_size * num_heads * self.max_seqlen_q)
            .w()?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };

        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        let is_causal = if window_size_left < 0 && window_size_right == 0 {
            1
        } else {
            0
        };
        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = self.max_seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = self.max_seqlen_k as i32;
        }

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
            let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
            let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
            let softmax_lse_ptr = *softmax_lse.device_ptr() as *const core::ffi::c_void;
            let seqlens_q_ptr = *seqlens_q.device_ptr() as *const core::ffi::c_int;
            let seqlens_k_ptr = *seqlens_k.device_ptr() as *const core::ffi::c_int;
            ffi::run_mha(
                q_ptr,
                k_ptr,
                v_ptr,
                dst_ptr,
                softmax_lse_ptr,
                /* alibi_slopes_ptr */ alibi_slopes_ptr,
                /* cu_seqlens_q_ptr */ seqlens_q_ptr,
                /* cu_seqlens_k_ptr */ seqlens_k_ptr,
                /* q_batch_stride */ 0,
                /* k_batch_stride */ 0,
                /* v_batch_stride */ 0,
                /* o_batch_stride */ 0,
                /* alibi_slopes_batch_stride */ 0,
                /* q_row_stride   */ q_stride[q_rank - 3] as u32,
                /* k_row_stride   */ k_stride[k_rank - 3] as u32,
                /* v_row_stride   */ v_stride[v_rank - 3] as u32,
                /* o_row_stride   */ o_stride[o_rank - 3] as u32,
                /* q_head_stride  */ q_stride[q_rank - 2] as u32,
                /* k_head_stride  */ k_stride[k_rank - 2] as u32,
                /* v_head_stride  */ v_stride[v_rank - 2] as u32,
                /* o_head_stride  */ o_stride[o_rank - 2] as u32,
                /* b */ batch_size as u32,
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size as u32,
                /* d_rounded */ head_size_rounded as u32,
                /* softmax_scale*/ self.softmax_scale,
                /* seqlen_q */ self.max_seqlen_q as u32,
                /* seqlen_k */ self.max_seqlen_k as u32,
                /* seqlen_q_rounded */ seqlen_q_rounded as u32,
                /* seqlen_k_rounded */ seqlen_k_rounded as u32,
                /* is_bf16 */ is_bf16,
                /* is_causal */ is_causal,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for FlashAttentionVarLen {
    fn name(&self) -> &'static str {
        "flash-attn-varlen"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for flash-attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
            dt => candle::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttentionVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: None,
        window_size_left,
        window_size_right,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_varlen_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttentionVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: None,
        window_size_left,
        window_size_right,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
pub fn flash_attn_varlen_alibi(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttentionVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v2 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_varlen_alibi_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttentionVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        alibi_slopes: Some(alibi_slopes.clone()),
        window_size_left,
        window_size_right,
    };
    q.apply_op3(k, v, op)
}

// /// Flash-attention v2 layer, with Key-Value cache.
// ///
// /// NOTE: We are not passing in each Key and Value tensor for the decoding phase. This is
// /// because we plan to use paged attention with flash attention.
// /// In that case, the key and value tensors at each decoding phase are stored within the
// /// kv cache tensor, separately. So we don't need to pass the key and value tensors to the
// /// flash attention kernel, directly.
// struct FlashAttentionKvCache {
//     /// Softmax scale
//     pub softmax_scale: f32,
//     /// Block table, used for paged attention algorithm
//     /// of shape [batch_size, max_num_block_per_sequence]
//     pub block_table: Option<Tensor>,
//     /// Alibi slopes, see https://nn.labml.ai/transformers/alibi/index.html,
//     /// of shape `[num_heads, ]` or `[batch_size, num_heads]`
//     pub alibi_slopes: Option<Tensor>,
//     /// Window size for left sided slicing attention
//     pub window_size_left: Option<usize>,
//     /// Window size for right sided slicing attention
//     pub window_size_right: Option<usize>,
//     /// Sequence lengths for the key tensor,
//     /// of shape `[batch_size, ]`
//     pub seqlens_k: Option<Tensor>,
//     /// Softcap parameter, used in Grok and Gemma2 models
//     pub softcap: Option<f32>,
// }

// impl FlashAttentionKvCache {
//     #[allow(clippy::too_many_arguments)]
//     fn cuda_fwd_t<
//         T: candle_core::cuda_backend::CudaDType
//             + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
//     >(
//         &self,
//         q: &candle_core::CudaStorage,
//         q_l: &Layout, // shape: `[batch_size, seqlen_q, num_heads, head_size]`, total_q := \sum_{i=0}^{b} s_i
//         kc: &candle_core::CudaStorage,
//         kc_l: &Layout, // shape: `[batch_size_cache, seqlen_k, num_heads_k, head_size]`, or `[num_blocks, page_block_size, num_heads_k, head_size]` if `self.block_table.is_some()`.
//         vc: &candle_core::CudaStorage,
//         vc_l: &Layout, // shape: `[batch_size_cache, seqlen_k, num_heads_k, head_size]`, or `[num_blocks, page_block_size, num_heads_k, head_size]` if `self.block_table.is_some()`.
//         is_bf16: bool,
//     ) -> Result<(candle_core::CudaStorage, Shape)> {
//         // https://github.com/Dao-AILab/flash-attention/blob/7551202cb2dd245432bc878447e19015c0af3c22/csrc/flash_attn/flash_api.cpp#L1284
//         let dev = q.device();

//         // Check GPU device compatibility
//         utils::check_gpu_compatibility(dev.ordinal())?;

//         if q.dtype() != kc.dtype() {
//             candle_core::bail!("query and key must have the same dtype");
//         }

//         if q.dtype() != vc.dtype() {
//             candle_core::bail!("query and value must have the same dtype");
//         }

//         let (block_table_ptr, block_table_layout) = if let Some(block_table) = &self.block_table {
//             let (block_table_storage, block_table_layout) = block_table.storage_and_layout();
//             let block_table_ptr = match &*block_table_storage {
//                 candle_core::Storage::Cuda(c) => {
//                     let cuda_slice = c.as_cuda_slice::<u32>()?;
//                     let block_table = cuda_slice.slice(block_table_layout.start_offset()..);
//                     let block_table_stride = block_table_layout.stride();
//                     let block_table_rank = block_table_stride.len();
//                     if block_table_stride[block_table_rank - 1] != 1 {
//                         candle_core::bail!("block_table must be contiguous")
//                     }
//                     *block_table.device_ptr() as *const i32
//                 }
//                 _ => candle_core::bail!("block_table must be a cuda tensor"),
//             };
//             // Clone block_table_storage to extend its lifetime
//             (block_table_ptr, Some(block_table_layout))
//         } else {
//             (std::ptr::null(), None)
//         };

//         let (batch_size, seqlen_q, num_heads, head_size_og) = q_l.shape().dims4()?;
//         let max_num_blocks_per_sequence = if let Some(layout) = block_table_layout {
//             let (b_sz, max_num_blocks_per_sequence) = layout.shape().dims2()?;
//             if b_sz != batch_size {
//                 candle_core::bail!(
//                     "shape mismatch of block_table (got {:?}) expected {:?})",
//                     layout.shape(),
//                     (batch_size, max_num_blocks_per_sequence)
//                 )
//             }
//             max_num_blocks_per_sequence
//         } else {
//             0
//         };

//         let (num_blocks, page_block_size, seqlen_k, num_heads_k) = if !block_table_ptr.is_null() {
//             let (num_blocks, page_block_size, num_heads_k, _head_size) = kc_l.shape().dims4()?;
//             (
//                 num_blocks,
//                 page_block_size,
//                 max_num_blocks_per_sequence * page_block_size,
//                 num_heads_k,
//             )
//         } else {
//             let (batch_size_cache, seqlen_k, num_heads_k, _head_size) = kc_l.shape().dims4()?;
//             (0, 1, seqlen_k, num_heads_k)
//         };

//         if !block_table_ptr.is_null() && page_block_size % 16 != 0 {
//             candle_core::bail!(
//                 "page_block_size must be a multiple of 16 when block_table is provided"
//             )
//         }

//         let batch_size_cache = if !block_table_ptr.is_null() {
//             batch_size
//         } else {
//             kc_l.dims()[0]
//         };

//         let mut window_size_left = self.window_size_left.map(|i| i as i32);
//         let mut window_size_right = self.window_size_right.map(|i| i as i32);

//         if let Some(w) = window_size_left {
//             if w >= seqlen_k as i32 {
//                 window_size_left = Some(-1);
//             }
//         }
//         if let Some(w) = window_size_right {
//             if w >= seqlen_k as i32 {
//                 window_size_right = Some(-1);
//             }
//         }

//         let mut window_size_left = window_size_left.unwrap_or(-1);
//         let mut window_size_right = window_size_right.unwrap_or(-1);

//         let mut is_causal = window_size_left < 0 && window_size_right == 0;
//         // causal=true is the same as causal=false in this case
//         if seqlen_q == 1 && self.alibi_slopes.is_none() {
//             is_causal = false;
//         }
//         if is_causal {
//             window_size_right = 0;
//         }

//         if window_size_left < 0 && window_size_right >= 0 {
//             window_size_left = seqlen_k as i32;
//         }
//         if window_size_right < 0 && window_size_left >= 0 {
//             window_size_right = seqlen_k as i32;
//         }

//         let out_shape = q_l.shape().clone();
//         let out_l = Layout::contiguous(&out_shape);

//         let q = q.as_cuda_slice::<T>()?;
//         let kc = kc.as_cuda_slice::<T>()?;
//         let vc = vc.as_cuda_slice::<T>()?;
//         let q = q.slice(q_l.start_offset()..);
//         let kc = kc.slice(kc_l.start_offset()..);
//         let vc = vc.slice(vc_l.start_offset()..);

//         let q_stride = q_l.stride();
//         let kc_stride = kc_l.stride();
//         let vc_stride = vc_l.stride();
//         let o_stride = out_l.stride();

//         let q_rank = q_stride.len();
//         let kc_rank = kc_stride.len();
//         let vc_rank = vc_stride.len();
//         let o_rank = o_stride.len();

//         if q_rank != 4 || kc_rank != 4 || vc_rank != 4 {
//             candle_core::bail!(
//                 "flash-attn expects input tensors of rank 4 (q: {q_rank}, k: {kc_rank}, v: {vc_rank})"
//             )
//         }

//         if q_stride[q_rank - 1] != 1 {
//             candle_core::bail!("the last dim of q must be contiguous {q_stride:?}")
//         }
//         if kc_stride[kc_rank - 1] != 1 {
//             candle_core::bail!("the last dim of k must be contiguous {kc_stride:?}")
//         }
//         if vc_stride[vc_rank - 1] != 1 {
//             candle_core::bail!("the last dim of v must be contiguous {vc_stride:?}")
//         }

//         let (alibi_slopes_ptr, alibi_slopes_batch_stride) =
//             if let Some(alibi_slopes) = &self.alibi_slopes {
//                 if alibi_slopes.dtype() != DType::F32 {
//                     candle_core::bail!(
//                         "DType mismatch alibi_slopes {:?}, expected {:?}",
//                         alibi_slopes.dtype(),
//                         DType::F32
//                     );
//                 }

//                 let alibi_slopes_batch_stride = if alibi_slopes.dims().len() == 2 {
//                     alibi_slopes.stride()[0]
//                 } else {
//                     0
//                 };

//                 let (alibi_slopes, alibi_slopes_layout) = alibi_slopes.storage_and_layout();

//                 if num_heads != alibi_slopes_layout.shape().dims1()? {
//                     candle_core::bail!(
//                         "shape mismatch alibi_slopes {:?}, expected {:?}",
//                         alibi_slopes_layout.shape(),
//                         (num_heads)
//                     );
//                 }

//                 let alibi_slopes = match &*alibi_slopes {
//                     candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
//                     _ => candle_core::bail!("alibi_slopes must be a cuda tensor"),
//                 };

//                 let alibi_slopes = alibi_slopes.slice(alibi_slopes_layout.start_offset()..);

//                 (
//                     *alibi_slopes.device_ptr() as *const core::ffi::c_void,
//                     alibi_slopes_batch_stride,
//                 )
//             } else {
//                 (std::ptr::null(), 0)
//             };

//         let head_size = utils::round_multiple(head_size_og, 8);
//         let head_size_rounded = utils::round_multiple(head_size, 32);
//         let seqlen_q_rounded = utils::round_multiple(seqlen_q, 128);
//         let seqlen_k_rounded = utils::round_multiple(seqlen_k, 128);

//         let elem_count = out_shape.elem_count();
//         let dst = unsafe { dev.alloc::<T>(elem_count) }.w()?;
//         let softmax_lse = dev
//             .alloc_zeros::<f32>(batch_size * num_heads * seqlen_q)
//             .w()?;

//         let is_bf16 = if is_bf16 { 1 } else { 0 };

//         unsafe {
//             let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
//             let kc_ptr = *kc.device_ptr() as *const core::ffi::c_void;
//             let vc_ptr = *vc.device_ptr() as *const core::ffi::c_void;
//             let dst_ptr = *dst.device_ptr() as *const core::ffi::c_void;
//             let softmax_lse_ptr = *softmax_lse.device_ptr() as *const core::ffi::c_void;
//             let cu_seqlens_k_ptr = if let Some(seqlens_k) = &self.seqlens_k {
//                 if seqlens_k.dims() != [batch_size] {
//                     candle_core::bail!(
//                         "shape mismatch of seqlens_k (got {:?}) expected {:?})",
//                         seqlens_k.dims(),
//                         [batch_size]
//                     )
//                 }
//                 if seqlens_k.dtype() != DType::U32 {
//                     candle_core::bail!(
//                         "DType mismatch seqlens_k {:?}, expected {:?}",
//                         seqlens_k.dtype(),
//                         DType::U32
//                     );
//                 }
//                 let (seqlens_k, seqlens_k_layout) = seqlens_k.storage_and_layout();
//                 let seqlens_k = match &*seqlens_k {
//                     candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
//                     _ => candle_core::bail!("seqlens_k must be a cuda tensor"),
//                 };
//                 let seqlens_k = seqlens_k.slice(seqlens_k_layout.start_offset()..);
//                 let seqlens_k_stride = seqlens_k_layout.stride();
//                 let seqlens_k_rank = seqlens_k_stride.len();
//                 if seqlens_k_stride[seqlens_k_rank - 1] != 1 {
//                     candle_core::bail!(
//                         "the last dim of seqlens_k must be contiguous {seqlens_k_stride:?}"
//                     )
//                 }
//                 *seqlens_k.device_ptr() as *const core::ffi::c_int
//             } else {
//                 std::ptr::null()
//             };
//             let is_seqlens_k_cumulative = self.seqlens_k.is_none();
//             let num_splits = utils::compute_num_splits(
//                 batch_size,
//                 num_heads,
//                 head_size,
//                 seqlen_k,
//                 seqlen_q,
//                 dev.ordinal(),
//             )?;
//             let block_table_batch_stride = if let Some(layout) = block_table_layout {
//                 layout.stride()[0] as u32
//             } else {
//                 0
//             };
//             ffi::run_mha(
//                 q_ptr,
//                 kc_ptr,
//                 vc_ptr,
//                 dst_ptr,
//                 softmax_lse_ptr,
//                 alibi_slopes_ptr,
//                 std::ptr::null(),
//                 cu_seqlens_k_ptr,
//                 is_seqlens_k_cumulative,
//                 q_stride[0] as u32,
//                 kc_stride[0] as u32,
//                 vc_stride[0] as u32,
//                 o_stride[0] as u32,
//                 alibi_slopes_batch_stride as u32,
//                 q_stride[q_rank - 3] as u32,
//                 /* k_row_stride   */ kc_stride[kc_rank - 3] as u32,
//                 /* v_row_stride   */ vc_stride[vc_rank - 3] as u32,
//                 /* o_row_stride   */ o_stride[o_rank - 3] as u32,
//                 /* q_head_stride  */ q_stride[q_rank - 2] as u32,
//                 /* k_head_stride  */ kc_stride[kc_rank - 2] as u32,
//                 /* v_head_stride  */ vc_stride[vc_rank - 2] as u32,
//                 /* o_head_stride  */ o_stride[o_rank - 2] as u32,
//                 num_splits,
//                 batch_size as u32,
//                 num_heads as u32,
//                 num_heads_k as u32,
//                 head_size as u32,
//                 head_size_rounded as u32,
//                 self.softmax_scale,
//                 self.softmax_scale * std::f32::consts::LOG2_E,
//                 block_table_ptr,
//                 block_table_batch_stride,
//                 page_block_size as i32,
//                 std::ptr::null(),
//                 seqlen_q as u32,
//                 seqlen_k as u32,
//                 seqlen_q_rounded as u32,
//                 seqlen_k_rounded as u32,
//                 is_bf16,
//                 is_causal as i32,
//                 window_size_left,
//                 window_size_right,
//                 0.0,
//                 false,
//                 !block_table_ptr.is_null(),
//             )
//         }

//         let dst = candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone());
//         Ok((dst, out_shape))
//     }
// }

// impl candle_core::CustomOp3 for FlashAttentionKvCache {
//     fn name(&self) -> &'static str {
//         "flash-attn"
//     }

//     fn cpu_fwd(
//         &self,
//         _: &CpuStorage,
//         _: &Layout,
//         _: &CpuStorage,
//         _: &Layout,
//         _: &CpuStorage,
//         _: &Layout,
//     ) -> Result<(CpuStorage, Shape)> {
//         candle_core::bail!("no cpu support for flash-attn")
//     }

//     fn cuda_fwd(
//         &self,
//         q: &candle_core::CudaStorage,
//         q_l: &Layout,
//         kc: &candle_core::CudaStorage,
//         kc_l: &Layout,
//         vc: &candle_core::CudaStorage,
//         vc_l: &Layout,
//     ) -> Result<(candle_core::CudaStorage, Shape)> {
//         match q.dtype() {
//             candle_core::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, kc, kc_l, vc, vc_l, false),
//             candle_core::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, kc, kc_l, vc, vc_l, true),
//             dt => candle_core::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
//         }
//     }
// }

// #[allow(clippy::too_many_arguments)]
// /// Flash-attention v2 layer with key and value tensors cached.
// ///
// /// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
// /// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
// /// than q, the number of heads in k and v has to be divisible by the number of heads in q.
// ///
// /// # Arguments
// ///
// /// * `q` - Query tensor with shape `[batch_size, seqlen_q, num_heads, head_size]`.
// /// * `k` - Key tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
// /// * `v` - Value tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
// ///
// /// The resulting tensor has dimensions `[batch_size, seqlen_q, num_heads, head_size]`.
// pub fn flash_attn_kv_cache(
//     q: &Tensor,
//     k: &Tensor,
//     v: &Tensor,
//     softmax_scale: f32,
//     causal: bool,
// ) -> Result<Tensor> {
//     let window_size_left = None;
//     let window_size_right = if causal { Some(0) } else { None };

//     let op = FlashAttentionKvCache {
//         softmax_scale,
//         alibi_slopes: None,
//         window_size_left,
//         window_size_right,
//         softcap: None,
//         block_table: None,
//         seqlens_k: None,
//     };
//     q.apply_op3(k, v, op)
// }

// #[allow(clippy::too_many_arguments)]
// /// Flash-attention v2 layer with key and value tensors cached.
// ///
// /// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
// /// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
// /// than q, the number of heads in k and v has to be divisible by the number of heads in q.
// ///
// /// # Arguments
// ///
// /// * `q` - Query tensor with shape `[batch_size, seqlen_q, num_heads, head_size]`.
// /// * `k` - Key tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
// /// * `v` - Value tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
// ///
// /// The resulting tensor has dimensions `[batch_size, seqlen_q, num_heads, head_size]`.
// ///
// /// # Causal mask
// ///
// /// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
// /// of  `Q @ K^T`
// pub fn flash_attn_kv_cache_windowed(
//     q: &Tensor,
//     k: &Tensor,
//     v: &Tensor,
//     seqlens_k: Option<&Tensor>,
//     softmax_scale: f32,
//     window_size_left: Option<usize>,
//     window_size_right: Option<usize>,
// ) -> Result<Tensor> {
//     let op = FlashAttentionKvCache {
//         softmax_scale,
//         alibi_slopes: None,
//         window_size_left,
//         window_size_right,
//         softcap: None,
//         block_table: None,
//         seqlens_k: seqlens_k.cloned(),
//     };
//     q.apply_op3(k, v, op)
// }

// #[allow(clippy::too_many_arguments)]
// /// Flash-attention v2 layer with key and value tensors cached.
// ///
// /// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
// /// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
// /// than q, the number of heads in k and v has to be divisible by the number of heads in q.
// ///
// /// # Arguments
// ///
// /// * `q` - Query tensor with shape `[batch_size, seqlen_q, num_heads, head_size]`.
// /// * `k` - Key tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
// /// * `v` - Value tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
// /// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
// ///
// /// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
// /// `seqlen_1 + seqlen_2`, etc.
// ///
// /// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
// pub fn flash_attn_kv_cache_alibi(
//     q: &Tensor,
//     k: &Tensor,
//     v: &Tensor,
//     alibi_slopes: &Tensor,
//     seqlens_k: Option<&Tensor>,
//     softmax_scale: f32,
//     causal: bool,
// ) -> Result<Tensor> {
//     let window_size_left = None;
//     let window_size_right = if causal { Some(0) } else { None };

//     let op = FlashAttentionKvCache {
//         softmax_scale,
//         alibi_slopes: Some(alibi_slopes.clone()),
//         window_size_left,
//         window_size_right,
//         softcap: None,
//         block_table: None,
//         seqlens_k: seqlens_k.cloned(),
//     };
//     q.apply_op3(k, v, op)
// }

// #[allow(clippy::too_many_arguments)]
// /// Flash-attention v2 layer with key and value tensors cached.
// ///
// /// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
// /// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
// /// than q, the number of heads in k and v has to be divisible by the number of heads in q.
// ///
// /// # Arguments
// ///
// /// * `q` - Query tensor with shape `[batch_size, seqlen_q, num_heads, head_size]`.
// /// * `k` - Key tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
// /// * `v` - Value tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
// /// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
// /// * `window_size_left` - Limit left attention to value tokens.
// /// * `window_size_right` - Limit right attention to value tokens.
// ///
// /// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
// ///
// /// # Causal mask
// ///
// /// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
// /// of  `Q @ K^T`
// pub fn flash_attn_kv_cache_alibi_windowed(
//     q: &Tensor,
//     k: &Tensor,
//     v: &Tensor,
//     alibi_slopes: &Tensor,
//     softmax_scale: f32,
//     window_size_left: Option<usize>,
//     window_size_right: Option<usize>,
// ) -> Result<Tensor> {
//     let op = FlashAttentionKvCache {
//         softmax_scale,
//         alibi_slopes: Some(alibi_slopes.clone()),
//         window_size_left,
//         window_size_right,
//         block_table: None,
//         seqlens_k: None,
//         softcap: None,
//     };
//     q.apply_op3(k, v, op)
// }

// #[allow(clippy::too_many_arguments)]
// /// Flash-attention v2 layer with key and value tensors cached.
// ///
// /// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
// /// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
// /// than q, the number of heads in k and v has to be divisible by the number of heads in q.
// ///
// /// # Arguments
// ///
// /// * `q` - Query tensor with shape `[batch_size, seqlen_q, num_heads, head_size]`.
// /// * `k` - Key tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
// /// * `v` - Value tensor with shape `[batch_size_cache, seqlen_k, num_heads_k, head_size]` or `[num_blocks, page_block_size, num_heads_k, head_size]` if block_table.is_some().
// /// * `alibi_slopes` - Alibi slopes tensor with shape `(num_heads_q)`.
// /// * `window_size_left` - Limit left attention to value tokens.
// /// * `window_size_right` - Limit right attention to value tokens.
// /// * `block_table` - Block table tensor with shape `[batch_size, max_num_block_per_sequence]`.
// /// * `softcap` - Softcap parameter, used in Grok and Gemma2 models.
// /// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
// ///
// /// The resulting tensor has dimensions `(total_q, num_heads_q, head_size)`.
// ///
// /// # Causal mask
// ///
// /// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
// /// of  `Q @ K^T`
// pub fn flash_attn_kv_cache_full(
//     q: &Tensor,
//     k: &Tensor,
//     v: &Tensor,
//     alibi_slopes: Option<&Tensor>,
//     softmax_scale: f32,
//     block_table: Option<&Tensor>,
//     seqlens_k: Option<&Tensor>,
//     softcap: Option<f32>,
//     causal: bool,
// ) -> Result<Tensor> {
//     let window_size_left = None;
//     let window_size_right = if causal { Some(0) } else { None };

//     let op = FlashAttentionKvCache {
//         softmax_scale,
//         alibi_slopes: alibi_slopes.cloned(),
//         window_size_left,
//         window_size_right,
//         block_table: block_table.cloned(),
//         seqlens_k: seqlens_k.cloned(),
//         softcap,
//     };
//     q.apply_op3(k, v, op)
// }

pub(crate) mod utils {

    use cuda_runtime_sys::*;

    use super::*;
    pub(crate) fn round_multiple(x: usize, m: usize) -> usize {
        (x + m - 1) / m * m
    }

    /// Find the number of splits that maximizes the occupancy. For example, if we have
    /// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
    /// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
    /// splits as that would incur more HBM reads/writes.
    /// So we find the best efficiency, then find the smallest number of splits that gets 85%
    /// of the best efficiency.
    pub(crate) fn num_splits_heuristic(
        batch_nheads_mblocks: usize,
        num_sms: usize,
        num_n_blocks: usize,
        max_splits: usize,
    ) -> usize {
        // If we have enough to almost fill the SMs, then just use 1 split
        if (batch_nheads_mblocks as f32) >= 0.8 * (num_sms as f32) {
            return 1;
        }

        let max_splits = max_splits.min(num_sms).min(num_n_blocks);
        let mut max_efficiency = 0.0;
        let mut efficiency = Vec::with_capacity(max_splits);

        let ceil_div = |a: usize, b: usize| -> usize { (a + b - 1) / b };

        let is_split_eligible = |num_splits: usize| -> bool {
            num_splits == 1
                || ceil_div(num_n_blocks, num_splits) != ceil_div(num_n_blocks, num_splits - 1)
        };

        for num_splits in 1..=max_splits {
            if !is_split_eligible(num_splits) {
                efficiency.push(0.0);
            } else {
                let n_waves = (batch_nheads_mblocks * num_splits) as f32 / num_sms as f32;
                let eff = n_waves / n_waves.ceil();
                if eff > max_efficiency {
                    max_efficiency = eff;
                }
                efficiency.push(eff);
            }
        }

        for num_splits in 1..=max_splits {
            if !is_split_eligible(num_splits) {
                continue;
            }
            if efficiency[num_splits - 1] >= 0.85 * max_efficiency {
                return num_splits;
            }
        }

        1
    }

    pub(crate) fn compute_num_splits(
        batch_size: usize,
        num_heads: usize,
        head_size: usize,
        max_seqlen_k: usize,
        max_seqlen_q: usize,
        device_ordinal: usize,
    ) -> Result<u32> {
        let block_n = if head_size <= 64 {
            256
        } else if head_size <= 128 {
            128
        } else {
            64
        };
        let num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
        // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
        // In any case we don't expect seqlen_q to be larger than 64 for inference.
        let num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
        let cuda_multiprocessor_count = get_multiprocessor_count(device_ordinal)?;
        let num_splits = num_splits_heuristic(
            batch_size * num_heads * num_m_blocks,
            cuda_multiprocessor_count * 2,
            num_n_blocks,
            128,
        );
        if num_splits > 128 {
            candle_core::bail!("num_splits > 128 not supported")
        }
        Ok(num_splits as u32)
    }

    pub(crate) fn get_multiprocessor_count(device_index: usize) -> Result<usize> {
        unsafe {
            let mut count = MaybeUninit::uninit();
            let error = cudaDeviceGetAttribute(
                count.as_mut_ptr(),
                cudaDeviceAttr::cudaDevAttrMultiProcessorCount,
                device_index as i32,
            );
            if error != cudaError::cudaSuccess {
                candle_core::bail!("CUDA error: {:?}", error)
            }
            Ok(count.assume_init() as usize)
        }
    }

    pub(crate) fn check_gpu_compatibility(device_index: usize) -> Result<()> {
        use core::ffi::c_int;
        let mut props = cudaDeviceProp::default();
        unsafe {
            let error =
                cudaGetDeviceProperties(&mut props as *mut cudaDeviceProp, device_index as c_int);
            if error != cudaError::cudaSuccess {
                candle_core::bail!("CUDA error: {:?}", error)
            }
            let is_sm8x = props.major == 8 && props.minor >= 0;
            let is_sm90 = props.major == 9 && props.minor == 0;

            if !(is_sm90 || is_sm8x) {
                candle_core::bail!("FlashAttention only supports Ampere GPUs or newer.")
            }
        }
        Ok(())
    }
}
