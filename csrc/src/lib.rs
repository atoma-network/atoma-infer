mod ffi;

use std::mem::MaybeUninit;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

pub struct FlashAttention {
    pub softmax_scale: f32,
    pub alibi_slopes: Option<Tensor>,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
    pub softcap: Option<f32>,
}

impl FlashAttention {
    fn cuda_fwd_t<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout,
        k: &candle_core::CudaStorage,
        k_l: &Layout,
        v: &candle_core::CudaStorage,
        v_l: &Layout,
        is_bf16: bool,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        // https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp#L341
        let device = q.device();

        check_gpu_compatibility(device.ordinal())?;

        if q.dtype() != k.dtype() {
            candle_core::bail!("query and key must have the same dtype");
        }

        if q.dtype() != v.dtype() {
            candle_core::bail!("query and value must have the same dtype");
        }

        // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case

        let q = q.as_cuda_slice::<T>()?;
        let k = k.as_cuda_slice::<T>()?;
        let v = v.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        let (b_sz, seqlen_q, num_heads, head_size_og) = q_l.shape().dims4()?;
        let (_b_sz, seqlen_k, num_heads_k, _head_size_og) = k_l.shape().dims4()?;

        let seqlenq_ngroups_swapped = seqlen_q == 1
            && num_heads > num_heads_k
            && self.window_size_left.is_none()
            && self.window_size_right.is_none()
            && head_size_og % 8 == 0
            && self.alibi_slopes.is_none();
        // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
        let (q_l, out_l, out_shape, seqlen_q, num_heads) = if seqlenq_ngroups_swapped {
            let ngroups = num_heads / num_heads_k;
            let new_shape = Shape::from((b_sz, ngroups, num_heads_k, head_size_og));

            // Create new layout for q, maintaining the original start_offset
            let new_q_l =
                Layout::contiguous_with_offset(&new_shape, q_l.start_offset()).transpose(1, 2)?;

            (
                new_q_l,
                Layout::contiguous(&new_shape),
                new_shape,
                ngroups,
                num_heads_k,
            )
        } else {
            let out_shape = q_l.shape().clone();
            (
                q_l.clone(),
                Layout::contiguous(&out_shape),
                out_shape,
                seqlen_q,
                num_heads,
            )
        };

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();
        let o_rank = o_stride.len();

        if q_rank != 4 || k_rank != 4 || v_rank != 4 {
            candle_core::bail!(
                "flash-attn expects input tensors of rank 4 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        }

        if q_stride[q_rank - 1] != 1 {
            candle_core::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle_core::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle_core::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let expected_kv = (b_sz, seqlen_k, num_heads_k, head_size_og);

        if expected_kv != k_l.shape().dims4()? {
            candle_core::bail!("shape mismatch q {:?} and k {:?}", q_l.shape(), k_l.shape())
        }
        if expected_kv != v_l.shape().dims4()? {
            candle_core::bail!("shape mismatch q {:?} and v {:?}", q_l.shape(), v_l.shape())
        }
        if head_size_og > 256 {
            candle_core::bail!("only supports head dimension at most 256 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle_core::bail!(
                "only supports head sizes that are a multiple of 8 (got {head_size_og})"
            )
        }
        if num_heads % num_heads_k != 0 {
            candle_core::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let alibi_slopes_ptr = if let Some(alibi_slopes) = &self.alibi_slopes {
            if alibi_slopes.dtype() != DType::F32 {
                candle_core::bail!(
                    "DType mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes.dtype(),
                    DType::F32
                );
            }

            let (alibi_slopes, alibi_slopes_layout) = alibi_slopes.storage_and_layout();

            if num_heads != alibi_slopes_layout.shape().dims1()? {
                candle_core::bail!(
                    "shape mismatch alibi_slopes {:?}, expected {:?}",
                    alibi_slopes_layout.shape(),
                    (num_heads)
                );
            }

            let alibi_slopes = match &*alibi_slopes {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle_core::bail!("alibi_slopes must be a cuda tensor"),
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

        let mut is_causal = if window_size_left < 0 && window_size_right == 0 {
            1
        } else {
            0
        };
        if seqlen_q == 1 && !self.alibi_slopes.is_some() {
            is_causal = 0;
        }

        let head_size = utils::round_multiple(head_size_og, 8);
        let head_size_rounded = utils::round_multiple(head_size, 32);
        let seqlen_q_rounded = utils::round_multiple(seqlen_q, 128);
        let seqlen_k_rounded = utils::round_multiple(seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let dst = unsafe { device.alloc::<T>(elem_count) }.w()?;
        let softmax_lse = device
            .alloc_zeros::<f32>(b_sz * 128 * num_heads * seqlen_q)
            .w()?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };

        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = seqlen_k as i32;
        }

        let num_splits = utils::compute_num_splits(
            b_sz,
            num_heads,
            head_size,
            seqlen_k,
            seqlen_q,
            device.ordinal(),
        )?;

        let mut softcap = self.softcap.unwrap_or(0.0);
        let (softmax_scale, scale_softmatx_log2) = if softcap > 0.0 {
            softcap = self.softmax_scale / softcap;
            (softcap, softcap * std::f32::consts::LOG2_E)
        } else {
            // Remove potential NaN
            softcap = 0.0;
            (
                self.softmax_scale,
                self.softmax_scale * std::f32::consts::LOG2_E,
            )
        };

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
                /* num_splits */ num_splits,
                /* b */ b_sz as u32,
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size as u32,
                /* d_rounded */ head_size_rounded as u32,
                /* softmax_scale*/ softmax_scale,
                /* scale_softmax_log2 */ scale_softmatx_log2,
                /* block_table */ std::ptr::null(),
                /* block_table_batch_stride */ 0,
                /* page_block_size */ 0,
                /* seqlen_q */ seqlen_q as u32,
                /* seqlen_k */ seqlen_k as u32,
                /* seqlen_q_rounded */ seqlen_q_rounded as u32,
                /* seqlen_k_rounded */ seqlen_k_rounded as u32,
                /* is_bf16 */ is_bf16,
                /* is_causal */ is_causal,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
                /* softcap */ softcap,
                false,
            )
        }

        let out_shape = if seqlenq_ngroups_swapped {
            Shape::from((b_sz, 1, num_heads_k * seqlen_q, head_size_og))
        } else {
            out_shape
        };

        let dst = candle_core::CudaStorage::wrap_cuda_slice(dst, device.clone());
        Ok((dst, out_shape))
    }
}

impl candle_core::CustomOp3 for FlashAttention {
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
        candle_core::bail!("no cpu support for flash-attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout,
        k: &candle_core::CudaStorage,
        k_l: &Layout,
        v: &candle_core::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match q.dtype() {
            candle_core::DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, k, k_l, v, v_l, false),
            candle_core::DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, k, k_l, v, v_l, true),
            dt => candle_core::bail!("flash-attn is only supported for f16/bf16 ({dt:?})"),
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
    softcap: Option<f32>,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttention {
        softmax_scale,
        alibi_slopes: None,
        window_size_left,
        window_size_right,
        softcap,
    };
    q.apply_op3(k, v, op)
}

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
        let mut efficiency = Vec::with_capacity(max_splits as usize);

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
                // println!("num_splits = {}, eff = {}", num_splits, eff);
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
            if efficiency[(num_splits - 1) as usize] >= 0.85 * max_efficiency {
                // println!("num_splits chosen = {}", num_splits);
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
            cuda_multiprocessor_count,
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
