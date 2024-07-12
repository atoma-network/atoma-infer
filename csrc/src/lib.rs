mod ffi;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, DType, Device, Layout, Result, Shape, Tensor};

pub struct FlashAttention {
    pub softmax_scale: f32,
    pub alibi_slopes: Option<Tensor>,
    pub window_size_left: i32,
    pub window_size_right: i32,
    pub softcap: Option<f32>,
}

fn round_multiple(x: usize, m: usize) -> usize {
    (x + m - 1) / m * m
}

/// Find the number of splits that maximizes the occupancy. For example, if we have
/// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
/// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
/// splits as that would incur more HBM reads/writes.
/// So we find the best efficiency, then find the smallest number of splits that gets 85%
/// of the best efficiency.
fn num_splits_heuristic(
    batch_nheads_mblocks: i32,
    num_sms: i32,
    num_n_blocks: i32,
    max_splits: i32,
) -> i32 {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks as f32) >= 0.8 * (num_sms as f32) {
        return 1;
    }

    let max_splits = max_splits.min(num_sms).min(num_n_blocks);
    let mut max_efficiency = 0.0;
    let mut efficiency = Vec::with_capacity(max_splits as usize);

    let ceil_div = |a: i32, b: i32| -> i32 { (a + b - 1) / b };

    let is_split_eligible = |num_splits: i32| -> bool {
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
    ) -> Result<(candle_core::CudaStorage, Shape<usize>)> {
        // https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp#L341
        let device = q.device();
        let mut out_shape = q_l.shape().clone();
        let mut out_l = Layout::contiguous(&out_shape);

        if q.dtype() != k.dtype() {
            candle_core::bail!("query and key must have the same dtype");
        }

        if q.dtype() != v.dtype() {
            candle_core::bail!("query and value must have the same dtype");
        }

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

        let q_rank = q_l.rank();
        let k_rank = k_l.rank();
        let v_rank = v_l.rank();
        let o_rank = out_l.rank();

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

        let mut is_causal = if window_size_left < 0 && window_size_right == 0 {
            true
        } else {
            false
        };
        if seqlen_q == 1 && !self.alibi_slopes.is_some() {
            is_causal = false;
        }

        // TODO: Can this case be safely removed?
        // let seqlenq_ngroups_swapped = seqlen_q == 1
        //     && num_heads > num_heads_k
        //     && window_size_left < 0
        //     && window_size_right < 0
        //     && head_size_og % 8 == 0
        //     && self.alibi_slopes.is_none();
        // // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
        // if seqlenq_ngroups_swapped {
        //     let ngroups = num_heads / num_heads_k;
        //     let new_shape = Shape::from((batch_size, num_heads_k, ngroups, head_size_og));
        //     let mut new_layout = Layout::contiguous(&new_shape);
        //     new_layout.transpose(1, 2);
        //     *q_l = new_layout;
        //     *out_l = new_layout;
        // }

        let head_size = round_multiple(head_size_og, 8);
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<T>(elem_count) }.w()?;
        let softmax_lse = dev
            .alloc_zeros::<f32>(b_sz * 128 * num_heads * seqlen_q)
            .w()?;

        let is_bf16 = if is_bf16 { true } else { false };

        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = seqlen_k as i32;
        }

        let num_splits = compute_num_splits(
            b_sz,
            num_heads,
            head_size,
            seqlen_k,
            seqlen_q,
            head_size_rounded,
        );

        let mut softcap = self.softcap.unwrap_or(0.0);
        let (softmax_scale, scale_softmatx_log2) = if (softcap > 0.0) {
            softcap = self.softmax_scale / softcap;
            (softcap, softcap * M_LOG2E)
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
                /* num_splits */ num_splits as u32,
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

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, device.clone());
        Ok((dst, out_shape))
    }
}

fn compute_num_splits(
    batch_size: i32,
    num_heads: i32,
    head_size: i32,
    max_seqlen_k: i32,
    max_seqlen_q: i32,
    head_size_rounded: i32,
) -> i32 {
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
    let num_splits = num_splits_heuristic(
        batch_size * num_heads * num_m_blocks,
        dprops.multiProcessorCount * 2,
        num_n_blocks,
        128,
    );
    if num_splits > 128 {
        candle_core::bail!(("num_splits > 128 not supported".into(),))
    }
    num_splits
}
