use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn run_mha(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        alibi_slopes_ptr: *const c_void,

        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,

        is_seqlens_k_cumulative: bool,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,
        alibi_slopes_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,

        num_splits: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,
        scale_softmatx_log2: f32,

        block_table: *const c_int,
        block_table_batch_stride: u32,
        page_block_size: c_int,

        seqused_k: *const c_int,
        seqlen_q: u32,
        seqlen_k: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,

        is_bf16: c_int,
        is_causal: c_int,

        window_size_left: c_int,
        window_size_right: c_int,
        softcap: f32,
        unpadded_lse: bool,
        force_split_kernel: bool,
    );
}
