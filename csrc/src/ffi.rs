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

        q_batch_stride: i64,
        k_batch_stride: i64,
        v_batch_stride: i64,
        o_batch_stride: i64,
        alibi_slopes_batch_stride: i64,

        q_row_stride: i64,
        k_row_stride: i64,
        v_row_stride: i64,
        o_row_stride: i64,

        q_head_stride: i64,
        k_head_stride: i64,
        v_head_stride: i64,
        o_head_stride: i64,

        num_splits: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,
        scale_softmatx_log2: f32,

        block_table: *const c_int,
        block_table_batch_stride: i64,
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

    pub(crate) fn copy_blocks_f16(
        key_cache_ptrs: *const c_void,
        value_cache_ptrs: *const c_void,
        block_mapping: *const c_void,
        num_layers: i64,
        num_pairs: i64,
        numel_per_block: i64,
        stream: *mut c_void,
    );

    pub(crate) fn copy_blocks_bf16(
        key_cache_ptrs: *const c_void,
        value_cache_ptrs: *const c_void,
        block_mapping: *const c_void,
        num_layers: i64,
        num_pairs: i64,
        numel_per_block: i64,
        stream: *mut c_void,
    );

    pub(crate) fn reshape_and_cache_flash(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        slot_mapping: *const i64,
        block_stride: i64,
        num_tokens: i64,
        num_heads: i64,
        head_size: i64,
        block_size: i64,
        key_stride: i64,
        value_stride: i64,
        dtype: i32,
    );
}
