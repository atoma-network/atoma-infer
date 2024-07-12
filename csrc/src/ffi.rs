use core::ffi::{c_int, c_void};

extern "C" {
    pub fn mha_fwd(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        alibi_slopes: *const c_void,
        p_dropout: f32,
        softmax_scale: f32,
        is_causal: bool,
        window_size_left: i32,
        window_size_right: i32,
        softcap: f32,
        return_softmax: bool,
        gen: *mut c_void,
    );

    pub fn mha_varlen_fwd(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        cu_seqlens_q: *const c_int,
        cu_seqlens_k: *const c_int,
        seqused_k: *const c_int,
        block_table: *const c_int,
        alibi_slopes: *const c_void,
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        p_dropout: f32,
        softmax_scale: f32,
        zero_tensors: bool,
        is_causal: bool,
        window_size_left: i32,
        window_size_right: i32,
        softcap: f32,
        return_softmax: bool,
        gen: *mut c_void,
    );

    pub fn mha_fwd_kvcache(
        q: *const c_void,
        kcache: *const c_void,
        vcache: *const c_void,
        k_: *mut c_void,
        v_: *mut c_void,
        seqlens_k_: *mut c_int,
        rotary_cos_: *mut c_void,
        rotary_sin_: *mut c_void,
        cache_batch_idx_: *mut c_int,
        block_table_: *mut c_int,
        alibi_slopes_: *const c_void,
        out_: *mut c_void,
        softmax_scale: f32,
        is_causal: bool,
        window_size_left: i32,
        window_size_right: i32,
        softcap: f32,
        is_rotary_interleaved: bool,
        num_splits: i32,
    );
}
