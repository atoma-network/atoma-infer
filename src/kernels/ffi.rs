use core::ffi::{c_int, c_long, c_void, torch::Tensor};

extern "C" {
    pub fn copy_blocks(
        key_caches: *const *const c_void,
        key_caches_len: usize,
        value_caches: *const *const c_void,
        value_caches_len: usize,
        block_mapping: *const c_void,
    );

    pub fn paged_attention_v1(
        out: *mut c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        num_kv_heads: i64,
        scale: f64,
        block_tables: *const c_void,
        seq_lens: *const c_void,
        block_size: i64,
        max_seq_len: i64,
        alibi_slopes: *const c_void,
        kv_cache_dtype: *const i8,
        kv_scale: f64,
        tp_rank: i64,
        blocksparse_local_blocks: i64,
        blocksparse_vert_stride: i64,
        blocksparse_block_size: i64,
        blocksparse_head_sliding_step: i64,
    );

    pub fn paged_attention_v2(
        out: *mut c_void,
        exp_sums: *mut c_void,
        max_logits: *mut c_void,
        tmp_out: *mut c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        num_kv_heads: i64,
        scale: f64,
        block_tables: *const c_void,
        seq_lens: *const c_void,
        block_size: i64,
        max_seq_len: i64,
        alibi_slopes: *const c_void,
        kv_cache_dtype: *const i8,
        kv_scale: f64,
        tp_rank: i64,
        blocksparse_local_blocks: i64,
        blocksparse_vert_stride: i64,
        blocksparse_block_size: i64,
        blocksparse_head_sliding_step: i64,
    );

    pub fn reshape_and_cache(
        key: *const c_void,
        value: *const c_void,
        key_cache: *mut c_void,
        value_cache: *mut c_void,
        slot_mapping: *const c_void,
        kv_cache_dtype: *const i8,
        kv_scale: f64,
    );

    pub fn swap_blocks(src: *const c_void, dst: *mut c_void, block_mapping: *const c_void);
}
