to do!(backend impl)



pub struct PagedAttention {
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    sequence_lengths: Tensor,
    max_sequence_length: usize,
    kv_cache_dtype: String,
    num_kv_heads: i64,
    scale: f64,
    alibi_slopes: Option<Tensor>,
    kv_scale: f64,
}

pub fn paged_attention(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    sequence_lengths: &Tensor,
    max_sequence_length: usize,
    kv_cache_dtype: String,
    num_kv_heads: usize,
    scale: f64,
    alibi_slopes: Option<Tensor>,
    kv_scale: f64,
) -> Result<Tensor> {
    let attention = PagedAttention {
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        block_tables: block_tables.clone(),
        sequence_lengths: sequence_lengths.clone(),
        max_sequence_length,
        kv_cache_dtype,
        num_kv_heads: num_kv_heads as i64,
        scale,
        alibi_slopes,
        kv_scale,
    };
    query.apply_op1(attention)
}
