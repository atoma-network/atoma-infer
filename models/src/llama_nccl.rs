use crate::{
    flash_attention::{FlashAttention, FlashAttentionMetadata},
    llama::{Config, Llama3RopeConfig, Llama3RopeType, LlamaConfig, LlamaEosToks},
    multi_gpu::{shard, TensorParallelColumnLinear, TensorParallelRowLinear},
};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{embedding, Embedding, Linear, Module, RmsNorm};
use cudarc::nccl::safe::Comm;
use std::rc::Rc;

/// Maximum sequence token length
const DEFAULT_MAX_SEQ_LEN: usize = 4096;

struct CausalSelfAttention {
    q_proj: TensorParallelColumnLinear,
    k_proj: TensorParallelColumnLinear,
    v_proj: TensorParallelColumnLinear,
    o_proj: TensorParallelRowLinear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    span: tracing::Span,
    span_rot: tracing::Span,
    cos_sin_cache: Cache,
    attention: FlashAttention,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, input_positions: &Tensor) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (b_sz, _num_heads, num_total_tokens, _hidden_size) = x.dims4()?; // [1, num_heads, num_total_tokens, hidden_size]

        if b_sz != 1 {
            candle_core::bail!("batch size must be 1, got {}", b_sz);
        }
        if input_positions.dims() != [1, num_total_tokens] {
            candle_core::bail!(
            "index_positions must be of shape [batch_size, sequence_length] = [{}, {}], got {:?}",
            b_sz,
            num_total_tokens,
            input_positions.dims()
        );
        }
        if input_positions.dtype() != DType::I64 {
            candle_core::bail!(
                "input_positions must be of dtype i64, got {:?}",
                input_positions.dtype()
            );
        }

        // select input positions tokens
        let cos = self
            .cos_sin_cache
            .cos
            .index_select(&input_positions.flatten(0, 1)?, 0)?;
        let sin = self
            .cos_sin_cache
            .sin
            .index_select(&input_positions.flatten(0, 1)?, 0)?;

        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(
        &mut self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_cache: &Tensor,
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let (batch_size, num_total_tokens, _hidden_size) = x.dims3()?;
        if batch_size != 1 {
            candle_core::bail!(
                "x must be of shape [1, num_total_tokens], got {:?}",
                x.dims()
            );
        }

        let _enter = self.span.enter();
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((
                batch_size,
                num_total_tokens,
                self.num_attention_heads,
                self.head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((
                batch_size,
                num_total_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.reshape((
            batch_size,
            num_total_tokens,
            self.num_key_value_heads,
            self.head_dim,
        ))?;

        let q = self.apply_rotary_embed(&q, input_positions)?;
        let k = self.apply_rotary_embed(&k, input_positions)?;

        // transpose the matrices back to [sequence_length, num_heads, head_dim]
        let q = q.transpose(1, 2)?.squeeze(0)?.contiguous()?;
        let k = k.transpose(1, 2)?.squeeze(0)?.contiguous()?;
        let v = v.squeeze(0)?;

        let o = self
            .attention
            .forward(&q, &k, &v, kv_cache, attention_metadata)?;

        let o = o.unsqueeze(0)?;
        let out = self.o_proj.forward(&o)?;

        Ok(out)
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        comm: Rc<Comm>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let rank = comm.rank();
        let span = tracing::span!(tracing::Level::TRACE, format!("attn-{}", rank));
        let span_rot = tracing::span!(tracing::Level::TRACE, format!("attn-rot-{}", rank));
        let q_proj = TensorParallelColumnLinear::load_multi(vb.clone(), &["q_proj"], comm.clone())?;
        let k_proj = TensorParallelColumnLinear::load_multi(vb.clone(), &["k_proj"], comm.clone())?;
        let v_proj = TensorParallelColumnLinear::load_multi(vb.clone(), &["v_proj"], comm.clone())?;
        let o_proj = TensorParallelRowLinear::load(vb.pp("o_proj"), comm.clone())?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            num_key_value_heads: cfg.num_key_value_heads() / comm.world_size(),
            head_dim,
            span,
            span_rot,
            attention: FlashAttention::new(
                cfg.num_attention_heads / comm.world_size(),
                cfg.num_key_value_heads / comm.world_size(),
                head_dim,
                1f32 / (head_dim as f32).sqrt(),
                None,
                None,
                dtype,
                device.clone(),
            )?,
            cos_sin_cache: Cache::new(dtype, cfg, device)?,
        })
    }
}

struct Mlp {
    c_fc1: TensorParallelColumnLinear,
    c_fc2: TensorParallelColumnLinear,
    c_proj: TensorParallelRowLinear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, _cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let span = tracing::span!(tracing::Level::TRACE, format!("mlp-{}"));
        let c_fc1 = TensorParallelColumnLinear::load(vb.pp("gate_proj"), comm.clone())?;
        let c_fc2 = TensorParallelColumnLinear::load(vb.pp("up_proj"), comm.clone())?;
        let c_proj = TensorParallelRowLinear::load(vb.pp("down_proj"), comm)?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &mut self,
        x: &Tensor,
        input_positions: &Tensor,
        cache: &Tensor,
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self
            .attn
            .forward(&x, input_positions, cache, attention_metadata)?
            + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(
        vb: VarBuilder,
        cache: &Cache,
        cfg: &Config,
        comm: Rc<Comm>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let rank = comm.rank();
        let span = tracing::span!(tracing::Level::TRACE, format!("block-{}", rank));
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cache, cfg, comm.clone())?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg, comm)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    cfg: Config,
}

impl Llama {
    /// Forward pass of multi-gpu Llama model, using
    /// flash attention kernels, with paged attention
    /// memory batching optimizations.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[1, num_total_tokens]`,
    ///     where `num_total_tokens = num_prefill_tokens + num_decode_tokens`
    /// * `input_positions` - Input positions tensor of shape `[1, num_total_tokens]`,
    ///     where `num_total_tokens = num_prefill_tokens + num_decode_tokens`.
    ///     it contains all input positions, so that rotary embeddings can be applied correctly
    /// * `selected_token_indices` - Selected token indices tensor of shape `[1, num_decode_tokens]`
    /// * `kv_caches` - KV caches with paged block arrangement for each model layer. Each tensor is of
    ///      shape `[num_blocks, block_size, num_heads, head_dim]`
    /// * `attention_metadata` - Flash attention metadata, that
    pub fn forward(
        &mut self,
        x: &Tensor,
        input_positions: &Tensor,
        selected_token_indices: &Tensor,
        kv_caches: &[&mut Tensor],
        attention_metadata: FlashAttentionMetadata,
    ) -> Result<Tensor> {
        if x.dims()[0] != 1 {
            candle_core::bail!(
                "x must be of shape [1, num_total_tokens], got {:?}",
                x.dims()
            );
        }
        let mut x = self.wte.forward(x)?;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            x = block.forward(&x, input_positions, kv_caches[i], &attention_metadata)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.index_select(selected_token_indices, 1)?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        comm: &Rc<Comm>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let wte = embedding(cfg, vb.pp("model.embed_tokens"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| {
                Block::load(
                    vb.pp(&format!("model.layers.{i}")),
                    cache,
                    cfg,
                    comm.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { 
            wte,
            blocks,
            ln_f: norm,
            lm_head,
            cfg: cfg.clone(),
        })
    }
}
