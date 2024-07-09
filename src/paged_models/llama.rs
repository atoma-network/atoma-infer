use crate::pagedattention::{PagedAttention, PagedAttentionMetadata};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};
use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use serde::Serialize;
use std::collections::HashMap;

const MAX_SEQ_LEN: usize = 4096;

#[derive(Clone, Debug, Serialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl LlamaConfig {
    /// This will return the number of key-value heads
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl Config {
    /// Creates a new Config for version 1 of the 7 billion parameter model.
    pub fn new_7b_v1(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
        }
    }

    /// Creates a new Config for version 2 of the 7 billion parameter model.
    pub fn new_7b_v2(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
        }
    }
}

#[derive(Clone, Debug)]
/// Cache for Llama model, used for storing masks and precomputed cosine and sine tensors.
pub struct Cache {
    masks: HashMap<usize, Tensor>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl Cache {
    /// Constructor for Cache, initializes cosine and sine tensors used in rotary embeddings.
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // Precomputed frequency tensor for complex exponentials (cis)
        let n_elem = config.hidden_size / config.num_attention_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(&theta, device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.len()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: HashMap::new(),
            cos,
            sin,
            device: device.clone(),
        })
    }
}

struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
    cos_sin_cache: Cache,
    attention: PagedAttention,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    /// Applies rotary embeddings to the input tensor.
    fn apply_rotary_embed(&self, x: &Tensor, input_positions: &Tensor) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (b_sz, _, seq_len, hidden_size) = x.dims4()?; // [batch_size, _, sequence_length, hidden_size]

        // Ensure input_positions tensor has the correct dimensions and type
        if input_positions.dims() != [b_sz, seq_len] {
            candle_core::bail!(
            "index_positions must be of shape [batch_size, sequence_length] = [{}, {}], got {:?}",
            b_sz,
            seq_len,
            input_positions.dims()
        );
        }
        if input_positions.dtype() != DType::I64 {
            candle_core::bail!(
                "index_positions must be of dtype i64, got {:?}",
                input_positions.dtype()
            );
        }

        // Select cosine and sine values based on input_positions
        let cos = self
            .cos_sin_cache
            .cos
            .index_select(&input_positions.flatten(0, 1)?, 0)?;
        let sin = self
            .cos_sin_cache
            .sin
            .index_select(&input_positions.flatten(0, 1)?, 0)?;

        // Reshape cos and sin to match the input tensor shape
        let cos = cos.reshape((b_sz, seq_len, hidden_size))?;
        let sin = sin.reshape((b_sz, seq_len, hidden_size))?;

        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    /// Forward pass for the causal self-attention layer.
    fn forward(
        &mut self,
        x: &Tensor,
        input_positions: &Tensor,
        cache: &Tensor,
        attention_metadata: &mut PagedAttentionMetadata,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = self.apply_rotary_embed(&q, input_positions)?;
        let k = self.apply_rotary_embed(&k, input_positions)?;

        let y = self
            .attention
            .forward(&q, &k, &v, cache, &attention_metadata)?;

        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    /// Repeats key-value tensors to match the number of attention heads.
    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        candle_transformers::utils::repeat_kv(
            x,
            self.num_attention_heads / self.num_key_value_heads,
        )
    }

    /// Loads the CausalSelfAttention layer from a variable builder.
    fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
            attention: PagedAttention::new(
                cfg.num_attention_heads,
                head_dim,
                1. / (head_dim as f32).sqrt(),
                Some(cfg.num_key_value_heads),
                None,
                vb.device(),
                None,
            )?,
            cos_sin_cache: Cache::new(true, dtype, cfg, device)?,
        })
    }
}

#[derive(Clone, Debug)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    /// Forward pass for the MLP layer.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    /// Loads the MLP layer from a variable builder.
    fn load(vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
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
    /// Forward pass for the Transformer block.
    fn forward(
        &mut self,
        x: &Tensor,
        input_positions: &Tensor,
        cache: &Tensor,
        attention_metadata: &mut PagedAttentionMetadata,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = self
            .attn
            .forward(&x, input_positions, cache, attention_metadata)?
            + residual;
        let residual = &x;
        let x = self.mlp.forward(&self.rms_2.forward(&x)?)? + residual;
        Ok(x)
    }

    /// Loads the Transformer block from a variable builder.
    fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg, dtype, device)?;
        let mlp = Mlp::load(&vb.pp("mlp"), cfg)?;
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
    dtype: DType,
    device: Device,
}

impl Llama {
    /// Forward pass for the Llama model.
    pub fn forward(
        &mut self,
        x: &Tensor,
        input_positions: &Tensor,
        selected_token_indices: &Tensor,
        kv_caches: Vec<Tensor>,
        attention_metadata: &mut PagedAttentionMetadata,
    ) -> Result<Tensor> {
        let mut x = self.wte.forward(x)?;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            x = block.forward(&x, input_positions, &kv_caches[i], attention_metadata)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.index_select(selected_token_indices, 1)?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    /// Loads the Llama model from a variable builder.
    pub fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(&format!("model.layers.{i}")), cfg, dtype, device).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            cfg: cfg.clone(),
            dtype,
            device: device.clone(),
        })
    }

    /// This Returns the configuration of the Llama model.
    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}