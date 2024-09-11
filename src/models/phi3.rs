/// A Rust implementation of the Phi3 model, a transformer-based language model.
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;
use std::sync::Arc;

use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};

use crate::flash_attention::{FlashAttention, FlashAttentionMetadata};
use candle_transformers::utils;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Phi3Config {
    pub vocab_size: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub resid_pdrop: f64,
    pub embd_pdrop: Option<f64>,
    pub attn_pdrop: Option<f64>,
    pub model_type: String,
    pub torch_dtype: Option<String>,
    pub rope_scaling: Option<serde_json::Value>,
    pub attention_bias: bool,
    pub initializer_range: f64,
    pub original_max_position_embeddings: usize,
    pub pad_token_id: Option<u32>,
    pub sliding_window: Option<usize>,
}

impl Phi3Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Phi3Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?
            .to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.index_select(&input_positions.flatten(0, 1)?, 0)?;
        let sin = self.sin.index_select(&input_positions.flatten(0, 1)?, 0)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

use std::fmt;

struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    attention: FlashAttention,
    dtype: DType,
    device: Device,
}

// Manually implement Debug
impl fmt::Debug for Attention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Attention")
            .field("qkv_proj", &self.qkv_proj)
            .field("o_proj", &self.o_proj)
            .field("num_heads", &self.num_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("num_kv_groups", &self.num_kv_groups)
            .field("head_dim", &self.head_dim)
            .field("rotary_emb", &self.rotary_emb)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

// Manually implement Clone
impl Clone for Attention {
    fn clone(&self) -> Self {
        Self {
            qkv_proj: self.qkv_proj.clone(),
            o_proj: self.o_proj.clone(),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            num_kv_groups: self.num_kv_groups,
            head_dim: self.head_dim,
            rotary_emb: self.rotary_emb.clone(),
            attention: FlashAttention::new(
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                1f32 / (self.head_dim as f32).sqrt(),
                None,
                None,
                self.dtype,
                self.device.clone(),
            ).unwrap(),
            dtype: self.dtype,
            device: self.device.clone(),
        }
    }
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Phi3Config, vb: VarBuilder, dtype: DType, device: Device) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = linear(cfg.hidden_size, op_size, vb.pp("qkv_proj"))?;
        let o_proj = linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            attention: FlashAttention::new(
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                head_dim,
                1f32 / (head_dim as f32).sqrt(),
                None,
                None,
                dtype,
                device.clone(),
            )?,
            dtype,
            device,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        input_positions: &Tensor,
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let query_pos = self.num_heads * self.head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?;
        let key_states = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, input_positions)?;

        let key_states = utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let attn_output = self.attention.forward(&query_states, &key_states, &value_states, &key_states, attention_metadata)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Phi3Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = linear(hidden_size, 2 * i_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear(i_size, hidden_size, vb.pp("down_proj"))?;
        let act_fn = match cfg.hidden_act.as_str() {
            "silu" => candle_nn::Activation::Silu,
            _ => return Err(candle_core::Error::Msg(format!("Unsupported activation function: {}", cfg.hidden_act))),
        };
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn,
            i_size,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = xs.apply(&self.gate_up_proj)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        up_states.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Phi3Config, vb: VarBuilder, dtype: DType, device: &Device) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"), dtype, device.clone())?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        input_positions: &Tensor,
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, input_positions, attention_metadata)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Phi3Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), vb.dtype(), vb.device())?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, input_positions: &Tensor, attention_metadata: &FlashAttentionMetadata) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, input_positions, attention_metadata)?
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::io::{Read, Write};
/// A Rust implementation of the Phi3 model, a transformer-based language model.
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;
use std::sync::Arc;

use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};

use crate::flash_attention::{FlashAttention, FlashAttentionMetadata};
use candle_transformers::utils;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Phi3Config {
    pub vocab_size: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub resid_pdrop: f64,
    pub embd_pdrop: Option<f64>,
    pub attn_pdrop: Option<f64>,
    pub model_type: String,
    pub torch_dtype: Option<String>,
    pub rope_scaling: Option<serde_json::Value>,
    pub attention_bias: bool,
    pub initializer_range: f64,
    pub original_max_position_embeddings: usize,
    pub pad_token_id: Option<u32>,
    pub sliding_window: Option<usize>,
}

impl Phi3Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Phi3Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?
            .to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.index_select(&input_positions.flatten(0, 1)?, 0)?;
        let sin = self.sin.index_select(&input_positions.flatten(0, 1)?, 0)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

use std::fmt;

struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    attention: FlashAttention,
    dtype: DType,
    device: Device,
}

// Manually implement Debug
impl fmt::Debug for Attention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Attention")
            .field("qkv_proj", &self.qkv_proj)
            .field("o_proj", &self.o_proj)
            .field("num_heads", &self.num_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("num_kv_groups", &self.num_kv_groups)
            .field("head_dim", &self.head_dim)
            .field("rotary_emb", &self.rotary_emb)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

// Manually implement Clone
impl Clone for Attention {
    fn clone(&self) -> Self {
        Self {
            qkv_proj: self.qkv_proj.clone(),
            o_proj: self.o_proj.clone(),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            num_kv_groups: self.num_kv_groups,
            head_dim: self.head_dim,
            rotary_emb: self.rotary_emb.clone(),
            attention: FlashAttention::new(
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                1f32 / (self.head_dim as f32).sqrt(),
                None,
                None,
                self.dtype,
                self.device.clone(),
            ).unwrap(),
            dtype: self.dtype,
            device: self.device.clone(),
        }
    }
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Phi3Config, vb: VarBuilder, dtype: DType, device: Device) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = linear(cfg.hidden_size, op_size, vb.pp("qkv_proj"))?;
        let o_proj = linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            attention: FlashAttention::new(
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                head_dim,
                1f32 / (head_dim as f32).sqrt(),
                None,
                None,
                dtype,
                device.clone(),
            )?,
            dtype,
            device,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        input_positions: &Tensor,
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let query_pos = self.num_heads * self.head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?;
        let key_states = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, input_positions)?;

        let key_states = utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let attn_output = self.attention.forward(&query_states, &key_states, &value_states, &key_states, attention_metadata)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Phi3Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = linear(hidden_size, 2 * i_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear(i_size, hidden_size, vb.pp("down_proj"))?;
        let act_fn = match cfg.hidden_act.as_str() {
            "silu" => candle_nn::Activation::Silu,
            _ => return Err(candle_core::Error::Msg(format!("Unsupported activation function: {}", cfg.hidden_act))),
        };
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn,
            i_size,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = xs.apply(&self.gate_up_proj)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        up_states.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Phi3Config, vb: VarBuilder, dtype: DType, device: &Device) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"), dtype, device.clone())?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        input_positions: &Tensor,
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, input_positions, attention_metadata)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Phi3Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), vb.dtype(), vb.device())?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, input_positions: &Tensor, attention_metadata: &FlashAttentionMetadata) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, input_positions, attention_metadata)?
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest;
    use serde_json::json;
    use std::env;

    #[test]
    #[serial]
    fn test_phi3_model_long() -> Result<()> {
        let prompt = "The History of France starts in ".to_string();

        let client = reqwest::blocking::Client::new();
        let api_url = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct/v1/chat/completions";
        let api_token = env::var("HF_API_TOKEN").expect("HF_API_TOKEN not set");

        println!("Starting inference using Hugging Face API");
        print!("{prompt}");

        let response = client.post(api_url)
            .header("Authorization", format!("Bearer {}", api_token))
            .header("Content-Type", "application/json")
            .json(&json!({
                "model": "microsoft/Phi-3-mini-4k-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "stream": false
            }))
            .send()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to send request: {}", e)))?;

        if response.status().is_success() {
            let result: serde_json::Value = response.json()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to parse JSON response: {}", e)))?;
            if let Some(content) = result["choices"][0]["message"]["content"].as_str() {
                println!("API Response: {}", content);
                // You can add assertions here to verify the response content
                assert!(!content.is_empty(), "Response content should not be empty");
                assert!(content.contains("France"), "Response should mention France");
            } else {
                return Err(candle_core::Error::Msg("Unexpected response format".to_string()));
            }
        } else {
            let error_text = response.text()
                .map_err(|e| candle_core::Error::Msg(format!("Failed to get error response text: {}", e)))?;
            return Err(candle_core::Error::Msg(format!("API Error: {}. Response: {}", response.status(), error_text)));
        }

        Ok(())
    }
}
