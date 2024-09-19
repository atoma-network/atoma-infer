use crate::flash_attention::{FlashAttention, FlashAttentionMetadata};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, Embedding, Activation, VarBuilder};

/// Mistral LLM, https://github.com/mistralai/mistral-src
use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use std::sync::Arc;

const DEFAULT_MAX_SEQ_LEN: usize = 4096;

fn default_use_flash_attn() -> bool {
    false
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum MistralRopeType {
    #[serde(rename = "mistral")]
    Mistral,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct MistralRopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: MistralRopeType,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum MistralEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct MistralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    #[serde(default = "default_use_flash_attn")]
    pub use_flash_attn: bool,
}

impl MistralConfig {
    // https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl MistralConfig {
    pub fn into_config(self) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            sliding_window: self.sliding_window,
            use_flash_attn: self.use_flash_attn,
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
    pub head_dim: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f64>,
    pub sliding_window: Option<usize>,
    pub use_flash_attn: bool,
    pub rope_theta: Option<f32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<MistralEosToks>,
    pub rope_scaling: Option<MistralRopeConfig>,
}

impl Config {
    // https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
    pub fn config_7b_v0_1(use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            head_dim: None,
            num_key_value_heads: 8,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: Some(10_000.),
            sliding_window: Some(4096),
            use_flash_attn,
        }
    }

    // https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca/blob/main/config.json
    // https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/config.json
    pub fn config_chat_ml(use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 32002,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            head_dim: None,
            num_key_value_heads: 8,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            sliding_window: Some(4096),
            use_flash_attn,
        }
    }

    // https://huggingface.co/amazon/MistralLite/blob/main/config.json
    pub fn config_amazon_mistral_lite(use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 32003,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            head_dim: None,
            num_key_value_heads: 8,
            hidden_act: Activation::Silu,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            sliding_window: Some(4096),
            use_flash_attn,
        }
    }

    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

pub struct Cache {
    cos: Tensor,
    sin: Tensor,
}
fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        let theta = match &config.rope_scaling {
            None
            | Some(MistralRopeConfig {
                rope_type: MistralRopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|inv_freq| inv_freq * rope_scaling.factor)
                    .collect()
            }
        };
        let cos = Tensor::empty(
            &[config.max_position_embeddings, config.head_dim],
            dtype,
            device,
        )?;
        let sin = Tensor::empty(
            &[config.max_position_embeddings, config.head_dim],
            dtype,
            device,
        )?;
        Ok(Self { cos, sin })
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
    cfg: Config,
    span: tracing::Span,
}

impl MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            cfg: cfg.clone(),
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    use_flash_attn: bool,
    cfg: Config,
    span: tracing::Span,
    span_rot: tracing::Span,
}

impl CausalSelfAttention {
    fn apply_rotary_embed(&self, x: &Tensor, input_positions: &Tensor) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (b_sz, _num_heads, num_total_tokens, _hidden_size) = x.dims4()?; // [1, num_heads, num_total_tokens, hidden_size]
        let seqlen_offset = 0;
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
        let cos = self.cos_sin_cache.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.cos_sin_cache.sin.narrow(0, seqlen_offset, seq_len)?;
    }

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
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let cos_sin_cache = rotary_emb.cos_sin_cache;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_key_value_heads / cfg.num_attention_heads,
            head_dim,
            cos_sin_cache,
            kv_cache: None,
            use_flash_attn: cfg.use_flash_attn,
            cfg: cfg.clone(),
            span,
            span_rot,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        input_positions: &Tensor,
    ) -> Result<Tensor> {
        let q = xs.apply(&self.q_proj)?;
        let k = xs.apply(&self.k_proj)?;
        let v = xs.apply(&self.v_proj)?;
        let q = self.apply_rotary_embed(&q, input_positions)?;
        let k = self.apply_rotary_embed(&k, input_positions)?;
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

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
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
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: Attention,
    rms_2: RmsNorm,
    mlp: MLP,
    span: tracing::Span,
    span_rot: tracing::Span,
    cfg: Config,
}

impl Block {
    fn forward(
        &mut self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        input_positions: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, attention_mask, input_positions)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }
    fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = Attention::load(vb.pp("self_attn"), cfg, dtype, device)?;
        let mlp = MLP::load(&vb.pp("mlp"), cfg)?;
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
            span_rot,
            cfg: cfg.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Mistral {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    sliding_window: Option<usize>,
    device: Device,
    dtype: DType,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Mistral {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(self.rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            rotary_emb,
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let sliding_window = self.sliding_window.unwrap_or(tgt_len + 1);
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((1, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(seq_len, seqlen_offset)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }
    pub fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cfg: cfg.clone(),
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            rotary_emb,
        })
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}

impl Default for MistralConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
        }
    }
}

mod tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionDecodingMetadata, FlashAttentionPrefillMetadata};
    use candle_core::IndexOp;
    use candle_transformers::generation::{LogitsProcessor, Sampling};
    use hf_hub::{api::sync::Api, Repo, RepoType};
    use rand::Rng;
    use serial_test::serial;
    use std::io::Write;
    use tokenizers::Tokenizer;

    const EOS_TOKEN: &str = "</s>";

    #[test]
    fn test_mistral_model() -> Result<()> {
        let prompt = "The capital of France is ".to_string();

        let dtype = DType::BF16;
        let device = Device::new_cuda(0).unwrap();
        let model_id = "mistralai/Mistral-7B-Instruct-v0.3".to_string();
        let revision = "main".to_string();
        let api = Api::new().expect("Failed to create the HF API");

        println!("loading the model weights from {model_id}");
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api
            .get("tokenizer.json")
            .expect("Failed to get tokenizer.json");
        let config_filename = api.get("config.json").expect("Failed to get config.json");
        let config: MistralConfig = serde_json::from_slice(
            &std::fs::read(config_filename).expect("Failed to read config.json"),
        )
        .expect("Failed to deserialize config.json");
        let config = config.into_config();

        let filenames = vec![api
            .get("model.safetensors")
            .expect("Failed to get model.safetensors")];
        let mut mistral_model = {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            Mistral::load(vb, &config, dtype, &device).expect("Failed to load the model")
        };
        let tokenizer =
            Tokenizer::from_file(tokenizer_filename).expect("Failed to load the tokenizer");
        let eos_token_id = config
            .eos_token_id
            .clone()
            .or_else(|| tokenizer.token_to_id(EOS_TOKEN).map(MistralEosToks::Single));

        let mut tokens = tokenizer
            .encode(prompt.clone(), true)
            .expect("Failed to encode the prompt")
            .get_ids()
            .to_vec();

        let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);
        println!("starting the inference loop");
        print!("{prompt}");

        let mut logits_processor = {
            let temperature = 0.8;
            let sampling = Sampling::All { temperature };
            LogitsProcessor::from_sampling(42, sampling)
        };

        let sample_len = 32;
        let start_gen = std::time::Instant::now();
        let mut token_generated = 0;

        // kv cache
        let num_blocks = 100;
        let block_size = 16;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let mut kv_cache = std::iter::repeat_with(|| {
            Tensor::zeros(
                (2, num_blocks, block_size, num_key_value_heads, head_dim),
                dtype,
                &device,
            )
        })
        .take(config.num_hidden_layers)
        .collect::<Result<Vec<_>>>()?;

        let kv_cache = kv_cache.iter_mut().collect::<Vec<_>>();

        // prefill forward pass
        let input_positions = Tensor::arange(0, tokens.len() as i64, &device)?.unsqueeze(0)?;
        let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
        let attention_metadata = FlashAttentionMetadata {
            context_lengths: Some(Tensor::from_vec(vec![tokens.len() as u32], (1,), &device)?),
            slot_mapping: Tensor::arange(0, tokens.len() as i64, &device)?,
            decoding_metadata: None,
            num_prefill_tokens: tokens.len(),
            num_decoding_tokens: 0,
            prefill_metadata: Some(FlashAttentionPrefillMetadata {
                block_tables: None,
                max_query_length: Some(tokens.len()),
                max_prefill_sequence_length: tokens.len(),
                query_start_locations: Some(Tensor::from_vec(
                    vec![0, tokens.len() as u32],
                    (2,),
                    &device,
                )?),
                sequence_start_locations: Some(Tensor::from_vec(
                    vec![0, tokens.len() as u32],
                    (2,),
                    &device,
                )?),
                sequence_lengths: Some(Tensor::from_vec(vec![tokens.len() as u32], (1,), &device)?),
            }),
        };
        let logits = mistral_model.forward(
            &input,
            &input_positions,
            &Tensor::new(vec![tokens.len() as u32 - 1], &device)?,
            &kv_cache,
            attention_metadata,
        )?;
        let logits = logits.squeeze(0)?.squeeze(0)?;

        let mut next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        // decoding loop
        for _ in 1..sample_len {
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let input_positions = Tensor::new(&[tokens.len() as i64 - 1], &device)?.unsqueeze(0)?;
            let selected_token_indices = Tensor::new(&[0u32], &device)?;
            let num_blocks = (tokens.len() / block_size) as i64 + 1;
            let attention_metadata = FlashAttentionMetadata {
                context_lengths: Some(Tensor::from_vec(vec![tokens.len() as u32], (1,), &device)?),
                slot_mapping: Tensor::arange(0, tokens.len() as i64, &device)?,
                decoding_metadata: Some(FlashAttentionDecodingMetadata {
                    block_tables: Some(
                        Tensor::arange(0, num_blocks, &device)?
                            .to_dtype(DType::U32)?
                            .reshape((1, num_blocks as usize))?,
                    ),
                    max_decoding_sequence_length: tokens.len(),
                    sequence_lengths: Some(Tensor::new(&[tokens.len() as u32], &device)?),
                }),
                prefill_metadata: None,
                num_prefill_tokens: 0,
                num_decoding_tokens: 1,
            };
            let logits = mistral_model
                .forward(
                    &input,
                    &input_positions,
                    &selected_token_indices,
                    &kv_cache,
                    attention_metadata,
                )?
                .squeeze(0)?
                .squeeze(0)?;

            next_token = logits_processor.sample(&logits)?;
            token_generated += 1;
            tokens.push(next_token);

            match eos_token_id {
                Some(MistralEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                    break;
                }
                Some(MistralEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                    break;
                }
                _ => (),
            }
            if let Some(t) = tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }

        if let Some(rest) = tokenizer.decode_rest().unwrap() {
            print!("{rest}");
        }

        let dt = start_gen.elapsed();
        println!(
            "\n\n{} tokens generated ({} token/s)\n",
            token_generated,
            (token_generated - 1) as f64 / dt.as_secs_f64(),
        );

        Ok(())
    }
}
