/// A Rust implementation of the Phi3 model, a transformer-based language model.
use candle_core::{DType, Device, Module, Result, Tensor};

use candle_nn::VarBuilder;
use std::sync::Arc;

use candle_transformers::models::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};

use crate::flash_attention::{FlashAttention, FlashAttentionMetadata};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MistralConfig {
    pub vocab_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<String>,
    pub max_position_embeddings: usize,
}

impl MistralConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    pub fn into_config(self) -> MistralConfig {
        MistralConfig {
            vocab_size: self.vocab_size,
            hidden_act: self.hidden_act,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &MistralConfig, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_embed(&self, x: &Tensor, input_positions: &Tensor) -> Result<Tensor> {
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
        let cos = self.cos.index_select(&input_positions.flatten(0, 1)?, 0)?;
        let sin = self.sin.index_select(&input_positions.flatten(0, 1)?, 0)?;

        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }
}

use std::fmt;

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
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
            .field("q_proj", &self.q_proj)
            .field("k_proj", &self.k_proj)
            .field("v_proj", &self.v_proj)
            .field("o_proj", &self.o_proj)
            .field("num_heads", &self.num_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("num_kv_groups", &self.num_kv_groups)
            .field("head_dim", &self.head_dim)
            .field("rotary_emb", &self.rotary_emb)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .finish()
    }
}

// Manually implement Clone
impl Clone for Attention {
    fn clone(&self) -> Self {
        Self {
            q_proj: self.q_proj.clone(),
            k_proj: self.k_proj.clone(),
            v_proj: self.v_proj.clone(),
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
            )
            .unwrap(),
            dtype: self.dtype,
            device: self.device.clone(),
        }
    }
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &MistralConfig,
        vb: VarBuilder,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let q_proj = linear(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            attention: FlashAttention::new(
                num_heads,
                num_kv_heads,
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
        kv_cache: &Tensor,
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let (batch_size, num_total_tokens, _hidden_size) = xs.dims3()?;
        if batch_size != 1 {
            candle_core::bail!(
                "x must be of shape [1, num_total_tokens], got {:?}",
                xs.dims()
            );
        }

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((batch_size, num_total_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key_states = key_states
            .reshape((
                batch_size,
                num_total_tokens,
                self.num_kv_heads,
                self.head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let value_states = value_states.reshape((
            batch_size,
            num_total_tokens,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        let query_states = self
            .rotary_emb
            .apply_rotary_embed(&query_states, input_positions)?;
        let key_states = self
            .rotary_emb
            .apply_rotary_embed(&key_states, input_positions)?;

        let query_states = query_states.transpose(1, 2)?.squeeze(0)?.contiguous()?;
        let key_states = key_states.transpose(1, 2)?.squeeze(0)?.contiguous()?;
        let value_states = value_states.squeeze(0)?;

        let attn_output = self.attention.forward(
            &query_states,
            &key_states,
            &value_states,
            kv_cache,
            attention_metadata,
        )?;

        attn_output.unsqueeze(0)?.apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &MistralConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
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
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &MistralConfig,
        vb: VarBuilder,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
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
        kv_cache: &Tensor,
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, input_positions, kv_cache, attention_metadata)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MistralModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl MistralModel {
    pub fn load(
        vb: VarBuilder,
        cfg: &MistralConfig,
        _dtype: DType,
        _device: &Device,
    ) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                vb.dtype(),
                vb.device(),
            )?;
            layers.push(layer)
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
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

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        selected_token_indices: &Tensor,
        kv_caches: &[&mut Tensor],
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let (b_size, _seq_len) = input_ids.dims2()?;
        if b_size != 1 {
            candle_core::bail!(
                "x must be of shape [1, num_total_tokens], got {:?}",
                input_ids.dims()
            );
        }
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (layer, kv_cache) in self.layers.iter_mut().zip(kv_caches.iter()) {
            xs = layer.forward(&xs, input_positions, kv_cache, attention_metadata)?
        }
        xs = self.norm.forward(&xs)?;
        let xs = xs.index_select(selected_token_indices, 1)?.contiguous()?;
        let logits = self.lm_head.forward(&xs)?;

        logits.to_dtype(DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionDecodingMetadata, FlashAttentionPrefillMetadata};
    use candle_core::IndexOp;
    use candle_transformers::generation::{LogitsProcessor, Sampling};
    use hf_hub::{
        api::sync::{Api, ApiBuilder},
        Repo, RepoType,
    };
    use serial_test::serial;
    use std::io::Write;
    use tokenizers::Tokenizer;

    const EOS_TOKEN: &str = "ӏ ";
    const BLOCK_SIZE: usize = 16;

    #[test]
    #[serial]
    fn test_mistral_model() -> Result<()> {
        let prompt = "The History of France ".to_string();

        let dtype = DType::BF16;
        let device = Device::new_cuda(0).unwrap();
        let model_id = "mistralai/Mistral-7B-Instruct-v0.3".to_string();
        let revision = "main".to_string();
        let api = if let Ok(token) = std::env::var("HUGGING_FACE_TOKEN") {
            println!("Using provided Hugging Face token");
            ApiBuilder::new()
                .with_token(Some(token))
                .build()
                .expect("Failed to create the HF API")
        } else {
            println!("HUGGING_FACE_TOKEN not set, proceeding without authentication");
            Api::new().expect("Failed to create the HF API")
        };

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

        let filenames = vec![
            api.get("model-00001-of-00003.safetensors")
                .expect("Failed to get model.safetensors"),
            api.get("model-00002-of-00003.safetensors")
                .expect("Failed to get model.safetensors"),
            api.get("model-00003-of-00003.safetensors")
                .expect("Failed to get model.safetensors"),
        ];
        let mut mistral_model = {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            MistralModel::load(vb, &config, dtype, &device).expect("Failed to load the model")
        };

        //tests
        let tokenizer =
            Tokenizer::from_file(tokenizer_filename).expect("Failed to load the tokenizer");
        let eos_token_id = config
            .eos_token_id
            .or_else(|| tokenizer.token_to_id(EOS_TOKEN));

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

        let num_layers = mistral_model.layers.len();
        let num_blocks = 100;
        let block_size = BLOCK_SIZE;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();

        let mut kv_caches = std::iter::repeat_with(|| {
            Tensor::zeros(
                (2, num_blocks, block_size, num_key_value_heads, head_dim),
                dtype,
                &device,
            )
        })
        .take(num_layers)
        .collect::<Result<Vec<_>>>()?;

        let kv_caches_refs: Vec<&mut Tensor> = kv_caches.iter_mut().collect();

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
            &kv_caches_refs,
            &attention_metadata,
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
            let _selected_token_indices = Tensor::new(&[0u32], &device)?;
            let num_blocks = (tokens.len() / BLOCK_SIZE) as i64 + 1;
            let attention_metadata = FlashAttentionMetadata {
                context_lengths: None,
                slot_mapping: Tensor::new(&[tokens.len() as i64 - 1], &device)?,
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
            let selected_token_indices = Tensor::new(&[0u32], &device)?;
            let logits = mistral_model
                .forward(
                    &input,
                    &input_positions,
                    &selected_token_indices,
                    &kv_caches_refs,
                    &attention_metadata,
                )?
                .squeeze(0)?
                .squeeze(0)?;

            next_token = logits_processor.sample(&logits)?;
            token_generated += 1;
            tokens.push(next_token);

            if Some(next_token) == eos_token_id {
                break;
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

    //  cargo test test_mistral_model_batch -- --exact
    #[test]
    #[serial]
    fn test_mistral_model_batch() -> Result<()> {
        let prompts = vec![
            "The History of France ".to_string(),
            "The History of Germany ".to_string(),
        ];

        let batch_size = prompts.len();

        let dtype = DType::BF16;
        let device = Device::new_cuda(0).unwrap();

        let model_id = "mistralai/Mistral-7B-Instruct-v0.3".to_string();
        let revision = "main".to_string();
        let api = if let Ok(token) = std::env::var("HUGGING_FACE_TOKEN") {
            println!("Using provided Hugging Face token");
            ApiBuilder::new()
                .with_token(Some(token))
                .build()
                .expect("Failed to create the HF API")
        } else {
            println!("HUGGING_FACE_TOKEN not set, proceeding without authentication");
            Api::new().expect("Failed to create the HF API")
        };

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

        let filenames = vec![
            api.get("model-00001-of-00003.safetensors")
                .expect("Failed to get model.safetensors"),
            api.get("model-00002-of-00003.safetensors")
                .expect("Failed to get model.safetensors"),
            api.get("model-00003-of-00003.safetensors")
                .expect("Failed to get model.safetensors"),
        ];
        let mut mistral_model = {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            MistralModel::load(vb, &config, dtype, &device).expect("Failed to load the model")
        };

        let tokenizer =
            Tokenizer::from_file(tokenizer_filename).expect("Failed to load the tokenizer");
        let eos_token_id = config
            .eos_token_id
            .or_else(|| tokenizer.token_to_id(EOS_TOKEN));

        let mut tokens = prompts
            .iter()
            .map(|prompt| {
                tokenizer
                    .encode(prompt.clone(), true)
                    .expect("Failed to encode the prompt")
                    .get_ids()
                    .to_vec()
            })
            .collect::<Vec<_>>();

        let mut tokenizers = std::iter::repeat_with(|| {
            candle_examples::token_output_stream::TokenOutputStream::new(tokenizer.clone())
        })
        .take(batch_size)
        .collect::<Vec<_>>();
        println!("starting the inference loop");
        for prompt in prompts.iter() {
            println!("{prompt}");
        }

        let mut logits_processors = {
            let temperature = 0.8;
            let sampling = Sampling::All { temperature };
            std::iter::repeat_with(|| LogitsProcessor::from_sampling(42, sampling.clone()))
                .take(prompts.len())
                .collect::<Vec<_>>()
        };

        let sample_len = 64;
        let start_gen = std::time::Instant::now();
        let mut token_generated = 0;

        let num_prefill_tokens = tokens.iter().map(|ts| ts.len()).sum::<usize>();
        let max_tokens_len = tokens.iter().map(|ts| ts.len()).max().unwrap();
        let token_size_allocation =
            ((max_tokens_len + sample_len + BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE;

        let num_layers = mistral_model.layers.len();
        let num_blocks = 100;
        let block_size = BLOCK_SIZE;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();

        let mut kv_caches = std::iter::repeat_with(|| {
            Tensor::zeros(
                (2, num_blocks, block_size, num_key_value_heads, head_dim),
                dtype,
                &device,
            )
        })
        .take(num_layers)
        .collect::<Result<Vec<_>>>()?;

        let kv_caches_refs: Vec<&mut Tensor> = kv_caches.iter_mut().collect();

        // prefill forward pass
        let input_positions = Tensor::from_vec(
            tokens
                .iter()
                .flat_map(|ts| (0..(ts.len() as i64)))
                .collect::<Vec<_>>(),
            (1, num_prefill_tokens),
            &device,
        )?;
        let input = Tensor::from_vec(
            tokens.clone().into_iter().flatten().collect(),
            (1, num_prefill_tokens),
            &device,
        )?;
        let sequence_start_locs = {
            let mut result = Vec::with_capacity(tokens.len() + 1);
            result.push(0); // Start with 0
            tokens.iter().fold(0, |acc, x| {
                let sum = acc + x.len() as u32;
                result.push(sum);
                sum
            });
            result
        };
        let context_lengths = Some(Tensor::from_vec(
            tokens.iter().map(|ts| ts.len() as u32).collect(),
            (tokens.len(),),
            &device,
        )?);
        let slot_mapping = Tensor::from_vec(
            tokens
                .iter()
                .enumerate()
                .flat_map(|(i, ts)| {
                    ((i * token_size_allocation) as i64)
                        ..((i * token_size_allocation + ts.len()) as i64)
                })
                .collect(),
            (num_prefill_tokens,),
            &device,
        )?;
        let query_start_locations = Some(Tensor::from_vec(
            sequence_start_locs.clone(),
            (tokens.len() + 1,),
            &device,
        )?);
        let sequence_start_locations = Some(Tensor::from_vec(
            sequence_start_locs,
            (tokens.len() + 1,),
            &device,
        )?);
        let sequence_lengths = Some(Tensor::from_vec(
            tokens.iter().map(|ts| ts.len() as u32).collect(),
            (tokens.len(),),
            &device,
        )?);
        let attention_metadata = FlashAttentionMetadata {
            context_lengths,
            slot_mapping,
            decoding_metadata: None,
            num_prefill_tokens,
            num_decoding_tokens: 0,
            prefill_metadata: Some(FlashAttentionPrefillMetadata {
                block_tables: None,
                max_query_length: Some(max_tokens_len),
                max_prefill_sequence_length: max_tokens_len,
                query_start_locations,
                sequence_start_locations,
                sequence_lengths,
            }),
        };

        let selected_token_indices = {
            let mut result = Vec::with_capacity(tokens.len());
            let mut i = 0;
            tokens.iter().fold(0, |acc, x| {
                let sum = if i == 0 {
                    i += 1;
                    acc + x.len() as u32 - 1
                } else {
                    acc + x.len() as u32
                };
                result.push(sum);
                sum
            });
            result
        };
        let selected_token_indices =
            Tensor::from_vec(selected_token_indices, (tokens.len(),), &device)?;
        let logits = mistral_model
            .forward(
                &input,
                &input_positions,
                &selected_token_indices,
                &kv_caches_refs,
                &attention_metadata,
            )?
            .squeeze(0)?;

        assert_eq!(logits.dims().len(), 2);
        assert_eq!(logits.dims()[0], batch_size);
        assert_eq!(logits.dims()[1], 32_768);

        let mut sentences = prompts.clone();

        (0..batch_size).for_each(|i| {
            let next_token = logits_processors[i].sample(&logits.i(i).unwrap()).unwrap();
            if let Some(t) = tokenizers[i].next_token(next_token).unwrap() {
                sentences[i].push_str(&t);
            }
            tokens[i].push(next_token);
        });
        token_generated += batch_size;

        let mut finished_sequences = Vec::with_capacity(batch_size);
        let mut active_indices: Vec<usize> = (0..batch_size).collect();

        // round division
        let total_num_blocks_per_sequence =
            ((token_size_allocation + block_size - 1) / block_size) as i64;

        // decoding loop
        for _ in 1..sample_len {
            let num_active = active_indices.len();
            if num_active == 0 {
                break; // All sequences have finished
            }

            let input = Tensor::from_vec(
                active_indices
                    .iter()
                    .map(|&i| *tokens[i].last().unwrap())
                    .collect(),
                (1, num_active),
                &device,
            )?;
            let input_positions = Tensor::from_vec(
                active_indices
                    .iter()
                    .map(|&i| tokens[i].len() as i64 - 1)
                    .collect(),
                (1, num_active),
                &device,
            )?;
            let selected_token_indices =
                Tensor::from_vec((0..num_active as u32).collect(), (num_active,), &device)?;
            let max_decoding_sequence_length = active_indices
                .iter()
                .map(|i| tokens[*i].len())
                .max()
                .unwrap();
            let num_blocks_per_sequence = active_indices
                .iter()
                .map(|i| ((tokens[*i].len() + 15) / BLOCK_SIZE) as i64)
                .collect::<Vec<_>>();
            let max_num_blocks = *num_blocks_per_sequence.iter().max().unwrap() as usize;

            let slot_mapping = Tensor::from_vec(
                active_indices
                    .iter()
                    .map(|&i| (i * token_size_allocation + tokens[i].len()) as i64 - 1)
                    .collect(),
                (num_active,),
                &device,
            )?;

            let block_tables = active_indices
                .iter()
                .zip(num_blocks_per_sequence.iter())
                .flat_map(|(i, num_blocks)| {
                    let mut range = ((*i as u32 * total_num_blocks_per_sequence as u32)
                        ..((*i as u32 * total_num_blocks_per_sequence as u32)
                            + *num_blocks as u32))
                        .collect::<Vec<_>>();
                    range.extend([0u32].repeat(max_num_blocks - *num_blocks as usize)); // pad to max_num_blocks
                    range
                });
            let block_tables = Some(Tensor::from_vec(
                block_tables.collect(),
                (active_indices.len(), max_num_blocks),
                &device,
            )?);
            let sequence_lengths = Some(Tensor::from_vec(
                active_indices
                    .iter()
                    .map(|&i| tokens[i].len() as u32)
                    .collect::<Vec<_>>(),
                (active_indices.len(),),
                &device,
            )?);

            let attention_metadata = FlashAttentionMetadata {
                context_lengths: None,
                slot_mapping,
                decoding_metadata: Some(FlashAttentionDecodingMetadata {
                    block_tables,
                    max_decoding_sequence_length,
                    sequence_lengths,
                }),
                prefill_metadata: None,
                num_prefill_tokens: 0,
                num_decoding_tokens: num_active,
            };
            let logits = mistral_model
                .forward(
                    &input,
                    &input_positions,
                    &selected_token_indices,
                    &kv_caches_refs,
                    &attention_metadata,
                )?
                .squeeze(0)?;

            let mut new_active_indices = Vec::new();
            for (idx, &i) in active_indices.iter().enumerate() {
                let next_token = logits_processors[i]
                    .sample(&logits.i(idx).unwrap())
                    .unwrap();
                if let Some(t) = tokenizers[i].next_token(next_token).unwrap() {
                    sentences[i].push_str(&t);
                }

                tokens[i].push(next_token);

                if Some(next_token) != eos_token_id {
                    new_active_indices.push(i);
                } else {
                    finished_sequences.push(tokens[i].clone());
                }
            }

            active_indices = new_active_indices;
            token_generated += num_active;
        }

        finished_sequences.extend(tokens);

        for i in 0..batch_size {
            if let Some(rest) = tokenizers[i].decode_rest().unwrap() {
                sentences[i].push_str(&rest);
            }
        }

        let dt = start_gen.elapsed();
        println!(
            "\n\n{} tokens generated ({} token/s)\n",
            token_generated,
            (token_generated - 1) as f64 / dt.as_secs_f64(),
        );

        for s in sentences {
            println!("{:?}", s);
        }

        Ok(())
    }

    #[test]
    #[serial]
    fn test_mistral_model_long() -> Result<()> {
        let prompt = "The History of France ".to_string();

        let dtype = DType::BF16;
        let device = Device::new_cuda(0).unwrap();
        let model_id = "mistralai/Mistral-7B-Instruct-v0.3".to_string();
        let revision = "main".to_string();

        // Create API with token
        let api = if let Ok(token) = std::env::var("HUGGING_FACE_TOKEN") {
            println!("Using provided Hugging Face token");
            ApiBuilder::new()
                .with_token(Some(token))
                .build()
                .expect("Failed to create the HF API")
        } else {
            println!("HUGGING_FACE_TOKEN not set, proceeding without authentication");
            Api::new().expect("Failed to create the HF API")
        };

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

        let filenames = vec![
            api.get("model-00001-of-00003.safetensors")
                .expect("Failed to get model.safetensors"),
            api.get("model-00002-of-00003.safetensors")
                .expect("Failed to get model.safetensors"),
            api.get("model-00003-of-00003.safetensors")
                .expect("Failed to get model.safetensors"),
        ];
        let mut mistral_model = {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            MistralModel::load(vb, &config, dtype, &device).expect("Failed to load the model")
        };
        let tokenizer =
            Tokenizer::from_file(tokenizer_filename).expect("Failed to load the tokenizer");
        let eos_token_id = config
            .eos_token_id
            .or_else(|| tokenizer.token_to_id(EOS_TOKEN));

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

        let sample_len = 512;
        let start_gen = std::time::Instant::now();
        let mut token_generated = 0;

        // kv cache
        let num_blocks = 100;
        let block_size = BLOCK_SIZE;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();
        let mut kv_caches = std::iter::repeat_with(|| {
            Tensor::zeros(
                (2, num_blocks, block_size, num_key_value_heads, head_dim),
                dtype,
                &device,
            )
        })
        .take(config.num_hidden_layers)
        .collect::<Result<Vec<_>>>()?;

        let kv_caches_refs: Vec<&mut Tensor> = kv_caches.iter_mut().collect();

        // prefill forward pass
        let input_positions = Tensor::arange(0, tokens.len() as i64, &device)?.unsqueeze(0)?;
        let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;

        let context_lengths = Tensor::from_vec(vec![tokens.len() as u32], (1,), &device)?;
        let slot_mapping = Tensor::arange(0, tokens.len() as i64, &device)?;
        let query_start_locations = Tensor::from_vec(vec![0, tokens.len() as u32], (2,), &device)?;
        let sequence_start_locations =
            Tensor::from_vec(vec![0, tokens.len() as u32], (2,), &device)?;
        let sequence_lengths = Tensor::from_vec(vec![tokens.len() as u32], (1,), &device)?;
        let _block_tables = Tensor::new::<&[u32; 0]>(&[], &device)?;

        let num_prefill_tokens = tokens.len();
        let num_decoding_tokens = 0;
        let max_query_length = tokens.len();
        let _max_decoding_sequence_length = 0;
        let max_prefill_sequence_length = tokens.len();
        let _num_prefill_sequences = 1;

        let attention_metadata = FlashAttentionMetadata {
            context_lengths: Some(context_lengths),
            slot_mapping,
            decoding_metadata: None,
            num_prefill_tokens,
            num_decoding_tokens,
            prefill_metadata: Some(FlashAttentionPrefillMetadata {
                block_tables: None,
                max_query_length: Some(max_query_length),
                max_prefill_sequence_length,
                query_start_locations: Some(query_start_locations),
                sequence_start_locations: Some(sequence_start_locations),
                sequence_lengths: Some(sequence_lengths),
            }),
        };
        let logits = mistral_model.forward(
            &input,
            &input_positions,
            &Tensor::new(vec![tokens.len() as u32 - 1], &device)?,
            &kv_caches_refs,
            &attention_metadata,
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
            let num_blocks = (tokens.len() / block_size) as i64 + 1;

            let _context_lengths = Tensor::new(&[0u32], &device)?;
            let slot_mapping = Tensor::new(&[tokens.len() as i64 - 1], &device)?;
            let _query_start_locations = Tensor::new(&[0u32, 1], &device)?;
            let _sequence_start_locations = Tensor::new(&[0, tokens.len() as u32], &device)?;
            let sequence_lengths = Tensor::new(&[tokens.len() as u32], &device)?;
            let block_tables = Tensor::arange(0, num_blocks, &device)?
                .to_dtype(DType::U32)?
                .reshape((1, num_blocks as usize))?;

            let num_prefill_tokens = 0;
            let num_decoding_tokens = 1;
            let _max_query_length = 1;
            let max_decoding_sequence_length = tokens.len();
            let _max_prefill_sequence_length = 0;
            let _num_prefill_sequences = 0;

            let attention_metadata = FlashAttentionMetadata {
                context_lengths: None,
                slot_mapping,
                decoding_metadata: Some(FlashAttentionDecodingMetadata {
                    block_tables: Some(block_tables),
                    max_decoding_sequence_length,
                    sequence_lengths: Some(sequence_lengths),
                }),
                prefill_metadata: None,
                num_prefill_tokens,
                num_decoding_tokens,
            };
            let logits = mistral_model
                .forward(
                    &input,
                    &input_positions,
                    &Tensor::new(&[0u32], &device)?,
                    &kv_caches_refs,
                    &attention_metadata,
                )?
                .squeeze(0)?
                .squeeze(0)?;

            next_token = logits_processor.sample(&logits)?;
            token_generated += 1;
            tokens.push(next_token);

            if Some(next_token) == eos_token_id {
                break;
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
