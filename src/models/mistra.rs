use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::with_tracing::{linear_no_bias, Linear, RmsNorm};

use crate::flash_attention::{FlashAttention, FlashAttentionMetadata};

fn default_use_flash_attn() -> bool {
    false
}

/// Mistral LLM, https://github.com/mistralai/mistral-src

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct MistralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: Option<usize>,
    pub num_key_value_heads: usize,
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
            rope_theta: 10_000.,
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

    // fn head_dim(&self) -> usize {
    //     self.head_dim
    //         .unwrap_or(self.hidden_size / self.num_attention_heads)
    // }
}

#[derive(Clone, Debug)]
pub struct Cache {
    cos: Tensor,
    sin: Tensor,
}

impl Cache {
    pub fn new(dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / (config.rope_theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (head_dim / 2,), device)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
            .to_dtype(dtype)?
            .reshape((config.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
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
    cos_sin_cache: Cache,
    attention: FlashAttention,
}

impl CausalSelfAttention {
    fn apply_rotary_embed(&self, x: &Tensor, input_positions: &Tensor) -> Result<Tensor> {
        let (b_sz, _num_heads, num_total_tokens, _hidden_size) = x.dims4()?;

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

        let q = q.transpose(1, 2)?.squeeze(0)?.contiguous()?;
        let k = k.transpose(1, 2)?.squeeze(0)?.contiguous()?;
        let v = v.squeeze(0)?;

        let o = self
            .attention
            .forward(&q, &k, &v, kv_cache, attention_metadata)?;

        let o = o.unsqueeze(0)?;
        self.o_proj.forward(&o)
    }

    fn load(vb: VarBuilder, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_attention_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let q_proj = linear_no_bias(hidden_sz, num_attention_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_attention_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attention: FlashAttention::new(
                num_attention_heads,
                num_key_value_heads,
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

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &MistralConfig, vb: VarBuilder) -> Result<Self> {
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
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

struct DecoderLayer {
    self_attn: CausalSelfAttention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &MistralConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn =
            CausalSelfAttention::load(vb.pp("self_attn"), cfg, vb.dtype(), vb.device())?;
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
pub struct MistralModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    sliding_window: Option<usize>,
    device: Device,
    dtype: DType,
}

impl MistralModel {
    pub fn new(cfg: &MistralConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, vb_l.pp(layer_idx))?;
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
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        input_positions: &Tensor,
        selected_token_indices: &Tensor,
        kv_caches: &[&mut Tensor],
        attention_metadata: FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let (_b_size, _seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (layer, kv_cache) in self.layers.iter_mut().zip(kv_caches.iter()) {
            xs = layer.forward(&xs, input_positions, kv_cache, &attention_metadata)?;
        }
        xs.index_select(selected_token_indices, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }
}
#[cfg(test)]
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
    #[serial]
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
        let mut llama_model = {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            MistralModel::load(vb, &config, dtype, &device).expect("Failed to load the model")
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

    #[test]
    #[serial]
    fn test_mistral_model_long() -> Result<()> {
        let prompt = "Once upon a time ".to_string();

        let dtype = DType::F32;
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
            MistralModel::load(vb, &config, dtype, &device).expect("Failed to load the model")
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

        let sample_len = 1024;
        let start_gen = std::time::Instant::now();
        let mut token_generated = 0;

        // kv cache
        let num_blocks = 100;
        let block_size = 64;
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

        let context_lengths = Tensor::from_vec(vec![tokens.len() as u32], (1,), &device)?;
        let slot_mapping = Tensor::arange(0, tokens.len() as i64, &device)?;
        let query_start_locations = Tensor::from_vec(vec![0, tokens.len() as u32], (2,), &device)?;
        let sequence_start_locations =
            Tensor::from_vec(vec![0, tokens.len() as u32], (2,), &device)?;
        let sequence_lengths = Tensor::from_vec(vec![tokens.len() as u32], (1,), &device)?;
        let block_tables = Tensor::arange(0, num_blocks, &device)?
            .to_dtype(DType::U32)?
            .reshape((1, num_blocks as usize))?;

        let num_prefill_tokens = tokens.len();
        let num_decoding_tokens = 0;
        let max_query_length = tokens.len();
        let max_decoding_sequence_length = 0;
        let max_prefill_sequence_length = tokens.len();
        let num_prefill_sequences = 1;

        let attention_metadata = FlashAttentionMetadata::new(
            context_lengths,
            slot_mapping,
            query_start_locations,
            num_prefill_tokens,
            num_decoding_tokens,
            max_query_length,
            max_decoding_sequence_length,
            max_prefill_sequence_length,
            num_prefill_sequences,
            sequence_start_locations,
            sequence_lengths,
            block_tables,
            false,
        )
        .expect("Failed to create `FlashAttentionMetadata` instance");
        let logits = llama_model.forward(
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

            let context_lengths = Tensor::new(&[0u32], &device)?;
            let slot_mapping = Tensor::arange(
                (99 * block_size) as i64,
                (99 * block_size) as i64 + (tokens.len() % block_size) as i64,
                &device,
            )?;
            let query_start_locations = Tensor::new(&[0u32, 1], &device)?;
            let sequence_start_locations = Tensor::new(&[0, tokens.len() as u32], &device)?;
            let sequence_lengths = Tensor::new(&[tokens.len() as u32], &device)?;
            let block_tables = Tensor::arange(0, num_blocks, &device)?
                .to_dtype(DType::U32)?
                .reshape((1, num_blocks as usize))?;

            let num_prefill_tokens = 0;
            let num_decoding_tokens = 1;
            let max_query_length = 1;
            let max_decoding_sequence_length = tokens.len();
            let max_prefill_sequence_length = 0;
            let num_prefill_sequences = 0;

            let attention_metadata = FlashAttentionMetadata::new(
                context_lengths,
                slot_mapping,
                query_start_locations,
                num_prefill_tokens,
                num_decoding_tokens,
                max_query_length,
                max_decoding_sequence_length,
                max_prefill_sequence_length,
                num_prefill_sequences,
                sequence_start_locations,
                sequence_lengths,
                block_tables,
                false,
            )
            .expect("Failed to create the `FlashAttentionMetadata` instance");
            let logits = llama_model
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

    #[test]
    #[serial]
    fn test_llama_model_random_block_order() -> Result<()> {
        let prompt = "The History of France starts in ".to_string();

        let dtype = DType::BF16;
        let device = Device::new_cuda(0).unwrap();
        let model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string();
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

        let sample_len = 512;
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

        // block tables number
        let mut allocated_blocks = Vec::<u32>::with_capacity(64);
        allocated_blocks.push(99); // first block is allocated, we set it to the last available block

        // prefill forward pass
        let input_positions = Tensor::arange(0, tokens.len() as i64, &device)?.unsqueeze(0)?;
        let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;

        let context_lengths = Tensor::from_vec(vec![tokens.len() as u32], (1,), &device)?;
        let slot_mapping = Tensor::arange(
            (99 * block_size) as i64,
            (99 * block_size) as i64 + (tokens.len() % block_size) as i64,
            &device,
        )?;
        let query_start_locations = Tensor::from_vec(vec![0, tokens.len() as u32], (2,), &device)?;
        let sequence_start_locations =
            Tensor::from_vec(vec![0, tokens.len() as u32], (2,), &device)?;
        let sequence_lengths = Tensor::from_vec(vec![tokens.len() as u32], (1,), &device)?;
        let block_tables = Tensor::new::<&[u32; 0]>(&[], &device)?;

        let num_prefill_tokens = tokens.len();
        let num_decoding_tokens = 0;
        let max_query_length = tokens.len();
        let max_decoding_sequence_length = 0;
        let max_prefill_sequence_length = tokens.len();
        let num_prefill_sequences = 1;

        let attention_metadata = FlashAttentionMetadata::new(
            context_lengths,
            slot_mapping,
            query_start_locations,
            num_prefill_tokens,
            num_decoding_tokens,
            max_query_length,
            max_decoding_sequence_length,
            max_prefill_sequence_length,
            num_prefill_sequences,
            sequence_start_locations,
            sequence_lengths,
            block_tables,
            false,
        )
        .expect("Failed to create `FlashAttentionMetadata` instance");
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

        let mut rng = rand::thread_rng();

        // decoding loop
        for _ in 1..sample_len {
            if tokens.len() % 16 == 1 {
                let mut num = rng.gen_range(0..100);
                while allocated_blocks.contains(&num) {
                    num = rng.gen_range(0..100);
                }
                allocated_blocks.push(num);
            }

            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let input_positions = Tensor::new(&[tokens.len() as i64 - 1], &device)?.unsqueeze(0)?;
            let selected_token_indices = Tensor::new(&[0u32], &device)?;
            let num_blocks = allocated_blocks.len();

            let context_lengths = Tensor::new(&[0u32], &device)?;
            let last_allocated_block = *allocated_blocks.last().unwrap();
            let slot_mapping = Tensor::new(
                &[(last_allocated_block as i64) * (block_size as i64)
                    + ((tokens.len() - 1) % block_size as usize) as i64],
                &device,
            )?;
            let query_start_locations = Tensor::new(&[0u32, 1], &device)?;
            let sequence_start_locations = Tensor::new(&[0, tokens.len() as u32], &device)?;
            let sequence_lengths = Tensor::new(&[tokens.len() as u32], &device)?;

            let block_tables =
                Tensor::from_vec(allocated_blocks.clone(), (1, num_blocks as usize), &device)?
                    .to_dtype(DType::U32)?
                    .reshape((1, num_blocks as usize))?;

            let num_prefill_tokens = 0;
            let num_decoding_tokens = 1;
            let max_query_length = 1;
            let max_decoding_sequence_length = tokens.len();
            let max_prefill_sequence_length = 0;
            let num_prefill_sequences = 0;

            let attention_metadata = FlashAttentionMetadata::new(
                context_lengths,
                slot_mapping,
                query_start_locations,
                num_prefill_tokens,
                num_decoding_tokens,
                max_query_length,
                max_decoding_sequence_length,
                max_prefill_sequence_length,
                num_prefill_sequences,
                sequence_start_locations,
                sequence_lengths,
                block_tables,
                false,
            )
            .expect("Failed to create the `FlashAttentionMetadata` instance");
            let logits = llama_model
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

    #[test]
    #[serial]
    fn test_llama_model_batch() -> Result<()> {
        let prompts = vec![
            "The capital of France is ".to_string(),
            "Modern music is especially focused on ".to_string(),
            "How many countries do exist ? ".to_string(),
            "Sailing requires advanced techniques on ".to_string(),
            "What are the best places to surf ? ".to_string(),
            "How many letters does the word 'Algarve' has ? ".to_string(),
            "Zero knowledge cryptography regards ".to_string(),
            "What is a large language model ? ".to_string(),
            "What is the best way to learn a new language ? ".to_string(),
            "Healthy food is vital for ".to_string(),
            "History books ".to_string(),
            "Once upon a time ".to_string(),
        ];

        let batch_size = prompts.len();

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
            Mistral::load(vb, &config, dtype, &device).expect("Failed to load the model")
        };
        let tokenizer =
            Tokenizer::from_file(tokenizer_filename).expect("Failed to load the tokenizer");
        let eos_token_id = config
            .eos_token_id
            .clone()
            .or_else(|| tokenizer.token_to_id(EOS_TOKEN).map(MistralEosToks::Single));

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

        let sample_len = 1024;
        let start_gen = std::time::Instant::now();
        let mut token_generated = 0;

        // KV cache
        let num_blocks = 1000;
        let block_size = 16;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let mut kv_caches = std::iter::repeat_with(|| {
            Tensor::zeros(
                (2, num_blocks, block_size, num_key_value_heads, head_dim),
                dtype,
                &device,
            )
        })
        .take(config.num_hidden_layers)
        .collect::<Result<Vec<_>>>()?;

        let kv_caches: Vec<_> = kv_caches.iter_mut().collect();

        let num_prefill_tokens = tokens.iter().map(|ts| ts.len()).sum::<usize>();
        let max_tokens_len = tokens.iter().map(|ts| ts.len()).max().unwrap();
        let token_size_allocation =
            ((max_tokens_len + sample_len + block_size) / block_size) * block_size;

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
                &kv_caches,
                attention_metadata,
            )?
            .squeeze(0)?;

        assert_eq!(logits.dims().len(), 2);
        assert_eq!(logits.dims()[0], batch_size);
        assert_eq!(logits.dims()[1], 32_000);

        let mut sentences = prompts.clone();

        (0..batch_size).for_each(|i| {
            let next_token = logits_processors[i].sample(&logits.i(i).unwrap()).unwrap();
            if let Some(t) = tokenizers[i].next_token(next_token).unwrap() {
                sentences[i].push_str(&t);
            }
            tokens[i].push(next_token);
        });
        token_generated += batch_size;

        // round division
        let total_num_blocks_per_sequence =
            ((token_size_allocation + block_size - 1) / block_size) as i64;

        let mut finished_sequences = Vec::with_capacity(batch_size);
        let mut active_indices: Vec<usize> = (0..batch_size).collect();

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
                .map(|i| ((tokens[*i].len() + 15) / block_size) as i64)
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
            let logits = llama_model
                .forward(
                    &input,
                    &input_positions,
                    &selected_token_indices,
                    &kv_caches,
                    attention_metadata,
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

                match eos_token_id {
                    Some(MistralEosToks::Single(eos_tok_id)) => {
                        if next_token != eos_tok_id {
                            new_active_indices.push(i);
                        } else {
                            finished_sequences.push(tokens[i].clone());
                        }
                    }
                    Some(MistralEosToks::Multiple(ref eos_ids)) => {
                        if eos_ids.contains(&next_token) {
                            finished_sequences.push(tokens[i].clone());
                        } else {
                            new_active_indices.push(i);
                        }
                    }
                    _ => (),
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
}
