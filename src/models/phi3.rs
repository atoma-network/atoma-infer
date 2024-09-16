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



impl Phi3Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    pub fn into_config(self) -> Phi3Config {
        Phi3Config {
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
    pub fn new(dtype: DType, cfg: &Phi3Config, dev: &Device) -> Result<Self> {
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

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        input_positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, _seq_len, _n_embd) = q.dims4()?;
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
            .finish()
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
        cfg: &Phi3Config,
        vb: VarBuilder,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
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

        let attn_output = self.attention.forward(
            &query_states,
            &key_states,
            &value_states,
            kv_cache,
            attention_metadata,
        )?;

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
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
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
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Phi3Config,
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

#[derive(Debug, Clone)]
pub struct Phi3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl Phi3Model {
    pub fn load(vb: VarBuilder, cfg: &Phi3Config, dtype: DType, device: &Device) -> Result<Self> {
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
        kv_caches: &[&mut Tensor],
        attention_metadata: &FlashAttentionMetadata,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (layer, kv_cache) in self.layers.iter_mut().zip(kv_caches.iter()) {
            xs = layer.forward(&xs, input_positions, kv_cache, attention_metadata)?
        }
        xs = self.norm.forward(&xs)?;

        // Squeeze the first dimension if it's 1
        let xs = if b_size == 1 { xs.squeeze(0)? } else { xs };

        // Select the last token's logits
        let logits = xs
            .narrow(0, seq_len - 1, 1)?
            .apply(&self.lm_head)?
            .squeeze(0)?;

        logits.to_dtype(DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionDecodingMetadata, FlashAttentionPrefillMetadata};
    use candle_core::IndexOp;
    use candle_transformers::generation::{LogitsProcessor, Sampling};
    use hf_hub::{api::sync::Api, Repo, RepoType};
    use serial_test::serial;
    use std::io::Write;
    use tokenizers::Tokenizer;

    const EOS_TOKEN: &str = "ӏ ";
    const BLOCK_SIZE: usize = 16;

    #[test]
    #[serial]

    //  cargo test test_phi3_model -- --exact
    fn test_phi3_model() -> Result<()> {
        let prompt = "The History of France ".to_string();

        let dtype = DType::BF16;
        let device = Device::new_cuda(0).unwrap();
        let model_id = "microsoft/Phi-3-mini-4k-instruct".to_string();
        let revision = "main".to_string();
        let api = Api::new().expect("Failed to create the HF API");

        println!("loading the model weights from {model_id}");
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api
            .get("tokenizer.json")
            .expect("Failed to get tokenizer.json");
        let config_filename = api.get("config.json").expect("Failed to get config.json");
        let config: Phi3Config = serde_json::from_slice(
            &std::fs::read(config_filename).expect("Failed to read config.json"),
        )
        .expect("Failed to deserialize config.json");
        let config = config.into_config();

        let filenames = vec![api
            .get("model.safetensors.index.json")
            .expect("Failed to get model.safetensors")];
        let mut phi3_model = {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            Phi3Model::load(vb, &config, dtype, &device).expect("Failed to load the model")
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
        let logits = phi3_model.forward(&input, &input_positions, &attention_metadata)?;
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
            let logits = phi3_model
                .forward(&input, &input_positions, &attention_metadata)?
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

    #[test]
    #[serial]
    fn test_phi3_model_batch() -> Result<()> {
        let prompts = vec![
            "The History of France ".to_string(),
            "The History of Germany ".to_string(),
        ];

        let dtype = DType::BF16;
        let device = Device::new_cuda(0).unwrap();
        let model_id = "microsoft/Phi-3-mini-4k-instruct".to_string();
        let revision = "main".to_string();
        let api = Api::new().expect("Failed to create the HF API");

        println!("loading the model weights from {model_id}");
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api
            .get("tokenizer.json")
            .expect("Failed to get tokenizer.json");
        let config_filename = api.get("config.json").expect("Failed to get config.json");
        let config: Phi3Config = serde_json::from_slice(
            &std::fs::read(config_filename).expect("Failed to read config.json"),
        )
        .expect("Failed to deserialize config.json");
        let config = config.into_config();

        let filenames = vec![api
            .get("model.safetensors.index.json")
            .expect("Failed to get model.safetensors")];
        let mut phi3_model = {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            Phi3Model::load(vb, &config, dtype, &device).expect("Failed to load the model")
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
        .take(10)
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
        let token_size_allocation = ((max_tokens_len + 64 + 16) / 16) * 16;

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
        let logits = phi3_model
            .forward(&input, &input_positions, &attention_metadata)?
            .squeeze(0)?;

        assert_eq!(logits.dims().len(), 2);
        assert_eq!(logits.dims()[0], 10);
        assert_eq!(logits.dims()[1], 32_000);

        let mut sentences = prompts.clone();

        (0..10).for_each(|i| {
            let next_token = logits_processors[i].sample(&logits.i(i).unwrap()).unwrap();
            if let Some(t) = tokenizers[i].next_token(next_token).unwrap() {
                sentences[i].push_str(&t);
            }
            tokens[i].push(next_token);
        });
        token_generated += 10;

        // decoding loop
        let mut finished_sequences = Vec::with_capacity(10);
        let mut active_indices: Vec<usize> = (0..10).collect();

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
                    let mut range = ((*i as u32 * max_num_blocks as u32)
                        ..((*i as u32 * max_num_blocks as u32) + *num_blocks as u32))
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
            let logits = phi3_model
                .forward(&input, &input_positions, &attention_metadata)?
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

        for i in 0..10 {
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
    fn test_phi3_model_long() -> Result<()> {
        let prompt = "The History of France ".to_string();

        let dtype = DType::BF16;
        let device = Device::new_cuda(0).unwrap();
        let model_id = "microsoft/Phi-3-mini-4k-instruct".to_string();
        let revision = "main".to_string();
        let api = Api::new().expect("Failed to create the HF API");

        println!("loading the model weights from {model_id}");
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = api
            .get("tokenizer.json")
            .expect("Failed to get tokenizer.json");
        let config_filename = api.get("config.json").expect("Failed to get config.json");
        let config: Phi3Config = serde_json::from_slice(
            &std::fs::read(config_filename).expect("Failed to read config.json"),
        )
        .expect("Failed to deserialize config.json");
        let config = config.into_config();

        let filenames = vec![api
            .get("model.safetensors.index.json")
            .expect("Failed to get model.safetensors")];
        let mut phi3_model = {
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            Phi3Model::load(vb, &config, dtype, &device).expect("Failed to load the model")
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
        let mut kv_cache = std::iter::repeat_with(|| {
            Tensor::zeros(
                (num_blocks, block_size, num_key_value_heads, head_dim),
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
        let block_tables = Tensor::new::<&[u32; 0]>(&[], &device)?;

        let num_prefill_tokens = tokens.len();
        let num_decoding_tokens = 0;
        let max_query_length = tokens.len();
        let max_decoding_sequence_length = 0;
        let max_prefill_sequence_length = tokens.len();
        let num_prefill_sequences = 1;

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
        let logits = phi3_model.forward(&input, &input_positions, &kv_cache, &attention_metadata)?;
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

            let context_lengths = Tensor::new(&[0u32], &device)?;
            let slot_mapping = Tensor::new(&[tokens.len() as i64 - 1], &device)?;
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
            let logits = phi3_model
                .forward(&input, &input_positions, &kv_cache, &attention_metadata)?
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
