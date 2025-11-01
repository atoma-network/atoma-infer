use crate::{
    flash_attention::{FlashAttention, FlashAttentionMetadata},
    llama::{Cache, Config},
    multi_gpu::{shard, TensorParallelColumnLinear, TensorParallelRowLinear},
};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;
use candle_nn::{Embedding, Linear, Module, RmsNorm};
use cudarc::nccl::safe::Comm;
use std::rc::Rc;

fn embedding(cfg: &Config, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, cfg.hidden_size))
}

fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), "weight")?;
    Ok(Linear::new(weight, None))
}

fn rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get_with_hints(size, "weight", shard(0, 0, 1))?;
    Ok(RmsNorm::new(weight, eps))
}

fn silu(xs: &Tensor) -> Result<Tensor> {
    xs / (xs.neg()?.exp()? + 1.0)?
}

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
    fn apply_rotary_embed(&self, x: &Tensor, input_positions: &Tensor) -> Result<Tensor> {
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
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let q_proj = TensorParallelColumnLinear::load(vb.pp("q_proj"), comm.clone())?;
        let k_proj = TensorParallelColumnLinear::load(vb.pp("k_proj"), comm.clone())?;
        let v_proj = TensorParallelColumnLinear::load(vb.pp("v_proj"), comm.clone())?;
        let o_proj = TensorParallelRowLinear::load(vb.pp("o_proj"), comm.clone())?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            num_key_value_heads: cfg.num_key_value_heads / comm.world_size(),
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
        let x = (silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, _cfg: &Config, comm: Rc<Comm>) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
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
        cfg: &Config,
        comm: Rc<Comm>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg, comm.clone(), dtype, device)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg, comm)?;
        let rms_1 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = rms_norm(
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
}

impl Llama {
    /// Forward pass of multi-gpu Llama model, using
    /// flash attention kernels, with paged attention
    /// memory batching optimizations.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[1, num_total_tokens]`, where `num_total_tokens =
    ///   num_prefill_tokens + num_decode_tokens`
    /// * `input_positions` - Input positions tensor of shape `[1, num_total_tokens]`, where
    ///   `num_total_tokens = num_prefill_tokens + num_decode_tokens`. it contains all input
    ///   positions, so that rotary embeddings can be applied correctly
    /// * `selected_token_indices` - Selected token indices tensor of shape `[1, num_decode_tokens]`
    /// * `kv_caches` - KV caches with paged block arrangement for each model layer. Each tensor is
    ///   of shape `[num_blocks, block_size, num_heads, head_dim]`
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
        let ln_f = rms_norm(cfg.hidden_size, 1e-5, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| {
                Block::load(
                    vb.pp(format!("model.layers.{i}")),
                    cfg,
                    comm.clone(),
                    dtype,
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::{LlamaConfig, LlamaEosToks};
    use candle_transformers::generation::{LogitsProcessor, Sampling};
    #[cfg(feature = "nccl")]
    use cudarc::{
        driver::safe::CudaDevice,
        nccl::safe::{Comm, Id},
    };
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
    use rand::Rng;
    use serial_test::serial;
    use std::io::Write;
    use tokenizers::Tokenizer;

    const EOS_TOKEN: &str = "</s>";

    #[test]
    #[serial]
    fn test_llama_nccl_model_random_block_order() -> Result<()> {
        const DEVICE_ID: usize = 0;
        let prompt = "The History of France starts in ".to_string();

        let dtype = DType::BF16;
        let device = Device::new_cuda(DEVICE_ID).unwrap();
        let model_id = "meta-llama/Llama-3.1-8B-Instruct".to_string();
        let revision = "main".to_string();
        let api_key = std::env::var("HF_API_KEY").expect("HF_API_KEY not set, please set it to run this test, with `export HF_API_KEY=<your key>`");

        println!("loading the model weights from {model_id}");
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .build()
            .expect("Failed to build the API");

        let api = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            revision,
        ));

        let tokenizer_filename = api
            .get("tokenizer.json")
            .expect("Failed to get tokenizer.json");
        let config_filename = api.get("config.json").expect("Failed to get config.json");
        let config: LlamaConfig = serde_json::from_slice(
            &std::fs::read(config_filename).expect("Failed to read config.json"),
        )
        .expect("Failed to deserialize config.json");
        let config = config.into_config();

        let filenames =
            candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?;

        let id = Id::new().unwrap();
        let cuda_device = CudaDevice::new(DEVICE_ID).expect("Failed to create the CUDA device");
        let comm = std::rc::Rc::new(
            Comm::from_rank(cuda_device, DEVICE_ID, 1, id)
                .expect("Failed to create the NCCL communicator"),
        );
        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(&filenames, dtype, &device)?
        };
        let mut llama_model =
            Llama::load(vb, &config, &comm, dtype, &device).expect("Failed to load the model");
        let tokenizer =
            Tokenizer::from_file(tokenizer_filename).expect("Failed to load the tokenizer");
        let eos_token_id = config
            .eos_token_id
            .clone()
            .or_else(|| tokenizer.token_to_id(EOS_TOKEN).map(LlamaEosToks::Single));

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
                Some(LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                    break;
                }
                Some(LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
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
