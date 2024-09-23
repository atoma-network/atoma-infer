use atoma_paged_attention::{
    models::{
        llama::{Config, LlamaConfig, LlamaEosToks},
        Llama,
    },
    FlashAttentionMetadata,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::{path::Path, time::Instant};
use tracing::info;

use crate::{
    model_executor::{
        ModelExecutor, ModelExecutorError, ModelFilePaths, ModelLoader, ModelLoaderError,
        ModelMetadata,
    },
    models::hub_load_safetensors,
};

/// Represents a Llama language model.
///
/// This struct encapsulates the configuration, device, data type, and the actual Llama model.
pub struct LlamaModel {
    /// The configuration for the Llama model.
    config: Config,
    /// The actual Llama model implementation.
    model: Llama,
}

impl ModelLoader for LlamaModel {
    fn fetch<T: AsRef<Path>>(
        api_key: String,
        cache_dir: T,
        model_id: String,
        revision: String,
    ) -> Result<ModelFilePaths, ModelLoaderError> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(api_key))
            .with_cache_dir(cache_dir.as_ref().to_path_buf())
            .build()?;

        let repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            revision,
        ));
        let config_file_path = repo.get("config.json")?;
        let tokenizer_file_path = repo.get("tokenizer.json")?;

        let model_weights_file_paths = if &model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0" {
            vec![repo.get("model.safetensors")?]
        } else {
            hub_load_safetensors(&repo, "model.safetensors.index.json")?
        };

        Ok(ModelFilePaths {
            config_path: config_file_path,
            tokenizer_path: tokenizer_file_path,
            weights_path: model_weights_file_paths,
        })
    }

    fn load(
        device: Device,
        dtype: DType,
        file_paths: &ModelFilePaths,
    ) -> Result<Self, ModelLoaderError>
    where
        Self: Sized,
    {
        info!("Loading Llama model ...");
        let start = Instant::now();

        let (model, config) = {
            let config: LlamaConfig =
                serde_json::from_slice(&std::fs::read(&file_paths.config_path)?)?;
            let config = config.into_config();

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    file_paths.weights_path.as_slice(),
                    dtype,
                    &device,
                )?
            };
            (Llama::load(vb, &config, dtype, &device)?, config)
        };
        info!("Loaded Llama model in {:?}", start.elapsed());

        Ok(Self { model, config })
    }
}

impl ModelMetadata for LlamaModel {
    fn alibi_slopes(&self) -> Option<&Tensor> {
        None
    }

    fn eos_token_ids(&self) -> Option<Vec<u32>> {
        match self.config.eos_token_id.clone() {
            Some(LlamaEosToks::Single(id)) => Some(vec![id]),
            Some(LlamaEosToks::Multiple(ids)) => Some(ids),
            None => None,
        }
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn num_attention_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    fn num_hidden_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn num_kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    fn softmax_scale(&self) -> f32 {
        let head_dim = self.hidden_size() / self.num_attention_heads();
        1f32 / (head_dim as f32).sqrt()
    }

    fn sliding_window(&self) -> Option<usize> {
        None
    }
}

impl ModelExecutor for LlamaModel {
    fn forward(
        &mut self,
        input: &Tensor,
        input_positions: &Tensor,
        selected_token_positions: &Tensor,
        kv_cache: Vec<&mut Tensor>,
        attention_metadata: FlashAttentionMetadata,
    ) -> Result<Tensor, ModelExecutorError> {
        Ok(self.model.forward(
            input,
            input_positions,
            selected_token_positions,
            &kv_cache,
            attention_metadata,
        )?)
    }
}
