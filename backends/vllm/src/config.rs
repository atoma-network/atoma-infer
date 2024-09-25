use config::Config;
use dotenv::dotenv;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

const KB: usize = 1 << 10;

/// `ModelConfig` - Configuration for serving a large language model.
///
/// This struct contains the necessary parameters to download, initialize,
/// and set up a model for execution using the vLLM (very Large Language Model) backend.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    /// HuggingFace API key for authentication and accessing private models.
    /// Required for downloading models from the HuggingFace model repository.
    pub api_key: String,
    /// Cache directory path where the model weights and associated files are stored.
    /// This directory is used to save downloaded model files for faster subsequent loading.
    pub cache_dir: String,
    /// Flag to determine whether to clear the cache directory when the service is stopped.
    /// If true, the stored model files will be removed upon service termination.
    pub flush_storage: bool,
    /// Model name or identifier as listed in the HuggingFace model repository.
    /// This should match the exact name of the model you want to use.
    pub model_name: String,
    /// Number of tokenizer workers, to run multiple tokenizers concurrently
    /// via round-robin scheduling.
    pub num_tokenizer_workers: usize,
    /// Specific revision or version of the model to use from the HuggingFace repository.
    /// This allows for reproducibility by ensuring a consistent model version.
    pub revision: String,
    /// List of GPU device IDs to use for model inference.
    /// For single-GPU setups, this should contain one ID. For multi-GPU setups,
    /// include all relevant GPU IDs for distributed inference.
    pub device_ids: Vec<usize>,
    /// The data type to use for model weights and computations.
    /// Common values include "bf16" (bfloat16) or "fp16" (float16) for half precision.
    pub dtype: String,
}

impl ModelConfig {
    /// Creates a new `ModelConfig`
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: String,
        cache_dir: String,
        flush_storage: bool,
        model_name: String,
        num_tokenizer_workers: usize,
        revision: String,
        device_ids: Vec<usize>,
        dtype: String,
    ) -> Self {
        Self {
            api_key,
            cache_dir,
            flush_storage,
            model_name,
            num_tokenizer_workers,
            revision,
            device_ids,
            dtype,
        }
    }
}

impl ModelConfig {
    /// Creates a new instance of `ModelsConfig` from a toml file.
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Self {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate inference configuration file");
        config
            .get::<Self>("inference")
            .expect("Failed to generated config file")
    }

    /// Creates a new instance of `ModelConfig` from a `.env` file.
    pub fn from_env_file() -> Self {
        dotenv().ok();

        let api_key = std::env::var("API_KEY").ok().unwrap_or_default();
        let cache_dir = std::env::var("CACHE_DIR")
            .unwrap_or_default()
            .parse()
            .unwrap();
        let flush_storage = std::env::var("FLUSH_STORAGE")
            .unwrap_or_default()
            .parse()
            .unwrap();
        let model_name = serde_json::from_str(
            &std::env::var("MODEL_NAME")
                .expect("Failed to retrieve models metadata, from .env file"),
        )
        .unwrap();
        let num_tokenizer_workers = serde_json::from_str(
            &std::env::var("NUM_TOKENIZER_WORKERS")
                .expect("Failed to retrieve models metadata, from .env file"),
        )
        .unwrap();
        let revision = serde_json::from_str(
            &std::env::var("REVISION").expect("Failed to retrieve models metadata, from .env file"),
        )
        .unwrap();
        let device_ids = serde_json::from_str(
            &std::env::var("DEVICE_IDS")
                .expect("Failed to retrieve models metadata, from .env file"),
        )
        .unwrap();
        let dtype = serde_json::from_str(
            &std::env::var("DTYPE").expect("Failed to retrieve models metadata, from .env file"),
        )
        .unwrap();

        Self {
            api_key,
            cache_dir,
            flush_storage,
            model_name,
            num_tokenizer_workers,
            revision,
            device_ids,
            dtype,
        }
    }
}

/// Configuration for the KV cache.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CacheConfig {
    /// The size of a block, in number of tokens
    pub(crate) block_size: usize,
    /// The KV cache dtype, if different from the model dtype
    pub(crate) cache_dtype: Option<String>,
    /// The fraction of GPU memory to use for the vLLM execution
    pub(crate) gpu_memory_utilization: f32,
    /// Fraction of total system memory to use for swap space, this value is
    /// used to calculate the swap space in bytes if the latter is not specified
    pub(crate) swap_space_fraction: Option<f32>,
    /// Size of the CPU swap space per GPU, in bytes, filled using system profiling if not specified
    pub(crate) swap_space_bytes: Option<usize>,
    /// Number of GPU blocks to use. This value overrides the profiled `num_gpu_blocks`, if specified
    num_gpu_blocks_override: Option<usize>,
    /// Total number of GPU blocks to use, filled using system profiling if not specified
    num_gpu_blocks: Option<usize>,
    /// Total number of CPU blocks to use, filled using system profiling if not specified
    num_cpu_blocks: Option<usize>,
    /// Sliding window (optional)
    sliding_window: Option<usize>,
}

impl CacheConfig {
    /// Creates a new instance of `CacheConfig` from a `.toml` file.
    pub fn from_file_path<P: AsRef<Path>>(config_file_path: P) -> Result<Self, CacheConfigError> {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate inference configuration file");
        let mut this = config
            .get::<Self>("cache")
            .expect("Failed to generated config file");

        if this.swap_space_bytes.is_none() {
            assert!(this.swap_space_fraction.is_some());
            let swap_space_fraction = this.swap_space_fraction.unwrap();
            this.swap_space_bytes = Some(utils::calculate_swap_space(swap_space_fraction)?);
        }

        if let Some(num_gpu_blocks_override) = this.num_gpu_blocks_override {
            this.num_gpu_blocks = Some(num_gpu_blocks_override);
        } else {
            let num_gpu_blocks =
                utils::calculate_num_gpu_blocks(this.block_size, this.gpu_memory_utilization)?;
            this.num_gpu_blocks = Some(num_gpu_blocks);
        }

        this.num_cpu_blocks = Some(this.swap_space_bytes.unwrap() / this.block_size);

        this.verify_args()?;
        this.verify_cache_dtype()?;

        Ok(this)
    }
}

impl CacheConfig {
    /// Constructor
    pub fn new(
        block_size: usize,
        cache_dtype: Option<String>,
        gpu_memory_utilization: f32,
        swap_space_fraction: f32,
        num_gpu_blocks_override: Option<usize>,
        sliding_window: Option<usize>,
    ) -> Result<Self, CacheConfigError> {
        let swap_space_bytes = utils::calculate_swap_space(swap_space_fraction)?;
        let num_gpu_blocks = utils::calculate_num_gpu_blocks(block_size, gpu_memory_utilization)?;
        let num_cpu_blocks = swap_space_bytes / block_size;

        let this = Self {
            block_size,
            gpu_memory_utilization,
            swap_space_fraction: Some(swap_space_fraction),
            swap_space_bytes: Some(swap_space_bytes),
            num_gpu_blocks_override,
            num_gpu_blocks: Some(num_gpu_blocks),
            num_cpu_blocks: Some(num_cpu_blocks),
            cache_dtype,
            sliding_window,
        };

        this.verify_args()?;
        this.verify_cache_dtype()?;

        Ok(this)
    }

    /// Verify `CacheConfig` arguments
    fn verify_args(&self) -> Result<(), CacheConfigError> {
        if self.gpu_memory_utilization > 1.0 {
            return Err(CacheConfigError::InvalidGpuMemoryUtilization(
                self.gpu_memory_utilization,
            ));
        }
        if self.swap_space_fraction.is_none() && self.swap_space_bytes.is_none() {
            return Err(CacheConfigError::InvalidSwapSpace(
                "Cannot leave unspecified both `swap_space_fraction` and `swap_space_bytes`"
                    .to_string(),
            ));
        }
        if self.num_gpu_blocks.is_none() || self.num_cpu_blocks.is_none() {
            return Err(CacheConfigError::InvalidNumBlocks(
                "Cannot leave unspecified either `num_gpu_blocks` and `num_cpu_blocks`".to_string(),
            ));
        }
        Ok(())
    }

    /// Verify `CacheConfig` cache dtype
    fn verify_cache_dtype(&self) -> Result<(), CacheConfigError> {
        if let Some(cache_dtype) = &self.cache_dtype {
            if !["auto", "bf16", "f16", "f32"].contains(&cache_dtype.as_str()) {
                return Err(CacheConfigError::InvalidCacheDtype(cache_dtype.clone()));
            }
        }
        Ok(())
    }

    /// Getter for `block_size`
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Getter for `gpu_memory_utilization`
    pub fn gpu_memory_utilization(&self) -> f32 {
        self.gpu_memory_utilization
    }

    /// Getter for `swap_space_bytes`
    pub fn swap_space_bytes(&self) -> usize {
        self.swap_space_bytes.unwrap()
    }

    /// Getter for `sliding_window`
    pub fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    /// Getter for `num_gpu_block_override`
    pub fn num_gpu_block_override(&self) -> Option<usize> {
        self.num_gpu_blocks_override
    }

    /// Getter for `num_gpu_blocks`
    pub fn num_gpu_blocks(&self) -> Option<usize> {
        self.num_gpu_blocks
    }

    /// Getter for `num_cpu_blocks`
    pub fn num_cpu_blocks(&self) -> Option<usize> {
        self.num_cpu_blocks
    }
}

#[derive(Debug, Error)]
pub enum CacheConfigError {
    #[error("Invalid GPU memory utilization: `{0}`")]
    InvalidGpuMemoryUtilization(f32),
    #[error("Invalid cache dtype: `{0}`")]
    InvalidCacheDtype(String),
    #[error("Invalid swap space fraction: `{0}`")]
    InvalidSwapSpaceFraction(f32),
    #[error("Invalid swap space: `{0}`")]
    InvalidSwapSpace(String),
    #[error("Failed to get GPU memory: `{0}`")]
    GpuMemoryQueryError(String),
    #[error("Failed to get system memory")]
    FailedToGetSystemMemory,
    #[error("Cannot leave unspecified either `num_gpu_blocks` and `num_cpu_blocks`")]
    InvalidNumBlocks(String),
}

/// Scheduler's configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SchedulerConfig {
    /// Maximum number of tokens to be processed in a single iteration
    /// (that is, one `step` of the scheduler).
    max_num_batched_tokens: usize,
    /// Maximum number of sequences to be processed in a single iteration
    /// (that is, one `step` of the scheduler).
    max_num_sequences: usize,
    /// Maximum length of a sequence to be scheduled
    /// (including prompt and generated text).
    max_model_len: usize,
    /// Apply a delay (of delay factor multiplied by previous prompt
    /// latency) before scheduling next prompt.
    delay_factor: f32,
    /// Enable chunked prefill. If true, prefill requests can be
    /// chunked based on the remaining max_num_batched_tokens.
    enable_chunked_prefill: bool,
}

impl SchedulerConfig {
    /// Constructor
    pub fn new(
        max_num_batched_tokens: usize,
        max_num_sequences: usize,
        max_model_len: usize,
        delay_factor: f32,
        enable_chunked_prefill: bool,
    ) -> Result<Self, SchedulerConfigError> {
        let this = Self {
            max_num_batched_tokens,
            max_num_sequences,
            max_model_len,
            delay_factor,
            enable_chunked_prefill,
        };

        this.verify_args()?;
        Ok(this)
    }

    fn verify_args(&self) -> Result<(), SchedulerConfigError> {
        if self.max_num_batched_tokens < self.max_model_len && !self.enable_chunked_prefill {
            return Err(SchedulerConfigError::FailedVerifySchedulerConfig(format!(
                "`max_num_batched_tokens` ({}) is smaller than `max_model_len` ({}). 
                 This effectively limits the maximum sequence length to `max_num_batched_tokens` and makes the scheduler reject longer sequences. 
                 Please increase `max_num_batched_tokens` or decrease `max_sequence_length`.",
                self.max_num_batched_tokens, self.max_model_len
            )));
        }

        if self.max_num_batched_tokens < self.max_num_sequences {
            return Err(SchedulerConfigError::FailedVerifySchedulerConfig(format!(
                "max_num_batched_tokens ({}) must be greater than or equal to max_num_sequences ({}).",
                self.max_num_batched_tokens, self.max_num_sequences
            )));
        }

        Ok(())
    }

    /// Getter for `delay_factor`
    pub fn delay_factor(&self) -> f32 {
        self.delay_factor
    }

    /// Getter for `enable_chunked_prefill`
    pub fn enable_chunked_prefill(&self) -> bool {
        self.enable_chunked_prefill
    }

    /// Getter for `max_model_len`
    pub fn max_model_len(&self) -> usize {
        self.max_model_len
    }

    /// Getter for `max_num_batched_tokens`
    pub fn max_num_batched_tokens(&self) -> usize {
        self.max_num_batched_tokens
    }

    /// Getter for `max_num_sequences`
    pub fn max_num_sequences(&self) -> usize {
        self.max_num_sequences
    }
}

impl SchedulerConfig {
    /// Creates a new instance of `SchedulerConfig` from a `.toml` file.
    pub fn from_file_path<P: AsRef<Path>>(
        config_file_path: P,
    ) -> Result<Self, SchedulerConfigError> {
        let builder = Config::builder().add_source(config::File::with_name(
            config_file_path.as_ref().to_str().unwrap(),
        ));
        let config = builder
            .build()
            .expect("Failed to generate inference configuration file");
        let this = config
            .get::<Self>("scheduler")
            .expect("Failed to generated config file");

        this.verify_args()?;
        Ok(this)
    }
}

#[derive(Debug, Error)]
pub enum SchedulerConfigError {
    #[error("Failed verify scheduler config: `{0}")]
    FailedVerifySchedulerConfig(String),
}

pub(crate) mod utils {
    use super::*;
    use cuda_runtime_sys::*;

    /// Calculate the swap space in bytes based on the total system memory
    pub(crate) fn calculate_swap_space(fraction: f32) -> Result<usize, CacheConfigError> {
        if fraction <= 0.0 || fraction > 1.0 {
            return Err(CacheConfigError::InvalidSwapSpaceFraction(fraction));
        }

        let total_memory = sys_info::mem_info()
            .map_err(|_| CacheConfigError::FailedToGetSystemMemory)?
            .total as usize
            * KB; // Convert from KB to bytes

        Ok((total_memory as f32 * fraction) as usize)
    }

    /// Calculate the number of GPU blocks to use
    pub(crate) fn calculate_num_gpu_blocks(
        block_size: usize,
        gpu_memory_utilization: f32,
    ) -> Result<usize, CacheConfigError> {
        unsafe {
            let mut device_count = 0;
            let result = cudaGetDeviceCount(&mut device_count);
            if result != cudaError::cudaSuccess || device_count == 0 {
                return Err(CacheConfigError::GpuMemoryQueryError(format!(
                    "Failed to get device count: {:?}",
                    result
                )));
            }

            let mut per_device_memory = Vec::with_capacity(device_count as usize);
            for device in 0..device_count {
                let result = cudaSetDevice(device);
                if result != cudaError::cudaSuccess {
                    return Err(CacheConfigError::GpuMemoryQueryError(format!(
                        "Failed to set device {}: {:?}",
                        device, result
                    )));
                }

                let mut free = 0;
                let mut total = 0;
                let result = cudaMemGetInfo(&mut free, &mut total);
                if result != cudaError::cudaSuccess {
                    return Err(CacheConfigError::GpuMemoryQueryError(format!(
                        "Failed to get memory info for device {}: {:?}",
                        device, result
                    )));
                }

                per_device_memory.push((free, total));
            }
            // We compute the minimum total memory across all devices to ensure
            // that each device has at least this amount of memory to serve the KV cache.
            // NOTE: we further assume that the model weights have been already loaded in GPU
            // VRAM memory, therefore free_memory already accounts for the model weights.
            let free_memory = per_device_memory
                .iter()
                .map(|(free, _)| free)
                .min()
                .unwrap();
            let num_gpu_blocks =
                (*free_memory as f32 * gpu_memory_utilization).floor() as usize / block_size;

            Ok(num_gpu_blocks)
        }
    }
}
