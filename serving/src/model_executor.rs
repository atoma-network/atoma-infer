use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use atoma_paged_attention::flash_attention::FlashAttentionMetadata;
use candle_core::{DType, Device, IndexOp, Tensor};
use futures::stream::FuturesUnordered;
use thiserror::Error;
use tokio::{
    sync::{
        mpsc,
        oneshot::{self, error::RecvError},
    },
    task::JoinHandle,
};
use tracing::{error, info, info_span, instrument, trace, Span};

use crate::{
    config::{CacheConfig, SchedulerConfig},
    sequence::{
        ExecuteModelRequest, LogProb, SequenceGroupMetadata, SequenceGroupMetrics,
        SequenceGroupOutput, SequenceOutput,
    },
    validation::NextTokenChooserParameters,
    worker::{ModelWorker, ModelWorkerError},
};

/// `FilePaths` - wrapper struct for the model's file paths
pub struct ModelFilePaths {
    /// Configuration file path
    pub config_path: PathBuf,
    /// Tokenizer file path
    pub tokenizer_path: PathBuf,
    /// Weights file path
    pub weights_path: Vec<PathBuf>,
}

/// `ModelLoader` trait - interface for fetching and loading a LLM model weights.
pub trait ModelLoader {
    /// Fetches model files from a remote source.
    ///
    /// # Arguments
    /// * `api_key` - Authentication key for the API.
    /// * `cache_dir` - Directory to store downloaded files.
    /// * `model_id` - Identifier for the model to fetch.
    /// * `revision` - Specific version or revision of the model.
    ///
    /// # Returns
    /// `Result<ModelFilePaths, ModelLoaderError>` containing paths to the fetched files.
    fn fetch<T: AsRef<Path>>(
        api_key: String,
        cache_dir: T,
        model_id: String,
        revision: String,
    ) -> Result<ModelFilePaths, ModelLoaderError>;

    /// Loads the model into memory.
    ///
    /// # Arguments
    /// * `device` - The device to load the model onto (e.g., CPU, GPU).
    /// * `dtype` - The data type for the model's parameters.
    /// * `file_paths` - Paths to the model files.
    ///
    /// # Returns
    /// `Result<Self, ModelLoaderError>` containing the loaded model.
    fn load(
        device: Device,
        dtype: DType,
        file_paths: &ModelFilePaths,
    ) -> Result<Self, ModelLoaderError>
    where
        Self: Sized;
}

/// `ModelMetadata` - Metadata for a LLM model
pub trait ModelMetadata: ModelLoader {
    /// Returns the ALiBi (Attention with Linear Biases) slopes, if applicable
    fn alibi_slopes(&self) -> Option<&Tensor>;
    /// Returns the End-of-Sequence (EOS) token IDs, if defined
    fn eos_token_ids(&self) -> Option<Vec<u32>>;
    /// Returns the size of the hidden layers in the model
    fn hidden_size(&self) -> usize;
    /// Returns the number of attention heads in the model
    fn num_attention_heads(&self) -> usize;
    /// Returns the number of hidden layers in the model
    fn num_hidden_layers(&self) -> usize;
    /// Returns the number of key-value heads in the model
    fn num_kv_heads(&self) -> usize;
    /// Returns the softmax scale, if applicable
    fn softmax_scale(&self) -> f32;
    /// Returns the sliding window size for sliding window attention, if applicable
    fn sliding_window(&self) -> Option<usize>;
}

/// `ModelExecutor` trait - interface for running AI inference
/// from a LLM
pub trait ModelExecutor: ModelLoader + ModelMetadata {
    /// Performs a forward pass through the model
    ///
    /// # Arguments
    /// * `input_tensor` - The input token IDs
    /// * `input_positions` - The positions of the input tokens
    /// * `selected_token_positions` - The positions of tokens to generate logits for
    /// * `kv_cache` - The key-value cache for attention layers
    /// * `attention_metadata` - Metadata for flash attention optimization
    ///
    /// # Returns
    /// A tensor of logits for the selected token positions
    fn forward(
        &mut self,
        input_tensor: &Tensor,
        input_positions: &Tensor,
        selected_token_positions: &Tensor,
        kv_cache: Vec<&mut Tensor>,
        attention_metadata: FlashAttentionMetadata,
    ) -> Result<Tensor, ModelExecutorError>;

    /// Samples next tokens for a batch of sequence groups
    ///
    /// # Arguments
    /// * `logits` - The output logits from the forward pass
    /// * `sequence_groups_metadata` - Metadata for each sequence group
    ///
    /// # Returns
    /// A vector of `SequenceGroupOutput` containing the sampled tokens and related information
    fn sample(
        &self,
        logits: &Tensor,
        sequence_groups_metadata: &[Arc<SequenceGroupMetadata>],
    ) -> Result<Vec<SequenceGroupOutput>, ModelExecutorError> {
        let total_num_sequences = sequence_groups_metadata
            .iter()
            .map(|metadata| metadata.sequence_data.len())
            .sum::<usize>();

        // 1. Check if the logits zeroth dimension matches the total number of sequences
        if logits.dims()[0] != total_num_sequences {
            return Err(ModelExecutorError::InvalidLogits(
                logits.dims()[0],
                total_num_sequences,
            ));
        }

        let mut sequence_group_outputs = Vec::with_capacity(sequence_groups_metadata.len());
        let mut logits_idx = 0;
        for sequence_group_metadata in sequence_groups_metadata.iter() {
            // 2. Retrieve the next token chooser and stopping criteria parameters,
            //    from the `SequenceGroupMetadata`, to be used for sampling
            let NextTokenChooserParameters {
                repetition_penalty,
                repeat_last_n,
                ..
            } = sequence_group_metadata.next_token_chooser_params;

            // 3. Allocate a `HashMap` to store each of the sequence group's outputs
            let mut sequence_outputs =
                HashMap::with_capacity(sequence_group_metadata.sequence_data.len());

            // 4. Iterate over each `SequenceData` in the `SequenceGroupMetadata`,
            //    to sample next tokens for each sequence
            for (sequence_id, sequence_data) in sequence_group_metadata.sequence_data.iter() {
                // 5. Select the given sequence logits, and apply a
                // repetition penalty if necessary
                let sequence_logits = if repetition_penalty == 1. {
                    logits.i(logits_idx)?.squeeze(0)?
                } else {
                    debug_assert!(repeat_last_n > 0, "repeat_last_n should be > 0");
                    let num_sequence_tokens = sequence_data.length();
                    let start_at = num_sequence_tokens
                        .checked_sub(repeat_last_n as usize)
                        .unwrap_or_default();
                    let context = sequence_data.get_token_ids();
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits.i(logits_idx)?.squeeze(0)?,
                        repetition_penalty,
                        &context[start_at..],
                    )?
                };

                // 6. Sample the next token
                // TODO: we should be able to sample `best_of` sequences
                //      simultaneously, so we can later generate multiple
                //      sequences at once, in parallel.
                let next_token = sequence_group_metadata
                    .logits_processor
                    .write()
                    .unwrap()
                    .sample(&sequence_logits)?;

                let is_stop_token = self
                    .eos_token_ids()
                    .map(|eid| eid.contains(&next_token))
                    .unwrap_or_default();

                // 7. Update the logits index
                logits_idx += 1;

                // 8. Update the `output`
                // TODO: we are not forking a parent sequence into a new
                //       sequence group, so we should not have to update
                let logprob = sequence_logits.to_vec1::<f32>()?[next_token as usize];
                sequence_outputs.insert(
                    *sequence_id,
                    SequenceOutput {
                        parent_sequence_id: *sequence_id,
                        output_token: next_token,
                        is_stop_token,
                        logprob: HashMap::from_iter([(
                            next_token,
                            LogProb::new(logprob, Some(1), None), // NOTE: we don't compute the decoded token at this point
                        )]),
                    },
                );
            }

            sequence_group_outputs.push(SequenceGroupOutput {
                outputs: sequence_outputs,
                sampled_token_ids: None,
                sampled_token_probs: None,
                logprobs: None,
                spec_decode_worker_metrics: None,
                sequence_group_metrics: SequenceGroupMetrics {
                    time_to_generate: None,
                    num_tokens_generated: total_num_sequences,
                },
            });
        }

        Ok(sequence_group_outputs)
    }
}

/// `ModelThreadCommand` - Encapsulates an AI inference request and a channel for sending the result
pub struct ModelThreadCommand {
    /// The AI inference request to be executed
    request: ExecuteModelRequest,
    /// A one-shot channel sender for communicating the generated output back to the main task
    sender: oneshot::Sender<Vec<SequenceGroupOutput>>,
}

/// `ModelThread` - Encapsulates the logic for running a model thread/task in the background.
/// It receives incoming requests and processes AI inference on them.
pub struct ModelThread<M: ModelExecutor> {
    /// The worker responsible for executing the model
    worker: ModelWorker<M>,
    /// Receiver for incoming model execution requests
    receiver: mpsc::UnboundedReceiver<ModelThreadCommand>,
    /// Tracing span for logging and diagnostics
    span: Span,
}

impl<M> ModelThread<M>
where
    M: ModelExecutor + Send + Sync,
{
    /// Main loop for processing incoming model execution requests.
    ///
    /// This method continuously listens for incoming `ModelThreadCommand`s. For each command:
    /// 1. It executes the model inference using the encapsulated request.
    /// 2. Measures the execution time.
    /// 3. Updates the output with execution metrics.
    /// 4. Sends the generated output back to the caller via the provided channel.
    ///
    /// The loop continues until the receiver channel is closed.
    ///
    /// # Returns
    /// - `Ok(())` if the loop completes normally (receiver closed)
    /// - `Err(ModelThreadError)` if there's an error during model execution
    #[instrument(skip(self))]
    pub fn run(mut self) -> Result<(), ModelThreadError> {
        let _enter = self.span.enter();
        info!("Start Model thread");

        while let Some(command) = self.receiver.blocking_recv() {
            let ModelThreadCommand { request, sender } = command;

            let execution_start_time = std::time::Instant::now();
            let mut output = match self.worker.execute_model(request) {
                Ok(output) => output,
                Err(e) => {
                    error!("Failed to run forward pass on model, with error: {e}");
                    return Err(ModelThreadError::ModelWorkerError(e));
                }
            };
            let execution_elapsed_time = execution_start_time.elapsed().as_secs_f32();
            for o in output.iter_mut() {
                o.sequence_group_metrics = SequenceGroupMetrics {
                    time_to_generate: Some(execution_elapsed_time),
                    num_tokens_generated: 1, // NOTE: without speculative decoding, we generate one token at a time for both prefill and decode sequences
                };
            }

            // Send responses back to the engine
            sender.send(output).ok();
        }

        Ok(())
    }
}

/// `ModelThreadDispatcher` - Manages incoming requests for background LLM inference tasks
pub struct ModelThreadDispatcher {
    /// Sender for `ModelThreadCommand`s to the model execution thread
    pub sender: mpsc::UnboundedSender<ModelThreadCommand>,
    /// Collection of receivers for AI inference outputs
    /// Yields when a new output is generated
    pub responses: FuturesUnordered<oneshot::Receiver<Vec<SequenceGroupOutput>>>,
    /// Join handle for the model execution thread
    pub join_handle: JoinHandle<Result<(), ModelThreadError>>,
}

impl ModelThreadDispatcher {
    /// Starts a new instance of a `ModelThreadDispatcher`.
    ///
    /// This function initializes and spawns a new model thread that continuously
    /// listens for incoming AI inference requests and processes them.
    ///
    /// # Arguments
    /// * `cache_config` - Configuration for the cache
    /// * `device` - The device (CPU/GPU) to run the model on
    /// * `dtype` - The data type for model computations
    /// * `model` - The model instance implementing `ModelExecutor`
    /// * `scheduler_config` - Configuration for the scheduler
    ///
    /// # Returns
    /// * `Result<Self, ModelThreadError>` - A new `ModelThreadDispatcher` instance or an error
    ///
    /// # Type Parameters
    /// * `M` - A type that implements `ModelExecutor + Send + Sync + 'static`
    #[instrument(skip_all)]
    pub(crate) fn start<M>(
        cache_config: CacheConfig,
        device: Device,
        dtype: DType,
        model: M,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, ModelThreadError>
    where
        M: ModelExecutor + Send + Sync + 'static,
    {
        let (sender, receiver) = mpsc::unbounded_channel();

        let join_handle = tokio::task::spawn_blocking(move || {
            let block_size = cache_config.block_size();
            let num_cpu_blocks = cache_config.num_cpu_blocks();
            let num_gpu_blocks = cache_config.num_gpu_blocks();
            let model_worker = ModelWorker::<M>::new(
                block_size,
                device,
                dtype,
                model,
                num_cpu_blocks,
                num_gpu_blocks,
                scheduler_config.enable_chunked_prefill(),
            )?;
            let model_thread = ModelThread {
                worker: model_worker,
                receiver,
                span: info_span!("model-thread"),
            };
            if let Err(e) = model_thread.run() {
                error!("Model thread error: {e}");
                if !matches!(e, ModelThreadError::Shutdown(_)) {
                    panic!("Fatal error occurred: {e}");
                }
            }

            Ok(())
        });

        let model_dispatcher = ModelThreadDispatcher {
            sender,
            responses: FuturesUnordered::new(),
            join_handle,
        };

        Ok(model_dispatcher)
    }

    /// Sends an `ExecuteModelRequest` to the model execution thread for processing.
    ///
    /// This method creates a `ModelThreadCommand` with the given request and a channel
    /// for receiving the result. It then sends this command to the model thread and
    /// adds the receiver to the `responses` queue for later retrieval.
    ///
    /// # Arguments
    /// * `request` - The `ExecuteModelRequest` to be processed by the model.
    ///
    /// # Effects
    /// * Sends a command to the model execution thread.
    /// * Adds a receiver to the `responses` queue.
    ///
    /// # Errors
    /// * Logs an error if the command cannot be sent, which may indicate that the model thread is shutting down.
    #[instrument(skip_all)]
    pub fn send(&self, request: ExecuteModelRequest) {
        trace!("Sending new `ExecuteModelRequest` to model executor task");

        let (sender, receiver) = oneshot::channel();
        let command = ModelThreadCommand { request, sender };

        if let Err(e) = self.sender.send(command) {
            error!("Could not send command to model core, it might be shutting down: {e}");
        }

        self.responses.push(receiver);
    }
}

#[derive(Debug, Error)]
pub enum ModelThreadError {
    #[error("Core thread shutdown: `{0}`")]
    Shutdown(RecvError),
    #[error("Send error")]
    SendError,
    #[error("Model loader error: `{0}`")]
    ModelLoaderError(#[from] ModelLoaderError),
    #[error("Model executor error: `{0}`")]
    ModelExecutorError(#[from] ModelExecutorError),
    #[error("Model worker error: `{0}`")]
    ModelWorkerError(#[from] ModelWorkerError),
}

#[derive(Debug, Error)]
pub enum ModelLoaderError {
    #[error("Api error: `{0}`")]
    ApiError(#[from] hf_hub::api::sync::ApiError),
    #[error("Candle error: `{0}`")]
    CandleError(#[from] candle_core::Error),
    #[error("Io error: `{0}`")]
    IoError(#[from] std::io::Error),
    #[error("Serde json error: `{0}`")]
    SerdeJsonError(#[from] serde_json::Error),
}

#[derive(Debug, Error)]
pub enum ModelExecutorError {
    #[error("Candle error: `{0}`")]
    CandleError(#[from] candle_core::Error),
    #[error(
        "Invalid logits or next token parameters (logits dims: {0}, next token params dims: {1})"
    )]
    InvalidLogits(usize, usize),
    #[error("Invalid next token parameters or stopping parameters (next token params dims: {0}, stopping params dims: {1})")]
    InvalidNextTokenParams(usize, usize),
}
