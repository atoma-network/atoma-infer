use std::{path::Path, str::FromStr, time::Instant};

use crate::{
    config::{
        CacheConfig, CacheConfigError, ModelConfig, SchedulerConfig, SchedulerConfigError,
        ValidationConfig,
    },
    llm_engine::{EngineError, GenerateRequestOutput, LlmEngine, StreamResponse},
    model_executor::{
        Config, ConfigError, ModelExecutor, ModelLoaderError, ModelThreadDispatcher,
        ModelThreadError,
    },
    scheduler::{Scheduler, SchedulerError},
    sequence::{Sequence, SequenceError, SequenceGroup},
    tokenizer::{TokenizerError, TokenizerWorker},
    types::GenerateRequest,
    validation::{ValidGenerateRequest, Validation, ValidationError},
};
use candle_core::{DType, DTypeParseError, Error as CandleError};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use metrics::{counter, gauge};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::{
    sync::{
        mpsc::{self, error::SendError, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    task::JoinHandle,
};
use tracing::{error, info, info_span, instrument, Span};

// TODO:
// 1. Add proper tokenizer shutdown logic, and other related services

/// Represents different types of requests that can be sent to the LLM service.
pub enum ServiceRequest {
    /// A request for generating text without streaming.
    ///
    /// # Fields
    /// * `GenerateRequest` - The request parameters for text generation.
    /// * `oneshot::Sender<GenerateRequestOutput>` - A channel to send back the generated output.
    GenerateRequest(GenerateRequest, oneshot::Sender<GenerateRequestOutput>),
    /// A request for generating text with streaming output.
    ///
    /// # Fields
    /// * `GenerateRequest` - The request parameters for text generation.
    /// * `flume::Sender<GenerateStreamingOutput>` - A channel to stream the generated output.
    GenerateStreamingRequest(GenerateRequest, flume::Sender<StreamResponse>),
}

/// Represents different types of requests that can be sent to the LLM engine.
pub enum EngineRequest {
    /// A request for generating text without streaming.
    ///
    /// # Fields
    /// * `SequenceGroup` - The sequence group to be processed.
    /// * `oneshot::Sender<GenerateRequestOutput>` - A channel to send back the generated output.
    GenerateRequest(SequenceGroup, oneshot::Sender<GenerateRequestOutput>),
    /// A request for generating text with streaming output.
    ///
    /// # Fields
    /// * `SequenceGroup` - The sequence group to be processed.
    /// * `flume::Sender<GenerateStreamingOutput>` - A channel to stream the generated output.
    GenerateStreamingRequest(SequenceGroup, flume::Sender<StreamResponse>),
}

/// `LlmService` - the entrypoint of the Atoma's inference service.
/// It receives requests from the Atoma's event subscriber
/// service. It validates and tokenizes such requests
/// and sends the valid request to the `LlmEngine`
pub struct LlmService {
    /// A receiver channel, it is responsible for
    /// receiving incoming requests from the OpenAI API server
    service_request_receiver: UnboundedReceiver<ServiceRequest>,
    /// Sender to communicate with the `LlmEngine` serviceservice_response_sender,
    engine_sender: UnboundedSender<EngineRequest>,
    /// Block size
    block_size: usize,
    /// Join handle for the background task
    /// running the `LlmEngine` instance
    llm_engine_handle: JoinHandle<Result<(), LlmServiceError>>,
    /// Model config
    model_config: ModelConfig,
    /// Starting time of the instance
    start_time: Instant,
    /// Request counter
    request_counter: u64,
    /// A request validation instance
    validation_service: Validation,
    /// Tokenizer handle
    tokenizer_handle: JoinHandle<Result<(), LlmServiceError>>,
    /// Shutdown signal
    shutdown_signal: mpsc::Receiver<()>,
    /// Tracing span
    span: Span,
}

impl LlmService {
    /// Starts the service
    #[instrument(skip_all)]
    pub async fn start<M, P: AsRef<Path>>(
        service_request_receiver: UnboundedReceiver<ServiceRequest>,
        config_path: P,
        shutdown_signal: mpsc::Receiver<()>,
    ) -> Result<Self, LlmServiceError>
    where
        M: ModelExecutor + Send + Sync + 'static,
    {
        let span = info_span!("llm-service");
        let _span = span.clone();
        let _enter = _span.enter();

        info!("Starting a new `LlmService` instance..");

        let model_config = ModelConfig::from_file_path(config_path.as_ref());
        // NOTE: for now we use a synchronous model loader, on the instance's thread, as the service
        // should not make any progress until the model is loaded in GPU memory
        let file_paths = M::fetch(
            model_config.api_key.clone(),
            model_config.cache_dir.clone(),
            model_config.model_name.clone(),
            model_config.revision.clone(),
        )?;
        let config = M::C::from_file_path(&file_paths.config_path)?;
        let tokenizer = Tokenizer::from_file(&file_paths.tokenizer_path)?;
        // NOTE: we load the model on GPU memory, as to properly compute the number of blocks
        // during the system profiling stage. See `compute_num_gpu_blocks` comments
        // in the file `config.rs` for more details.
        // TODO: support multi-GPUs
        let devices_ids = model_config.device_ids.clone();
        let dtype = DType::from_str(&model_config.dtype)?;
        let num_kv_heads = config.num_kv_heads();
        let hidden_dim = config.hidden_dim();
        let num_hidden_layers = config.num_hidden_layers();

        // 1. Initialize the `ModelThreadDispatcher`'s. We need to communicate with the background
        // task in order to know when the model is loaded for each GPU device. This is necessary
        // to then compute the total number of blocks that can be used for the KV cache, per each GPU device.

        // 2. We create oneshot channels to communicate with the `ModelThreadDispatcher`'s
        // corresponding GPU device background threads.
        let mut initializer_senders = Vec::with_capacity(devices_ids.len());
        let mut initializer_receivers = Vec::with_capacity(devices_ids.len());
        for _ in 0..devices_ids.len() {
            let (sender, receiver) = oneshot::channel();
            initializer_senders.push(sender);
            initializer_receivers.push(receiver);
        }

        // 3. We create broadcast channels to send both the cache and scheduler configs
        // to each GPU background threads.
        let (config_sender, _) = tokio::sync::broadcast::channel(1);
        let mut config_receivers = Vec::with_capacity(devices_ids.len());
        for _ in 0..devices_ids.len() {
            let config_receiver = config_sender.subscribe();
            config_receivers.push(config_receiver);
        }

        let now = Instant::now();
        // 4. Start the `ModelThreadDispatcher`
        let model_thread_dispatcher = ModelThreadDispatcher::start::<M>(
            config.clone(),
            devices_ids.clone(),
            dtype,
            file_paths,
            initializer_senders,
            config_receivers,
        )?;

        // 5. Wait for all the model thread dispatchers to load the model weights
        for receiver in initializer_receivers {
            receiver
                .await
                .expect("Failed to receive on channel to stop model thread dispatcher");
        }

        info!(
            "All model thread dispatchers have loaded the model weights, in total {:?} seconds",
            now.elapsed().as_secs_f32()
        );

        // 6. Compute the cache and scheduler configs and send them to all the model thread dispatchers, now
        // that the model weights are loaded in each GPU device memory, so we can properly compute the total
        // number of blocks that are available for the KV cache, per each GPU device.
        let cache_config = CacheConfig::from_file_path(
            config_path.as_ref(),
            num_kv_heads,
            hidden_dim,
            num_hidden_layers,
            &devices_ids,
        )?;
        let scheduler_config = SchedulerConfig::from_file_path(config_path.as_ref())?;

        // 7. Send the cache and scheduler configs to all the model thread dispatchers
        config_sender
            .send((cache_config.clone(), scheduler_config.clone()))
            .map_err(|e| {
                LlmServiceError::BroadcastSenderError(format!(
                    "Failed to send config to model threads, with error: {e}",
                ))
            })?;

        let validation_config = ValidationConfig::from_file_path(config_path.as_ref());

        let (tokenizer_sender, tokenizer_receiver) = mpsc::unbounded_channel();
        let validation_service = Validation::new(
            validation_config.best_of,
            validation_config.max_stop_sequences,
            validation_config.max_top_n_tokens,
            validation_config.max_input_length,
            validation_config.max_total_tokens,
            tokenizer_sender,
        );

        let block_size = cache_config.block_size;
        let scheduler = Scheduler::new(cache_config.clone(), scheduler_config.clone())?;

        let start_time = Instant::now();

        let cloned_tokenizer = tokenizer.clone();
        let tokenizer_handle = tokio::spawn(async move {
            Ok(TokenizerWorker::start(
                cloned_tokenizer,
                tokenizer_receiver,
                model_config.num_tokenizer_workers,
            )
            .await?)
        });

        let (engine_sender, engine_receiver) = mpsc::unbounded_channel();
        let llm_engine_handle = tokio::spawn(async move {
            let llm_engine = LlmEngine::new(
                model_thread_dispatcher,
                engine_receiver,
                scheduler,
                tokenizer,
            );

            llm_engine.run().await?;
            Ok::<_, LlmServiceError>(())
        });

        Ok(Self {
            service_request_receiver,
            engine_sender,
            block_size,
            llm_engine_handle,
            model_config,
            request_counter: 0,
            start_time,
            validation_service,
            shutdown_signal,
            tokenizer_handle,
            span,
        })
    }

    /// Main loop - awaits for incoming requests from the atoma
    ///     event subscriber channel. It then validates the request
    ///     and once the request is validated, it sends it to the
    ///     `LlmEngine` background task
    #[instrument(skip_all)]
    pub async fn run(mut self) -> Result<(), LlmServiceError> {
        loop {
            tokio::select! {
                Some(service_request) = self.service_request_receiver.recv() => {
                    match service_request {
                        ServiceRequest::GenerateRequest(request, response_sender) => {
                            let sequence_group = match self.handle_request(request).await {
                                Ok(sequence_group) => sequence_group,
                                Err(e) => {
                                    error!("Failed to handle request, with error: {e}");
                                    continue;
                                    // TODO: we need to handle errors more appropriately,
                                    //       we want to commit to these errors, as validation
                                    //       errors should also be committed to, by the node.
                                }
                            };
                            self
                                .engine_sender
                                .send(EngineRequest::GenerateRequest(sequence_group, response_sender))
                                .map_err(Box::new)?;
                        }
                        ServiceRequest::GenerateStreamingRequest(request, response_sender) => {
                            let sequence_group = match self.handle_request(request).await {
                                Ok(sequence_group) => sequence_group,
                                Err(e) => {
                                    error!("Failed to handle request, with error: {e}");
                                    continue;
                                    // TODO: we need to handle errors more appropriately,
                                    //       we want to commit to these errors, as validation
                                    //       errors should also be committed to, by the node.
                                }
                            };
                                self
                                    .engine_sender
                                    .send(EngineRequest::GenerateStreamingRequest(sequence_group, response_sender))
                                    .map_err(Box::new)?;
                        }
                    };
                },
                _ = self.shutdown_signal.recv() => {
                    info!("Received shutdown signal, stopping `LlmService` instance..");
                    break;
                }
            }
        }
        self.stop().await
    }

    /// Handles a new received `GenerateRequest`
    /// and produces a valid `SequenceGroup` from it
    #[instrument(skip_all)]
    async fn handle_request(
        &mut self,
        request: GenerateRequest,
    ) -> Result<SequenceGroup, LlmServiceError> {
        let _enter = self.span.enter();

        info!("Received new request, with id = {}", request.request_id);

        let arrival_time = Instant::now();
        let request_id = request.request_id.clone();
        let valid_request = match self.process_received_request(request).await {
            Ok(valid_request) => valid_request,
            Err(e) => {
                error!("Failed to validate request with id = {request_id}, due to error: {e}");
                return Err(e);
            }
        };

        counter!("llm-service-requests-total").increment(1);
        gauge!("llm-service-request-validation-time").set(arrival_time.elapsed().as_secs_f32());
        info!(
            request_id = %request_id,
            validation_time = ?arrival_time.elapsed(),
            "Received and validated new request"
        );

        let sequence_id = self.request_counter;

        let sampling =
            if !valid_request.parameters.do_sample || valid_request.parameters.temperature == 1.0 {
                Sampling::ArgMax
            } else if valid_request.parameters.top_p == 1.0 && valid_request.parameters.top_k == 0 {
                Sampling::All {
                    temperature: valid_request.parameters.temperature as f64,
                }
            } else if valid_request.parameters.top_k == 0 && valid_request.parameters.top_p < 1.0 {
                Sampling::TopP {
                    p: valid_request.parameters.top_p as f64,
                    temperature: valid_request.parameters.temperature as f64,
                }
            } else if valid_request.parameters.top_k != 0 && valid_request.parameters.top_p == 1.0 {
                Sampling::TopK {
                    k: valid_request.parameters.top_k as usize,
                    temperature: valid_request.parameters.temperature as f64,
                }
            } else {
                Sampling::TopKThenTopP {
                    k: valid_request.parameters.top_k as usize,
                    p: valid_request.parameters.top_p as f64,
                    temperature: valid_request.parameters.temperature as f64,
                }
            };

        let logits_processor =
            LogitsProcessor::from_sampling(valid_request.parameters.random_seed, sampling);

        let sequence = Sequence::new(
            sequence_id,
            valid_request.inputs.clone(),
            valid_request.encoding.get_ids().to_vec(),
            self.block_size,
            valid_request.return_full_text,
        )?;
        let sequence_group = SequenceGroup::new(
            request_id,
            vec![sequence],
            arrival_time,
            valid_request.parameters.clone(),
            valid_request.stopping_parameters.clone(),
            logits_processor,
        )?;

        Ok(sequence_group)
    }

    /// Processes newly arrived inputs
    #[instrument(skip_all)]
    async fn process_received_request(
        &self,
        request: GenerateRequest,
    ) -> Result<ValidGenerateRequest, LlmServiceError> {
        Ok(self.validation_service.validate(request).await?)
    }

    /// Stops the running instance
    #[instrument(skip(self))]
    pub async fn stop(self) -> Result<(), LlmServiceError> {
        info!(
            "Stopping the `LlmService` instance, running time: {:?}",
            self.start_time.elapsed()
        );

        // Flush storage if configured
        if self.model_config.flush_storage {
            match tokio::fs::remove_dir_all(&self.model_config.cache_dir).await {
                Ok(()) => info!("Successfully removed storage folder"),
                Err(e) => error!("Failed to remove storage folder, on shutdown: {e}"),
            }
        }

        // Abort the background tasks
        self.llm_engine_handle.abort();

        // Awaits for the task to finish and handle any errors
        match self.llm_engine_handle.await {
            Ok(result) => match result {
                Ok(()) => info!("`LlmService` background task finished successfully"),
                Err(e) => error!("`LlmService` background task failed, with error: {e}"),
            },
            Err(e) => error!("Failed to abort the background task: {e}"),
        }

        self.tokenizer_handle.abort();

        match self.tokenizer_handle.await {
            Ok(result) => match result {
                Ok(()) => info!("`LlmService` tokenizer background task finished successfully"),
                Err(e) => error!("`LlmService` tokenizer background task failed, with error: {e}"),
            },
            Err(e) => error!("Failed to abort the tokenizer background task: {e}"),
        }

        info!("`LlmService` stopped successfully");
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum LlmServiceError {
    #[error("Boxed error: `{0}`")]
    BoxedError(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("Broadcast sender error: `{0}`")]
    BroadcastSenderError(String),
    #[error("Cache config error: `{0}`")]
    CacheConfigError(#[from] CacheConfigError),
    #[error("Candle error: `{0}`")]
    CandleError(#[from] CandleError),
    #[error("Config error: `{0}`")]
    ConfigError(#[from] ConfigError),
    #[error("DType parse error: `{0}`")]
    DTypeParseError(#[from] DTypeParseError),
    #[error("Model loader error: `{0}`")]
    ModelLoaderError(#[from] ModelLoaderError),
    #[error("Model thread error: `{0}`")]
    ModelThreadError(#[from] ModelThreadError),
    #[error("Engine error: `{0}`")]
    EngineError(#[from] EngineError),
    #[error("Validation error: `{0}`")]
    ValidationError(#[from] ValidationError),
    #[error("Scheduler error: `{0}`")]
    SchedulerError(#[from] SchedulerError),
    #[error("Scheduler config error: `{0}`")]
    SchedulerConfigError(#[from] SchedulerConfigError),
    #[error("Sequence error: `{0}`")]
    SequenceError(#[from] SequenceError),
    #[error("Send error: `{0}`")]
    SendError(#[from] Box<SendError<EngineRequest>>),
    #[error("Tokenizer error: `{0}`")]
    TokenizerError(#[from] TokenizerError),
}
