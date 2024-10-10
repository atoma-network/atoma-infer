#[cfg(feature = "nccl")]
use cudarc::nccl::Comm;
#[cfg(feature = "nccl")]
use std::rc::Rc;
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[cfg(not(feature = "nccl"))]
mod llama;
#[cfg(feature = "nccl")]
mod llama_nccl;

use candle_core::{DType, Device, Tensor};
use futures::{stream::FuturesUnordered, StreamExt};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use models::FlashAttentionMetadata;
use rand::Rng;
use tokio::sync::{mpsc, oneshot};
use tracing::info;

use crate::{
    llm_service::{LlmService, ServiceRequest},
    model_executor::{
        Config, ConfigError, ModelExecutor, ModelExecutorError, ModelFilePaths, ModelLoader,
        ModelLoaderError,
    },
    sequence::ExecuteModelRequest,
    types::{GenerateParameters, GenerateRequest},
};

const MAX_ELAPSED_INTERNAL: u64 = 50;
const VOCAB_SIZE: usize = 128;

struct MockModel {}

impl Config for () {
    fn alibi_slopes(&self) -> Option<&Tensor> {
        unimplemented!()
    }

    fn eos_token_ids(&self) -> Option<Vec<u32>> {
        unimplemented!()
    }

    fn hidden_dim(&self) -> usize {
        unimplemented!()
    }

    fn num_attention_heads(&self) -> usize {
        unimplemented!()
    }

    fn num_hidden_layers(&self) -> usize {
        unimplemented!()
    }

    fn num_kv_heads(&self) -> usize {
        unimplemented!()
    }

    fn sliding_window(&self) -> Option<usize> {
        unimplemented!()
    }

    fn softmax_scale(&self) -> f32 {
        unimplemented!()
    }

    fn from_file_path(_: &PathBuf) -> Result<Self, ConfigError> {
        unimplemented!()
    }
}

impl ModelLoader for MockModel {
    type C = ();

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
        let tokenizer_file_path = repo.get("tokenizer.json")?;

        Ok(ModelFilePaths {
            config_path: "".into(),
            tokenizer_path: tokenizer_file_path,
            weights_path: vec![],
        })
    }

    #[cfg(not(feature = "nccl"))]
    fn load(
        _: Self::C,
        _: &Device,
        _: DType,
        _: &ModelFilePaths,
    ) -> Result<Self, ModelLoaderError> {
        Ok(Self {})
    }

    #[cfg(feature = "nccl")]
    fn load(
        _: Self::C,
        _: &Device,
        _: DType,
        _: &ModelFilePaths,
        _: &Rc<Comm>,
    ) -> Result<Self, ModelLoaderError> {
        unimplemented!()
    }
}

impl From<ExecuteModelRequest> for Vec<u32> {
    fn from(value: ExecuteModelRequest) -> Self {
        value
            .sequence_groups_metadata
            .first()
            .unwrap()
            .sequence_data
            .values()
            .next()
            .unwrap()
            .get_token_ids()
    }
}

impl ModelExecutor for MockModel {
    fn forward(
        &mut self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: Vec<&mut Tensor>,
        attention_metadata: FlashAttentionMetadata,
    ) -> Result<Tensor, ModelExecutorError> {
        let mut rng = rand::thread_rng();
        std::thread::sleep(Duration::from_secs(2)); // mimic forward pass
        let batch_size = attention_metadata
            .context_lengths
            .expect("Context lengths should be set")
            .dims()[0];
        let logits = (0..(batch_size * VOCAB_SIZE))
            .map(|_| rng.gen_range(0.0..1.0) as f32)
            .collect::<Vec<_>>();

        Ok(Tensor::new(logits, &Device::Cpu)?.reshape((batch_size, VOCAB_SIZE))?)
    }

    fn config(&self) -> &Self::C {
        &()
    }
}

#[tokio::test]
async fn test_llm_engine() {
    init_tracing();

    const NUM_REQUESTS: usize = 128;
    const MAX_NUM_SEQUENCES: usize = 32;
    const NUM_RUNS: usize = NUM_REQUESTS / MAX_NUM_SEQUENCES;

    let (shutdown_signal_sender, shutdown_signal_receiver) = mpsc::channel(1);

    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("tests")
        .join("test_config_enable_chunked_prefill.toml");

    let (service_request_sender, service_request_receiver) = mpsc::unbounded_channel();
    let service = LlmService::start::<MockModel, PathBuf>(
        service_request_receiver,
        config_path,
        shutdown_signal_receiver,
    )
    .await
    .expect("Failed to start LLM service");

    tokio::spawn(async move {
        service.run().await.expect("Fail to run llm service");
    });

    info!("Sending request through atoma_event_subscriber_sender");

    let requests = (0..NUM_REQUESTS).map(|i| GenerateRequest {
        request_id: format!("{}", i),
        inputs: "Hello world, from the Caribbean".to_string(),
        parameters: GenerateParameters {
            best_of: None,
            temperature: Some(1.2),
            repetition_penalty: Some(1.1),
            frequency_penalty: Some(1.1),
            repeat_last_n: Some(8),
            top_k: Some(8),
            top_p: Some(0.8),
            typical_p: None,
            do_sample: true,
            max_new_tokens: Some(16),
            return_full_text: Some(true),
            stop: vec!["STOP".to_string()],
            truncate: None,
            decoder_input_details: true,
            random_seed: Some(42),
            top_n_tokens: None,
            n: 1,
        },
    });

    let mut futures = FuturesUnordered::new();
    for request in requests {
        let (sender, receiver) = oneshot::channel();
        service_request_sender
            .send(ServiceRequest::GenerateRequest(request, sender))
            .expect("Failed to send request");
        futures.push(receiver);
    }

    let mut number_of_responses = 0;

    let start = Instant::now();
    let mut elapsed_times = Vec::with_capacity(100);

    while let Some(responses) = futures.next().await {
        let responses = responses.unwrap();
        elapsed_times.push(start.elapsed());
        for response in responses.inference_outputs.iter() {
            number_of_responses += 1;
            info!("Got new response: {response:?}");
        }
        info!("Number of responses {number_of_responses}")
    }

    info!("Elapsed times: {elapsed_times:?}");

    assert_eq!(number_of_responses, NUM_REQUESTS);
    assert_eq!(elapsed_times.len(), NUM_RUNS);

    // Give enough variability time for different machines
    let max_elapsed_interval = Duration::from_secs(MAX_ELAPSED_INTERNAL);
    for i in 0..(NUM_RUNS - 1) {
        let left_run_time = elapsed_times[i];
        let right_run_time = elapsed_times[i + 1];
        assert!(right_run_time - left_run_time <= max_elapsed_interval);
    }

    shutdown_signal_sender.send(()).await.unwrap();
}

#[tokio::test]
async fn test_llm_engine_with_enable_chunking() {
    init_tracing();

    const NUM_REQUESTS: usize = 128;
    const MAX_NUM_SEQUENCES: usize = 32;
    const NUM_RUNS: usize = NUM_REQUESTS / MAX_NUM_SEQUENCES;

    let (service_request_sender, service_request_receiver) = mpsc::unbounded_channel();
    let (shutdown_signal_sender, shutdown_signal_receiver) = mpsc::channel(1);

    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("tests")
        .join("test_config_enable_chunked_prefill.toml");

    let service = LlmService::start::<MockModel, PathBuf>(
        service_request_receiver,
        config_path,
        shutdown_signal_receiver,
    )
    .await
    .expect("Failed to start LLM service");

    tokio::spawn(async move {
        service.run().await.expect("Fail to run llm service");
    });

    info!("Sending request through atoma_event_subscriber_sender");

    let requests = (0..NUM_REQUESTS).map(|i| GenerateRequest {
        request_id: format!("{}", i),
        inputs: "Hello world, from the Caribbean".to_string(),
        parameters: GenerateParameters {
            best_of: None,
            temperature: Some(1.2),
            repetition_penalty: Some(1.1),
            frequency_penalty: Some(1.1),
            repeat_last_n: Some(8),
            top_k: Some(8),
            top_p: Some(0.8),
            typical_p: None,
            do_sample: true,
            max_new_tokens: Some(16),
            return_full_text: Some(true),
            stop: vec!["STOP".to_string()],
            truncate: None,
            decoder_input_details: true,
            random_seed: Some(42),
            top_n_tokens: None,
            n: 1,
        },
    });

    let mut futures = FuturesUnordered::new();
    for request in requests {
        let (sender, receiver) = oneshot::channel();
        service_request_sender
            .send(ServiceRequest::GenerateRequest(request, sender))
            .expect("Failed to send request");
        futures.push(receiver);
    }

    let mut number_of_responses = 0;

    let start = Instant::now();
    let mut elapsed_times = Vec::with_capacity(100);

    while let Some(responses) = futures.next().await {
        let responses = responses.unwrap();
        elapsed_times.push(start.elapsed());
        for response in responses.inference_outputs.iter() {
            number_of_responses += 1;
            info!("Got new response: {response:?}");
        }
        info!("Number of responses {number_of_responses}")
    }
    info!("Elapsed times: {elapsed_times:?}");

    assert_eq!(number_of_responses, NUM_REQUESTS);
    assert_eq!(elapsed_times.len(), 2 * NUM_RUNS);

    // Give enough variability time for different machines
    let max_elapsed_interval = Duration::from_secs(MAX_ELAPSED_INTERNAL);
    for i in 0..(2 * NUM_RUNS - 1) {
        let left_run_time = elapsed_times[i];
        let right_run_time = elapsed_times[i + 1];
        // Give enough variability time for different machines
        assert!(right_run_time - left_run_time <= max_elapsed_interval);
    }

    shutdown_signal_sender.send(()).await.unwrap();
}

pub fn init_tracing() {
    let _ = tracing_subscriber::fmt::try_init();
}
