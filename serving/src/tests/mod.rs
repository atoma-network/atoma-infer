use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[cfg(test)]
mod llama;

use atoma_paged_attention::FlashAttentionMetadata;
use candle_core::{DType, Device, Tensor};
use rand::Rng;
use tokio::sync::mpsc;
use tracing::info;

use crate::{
    config::{CacheConfig, SchedulerConfig},
    llm_service::LlmService,
    model_executor::{
        ModelExecutor, ModelExecutorError, ModelFilePaths, ModelLoader, ModelLoaderError,
        ModelMetadata,
    },
    sequence::ExecuteModelRequest,
    types::{GenerateParameters, GenerateRequest},
    validation::Validation,
};

const BLOCK_SIZE: usize = 16;
const MAX_ELAPSED_INTERNAL: u64 = 50;
const MAX_STOP_SEQUENCES: usize = 1;
const MAX_TOP_N_TOKENS: u32 = 0;
const MAX_INPUT_LENGTH: usize = 16;
const MAX_TOTAL_TOKENS: u32 = 32;
const NUM_CPU_BLOCKS: usize = 4096;
const NUM_GPU_BLOCKS: usize = 4096;
const EOS_TOKEN_ID: u32 = 2048;
const VOCAB_SIZE: usize = 128;

struct MockModel {}

impl ModelLoader for MockModel {
    fn fetch<T: AsRef<Path>>(
        _: String,
        _: T,
        _: String,
        _: String,
    ) -> Result<ModelFilePaths, ModelLoaderError> {
        Ok(ModelFilePaths {
            config_path: "".into(),
            tokenizer_path: "".into(),
            weights_path: vec![],
        })
    }

    fn load(_: Device, _: DType, _: &ModelFilePaths) -> Result<Self, ModelLoaderError> {
        Ok(Self {})
    }
}

impl ModelMetadata for MockModel {
    fn alibi_slopes(&self) -> Option<&Tensor> {
        None
    }

    fn eos_token_ids(&self) -> Option<Vec<u32>> {
        Some(vec![EOS_TOKEN_ID])
    }

    fn hidden_size(&self) -> usize {
        512
    }

    fn num_attention_heads(&self) -> usize {
        8
    }

    fn num_hidden_layers(&self) -> usize {
        8
    }

    fn num_kv_heads(&self) -> usize {
        8
    }

    fn sliding_window(&self) -> Option<usize> {
        None
    }

    fn softmax_scale(&self) -> f32 {
        1.0
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
}

#[tokio::test]
async fn test_llm_engine() {
    init_tracing();

    const NUM_REQUESTS: usize = 128;
    const MAX_NUM_SEQUENCES: usize = 32;
    const NUM_RUNS: usize = NUM_REQUESTS / MAX_NUM_SEQUENCES;

    let (atoma_client_sender, mut atoma_client_receiver) = mpsc::unbounded_channel();
    let (atoma_event_subscriber_sender, atoma_event_subscriber_receiver) =
        mpsc::unbounded_channel();

    let cache_config = CacheConfig::new(
        BLOCK_SIZE,
        1.0,
        1,
        None,
        None,
        NUM_CPU_BLOCKS,
        NUM_GPU_BLOCKS,
    )
    .expect("Failed to create cache config");

    let scheduler_config = SchedulerConfig::new(512, MAX_NUM_SEQUENCES, 512, 0.0, false, 0)
        .expect("Failed to create scheduler config");

    let (tokenizer_sender, tokenizer_receiver) = mpsc::unbounded_channel();
    let validation = Validation::new(
        1,
        MAX_STOP_SEQUENCES,
        MAX_TOP_N_TOKENS,
        MAX_INPUT_LENGTH,
        MAX_TOTAL_TOKENS,
        tokenizer_sender,
    );
    let (_, shutdown_signal) = mpsc::channel(1);

    let service = LlmService::start::<MockModel, PathBuf>(
        "".to_string(),
        atoma_event_subscriber_receiver,
        atoma_client_sender,
        cache_config,
        "./cache/".into(),
        Device::Cpu,
        DType::F16,
        true,
        "anthony/tokenizers-test".to_string(),
        4,
        "".to_string(),
        scheduler_config,
        tokenizer_receiver,
        validation,
        shutdown_signal,
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

    for request in requests {
        atoma_event_subscriber_sender
            .send(request)
            .expect("Failed to send request");
    }

    let mut number_of_responses = 0;

    let start = Instant::now();
    let mut elapsed_times = Vec::with_capacity(100);

    for _ in 0..(NUM_RUNS) {
        let responses = atoma_client_receiver.recv().await.unwrap();
        elapsed_times.push(start.elapsed());
        for response in responses.iter() {
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
}

#[tokio::test]
async fn test_llm_engine_with_enable_chunking() {
    init_tracing();

    const NUM_REQUESTS: usize = 128;
    const MAX_NUM_SEQUENCES: usize = 32;
    const NUM_RUNS: usize = NUM_REQUESTS / MAX_NUM_SEQUENCES;

    let (atoma_client_sender, mut atoma_client_receiver) = mpsc::unbounded_channel();
    let (atoma_event_subscriber_sender, atoma_event_subscriber_receiver) =
        mpsc::unbounded_channel();

    let cache_config = CacheConfig::new(
        BLOCK_SIZE,
        1.0,
        1,
        None,
        None,
        NUM_CPU_BLOCKS,
        NUM_GPU_BLOCKS,
    )
    .expect("Failed to create cache config");

    let scheduler_config = SchedulerConfig::new(512, MAX_NUM_SEQUENCES, 512, 0.0, true, 0)
        .expect("Failed to create scheduler config");

    let (tokenizer_sender, tokenizer_receiver) = mpsc::unbounded_channel();
    let validation = Validation::new(
        1,
        MAX_STOP_SEQUENCES,
        MAX_TOP_N_TOKENS,
        MAX_INPUT_LENGTH,
        MAX_TOTAL_TOKENS,
        tokenizer_sender,
    );
    let (_, shutdown_signal) = mpsc::channel(1);

    let service = LlmService::start::<MockModel, PathBuf>(
        "".to_string(),
        atoma_event_subscriber_receiver,
        atoma_client_sender,
        cache_config,
        "./cache/".into(),
        Device::Cpu,
        DType::F16,
        true,
        "anthony/tokenizers-test".to_string(),
        4,
        "".to_string(),
        scheduler_config,
        tokenizer_receiver,
        validation,
        shutdown_signal,
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

    for request in requests {
        atoma_event_subscriber_sender
            .send(request)
            .expect("Failed to send request");
    }

    let mut number_of_responses = 0;

    let start = Instant::now();
    let mut elapsed_times = Vec::with_capacity(100);

    for _ in 0..(2 * NUM_RUNS) {
        let responses: Vec<crate::llm_engine::GenerateRequestOutput> =
            atoma_client_receiver.recv().await.unwrap();
        elapsed_times.push(start.elapsed());
        for response in responses.iter() {
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
}

pub fn init_tracing() {
    let _ = tracing_subscriber::fmt::try_init();
}
