use crate::models::llama::LlamaModel;
use crate::{
    llm_service::LlmService,
    types::{GenerateParameters, GenerateRequest},
};
use futures::{stream::FuturesUnordered, StreamExt};
use std::{path::PathBuf, time::Instant};
use tracing::info;

#[tokio::test]
async fn test_llama_model() {
    crate::tests::init_tracing();

    let (shutdown_signal_sender, shutdown_signal_receiver) = tokio::sync::mpsc::channel(1);
    let (service_request_sender, service_request_receiver) = tokio::sync::mpsc::unbounded_channel();

    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("tests")
        .join("test_config.toml");

    let llm_service = LlmService::start::<LlamaModel, _>(
        service_request_receiver,
        config_path,
        shutdown_signal_receiver,
    )
    .await
    .expect("Failed to start LLM service");

    tokio::spawn(async move {
        llm_service.run().await.expect("Fail to run llm service");
    });

    let prompts = vec!["The capital of France is ".to_string()];

    let start = Instant::now();
    let mut futures = FuturesUnordered::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let request_id = format!("{}", i);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        service_request_sender
            .send((
                GenerateRequest {
                    request_id,
                    inputs: prompt.clone(),
                    parameters: GenerateParameters {
                        best_of: None,
                        temperature: Some(1.2),
                        repetition_penalty: Some(1.1),
                        frequency_penalty: Some(1.1),
                        repeat_last_n: Some(64),
                        top_k: Some(64),
                        top_p: Some(0.8),
                        typical_p: None,
                        do_sample: true,
                        max_new_tokens: Some(512),
                        return_full_text: Some(true),
                        stop: vec!["STOP".to_string()],
                        truncate: None,
                        decoder_input_details: true,
                        random_seed: Some(42),
                        top_n_tokens: None,
                        n: 1,
                    },
                },
                sender,
            ))
            .expect("Failed to send request with id = {i}");
        futures.push(receiver);
    }

    let mut received_responses = 0;
    while let Some(responses) = futures.next().await {
        let responses = responses.unwrap();
        let finished_time = responses.metrics.read().unwrap().finished_time.unwrap();
        let elapsed_time = finished_time.duration_since(start);
        for output in responses.inference_outputs {
            let text = output.output_text;
            info!("\n\nReceived response: {text:?}\n, within time: {elapsed_time:?}\n\n");
        }

        received_responses += 1;
        if received_responses == prompts.len() {
            break;
        }
    }

    shutdown_signal_sender.send(()).await.unwrap();
}
