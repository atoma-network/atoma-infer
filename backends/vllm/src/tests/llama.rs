use crate::{
    llm_service::LlmService,
    models::llama::LlamaModel,
    types::{GenerateParameters, GenerateRequest},
    validation::Validation,
};
use std::{path::PathBuf, time::Instant};
use tracing::info;

const MAX_STOP_SEQUENCES: usize = 1;
const MAX_TOP_N_TOKENS: u32 = 4;
const MAX_INPUT_LENGTH: usize = 512;
const MAX_TOTAL_TOKENS: u32 = 2048;

#[tokio::test]
async fn test_llama_model() {
    crate::tests::init_tracing();

    let (_shutdown_signal_sender, shutdown_signal_receiver) = tokio::sync::mpsc::channel(1);

    let (atoma_event_subscriber_sender, atoma_event_subscriber_receiver) =
        tokio::sync::mpsc::unbounded_channel();
    let (atoma_client_sender, mut atoma_client_receiver) = tokio::sync::mpsc::unbounded_channel();
    let (tokenizer_sender, tokenizer_receiver) = tokio::sync::mpsc::unbounded_channel();

    let validation_service = Validation::new(
        1,
        MAX_STOP_SEQUENCES,
        MAX_TOP_N_TOKENS,
        MAX_INPUT_LENGTH,
        MAX_TOTAL_TOKENS,
        tokenizer_sender,
    );

    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("tests")
        .join("test_config.toml");

    let llm_service = LlmService::start::<LlamaModel, _>(
        atoma_event_subscriber_receiver,
        atoma_client_sender,
        config_path,
        tokenizer_receiver,
        validation_service,
        shutdown_signal_receiver,
    )
    .await
    .expect("Failed to start LLM service");

    tokio::spawn(async move {
        llm_service.run().await.expect("Fail to run llm service");
    });

    let prompts = vec!["The capital of France is ".to_string()];

    let start = Instant::now();
    for (i, prompt) in prompts.iter().enumerate() {
        atoma_event_subscriber_sender
            .send(GenerateRequest {
                request_id: format!("{}", i),
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
            })
            .expect("Failed to send request with id = {i}");
    }

    let mut received_responses = 0;
    while received_responses < prompts.len() {
        let responses: Vec<crate::llm_engine::GenerateRequestOutput> =
            atoma_client_receiver.recv().await.unwrap();
        for inference_outputs in responses {
            let finished_time = inference_outputs
                .metrics
                .read()
                .unwrap()
                .finished_time
                .unwrap();
            let elapsed_time = finished_time.duration_since(start);
            for output in inference_outputs.inference_outputs {
                let text = output.output_text;
                info!("\n\nReceived response: {text:?}\n, within time: {elapsed_time:?}\n\n");
            }
        }

        received_responses += 1;
    }
}
