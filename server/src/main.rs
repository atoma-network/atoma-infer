use std::{env, sync::Arc};

#[cfg(feature = "vllm")]
use atoma_backends::{
    GenerateRequest, GenerateRequestOutput, LlamaModel, LlmService, LlmServiceError, Validation,
};
use axum::{
    extract::State,
    http::{header, HeaderMap},
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use serde_json::json;
use tokio::{
    net::TcpListener,
    sync::{
        mpsc::{self, UnboundedSender},
        oneshot,
    },
    task::JoinHandle,
};

use api::{
    chat_completions::{Model, RequestBody},
    validate_schema::validate_with_schema,
};

pub mod api;
#[cfg(test)]
pub mod tests;

// TODO: Add version path prefix, eg. `/v1` although maybe something along the lines of `/beta` would be more fitting?
/// The URL path to POST JSON for model chat completions.
pub const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
pub const DEFAULT_SERVER_ADDRESS: &str = "0.0.0.0";
pub const DEFAULT_SERVER_PORT: &str = "8080";
pub const AUTH_BEARER_PREFIX: &str = "Bearer ";

#[derive(Clone)]
pub struct LlmServiceState {
    join_handle: Arc<JoinHandle<anyhow::Result<()>>>,
    llm_service_sender: UnboundedSender<(GenerateRequest, oneshot::Sender<GenerateRequestOutput>)>,
    shutdown_signal_sender: mpsc::Sender<()>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // TODO: Write a clap cli for passing arguments
    let address =
        env::var("ATOMA_NODE_INFERENCE_SERVER_ADDRESS").unwrap_or(DEFAULT_SERVER_ADDRESS.into());
    let config_path = env::var("LLM_SERVICE_CONFIG_PATH").unwrap();
    let port = env::var("ATOMA_NODE_INFERENCE_SERVER_PORT").unwrap_or(DEFAULT_SERVER_PORT.into());
    let listener = TcpListener::bind(format!("{address}:{port}")).await?;

    let (llm_service_sender, llm_service_receiver) = mpsc::unbounded_channel();
    let (shutdown_signal_sender, shutdown_signal_receiver) = mpsc::channel(1);
    // TODO: Add model dispatcher
    let llm_service = LlmService::start::<LlamaModel, _>(
        llm_service_receiver,
        config_path,
        shutdown_signal_receiver,
    )
    .await
    .map_err(|e| anyhow::anyhow!("Failed to start `LlmService`, with error: {e}"))?;

    let join_handle = tokio::spawn(async move {
        llm_service
            .run()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to run `LlmService`, with error: {e}"))
    });

    let llm_service_state = LlmServiceState {
        join_handle: Arc::new(join_handle),
        llm_service_sender,
        shutdown_signal_sender,
    };
    run_server(listener, llm_service_state).await
}

pub async fn run_server(
    listener: TcpListener,
    llm_service_state: LlmServiceState,
) -> anyhow::Result<()> {
    let http_router = Router::new()
        .route(CHAT_COMPLETIONS_PATH, post(completion_handler))
        .route(
            &format!("{CHAT_COMPLETIONS_PATH}/validate"),
            post(validate_completion_handler),
        )
        .with_state(llm_service_state);

    Ok(axum::serve(listener, http_router.into_make_service()).await?)
}

/// Deserialize the `[RequestBody]` from JSON, and pass the information to the specified model,
/// returning the generated content as a `[Response]` in JSON.
pub async fn completion_handler(
    app_state: State<LlmServiceState>,
    headers: HeaderMap,
    Json(request): Json<RequestBody>,
) -> impl IntoResponse {
    let _auth_key = headers
        .get(header::AUTHORIZATION)
        .and_then(|auth_value| -> Option<&str> {
            auth_value
                .to_str()
                .ok()
                .and_then(|auth_header_str| auth_header_str.strip_prefix(AUTH_BEARER_PREFIX))
        })
        .unwrap_or_default();
    let model = match &request.model() {
        // TODO: Use information from the deserialized request
        // and return the response from the model
        Model::Llama3 => Json(json!({"status": "success"})),
    };
    let messages = request.messages();
    let frequency_penalty = request.frequency_penalty();
    let logit_bias = request.logit_bias();
    let logprobs = request.logprobs();
    let top_logprobs = request.top_logprobs();
    let max_completion_tokens = request.max_completion_tokens();
    let n = request.n();
    let presence_penalty = request.presence_penalty();
    let seed = request.seed();
    let stop = request.stop();
    let stream = request.stream();
    let temperature = request.temperature();
    let top_p = request.top_p();
    let tools = request.tools();
    let user = request.user();

    // let (sender, receiver) = oneshot::channel();
    // let generate_request = GenerateRequest {
    //     messages,
    //     frequency_penalty,
    //     logit_bias,
    //     logprobs,
    //     max_completion_tokens,
    //     n,
    //     presence_penalty,
    //     seed,
    //     stop,
    //     stream,
    // };

    // app_state.sender.send((generate_request, sender))?;
    // let outputs = receiver.await?;

    Json(json!({"status": "success"}))
}

pub async fn validate_completion_handler(
    Json(instance): Json<serde_json::Value>,
) -> impl IntoResponse {
    let schema = include_bytes!("../request_schema.json");
    let schema = serde_json::from_slice(schema).expect("failed to read schema file");
    let Ok(validator) = jsonschema::draft7::new(&schema) else {
        return Json(json!({
            "status": "failed",
            "reason": "failed to create validator from json schema"
        }));
    };
    if let Some(errors) = validate_with_schema(validator, instance) {
        return Json(json!(errors));
    }
    Json(json!({"status": "success"}))
}
