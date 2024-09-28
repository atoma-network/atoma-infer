use std::{
    env,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "vllm")]
use atoma_backends::{GenerateRequest, GenerateRequestOutput, LlamaModel, LlmService};
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
    signal,
    sync::{
        mpsc::{self, UnboundedSender},
        oneshot,
    },
    task::JoinHandle,
};
use tracing::{error, info};

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
pub struct AppState {
    request_counter: Arc<AtomicU64>,
    llm_service_sender: UnboundedSender<(GenerateRequest, oneshot::Sender<GenerateRequestOutput>)>,
    shutdown_signal_sender: mpsc::Sender<()>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

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

    let app_state = AppState {
        request_counter: Arc::new(AtomicU64::new(0)),
        llm_service_sender,
        shutdown_signal_sender,
    };
    run_server(listener, app_state, join_handle).await
}

pub async fn run_server(
    listener: TcpListener,
    app_state: AppState,
    join_handle: JoinHandle<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let shutdown_signal_sender = app_state.shutdown_signal_sender.clone();
    let http_router = Router::new()
        .route(CHAT_COMPLETIONS_PATH, post(completion_handler))
        .route(
            &format!("{CHAT_COMPLETIONS_PATH}/validate"),
            post(validate_completion_handler),
        )
        .with_state(app_state);

    let shutdown_signal = async {
        signal::ctrl_c()
            .await
            .expect("Failed to parse Ctrl+C signal");
        info!("Shutting down server...");
    };

    let server = axum::serve(listener, http_router.into_make_service())
        .with_graceful_shutdown(shutdown_signal);

    info!("OpenAI API server running, press Ctrl+C to shut it down");
    server.await?;

    // Shutdown the app state
    shutdown_signal_sender.send(()).await?;

    // Wait for LLM service to gracefully shut down
    match tokio::time::timeout(Duration::from_secs(30), join_handle).await {
        Ok(Ok(_)) => {
            info!("LlmService shutdown successfully");
        }
        Ok(Err(e)) => {
            error!("LlmService encountered an error during shutdown: {:?}", e);
        }
        Err(_) => {
            error!("LlmService shutdown timed out");
        }
    }
    info!("Server and LlmService shutdown complete");

    Ok(())
}

/// Deserialize the `[RequestBody]` from JSON, and pass the information to the specified model,
/// returning the generated content as a `[Response]` in JSON.
pub async fn completion_handler(
    app_state: State<AppState>,
    headers: HeaderMap,
    Json(request): Json<RequestBody>,
) -> impl IntoResponse {
    let request_number = app_state.request_counter.fetch_add(1, Ordering::SeqCst);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_nanos();
    let request_id = format!("{request_number}-{now}");

    let _auth_key = headers
        .get(header::AUTHORIZATION)
        .and_then(|auth_value| -> Option<&str> {
            auth_value
                .to_str()
                .ok()
                .and_then(|auth_header_str| auth_header_str.strip_prefix(AUTH_BEARER_PREFIX))
        })
        .unwrap_or_default();
    // let model = match &request.model() {
    //     // TODO: Use information from the deserialized request
    //     // and return the response from the model
    //     Model::Llama3 => Json(json!({"status": "success"})),
    // };
    // let messages = request.messages();
    // let frequency_penalty = request.frequency_penalty();
    // let logit_bias = request.logit_bias();
    // let logprobs = request.logprobs();
    // let top_logprobs = request.top_logprobs();
    // let max_completion_tokens = request.max_completion_tokens();
    // let n = request.n();
    // let presence_penalty = request.presence_penalty();
    // let seed = request.seed();
    // let stop = request.stop();
    // let stream = request.stream();
    // let temperature = request.temperature();
    // let top_p = request.top_p();
    // let tools = request.tools();
    // let user = request.user();

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
