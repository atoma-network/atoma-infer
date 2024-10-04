use clap::Parser;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "vllm")]
use atoma_backends::{GenerateRequest, GenerateRequestOutput, LlamaModel, LlmService};
use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
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
    chat_completions::{ChatCompletionResponse, RequestBody},
    validate_schema::validate_with_schema,
};

pub mod api;
mod config;
#[cfg(test)]
pub mod tests;
use config::CONFIG;

// TODO: Add version path prefix, eg. `/v1` although maybe something along the lines of `/beta` would be more fitting?
pub const AUTH_BEARER_PREFIX: &str = "Bearer "; // TODO: This will always be bearer, as per the OpenAI spec. Maybe remove this constant?

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short, long)]
    config_path: String,
}

#[derive(Clone)]
pub struct AppState {
    request_counter: Arc<AtomicU64>,
    llm_service_sender: UnboundedSender<(GenerateRequest, oneshot::Sender<GenerateRequestOutput>)>,
    shutdown_signal_sender: mpsc::Sender<()>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Args::parse();

    // TODO: Write a clap cli for passing arguments
    let address = &CONFIG.server_address;
    let port = &CONFIG.server_port;
    let config_path = cli.config_path;
    let listener = TcpListener::bind(format!("{address}:{port}")).await?;
    app_start(listener, config_path).await
}

/// Starts the server and LLM service. Used also in integration tests.
pub async fn app_start(listener: TcpListener, config_path: String) -> anyhow::Result<()> {
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
        .route(&CONFIG.chat_completions_path, post(completion_handler))
        .route(
            &format!("{}/validate", &CONFIG.chat_completions_path),
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

/// Handles chat completion requests by processing the input, sending it to the LLM service,
/// and returning the generated response.
///
/// # Arguments
///
/// * `app_state` - The shared application state containing the request counter and LLM service sender.
/// * `headers` - The HTTP headers of the incoming request.
/// * `request` - The deserialized JSON request body containing the chat completion parameters.
///
/// # Returns
///
/// Returns a `Result` containing either:
/// - `Ok(Json<ChatCompletionResponse>)`: The successful chat completion response.
/// - `Err((StatusCode, Json<serde_json::Value>))`: An error response with appropriate status code and message.
///
/// # Flow
///
/// 1. Generates a unique request ID.
/// 2. Extracts the authorization key from headers (currently unused).
/// 3. Converts the request to a `GenerateRequest` for the LLM service.
/// 4. Sends the request to the LLM service and awaits the response.
/// 5. Converts the LLM service output to a `ChatCompletionResponse`.
/// 6. Returns the response or an error if any step fails.
///
/// # Error Handling
///
/// - Returns a 500 Internal Server Error if:
///   - Sending the request to the LLM service fails.
///   - Receiving the response from the LLM service fails.
///   - Converting the LLM output to a `ChatCompletionResponse` fails.
pub async fn completion_handler(
    app_state: State<AppState>,
    headers: HeaderMap,
    Json(request): Json<RequestBody>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<serde_json::Value>)> {
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
    let model = request.model().to_string();
    let generate_request = request.to_generate_request(request_id.clone());
    let (sender, receiver) = oneshot::channel();

    if let Err(send_error) = app_state
        .llm_service_sender
        .send((generate_request, sender))
    {
        error!("Failed to send request to LLM Service: {}", send_error);
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Internal server error: failed to process request",
                    "type": "internal_error",
                    "request_id": request_id,
                }
            })),
        ));
    }

    let outputs = match receiver.await {
        Ok(outputs) => outputs,
        Err(_) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "error": {
                        "message": "Failed to receive response from LLM Service",
                        "type": "internal_error",
                        "request_id": request_id,
                    }
                })),
            ));
        }
    };

    let chat_response = ChatCompletionResponse::try_from((model, outputs)).map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": err,
                    "type": "internal_error",
                    "request_id": request_id,
                }
            })),
        )
    })?;

    Ok(Json(chat_response))
}

/// Validates the incoming JSON request body against the OpenAI Chat Completion API schema.
///
/// This handler is used to validate the structure and content of a request body
/// before it's processed by the actual completion handler. It helps clients
/// ensure their requests are properly formatted.
///
/// # Arguments
///
/// * `instance` - The JSON request body to validate, extracted from the request.
///
/// # Returns
///
/// Returns a JSON response indicating whether the validation was successful or not.
/// If validation fails, it returns details about the validation errors.
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
