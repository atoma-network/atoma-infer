use std::env;

use axum::{response::IntoResponse, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::net::TcpListener;

use api::chat_completions::{RequestBody, Model};

pub mod api;

// TODO: Add version path prefix, eg. `/v1` although maybe something along the lines of `/beta` would be more fitting?
/// The URL path to POST JSON for model chat completions.
pub const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
pub const DEFAULT_SERVER_ADDRESS: &str = "0.0.0.0";
pub const DEFAULT_SERVER_PORT: &str = "8080";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // TODO: Write a clap cli for passing arguments
    let address = env::var("ATOMA_NODE_INFERENCE_SERVER_ADDRESS").unwrap_or(DEFAULT_SERVER_ADDRESS.into());
    let port = env::var("ATOMA_NODE_INFERENCE_SERVER_PORT").unwrap_or(DEFAULT_SERVER_PORT.into());
    let listener = TcpListener::bind(format!("{address}:{port}")).await?;
    run_server(listener).await
}

pub async fn run_server(listener: TcpListener) -> anyhow::Result<()> {
    let http_router = Router::new()
        .route(CHAT_COMPLETIONS_PATH, post(completion_handler))
        .route(
            &format!("{CHAT_COMPLETIONS_PATH}/validate"),
            post(validate_completion_handler),
        );

    Ok(axum::serve(listener, http_router.into_make_service()).await?)
}

/// Deserialize the `[RequestBody]` from JSON, and pass the information to the specified model,
/// returning the generated content as a `[Response]` in JSON.
pub async fn completion_handler(Json(request): Json<RequestBody>) -> impl IntoResponse {
    match &request.model() {
        // TODO: Use information from the deserialized request
        // and return the response from the model
        Model::Llama3 => Json(json!({"status": "success"})),
    }
}

/// Given some `serde_json::Value` schema, construct a validator for some JSON instance returning a detailed error message upon failure.
/// This can be used for testing but also for users who may need to debug their request body.
pub fn validate_with_schema(
    schema: serde_json::Value,
    instance: serde_json::Value,
) -> anyhow::Result<()> {
    use std::fmt::Write;
    let validator = jsonschema::draft7::new(&schema)?;
    let result = validator.validate(&instance);
    if let Err(mut errors) = result {
        let err_buf = String::with_capacity(errors.as_mut().count());
        let err_buf = errors.fold(err_buf, |mut acc, err| {
            writeln!(&mut acc, "{err:?}").unwrap();
            acc
        });
        anyhow::bail!("AtomaAPI JSON verification failed:\n\n{err_buf}");
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename(
    serialize = "requestValidationError",
    deserialize = "requestValidationError"
))]
pub struct RequestValidationError {
    #[serde(rename(serialize = "type", deserialize = "type"))]
    r#type: RequestValidationType,
    message: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename(
    serialize = "requestValidationType",
    deserialize = "requestValidationType"
))]
pub enum RequestValidationType {
    #[serde(rename(serialize = "completion", deserialize = "completion"))]
    Completion,
}

pub async fn validate_completion_handler(
    Json(instance): Json<serde_json::Value>,
) -> impl IntoResponse {
    let schema = include_bytes!("../request_schema.json");
    let schema = serde_json::from_slice(schema).expect("failed to read schema file");
    if let Err(err) = validate_with_schema(schema, instance) {
        let err = RequestValidationError {
            r#type: RequestValidationType::Completion,
            message: format!("{err:?}"),
        };
        return Json(json!(err));
    }
    Json(json!({"status": "success"}))
}

