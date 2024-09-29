use std::env;

use axum::{
    http::{header, HeaderMap},
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use serde_json::json;
use tokio::net::TcpListener;

use api::{
    chat_completions::{Model, RequestBody},
    validate_schema::validate_with_schema,
};

pub mod api;
mod config;
#[cfg(test)]
pub mod tests;
use config::CONFIG;

// TODO: Add version path prefix, eg. `/v1` although maybe something along the lines of `/beta` would be more fitting?
pub const AUTH_BEARER_PREFIX: &str = "Bearer "; // TODO: This will always be bearer, as per the OpenAI spec. Maybe remove this constant?

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // TODO: Write a clap cli for passing arguments
    let address = CONFIG.server_address.clone();
    let port = CONFIG.server_port.clone();
    let listener = TcpListener::bind(format!("{address}:{port}")).await?;
    run_server(listener).await
}

pub async fn run_server(listener: TcpListener) -> anyhow::Result<()> {
    let http_router = Router::new()
        .route(&CONFIG.chat_completions_path, post(completion_handler))
        .route(
            &format!("{}/validate", CONFIG.chat_completions_path.clone()),
            post(validate_completion_handler),
        );

    Ok(axum::serve(listener, http_router.into_make_service()).await?)
}

/// Deserialize the `[RequestBody]` from JSON, and pass the information to the specified model,
/// returning the generated content as a `[Response]` in JSON.
pub async fn completion_handler(
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
    match &request.model() {
        // TODO: Use information from the deserialized request
        // and return the response from the model
        Model::Llama3 => Json(json!({"status": "success"})),
    }
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
