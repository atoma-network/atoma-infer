use std::env;

use axum::{response::IntoResponse, routing::post, Json, Router};
use jsonschema::{error::ValidationErrorKind, ValidationError};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::net::TcpListener;

use api::chat_completions::{Model, RequestBody};

pub mod api;

// TODO: Add version path prefix, eg. `/v1` although maybe something along the lines of `/beta` would be more fitting?
/// The URL path to POST JSON for model chat completions.
pub const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
pub const DEFAULT_SERVER_ADDRESS: &str = "0.0.0.0";
pub const DEFAULT_SERVER_PORT: &str = "8080";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // TODO: Write a clap cli for passing arguments
    let address =
        env::var("ATOMA_NODE_INFERENCE_SERVER_ADDRESS").unwrap_or(DEFAULT_SERVER_ADDRESS.into());
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
    validator: jsonschema::Validator,
    instance: serde_json::Value,
) -> Option<Vec<RequestValidationError>> {
    let result = validator.validate(&instance);
    if let Err(errors) = result {
        let errors = errors
            .map(|err| RequestValidationError::from(err))
            .collect();
        return Some(errors);
    }
    None
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename(
    serialize = "requestValidationError",
    deserialize = "requestValidationError"
))]
pub struct RequestValidationError {
    instance: serde_json::Value,
    kind: String,
    instance_path: Vec<String>,
    schema_path: Vec<String>,
}

impl<'a> From<ValidationError<'a>> for RequestValidationError {
    fn from(value: ValidationError) -> RequestValidationError {
        RequestValidationError {
            instance: value.instance.into_owned(),
            kind: value.kind.err_string(),
            instance_path: value.instance_path.into_vec(),
            schema_path: value.schema_path.into_vec(),
        }
    }
}

trait ErrString {
    fn err_string(&self) -> String;
}
impl ErrString for ValidationErrorKind {
    fn err_string(&self) -> String {
        match self {
            ValidationErrorKind::AdditionalItems { limit } => format!("The input array contains more items than expected: limit is {limit}"),
            ValidationErrorKind::AdditionalProperties { unexpected } => format!("Unexpected properties: {unexpected:?}"),
            ValidationErrorKind::AnyOf => "The input value is not valid under any of the schemas listed in the 'anyOf' keyword".into(),
            ValidationErrorKind::BacktrackLimitExceeded { error } => format!("error: {error:?}"),
            ValidationErrorKind::Constant { expected_value } => format!("expected value: {expected_value}"),
            ValidationErrorKind::Contains => "The input array doesn’t contain items conforming to the specified schema".into(),
            ValidationErrorKind::ContentEncoding {
                content_encoding,
            } => format!("The input value does not respect the defined contentEncoding: {content_encoding}"),
            ValidationErrorKind::ContentMediaType {
                content_media_type,
            } => format!("The input value does not respect the defined contentMediaType: {content_media_type}"),
            ValidationErrorKind::Custom {
                message,
            } => format!("Custom error message for user-defined validation: {message}"),
            ValidationErrorKind::Enum {
                options,
            } => format!("The input value doesn’t match any of specified options: {options:?}"),
            ValidationErrorKind::ExclusiveMaximum {
                limit,
            } => format!("Value is too large: {limit:?}"),
            ValidationErrorKind::ExclusiveMinimum {
                limit,
            } => format!("Value is too small: {limit:?}"),
            ValidationErrorKind::FalseSchema => "Everything is invalid for false schema".into(),
            ValidationErrorKind::FileNotFound {
                error,
            } => format!("If the referenced file is not found during ref resolution: err: {error:?}"),
            ValidationErrorKind::Format {
                format,
            } => format!("When the input doesn’t match to the specified format: {format}"),
            ValidationErrorKind::FromUtf8 {
                error,
            } => format!("May happen in contentEncoding validation if base64 encoded data is invalid: err: {error:?}"),
            ValidationErrorKind::Utf8 {
                error,
            } => format!("Invalid UTF-8 string during percent encoding when resolving happens: err: {error:?}"),
            ValidationErrorKind::JSONParse {
                error,
            } => format!("May happen during ref resolution when remote document is not a valid JSON: err: {error:?}"),
            ValidationErrorKind::InvalidReference {
                reference,
            } => format!("reference value is not valid: {reference}"),
            ValidationErrorKind::InvalidURL {
                error,
            } => format!("Invalid URL, e.g. invalid port number or IP address: err: {error:?}"),
            ValidationErrorKind::MaxItems {
                limit,
            } => format!("Too many items in an array: {limit}"),
            ValidationErrorKind::Maximum {
                limit,
            } => format!("Value is too large: {limit:?}"),
            ValidationErrorKind::MaxLength {
                limit,
            } => format!("String is too long: {limit}"),
            ValidationErrorKind::MaxProperties {
                limit,
            } => format!("Too many properties in an object: {limit}"),
            ValidationErrorKind::MinItems {
                limit,
            } => format!("Too few items in an array: {limit}"),
            ValidationErrorKind::Minimum {
                limit,
            } => format!("Value is too small: {limit:?}"),
            ValidationErrorKind::MinLength {
                limit,
            } => format!("String is too short: {limit}"),
            ValidationErrorKind::MinProperties {
                limit,
            } => format!("Not enough properties in an object: {limit}"),
            ValidationErrorKind::MultipleOf {
                multiple_of,
            } => format!("Number is not a multiple of another number: multiple of {multiple_of}"),
            ValidationErrorKind::Not {
                schema,
            } => format!("Negated schema failed validation: {schema:?}"),
            ValidationErrorKind::OneOfMultipleValid => "The given schema is valid under more than one of the schemas listed in the ‘oneOf’ keyword".into(),
            ValidationErrorKind::OneOfNotValid => "The given schema is not valid under any of the schemas listed in the ‘oneOf’ keyword".into(),
            ValidationErrorKind::Pattern {
                pattern,
            } => format!("The input doesn’t match to a pattern: {pattern}"),
            ValidationErrorKind::PropertyNames {
                error,
            } => format!("Object property names are invalid: error: {error:?}"),
            ValidationErrorKind::Required {
                property,
            } => format!("A required property is missing: {property:?}"),
            ValidationErrorKind::Schema => "Resolved schema failed to compile".into(),
            ValidationErrorKind::Type {
                kind,
            } => format!("The input value doesn’t match one or multiple required types: {kind:?}"),
            ValidationErrorKind::UnevaluatedProperties {
                unexpected,
            } => format!("Unexpected properties: {unexpected:?}"),
            ValidationErrorKind::UniqueItems => "The input array has non-unique elements".into(),
            ValidationErrorKind::UnknownReferenceScheme {
                scheme,
            } => format!("Reference contains unknown scheme: {scheme}"),
            ValidationErrorKind::Resolver {
                url,
                error,
            } => format!("Error during schema reference resolution: URL: {url:?}, err: {error:?}"),
        }
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
