use jsonschema::{error::ValidationErrorKind, ValidationError};
use serde::{Deserialize, Serialize};

/// Given some `serde_json::Value` schema, construct a validator for some JSON instance returning a detailed error message upon failure.
/// This can be used for testing but also for users who may need to debug their request body.
pub fn validate_with_schema(
    validator: jsonschema::Validator,
    instance: serde_json::Value,
) -> Option<Vec<RequestValidationError>> {
    let result = validator.validate(&instance);
    if let Err(errors) = result {
        let errors = errors.map(RequestValidationError::from).collect();
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
            } => format!("The referenced file is not found during ref resolution: err: {error:?}"),
            ValidationErrorKind::Format {
                format,
            } => format!("The input doesn’t match to the specified format: {format}"),
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
            } => format!("Reference value is not valid: {reference}"),
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
