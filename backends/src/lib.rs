#[cfg(feature = "vllm")]
pub use atoma_vllm_backend::{
    llm_engine::GenerateRequestOutput,
    llm_service::{LlmService, LlmServiceError},
    models::llama::LlamaModel,
    types::{GenerateParameters, GenerateRequest},
    validation::Validation,
};

#[cfg(not(feature = "vllm"))]
compile_error!("vllm backend is not enabled. Please compile with --features vllm");
