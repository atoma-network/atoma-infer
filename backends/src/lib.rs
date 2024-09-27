#[cfg(feature = "vllm")]
pub use atoma_vllm_backend::{
    llm_engine::GenerateRequestOutput,
    llm_service::{LlmService, LlmServiceError},
    types::GenerateRequest,
    validation::Validation,
    models::llama::LlamaModel,
};

#[cfg(not(feature = "vllm"))]
compile_error!("vllm backend is not enabled. Please compile with --features vllm");
