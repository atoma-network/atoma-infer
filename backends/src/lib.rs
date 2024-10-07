#[cfg(feature = "vllm")]
pub use atoma_vllm_backend::{
    llm_engine::GenerateRequestOutput,
    llm_service::{LlmService, LlmServiceError},
    types::{GenerateParameters, GenerateRequest},
    validation::Validation,
};

#[cfg(feature = "nccl")]
pub use atoma_vllm_backend::models::llama_nccl::LlamaModel;

#[cfg(not(feature = "nccl"))]
pub use atoma_vllm_backend::models::llama::LlamaModel;

#[cfg(not(feature = "vllm"))]
compile_error!("vllm backend is not enabled. Please compile with --features vllm");
