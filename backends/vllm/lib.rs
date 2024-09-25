//! This crate contains logic for fast inference servince, based on PagedAttention and the vLLM implementation. We refer the reader to
//! https://arxiv.org/pdf/2309.06180 for the detailed architecture of the service. We were highly inspired by the complete, in production, Python implementation
//! of vLLM, in https://github.com/vllm-project/vllm.

pub mod block;
pub mod block_allocator;
pub mod block_manager;
pub mod config;
pub mod evictor;
pub mod llm_engine;
pub mod llm_service;
pub mod model_executor;
pub mod models;
pub mod policy;
pub mod sampling_params;
pub mod scheduler;
pub mod sequence;
#[cfg(test)]
pub mod tests;
pub mod tokenizer;
pub mod types;
pub mod validation;
pub mod worker;
