#[cfg(feature = "vllm")]
pub use atoma_vllm_backend;

#[cfg(not(feature = "vllm"))]
compile_error!("vllm backend is not enabled. Please compile with --features vllm");
