use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;
use std::sync::Arc;

use super::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use crate::flash_attention::{FlashAttention, FlashAttentionMetadata};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Phi3Config {
    pub vocab_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<String>,
    pub max_position_embeddings: usize,
}
