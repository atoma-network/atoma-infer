use std::sync::{RwLockReadGuard, RwLockWriteGuard};

use serde::{Deserialize, Serialize};

pub trait ReadLock {
    type Error;
    type Inner;
    fn read_lock(&self) -> Result<RwLockReadGuard<Self::Inner>, Self::Error>;
}

pub trait WriteLock {
    type Error;
    type Inner;
    fn write_lock(&self) -> Result<RwLockWriteGuard<Self::Inner>, Self::Error>;
}

/// `GenerateRequest` - LLM inference request
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateRequest {
    /// The request id
    pub request_id: String,
    /// Inputs in the form of a `String`
    pub inputs: String,
    /// Generation parameters
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
}

/// `GenerateParameters` - Parameters used for
/// LLM inference
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GenerateParameters {
    /// Generate `best_of` sequences and return the one with the highest token logprobs
    pub best_of: Option<usize>,
    /// Temperature is used for modeling the logits distribution
    pub temperature: Option<f32>,
    /// The parameter for repetition penalty. 1.0 means no penalty.
    /// See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    pub repetition_penalty: Option<f32>,
    /// The parameter for frequency penalty. 1.0 means no penalty
    /// Penalize new tokens based on their existing frequency in the text so far,
    /// decreasing the model's likelihood to repeat the same line verbatim
    pub frequency_penalty: Option<f32>,
    /// Controls the number of tokens in the history to consider for penalizing repetition.
    /// A larger value will look further back in the generated text to prevent repetitions,
    /// while a smaller value will only consider recent tokens.
    pub repeat_last_n: Option<u32>,
    /// The number of highest probability vocabulary tokens to keep for top-k-filtering
    pub top_k: Option<u32>,
    /// Top-p value for nucleus sampling
    pub top_p: Option<f32>,
    /// Typical Decoding mass
    /// See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information.
    pub typical_p: Option<f32>,
    /// Activate logits sampling.
    pub do_sample: bool,
    /// Maximum number of tokens to generate.
    pub max_new_tokens: Option<u32>,
    /// Whether to prepend the prompt to the generated text
    pub return_full_text: Option<bool>,
    /// Stop generating tokens if a member of `stop` is generated.
    pub stop: Vec<String>,
    /// Truncate inputs tokens to the given size.
    pub truncate: Option<usize>,
    /// Whether to return decoder input token logprobs and ids.
    pub decoder_input_details: bool,
    /// Random sampling seed.
    pub random_seed: Option<u64>,
    /// The number of highest probability vocabulary tokens to keep for top-n-filtering.
    pub top_n_tokens: Option<u32>,
    /// Top n sequences to generate
    pub n: usize,
}

fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        best_of: None,
        temperature: None,
        repetition_penalty: None,
        frequency_penalty: None,
        top_k: None,
        top_p: None,
        typical_p: None,
        do_sample: true,
        max_new_tokens: Some(100),
        return_full_text: None,
        stop: Vec::new(),
        truncate: None,
        repeat_last_n: None,
        decoder_input_details: false,
        random_seed: None,
        top_n_tokens: None,
        n: 1,
    }
}
