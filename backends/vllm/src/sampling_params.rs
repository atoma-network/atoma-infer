use std::collections::HashSet;

use candle_core::Tensor;
use thiserror::Error;

/// Sampling epsilon
const SAMPLING_EPS: f32 = 1e-5;

/// Sampling strategy
pub enum SamplingStrategy {
    Beam,
    Greedy,
    Random,
    RandomSeed,
}

/// `LogitsProcessor` is a function that takes a list of previously generated
/// tokens and a tensor of the logits for the next token, and returns a modified
/// tensor of logits to sample from.
type LogitsProcessor = fn(&Vec<i32>, &Tensor) -> Tensor;

/// `EarlyStopping` - controls the stopping condition for beam search
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EarlyStopping {
    True,
    False,
    Never,
}

impl Default for EarlyStopping {
    fn default() -> Self {
        Self::False
    }
}

/// Sampling parameters for text generation.
///
/// Overall, we follow the sampling parameters from the OpenAI text completion
/// API (https://platform.openai.com/docs/api-reference/completions/create).
/// In addition, we support beam search, which is not supported by OpenAI.
///
/// Args:
///    n: Number of output sequences to return for the given prompt.
///    best_of: Number of output sequences that are generated from the prompt.
///        From these `best_of` sequences, the top `n` sequences are returned.
///        `best_of` must be greater than or equal to `n`. This is treated as
///        the beam width when `use_beam_search` is true. By default, `best_of`
///        is set to `n`.
///    presence_penalty: Float that penalizes new tokens based on whether they
///        appear in the generated text so far. Values > 0 encourage the model
///        to use new tokens, while values < 0 encourage the model to repeat
///        tokens.
///    frequency_penalty: Float that penalizes new tokens based on their
///        frequency in the generated text so far. Values > 0 encourage the
///        model to use new tokens, while values < 0 encourage the model to
///        repeat tokens.
///    repetition_penalty: Float that penalizes new tokens based on whether
///        they appear in the prompt and the generated text so far. Values > 1
///        encourage the model to use new tokens, while values < 1 encourage
///        the model to repeat tokens.
///    temperature: Float that controls the randomness of the sampling. Lower
///        values make the model more deterministic, while higher values make
///        the model more random. Zero means greedy sampling.
///    top_p: Float that controls the cumulative probability of the top tokens
///        to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
///    top_k: Integer that controls the number of top tokens to consider. Set
///        to -1 to consider all tokens.
///    min_p: Float that represents the minimum probability for a token to be
///        considered, relative to the probability of the most likely token.
///        Must be in [0, 1].
///    seed: Random seed to use for the generation.
///    use_beam_search: Whether to use beam search instead of sampling.
///    length_penalty: Float that penalizes sequences based on their length.
///        Used in beam search.
///    early_stopping: Controls the stopping condition for beam search. It
///        accepts the following values: `True`, where the generation stops as
///        soon as there are `best_of` complete candidates; `False`, where an
///        heuristic is applied and the generation stops when is it very
///        unlikely to find better candidates; `Never`, where the beam search
///        procedure only stops when there cannot be better candidates
///        (canonical beam search algorithm).
///    stop: List of strings that stop the generation when they are generated.
///        The returned output will not contain the stop strings.
///    stop_token_ids: List of tokens that stop the generation when they are
///        generated. The returned output will contain the stop tokens unless
///        the stop tokens are special tokens.
///    include_stop_str_in_output: Whether to include the stop strings in
///        output text. Defaults to False.
///    ignore_eos: Whether to ignore the EOS token and continue generating
///        tokens after the EOS token is generated.
///    max_tokens: Maximum number of tokens to generate per output sequence.
///    min_tokens: Minimum number of tokens to generate per output sequence
///        before EOS or stop_token_ids can be generated
///    logprobs: Number of log probabilities to return per output token.
///        Note that the implementation follows the OpenAI API: The return
///        result includes the log probabilities on the `logprobs` most likely
///        tokens, as well the chosen tokens. The API will always return the
///        log probability of the sampled token, so there  may be up to
///        `logprobs+1` elements in the response.
///    prompt_logprobs: Number of log probabilities to return per prompt token.
///    detokenize: Whether to detokenize the output. Defaults to True.
///    skip_special_tokens: Whether to skip special tokens in the output.
///    spaces_between_special_tokens: Whether to add spaces between special
///        tokens in the output.  Defaults to True.
///    logits_processors: List of functions that modify logits based on
///        previously generated tokens.
///    truncate_prompt_tokens: If set to an integer k, will use only the last k
///        tokens from the prompt (i.e., left truncation). Defaults to None
///        (i.e., no truncation).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingParams {
    /// Number of output sequences to return for the given prompt.
    pub n: usize,
    /// Best of
    pub best_of: usize,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Temperature
    pub temperature: f32,
    /// Top p
    pub top_p: f32,
    /// Top k
    pub top_k: i32,
    /// Min p
    pub min_p: f32,
    /// Random seed
    pub seed: Option<u64>,
    /// Use beam search (bool)
    pub use_beam_search: bool,
    /// Length penalty
    pub length_penalty: f32,
    /// Early stopping
    pub early_stopping: EarlyStopping,
    /// Stop
    pub stop: Vec<String>,
    /// Stop token ids
    pub stop_token_ids: HashSet<u32>,
    /// Include stop string in output
    pub include_stop_str_in_output: bool,
    /// Ignore EOS token
    pub ignore_eos: bool,
    /// Maximum number of tokens
    pub max_tokens: Option<usize>,
    /// Minimum number of tokens
    pub min_tokens: usize,
    /// Log probabilities
    pub logprobs: Option<usize>,
    /// Prompt log probabilities
    pub prompt_logprobs: Option<usize>,
    /// Detokenize
    pub detokenize: bool,
    /// Skip special tokens
    pub skip_special_tokens: bool,
    /// Spaces between special tokens
    pub spaces_between_special_tokens: bool,
    /// Logits processors
    pub logits_processors: Vec<LogitsProcessor>,
    /// Truncate prompt tokens
    pub truncate_prompt_tokens: Option<usize>,
    /// Output text buffer length - Number of characters to hold back for stop string evaluation
    /// until sequence is finished.
    pub output_text_buffer_length: usize,
}

impl SamplingParams {
    /// Constructor
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: usize,
        best_of: Option<usize>,
        presence_penalty: f32,
        frequency_penalty: f32,
        repetition_penalty: f32,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        min_p: f32,
        seed: Option<u64>,
        use_beam_search: bool,
        length_penalty: f32,
        early_stopping: EarlyStopping,
        stop: Vec<String>,
        stop_token_ids: HashSet<u32>,
        include_stop_str_in_output: bool,
        ignore_eos: bool,
        max_tokens: Option<usize>,
        min_tokens: usize,
        logprobs: Option<usize>,
        prompt_logprobs: Option<usize>,
        detokenize: bool,
        skip_special_tokens: bool,
        spaces_between_special_tokens: bool,
        logits_processors: Vec<LogitsProcessor>,
        truncate_prompt_tokens: Option<usize>,
    ) -> Result<Self, SamplingParamsError> {
        let output_text_buffer_length = if !stop.is_empty() && include_stop_str_in_output {
            stop.iter().map(|s| s.len()).max().unwrap_or(1) - 1
        } else {
            0
        };

        let mut this = Self {
            n,
            best_of: best_of.unwrap_or(n),
            presence_penalty,
            frequency_penalty,
            repetition_penalty,
            temperature,
            top_p,
            top_k,
            min_p,
            seed,
            use_beam_search,
            length_penalty,
            early_stopping,
            stop,
            stop_token_ids,
            include_stop_str_in_output,
            ignore_eos,
            max_tokens,
            min_tokens,
            logprobs,
            prompt_logprobs,
            detokenize,
            skip_special_tokens,
            spaces_between_special_tokens,
            logits_processors,
            truncate_prompt_tokens,
            output_text_buffer_length,
        };

        this.verify_args()?;
        if this.use_beam_search {
            this.verify_beam_search()?;
        } else {
            this.verify_non_beam_search()?;
            if this.temperature < SAMPLING_EPS {
                // Zero temperature means greedy sampling.
                this.top_p = 1.0;
                this.top_k = -1;
                this.min_p = 0.0;
                this.verify_greedy_sampling()?;
            }
        }

        Ok(this)
    }

    /// Verify arguments
    fn verify_args(&self) -> Result<(), SamplingParamsError> {
        if self.n < 1 {
            return Err(SamplingParamsError::VerifyArgumentsError(format!(
                "n must be at least 1, got {}.",
                self.n
            )));
        }
        if self.best_of < self.n {
            return Err(SamplingParamsError::VerifyArgumentsError(format!(
                "best_of must be greater than or equal to n, got n={} and best_of={}.",
                self.n, self.best_of
            )));
        }
        if !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err(SamplingParamsError::VerifyArgumentsError(format!(
                "presence_penalty must be in [-2, 2], got {}.",
                self.presence_penalty
            )));
        }
        if !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err(SamplingParamsError::VerifyArgumentsError(format!(
                "frequency_penalty must be in [-2, 2], got {}.",
                self.frequency_penalty
            )));
        }
        if !(0.0..=2.0).contains(&self.repetition_penalty) || self.repetition_penalty <= 0.0 {
            return Err(SamplingParamsError::VerifyArgumentsError(format!(
                "repetition_penalty must be in (0, 2], got {}.",
                self.repetition_penalty
            )));
        }
        if self.temperature < 0.0 {
            return Err(SamplingParamsError::VerifyArgumentsError(format!(
                "temperature must be non-negative, got {}.",
                self.temperature
            )));
        }
        if !(0.0..=1.0).contains(&self.top_p) || self.top_p <= 0.0 {
            return Err(SamplingParamsError::VerifyArgumentsError(format!(
                "top_p must be in (0, 1], got {}.",
                self.top_p
            )));
        }
        if self.top_k < -1 || self.top_k == 0 {
            return Err(SamplingParamsError::VerifyArgumentsError(format!(
                "top_k must be -1 (disable), or at least 1, got {}.",
                self.top_k
            )));
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(SamplingParamsError::VerifyArgumentsError(format!(
                "min_p must be in [0, 1], got {}.",
                self.min_p
            )));
        }
        if let Some(max_tokens) = self.max_tokens {
            if max_tokens < 1 {
                return Err(SamplingParamsError::VerifyArgumentsError(format!(
                    "max_tokens must be at least 1, got {}.",
                    max_tokens
                )));
            }
        }
        if let Some(max_tokens) = self.max_tokens {
            if self.min_tokens > max_tokens {
                return Err(SamplingParamsError::VerifyArgumentsError(format!(
                    "min_tokens must be less than or equal to max_tokens={}, got {}.",
                    max_tokens, self.min_tokens
                )));
            }
        }
        if let Some(truncate_prompt_tokens) = self.truncate_prompt_tokens {
            if truncate_prompt_tokens < 1 {
                return Err(SamplingParamsError::VerifyArgumentsError(format!(
                    "truncate_prompt_tokens must be >= 1, got {}.",
                    truncate_prompt_tokens
                )));
            }
        }
        if self.stop.iter().any(|s| s.is_empty()) {
            return Err(SamplingParamsError::VerifyArgumentsError(
                "stop cannot contain an empty string.".to_string(),
            ));
        }
        if !self.stop.is_empty() && !self.detokenize {
            return Err(SamplingParamsError::VerifyArgumentsError(
                "stop strings are only supported when detokenize is True. Set detokenize=True to use stop.".to_string()
            ));
        }
        Ok(())
    }

    /// Verify beam search
    fn verify_beam_search(&self) -> Result<(), SamplingParamsError> {
        if self.best_of == 1 {
            return Err(SamplingParamsError::VerifyBeamSearchError(format!(
                "best_of must be greater than 1 when using beam search. Got {}.",
                self.best_of
            )));
        }
        if self.temperature > SAMPLING_EPS {
            return Err(SamplingParamsError::VerifyBeamSearchError(
                "temperature must be 0 when using beam search.".to_string(),
            ));
        }
        if self.top_p < 1.0 - SAMPLING_EPS {
            return Err(SamplingParamsError::VerifyBeamSearchError(
                "top_p must be 1 when using beam search.".to_string(),
            ));
        }
        if self.top_k != -1 {
            return Err(SamplingParamsError::VerifyBeamSearchError(
                "top_k must be -1 when using beam search.".to_string(),
            ));
        }
        Ok(())
    }

    /// Verify non beam search
    fn verify_non_beam_search(&self) -> Result<(), SamplingParamsError> {
        if self.early_stopping != EarlyStopping::False {
            return Err(SamplingParamsError::VerifyNonBeamSearchError(
                "early_stopping is not effective and must be False when not using beam search."
                    .to_string(),
            ));
        }

        if (self.length_penalty < 1.0 - SAMPLING_EPS) || (self.length_penalty > 1.0 + SAMPLING_EPS)
        {
            return Err(SamplingParamsError::VerifyNonBeamSearchError(
                "length_penalty is not effective and must be the default value of 1.0 when not using beam search.".to_string(),
            ));
        }
        Ok(())
    }

    /// Verify greedy sampling
    fn verify_greedy_sampling(&self) -> Result<(), SamplingParamsError> {
        if self.best_of > 1 {
            return Err(SamplingParamsError::VerifyGreedySamplingError(format!(
                "best_of must be 1 when using greedy sampling. Got {}.",
                self.best_of
            )));
        }
        Ok(())
    }

    /// Update sampling parameter from generation configuration
    #[allow(dead_code)]
    fn update_from_generation_config(&mut self, eos_token_ids: Vec<u32>) {
        if !self.ignore_eos {
            self.stop_token_ids.extend(eos_token_ids.iter())
        }
    }

    /// Sampling strategy
    pub fn sampling_strategy(&self) -> SamplingStrategy {
        if self.use_beam_search {
            SamplingStrategy::Beam
        } else if self.temperature < SAMPLING_EPS {
            SamplingStrategy::Greedy
        } else if self.seed.is_some() {
            SamplingStrategy::RandomSeed
        } else {
            SamplingStrategy::Random
        }
    }
}

#[derive(Debug, Error)]
pub enum SamplingParamsError {
    #[error("Verify arguments error: `{0}`")]
    VerifyArgumentsError(String),
    #[error("Verify beam search error: `{0}`")]
    VerifyBeamSearchError(String),
    #[error("Verify non beam search error: `{0}`")]
    VerifyNonBeamSearchError(String),
    #[error("Verify greedy sampling error: `{0}`")]
    VerifyGreedySamplingError(String),
}
