use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use thiserror::Error;
use tracing::{debug, error, info, info_span, instrument, trace, Span};

use crate::{
    block::{BlockError, LogicalTokenBlock},
    types::{ReadLock, WriteLock},
    validation::{NextTokenChooserParameters, StoppingCriteriaParameters},
};

/// Represents log probabilities, token ranks, and decoded tokens for a given token.
#[derive(Clone, Debug, PartialEq)]
pub struct LogProb {
    /// The log probability of the token.
    logprob: f32,
    /// The rank of the token in the model's vocabulary, if available.
    rank: Option<u32>,
    /// The decoded string representation of the token, if available.
    decoded_token: Option<String>,
}

impl LogProb {
    /// Constructor
    pub fn new(logprob: f32, rank: Option<u32>, decoded_token: Option<String>) -> Self {
        Self {
            logprob,
            rank,
            decoded_token,
        }
    }

    /// Getter for `logprob`
    pub fn logprob(&self) -> f32 {
        self.logprob
    }

    /// Getter for `rank`
    pub fn rank(&self) -> Option<u32> {
        self.rank
    }

    /// Getter for `decoded_token`
    pub fn decoded_token(&self) -> Option<String> {
        self.decoded_token.clone()
    }
}

/// `SequenceStatus` represents the current status of a `Sequence` in the generation process.
///
/// `Waiting:` The sequence is waiting to be processed.
/// `Running:` The sequence is currently being processed.
/// `Swapped:` The sequence has been temporarily swapped out of memory.
/// `FinishedStopped:` The sequence has finished processing generation due to meeting a stopping criterion.
/// `FinishedLengthCapped:` The sequence has finished generation due to reaching the maximum length.
/// `FinishedAborted:` The sequence has been aborted due to an error or user intervention.
/// `FinishedIgnored:` The sequence has been ignored and will not be processed further.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Swapped,
    FinishedStopped,
    FinishedLengthCapped,
    FinishedAborted,
    FinishedIgnored,
}

impl SequenceStatus {
    /// Checks if the sequence has finished processing.
    ///
    /// Returns `true` if the sequence has reached a terminal state (aborted, ignored, length capped, or stopped),
    /// and `false` if it's still in progress (waiting, running, or swapped).
    ///
    /// # Examples
    ///
    /// ```
    /// use atoma_vllm::sequence::SequenceStatus;
    ///
    /// assert!(SequenceStatus::FinishedStopped.is_finished());
    /// assert!(!SequenceStatus::Running.is_finished());
    /// ```
    pub fn is_finished(&self) -> bool {
        match self {
            Self::FinishedAborted
            | Self::FinishedIgnored
            | Self::FinishedLengthCapped
            | Self::FinishedStopped => true,
            Self::Waiting | Self::Running | Self::Swapped => false,
        }
    }

    /// Returns the reason why the sequence finished, if applicable.
    ///
    /// # Returns
    /// - `Some(String)`: A string describing the reason for finishing, if the sequence has finished.
    /// - `None`: If the sequence has not finished (i.e., it's still waiting, running, or swapped).
    ///
    /// # Examples
    ///
    /// ```
    /// use atoma_vllm::sequence::SequenceStatus;
    ///
    /// assert_eq!(SequenceStatus::FinishedStopped.finished_reason(), Some("stopped".to_string()));
    /// assert_eq!(SequenceStatus::Running.finished_reason(), None);
    /// ```
    pub fn finished_reason(&self) -> Option<String> {
        match self {
            Self::FinishedAborted => Some("aborted".into()),
            Self::FinishedIgnored => Some("ignored".into()),
            Self::FinishedLengthCapped => Some("length_capped".into()),
            Self::FinishedStopped => Some("stopped".into()),
            Self::Waiting | Self::Running | Self::Swapped => None,
        }
    }
}

/// Represents the current stage of processing for a `Sequence`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequenceStage {
    /// The initial stage where the prompt is being processed.
    /// During this stage, the model computes attention for all input tokens simultaneously.
    Prefill,
    /// The stage where new tokens are generated one at a time.
    /// This stage follows the Prefill stage and uses cached attention from previous tokens.
    Decode,
}

/// Metrics tracking various time points and durations for a request's lifecycle.
#[derive(Clone, Debug)]
pub struct RequestMetrics {
    /// The time when the request was received by the service.
    pub arrival_time: Instant,
    /// The time when the most recent token was generated.
    pub last_token_time: Instant,
    /// The time when the request was first scheduled for processing.
    /// This is None if the request hasn't been scheduled yet.
    pub first_scheduled_time: Option<Instant>,
    /// The time when the first token was generated for this request.
    /// This is None if no tokens have been generated yet.
    pub first_token_time: Option<Instant>,
    /// The duration the request spent waiting in the queue before being processed.
    /// This is None if the request hasn't been scheduled yet.
    pub time_in_queue: Option<Duration>,
    /// The time when token generation for this request was completed.
    /// This is None if the request is still in progress.
    pub finished_time: Option<Instant>,
}

/// `SequenceData` - Represents the data associated with a `Sequence`
///
/// This struct holds information about the prompt and generated output tokens,
/// as well as metadata about the sequence's processing state.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceData {
    /// The token IDs of the initial prompt
    prompt_token_ids: Vec<u32>,
    /// The token IDs of the generated output
    output_token_ids: Vec<u32>,
    /// The cumulative log probability of the generated tokens
    cumulative_logprob: f32,
    /// The number of tokens that have been processed so far
    num_computed_tokens: usize,
    /// The current processing stage of the sequence
    stage: SequenceStage,
    /// Tracing span
    span: Span,
}

impl SequenceData {
    /// Constructor
    pub fn new(prompt_token_ids: Vec<u32>, output_token_ids: Vec<u32>) -> Self {
        Self {
            prompt_token_ids,
            output_token_ids,
            cumulative_logprob: 0.0,
            num_computed_tokens: 0,
            stage: SequenceStage::Prefill,
            span: info_span!("sequence-data"),
        }
    }

    /// Adds a new generated output token id to the sequence data.
    ///
    /// This method appends the given token ID to the output tokens and updates
    /// the cumulative log probability.
    ///
    /// # Arguments
    ///
    /// * `token_id` - The ID of the new token to add.
    /// * `logprob` - The log probability of the new token.
    ///
    /// # Effects
    ///
    /// * Appends `token_id` to `output_token_ids`.
    /// * Adds `logprob` to `cumulative_logprob`.
    #[instrument(skip_all)]
    pub fn add_token_id(&mut self, token_id: u32, logprob: f32) {
        let _enter = self.span.enter();
        trace!("Adding token id to `SequenceData`..");
        self.output_token_ids.push(token_id);
        self.cumulative_logprob += logprob;
    }

    /// Returns the total number of tokens in the sequence.
    ///
    /// This method calculates the sum of prompt tokens and generated output tokens.
    ///
    /// # Returns
    ///
    /// * `usize` - The total number of tokens in the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// let sequence_data = SequenceData::new(vec![1, 2, 3], vec![4, 5]);
    /// assert_eq!(sequence_data.length(), 5);
    /// ```
    pub fn length(&self) -> usize {
        self.prompt_token_ids.len() + self.output_token_ids.len()
    }

    /// Returns the length of the prompt token ids.
    ///
    /// This method returns the number of tokens in the initial prompt.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of tokens in the prompt.
    pub fn get_prompt_len(&self) -> usize {
        self.prompt_token_ids.len()
    }

    /// Returns the length of the output token ids.
    ///
    /// This method returns the number of tokens that have been generated as output.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of tokens in the output.
    pub fn get_output_len(&self) -> usize {
        self.output_token_ids.len()
    }

    /// Returns all token ids, including both prompt and output.
    ///
    /// This method combines the prompt token ids and the output token ids into a single vector.
    ///
    /// # Returns
    ///
    /// * `Vec<u32>` - A vector containing all token ids, with prompt tokens followed by output tokens.
    pub fn get_token_ids(&self) -> Vec<u32> {
        let mut output = Vec::with_capacity(self.length());
        output.extend(&self.prompt_token_ids);
        output.extend(&self.output_token_ids);
        output
    }

    /// Get prefix token IDs for the sequence.
    ///
    /// This function returns a tuple containing two vectors:
    /// 1. Prefix tokens from the prompt
    /// 2. Prefix tokens from the generated output (if any)
    ///
    /// The total number of tokens returned will be equal to `num_tokens`,
    /// unless `num_tokens` exceeds the total number of tokens in the sequence.
    ///
    /// # Arguments
    ///
    /// * `num_tokens` - The number of prefix tokens to return
    ///
    /// # Returns
    ///
    /// A tuple `(prompt_prefix, output_prefix)`, where:
    /// - `prompt_prefix` is a vector of token IDs from the prompt
    /// - `output_prefix` is a vector of token IDs from the generated output
    ///
    /// # Examples
    ///
    /// ```
    /// let sequence_data = SequenceData::new(vec![1, 2, 3], vec![4, 5, 6]);
    ///
    /// assert_eq!(sequence_data.get_prefix_token_ids(2), (vec![1, 2], vec![]));
    /// assert_eq!(sequence_data.get_prefix_token_ids(4), (vec![1, 2, 3], vec![4]));
    /// assert_eq!(sequence_data.get_prefix_token_ids(10), (vec![1, 2, 3], vec![4, 5, 6]));
    /// ```
    pub fn get_prefix_token_ids(&self, num_tokens: usize) -> (Vec<u32>, Vec<u32>) {
        let prompt_len = self.get_prompt_len();
        if num_tokens > prompt_len {
            (
                self.prompt_token_ids.clone(),
                self.output_token_ids[..(num_tokens - prompt_len)].to_vec(),
            )
        } else {
            (self.prompt_token_ids[..num_tokens].to_vec(), vec![])
        }
    }

    /// Returns the number of tokens that have been computed so far.
    ///
    /// This includes both prefill tokens and any generated tokens that have been processed.
    ///
    /// # Returns
    ///
    /// * `usize` - The total number of computed tokens.
    pub fn get_num_computed_tokens(&self) -> usize {
        self.num_computed_tokens
    }

    /// Computes the number of tokens that have not yet been processed.
    ///
    /// This method calculates the difference between the total length of the sequence
    /// (including both prompt and generated output tokens) and the number of tokens
    /// that have already been computed.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of uncomputed tokens.
    ///
    /// # Note
    ///
    /// This method uses `length()` which includes both `prompt_len` and `output_len`
    /// instead of just `prompt_len`. This is necessary because during recomputation,
    /// we need to prefill for both the prompt and any previously generated output.
    pub fn get_num_uncomputed_tokens(&self) -> usize {
        // NOTE: we use `length()` which includes `prompt_len + output_len` instead
        // of `prompt_len` here. This is because during recompute we need to
        // prefill for both prompt and output.
        self.length() - self.get_num_computed_tokens()
    }

    /// Updates the number of computed tokens for this sequence.
    ///
    /// This method is called after processing a batch of tokens to update the sequence's state.
    /// It transitions the sequence from the Prefill stage to the Decode stage when all tokens
    /// (including both prompt and any generated output) have been computed.
    ///
    /// # Arguments
    ///
    /// * `num_new_computed_tokens` - The number of new tokens that have been computed in the current batch.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the update was successful.
    /// * `Err(SequenceError::InvalidNumberGeneratedTokens)` if the update would result in more computed tokens than the total sequence length.
    ///
    /// # Effects
    ///
    /// * Increments `num_computed_tokens` by `num_new_computed_tokens`.
    /// * May change `stage` from `Prefill` to `Decode` if all tokens have been computed.
    ///
    /// # Logging
    ///
    /// * Traces the update operation.
    /// * Logs debug information about the sequence state.
    /// * Logs an error if the update would exceed the total sequence length.
    #[instrument(skip(self))]
    pub fn update_num_computed_tokens(
        &mut self,
        num_new_computed_tokens: usize,
    ) -> Result<(), SequenceError> {
        trace!(
            "Update number of computed tokens {} by {}",
            self.num_computed_tokens,
            num_new_computed_tokens
        );

        debug!(
            "Parameters: self.num_computed_tokens = {}, self.length() = {}, num_new_computed_tokens = {}, self.get_num_uncomputed_tokens() = {}", 
            self.num_computed_tokens,
            self.length(),
            num_new_computed_tokens,
            self.get_num_uncomputed_tokens()
        );

        self.num_computed_tokens += num_new_computed_tokens;
        if self.num_computed_tokens <= self.length() {
            if self.get_num_uncomputed_tokens() == 0 {
                // Prompt tokens attention layers have been now computed, so sequence transits to decode stage
                self.stage = SequenceStage::Decode;
            }
            return Ok(());
        }
        error!("Failed to update number of computed tokens: self.num_computed_tokens = {}, self.length() = {}", self.num_computed_tokens, self.length());
        Err(SequenceError::InvalidNumberGeneratedTokens)
    }

    /// Resets the state for recomputation.
    ///
    /// This method should be called when a sequence needs to be restarted from the beginning,
    /// for example, when a sequence is preempted.
    ///
    /// # Effects
    /// - Sets `num_computed_tokens` to 0.
    /// - Sets `stage` to `SequenceStage::Prefill`.
    pub fn reset_state_for_recompute(&mut self) {
        self.num_computed_tokens = 0;
        self.stage = SequenceStage::Prefill
    }

    /// Returns the ID of the last generated token.
    ///
    /// # Returns
    /// - `Some(u32)`: The ID of the last generated token if any tokens have been generated.
    /// - `Some(u32)`: The ID of the last prompt token if no tokens have been generated.
    /// - `None`: If there are no prompt tokens and no generated tokens.
    pub fn get_last_token_id(&self) -> Option<u32> {
        if self.output_token_ids.is_empty() {
            return self.prompt_token_ids.last().copied();
        }
        self.output_token_ids.last().copied()
    }

    /// Returns a clone of the prompt token IDs.
    ///
    /// # Returns
    /// A `Vec<u32>` containing the token IDs of the prompt.
    pub fn prompt_token_ids(&self) -> Vec<u32> {
        self.prompt_token_ids.clone()
    }

    /// Returns a clone of the output token IDs.
    ///
    /// # Returns
    /// A `Vec<u32>` containing the token IDs of the generated output.
    pub fn output_token_ids(&self) -> Vec<u32> {
        self.output_token_ids.clone()
    }

    /// Returns the current processing stage of the sequence.
    ///
    /// # Returns
    /// The `SequenceStage` enum value representing the current stage (Prefill or Decode).
    pub fn stage(&self) -> SequenceStage {
        self.stage
    }
}

/// `Sequence` - Represents a single sequence in the generation process, storing its data, status, and block information.
#[derive(Clone, Debug, PartialEq)]
pub struct Sequence {
    /// Unique identifier for the sequence.
    sequence_id: u64,
    /// The initial input text for the sequence.
    pub prompt: String,
    /// Token IDs corresponding to the prompt.
    pub prompt_token_ids: Vec<u32>,
    /// Detailed data about the sequence's tokens and processing state.
    pub sequence_data: SequenceData,
    /// Size of each logical token block. Should match the block size used by the block manager and cache engine.
    block_size: usize,
    /// Vector of logical token blocks representing the sequence's tokens.
    pub logical_token_blocks: Vec<LogicalTokenBlock>,
    /// The generated text output.
    pub output_text: String,
    /// Log probabilities for each generated token, mapping token IDs to their `LogProb`.
    pub output_logprobs: Vec<HashMap<u32, LogProb>>,
    /// Current status of the sequence (e.g., waiting, running, finished).
    sequence_status: SequenceStatus,
    /// Reason for stopping generation, if applicable.
    pub stop_reason: Option<u32>,
    /// Vector of generated token strings.
    pub tokens: Vec<String>,
    /// Tracing span for the sequence.
    span: Span,
}

impl Sequence {
    /// Constructor
    pub fn new(
        sequence_id: u64,
        prompt: String,
        prompt_token_ids: Vec<u32>,
        block_size: usize,
        return_full_text: bool,
    ) -> Result<Self, SequenceError> {
        let output_text = if return_full_text {
            prompt.clone()
        } else {
            String::new()
        };

        let mut am = Self {
            sequence_id,
            prompt,
            prompt_token_ids: prompt_token_ids.clone(),
            sequence_data: SequenceData::new(prompt_token_ids.clone(), vec![]),
            logical_token_blocks: vec![],
            block_size,
            output_logprobs: vec![],
            output_text,
            sequence_status: SequenceStatus::Waiting,
            stop_reason: None,
            tokens: vec![],
            span: info_span!("sequence"),
        };

        // Initialize the logical token blocks with the prompt token ids.
        am.append_tokens_to_blocks(&prompt_token_ids)?;
        Ok(am)
    }

    /// Get `output_text`
    pub fn get_output_text(&self) -> String {
        self.output_text.clone()
    }

    /// Computes the hash of a block given its logical index.
    ///
    /// This function calculates a hash based on the prefix tokens up to the given logical block index.
    /// The hash is used to uniquely identify the content of a block for caching purposes.
    ///
    /// # Arguments
    ///
    /// * `logical_idx` - The logical index of the block to hash.
    ///
    /// # Returns
    ///
    /// A 64-bit hash value representing the content of the block.
    ///
    /// # Note
    ///
    /// This implementation may produce incorrect hashes when the block size is greater than the prompt size.
    /// A more robust implementation should be considered for such cases.
    ///
    /// NOTE: This is especially relevant for prefix caching
    pub fn hash_of_block(&self, logical_idx: usize) -> u64 {
        // TODO: This can produce incorrect hash when block size > prompt size
        let num_tokens = self.num_hashed_tokens_of_block(logical_idx);
        let hashed_tokens = self.sequence_data.get_prefix_token_ids(num_tokens);

        let mut hasher = DefaultHasher::new();
        hashed_tokens.hash(&mut hasher);

        hasher.finish()
    }

    /// Calculates the number of tokens that should be hashed for a given logical block index.
    ///
    /// This method is used to determine how many tokens should be considered when computing
    /// the hash of a block, which is important for caching and identifying unique block states.
    ///
    /// # Arguments
    ///
    /// * `logical_idx` - The logical index of the block.
    ///
    /// # Returns
    ///
    /// The number of tokens that should be hashed for the given logical block index.
    pub fn num_hashed_tokens_of_block(&self, logical_idx: usize) -> usize {
        self.block_size * (logical_idx + 1)
    }

    /// Resets the internal state of the sequence for recomputation.
    ///
    /// This method is called when the sequence needs to be reprocessed from the beginning,
    /// such as when handling preemption or when restarting generation. It delegates the
    /// reset operation to the underlying `sequence_data`.
    pub fn reset_state_for_recompute(&mut self) {
        self.sequence_data.reset_state_for_recompute()
    }

    /// Appends a new logical block to the sequence.
    ///
    /// This method creates a new `LogicalTokenBlock` and adds it to the `logical_token_blocks` vector.
    /// It's used when the sequence needs to expand to accommodate more tokens.
    ///
    /// # Effects
    ///
    /// * Creates a new `LogicalTokenBlock` with the next available index and the sequence's block size.
    /// * Appends the new block to the `logical_token_blocks` vector.
    fn append_logical_block(&mut self) {
        let block = LogicalTokenBlock::new(self.logical_token_blocks.len(), self.block_size);
        self.logical_token_blocks.push(block)
    }

    /// Appends tokens to the logical blocks of the `Sequence`.
    ///
    /// This method iterates through the given token IDs and appends them to the logical blocks.
    /// If necessary, it creates new logical blocks to accommodate all tokens.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - A slice of token IDs to be appended.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the operation is successful.
    /// * `Err(SequenceError)` if there's an error appending tokens to a block.
    ///
    /// # Algorithm
    ///
    /// 1. Initialize a cursor at the start of `token_ids`.
    /// 2. While there are tokens to process:
    ///    a. If there are no logical blocks, create a new one.
    ///    b. Get the last block (creating a new one if the last is full).
    ///    c. Determine how many tokens can fit in the current block.
    ///    d. Append as many tokens as possible to the current block.
    ///    e. Move the cursor forward.
    ///
    /// # Note
    ///
    /// This method ensures that all tokens are appended, even if multiple new blocks need to be created.
    fn append_tokens_to_blocks(&mut self, token_ids: &[u32]) -> Result<(), SequenceError> {
        let mut cursor = 0;
        while cursor < token_ids.len() {
            if self.logical_token_blocks.is_empty() {
                self.append_logical_block();
            }

            let last_block = if self.is_last_block_full() {
                self.append_logical_block();
                // DON'T PANIC: at this point in the logic, we already checked that `self.logical_token_blocks` is not empty
                self.logical_token_blocks.last_mut().unwrap()
            } else {
                // DON'T PANIC: at this point in the logic, we already checked that `self.logical_token_blocks` is not empty
                self.logical_token_blocks.last_mut().unwrap()
            };

            let num_empty_slots = last_block.get_num_empty_slots();
            let start = cursor;
            let end = token_ids.len().min(cursor + num_empty_slots);
            last_block.append_tokens(&token_ids[start..end])?;
            cursor += num_empty_slots;
        }
        Ok(())
    }

    /// Checks if the last logical token block in the `Sequence` is full.
    ///
    /// # Returns
    ///
    /// - `true` if the last block exists and is full.
    /// - `false` if the last block exists and is not full, or if there are no blocks.
    ///
    /// # Note
    ///
    /// This method is used to determine if a new block needs to be appended
    /// when adding more tokens to the sequence.
    fn is_last_block_full(&self) -> bool {
        self.logical_token_blocks
            .last()
            .map(|b| b.is_full())
            .unwrap_or(false)
    }

    /// Gets the total number of logical token blocks in this sequence.
    ///
    /// # Returns
    ///
    /// The number of `LogicalTokenBlock`s in the `logical_token_blocks` vector.
    ///
    /// # Note
    ///
    /// This count includes all blocks, regardless of whether they are full or partially filled.
    /// It's useful for understanding the current memory usage and structure of the sequence.
    pub fn get_num_total_logical_token_blocks(&self) -> usize {
        self.logical_token_blocks.len()
    }

    /// Appends a single token to the `Sequence`
    ///
    /// # Arguments
    ///
    /// * `token_id` - The ID of the token to be added
    /// * `logprobs` - A HashMap containing log probabilities for tokens, where the key is the token ID
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the token was successfully added
    /// * `Err(SequenceError)` if there was an error appending the token to the blocks
    ///
    /// # Effects
    ///
    /// If the `token_id` exists in `logprobs`:
    /// * Appends the token to the logical blocks
    /// * Adds the token and its log probability to the sequence data
    /// * Pushes the entire `logprobs` HashMap to `output_logprobs`
    #[instrument(skip_all)]
    pub fn add_token_id(
        &mut self,
        token_id: u32,
        logprobs: HashMap<u32, LogProb>,
    ) -> Result<(), SequenceError> {
        if logprobs.contains_key(&token_id) {
            self.append_tokens_to_blocks(&[token_id])?;
            // DON'T PANIC: we have already verified that `token_id` is a valid key in `logprobs`
            let logprob = logprobs.get(&token_id).unwrap().logprob;
            self.sequence_data.add_token_id(token_id, logprob);
            self.output_logprobs.push(logprobs);
        }
        Ok(())
    }

    /// Returns the total length of the sequence, including both prompt and generated tokens
    ///
    /// # Returns
    ///
    /// * `usize` - The total number of tokens in the sequence
    pub fn length(&self) -> usize {
        self.sequence_data.length()
    }

    /// Getter for `block_size`
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the length of the prompt in the underlying `SequenceData`.
    ///
    /// This method provides the number of tokens in the initial prompt.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of tokens in the prompt.
    pub fn get_prompt_len(&self) -> usize {
        self.sequence_data.get_prompt_len()
    }

    /// Returns the length of the generated output in the underlying `SequenceData`.
    ///
    /// This method provides the number of tokens that have been generated as output.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of tokens in the generated output.
    pub fn get_output_len(&self) -> usize {
        self.sequence_data.get_output_len()
    }

    /// Retrieves the prompt token IDs from the underlying `SequenceData`.
    ///
    /// This method returns a vector of token IDs representing the initial prompt.
    ///
    /// # Returns
    ///
    /// * `Vec<u32>` - A vector containing the token IDs of the prompt.
    pub fn prompt_token_ids(&self) -> Vec<u32> {
        self.sequence_data.prompt_token_ids()
    }

    /// Retrieves the ID of the last token in the sequence.
    ///
    /// This method returns the ID of the most recently generated token, or the last token of the prompt if no tokens have been generated yet.
    ///
    /// # Returns
    ///
    /// * `Option<u32>` - The ID of the last token, or `None` if the sequence is empty.
    pub fn get_last_token_id(&self) -> Option<u32> {
        self.sequence_data.get_last_token_id()
    }

    /// Retrieves the output token IDs from the underlying `SequenceData`.
    ///
    /// This method returns a vector of token IDs representing the generated output.
    ///
    /// # Returns
    ///
    /// * `Vec<u32>` - A vector containing the token IDs of the generated output.
    pub fn output_token_ids(&self) -> Vec<u32> {
        self.sequence_data.output_token_ids()
    }

    /// Returns all token IDs in the sequence, including both prompt and generated tokens.
    ///
    /// # Returns
    ///
    /// * `Vec<u32>` - A vector containing all token IDs in the sequence.
    pub fn get_token_ids(&self) -> Vec<u32> {
        self.sequence_data.get_token_ids()
    }

    /// Returns the cumulative log probability of all generated tokens in the sequence.
    ///
    /// # Returns
    ///
    /// * `f32` - The sum of log probabilities for all generated tokens.
    pub fn cumulative_logprob(&self) -> f32 {
        self.sequence_data.cumulative_logprob
    }

    /// Returns the current status of the sequence.
    ///
    /// # Returns
    ///
    /// * `SequenceStatus` - The current status of the sequence (e.g., Waiting, Running, Finished).
    pub fn get_sequence_status(&self) -> SequenceStatus {
        self.sequence_status
    }

    /// Updates the status of the sequence.
    ///
    /// # Arguments
    ///
    /// * `sequence_status` - The new `SequenceStatus` to set.
    pub fn set_sequence_status(&mut self, sequence_status: SequenceStatus) {
        self.sequence_status = sequence_status
    }

    /// Checks if the sequence has finished processing.
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if the sequence has reached a terminal state, `false` otherwise.
    pub fn is_finished(&self) -> bool {
        self.sequence_status.is_finished()
    }

    /// Calculate the beam search score with length penalty.
    ///
    /// Adapted from
    ///
    /// https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    pub fn get_beam_search_score(
        &self,
        length_penalty: Option<f32>,
        mut sequence_length: Option<usize>,
        eos_token_id: Option<u32>,
    ) -> f32 {
        let length_penalty = length_penalty.unwrap_or(1.0);
        // NOTE: HF implementation does not count the EOS token
        // towards the length, we align with that here for testing.
        if sequence_length.is_none() {
            sequence_length = Some(self.length());
            if eos_token_id.is_some() && self.get_last_token_id() == eos_token_id {
                sequence_length = sequence_length.map(|l| l - 1);
            }
        }
        // DON'T PANIC: sequence length already enforced to be non null
        self.cumulative_logprob() / (sequence_length.unwrap() as f32 * length_penalty)
    }

    /// Creates a new `Sequence` by forking the current one.
    ///
    /// This method creates a deep copy of the current `Sequence` with a new sequence ID.
    /// It's typically used in beam search or other scenarios where multiple sequence
    /// variations need to be explored.
    ///
    /// # Arguments
    ///
    /// * `new_sequence_id` - The ID to assign to the new forked sequence.
    ///
    /// # Returns
    ///
    /// A new `Sequence` instance that is a clone of the current one, but with a new ID.
    ///
    /// # Example
    ///
    /// ```
    /// let original_sequence = Sequence::new(/* ... */);
    /// let forked_sequence = original_sequence.fork(new_sequence_id);
    /// assert_ne!(original_sequence.sequence_id(), forked_sequence.sequence_id());
    /// ```
    #[instrument(skip_all)]
    pub fn fork(&self, new_sequence_id: u64) -> Self {
        let _enter = self.span.enter();
        trace!("Forking sequence..");
        let mut new_seq = self.clone();
        new_seq.sequence_id = new_sequence_id;
        new_seq
    }

    /// Get the number of new tokens to be computed in the next iteration.
    ///
    /// This method determines how many tokens should be processed in the upcoming
    /// computation step, based on the current stage of the sequence.
    ///
    /// # Returns
    /// - For the Decode stage: Always returns 1, as we generate one new token at a time.
    /// - For the Prefill stage: Returns the number of remaining uncomputed tokens in the prompt.
    ///
    /// # Examples
    ///
    /// ```
    /// let sequence = Sequence::new(/* ... */);
    ///
    /// // During prefill stage with 5 uncomputed tokens
    /// assert_eq!(sequence.get_num_new_tokens(), 5);
    ///
    /// // After transitioning to decode stage
    /// assert_eq!(sequence.get_num_new_tokens(), 1);
    /// ```
    pub fn get_num_new_tokens(&self) -> usize {
        if self.sequence_data.stage == SequenceStage::Decode {
            return 1;
        }
        self.sequence_data.get_num_uncomputed_tokens()
    }

    /// Checks if the sequence is in the `Prefill` stage.
    ///
    /// # Returns
    ///
    /// `true` if the sequence is in the Prefill stage, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// let sequence = Sequence::new(/* ... */);
    /// assert!(sequence.is_prefill());
    /// // After processing some tokens...
    /// assert!(!sequence.is_prefill());
    /// ```
    pub fn is_prefill(&self) -> bool {
        self.sequence_data.stage == SequenceStage::Prefill
    }

    /// Returns the unique identifier of the sequence.
    ///
    /// # Returns
    ///
    /// The `u64` sequence ID.
    ///
    /// # Example
    ///
    /// ```
    /// let sequence = Sequence::new(42, /* ... */);
    /// assert_eq!(sequence.sequence_id(), 42);
    /// ```
    pub fn sequence_id(&self) -> u64 {
        self.sequence_id
    }

    /// Returns a clone of the internal `SequenceData`.
    ///
    /// # Returns
    ///
    /// A cloned `SequenceData` instance.
    ///
    /// # Note
    ///
    /// This method performs a deep copy of the `SequenceData`. Consider using
    /// a reference if you don't need ownership of the data.
    ///
    /// # Example
    ///
    /// ```
    /// let sequence = Sequence::new(/* ... */);
    /// let data = sequence.sequence_data();
    /// assert_eq!(data.length(), sequence.length());
    /// ```
    pub fn sequence_data(&self) -> SequenceData {
        self.sequence_data.clone()
    }
}

pub type SyncSequence = Arc<RwLock<Sequence>>;

impl ReadLock for SyncSequence {
    type Error = SequenceError;
    type Inner = Sequence;

    fn read_lock(&self) -> Result<std::sync::RwLockReadGuard<Self::Inner>, Self::Error> {
        self.read()
            .map_err(|e| SequenceError::ReadLockError(e.to_string()))
    }
}

impl WriteLock for SyncSequence {
    type Error = SequenceError;
    type Inner = Sequence;

    fn write_lock(&self) -> Result<std::sync::RwLockWriteGuard<Self::Inner>, Self::Error> {
        self.write()
            .map_err(|e| SequenceError::WriteLockError(e.to_string()))
    }
}

/// Represents the type of multi-modal data in a request.
#[derive(Clone, Debug)]
pub enum MultiModalType {
    /// Audio data, such as speech or music.
    Audio,
    /// Image data, including photographs, diagrams, or other visual content.
    Image,
    /// Video data, which may include both visual and audio components.
    Video,
}

/// Represents multi-modal data for requests that involve multiple types of input.
///
/// This struct is used to encapsulate different types of data (such as images, audio, or video)
/// that can be processed alongside text in multi-modal language models.
#[derive(Clone, Debug)]
pub struct MultiModalData {
    /// The type of multi-modal data (e.g., audio, image, video).
    pub r#type: MultiModalType,
    /// The actual multi-modal data as a tensor.
    pub data: Tensor,
}

/// Represents a group of sequences generated from the same prompt.
///
/// This struct manages multiple related sequences, their generation parameters,
/// and associated metrics for a single request.
#[derive(Clone)]
pub struct SequenceGroup {
    /// Unique identifier for the request.
    pub request_id: String,
    /// Map of sequence IDs to their corresponding `Sequence` instances.
    pub sequences: HashMap<u64, Arc<RwLock<Sequence>>>,
    /// Metrics tracking various time points and durations for the request.
    pub metrics: Arc<RwLock<RequestMetrics>>,
    /// Log probabilities for the prompt tokens, if available.
    pub prompt_logprobs: Option<LogProb>,
    /// Parameters for choosing the next token in the sequence.
    next_token_chooser_params: NextTokenChooserParameters,
    /// Criteria for stopping sequence generation.
    stopping_criteria: StoppingCriteriaParameters,
    /// Processor for modifying logits during token generation.
    pub logits_processor: Arc<RwLock<LogitsProcessor>>,
    /// Span for tracing and logging
    pub span: Span,
}

impl SequenceGroup {
    /// Constructor
    pub fn new(
        request_id: String,
        sequences: Vec<Sequence>,
        arrival_time: Instant,
        next_token_chooser_params: NextTokenChooserParameters,
        stopping_criteria: StoppingCriteriaParameters,
        logits_processor: LogitsProcessor,
    ) -> Result<Self, SequenceError> {
        if sequences.is_empty() {
            return Err(SequenceError::ConstructorError(
                "Empty vector of `Sequence`s".into(),
            ));
        }
        Ok(Self {
            request_id,
            sequences: sequences
                .into_iter()
                .map(|s| (s.sequence_id, Arc::new(RwLock::new(s))))
                .collect(),
            metrics: Arc::new(RwLock::new(RequestMetrics {
                arrival_time,
                last_token_time: arrival_time,
                finished_time: None,
                first_scheduled_time: None,
                first_token_time: None,
                time_in_queue: None,
            })),
            prompt_logprobs: None,
            next_token_chooser_params,
            stopping_criteria,
            logits_processor: Arc::new(RwLock::new(logits_processor)),
            span: info_span!("sequence_group"),
        })
    }

    /// Returns the prompt of the `SequenceGroup`.
    ///
    /// This function retrieves the prompt from the first sequence in the group.
    /// All sequences in a `SequenceGroup` should have the same prompt.
    ///
    /// # Returns
    ///
    /// - `String`: The prompt text of the sequence group.
    ///
    /// # Note
    ///
    /// If the `SequenceGroup` is empty, this function returns an empty string.
    ///
    /// # Example
    ///
    /// ```
    /// let seq_group = SequenceGroup::new(/* ... */);
    /// let prompt = seq_group.prompt();
    /// println!("Prompt: {}", prompt);
    /// ```
    pub fn prompt(&self) -> String {
        self.sequences
            .iter()
            .next()
            .map(|(_, s)| s.read().unwrap().prompt.clone())
            .unwrap_or_default()
    }

    /// Adds a token ID to a specific `Sequence` within this `SequenceGroup`.
    ///
    /// This method attempts to add a new token to the sequence identified by `sequence_id`.
    /// It also updates the associated log probabilities for the token.
    ///
    /// # Arguments
    ///
    /// * `sequence_id` - The unique identifier of the target sequence.
    /// * `token_id` - The ID of the token to be added.
    /// * `logprobs` - A map of token IDs to their log probabilities.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the token was successfully added.
    /// * `Err(SequenceError::MissingSequence)` if no sequence with the given ID was found.
    ///
    /// # Errors
    ///
    /// This method can return an error if:
    /// - The specified sequence is not found in the group.
    /// - There's an issue acquiring the write lock for the sequence.
    /// - The `add_token_id` operation on the sequence fails.
    ///
    /// # Tracing
    ///
    /// This method is instrumented with tracing. It logs a trace event when adding the token
    /// and an error event if the sequence is not found.
    #[instrument(skip(self))]
    pub fn add_token_id_to_seq(
        &self,
        sequence_id: u64,
        token_id: u32,
        logprobs: HashMap<u32, LogProb>,
    ) -> Result<(), SequenceError> {
        let _enter = self.span.enter();
        trace!("Adding token id to sequence in sequence group...");
        if let Some(sequence) = self.sequences.get(&sequence_id) {
            sequence.write_lock()?.add_token_id(token_id, logprobs)?;
            return Ok(());
        }
        error!("Missing sequence, with id = {sequence_id}");
        Err(SequenceError::MissingSequence(sequence_id))
    }

    /// Returns the prompt token IDs for the `SequenceGroup`.
    ///
    /// All sequences in a `SequenceGroup` share the same prompt, so this method
    /// retrieves the prompt token IDs from the first sequence in the group.
    ///
    /// # Returns
    ///
    /// - `Vec<u32>`: A vector containing the token IDs of the prompt.
    ///
    /// # Note
    ///
    /// If the `SequenceGroup` is empty, this function returns an empty vector.
    ///
    /// # Example
    ///
    /// ```
    /// let seq_group = SequenceGroup::new(/* ... */);
    /// let prompt_tokens = seq_group.prompt_token_ids();
    /// println!("Prompt tokens: {:?}", prompt_tokens);
    /// ```
    pub fn prompt_token_ids(&self) -> Vec<u32> {
        self.sequences
            .iter()
            .next()
            .map(|(_, s)| s.read().unwrap().prompt_token_ids.clone())
            .unwrap_or_default()
    }

    /// Calculates the latency since the last token generation and updates the last token time.
    ///
    /// This method performs two main tasks:
    /// 1. It calculates the duration between the current time (`now`) and the last token generation time.
    /// 2. It updates the `last_token_time` in the sequence group's metrics to the current time.
    ///
    /// # Arguments
    ///
    /// * `now` - An `Instant` representing the current time.
    ///
    /// # Returns
    ///
    /// * `Ok(Duration)` - The time elapsed since the last token was generated.
    /// * `Err(SequenceError::WhileInPrefix)` - If the sequence group is still in the prefill stage.
    ///
    /// # Errors
    ///
    /// Returns an error if called during the prefill stage, as latency is only meaningful
    /// for token-by-token generation after the initial prompt processing.
    ///
    /// # Thread Safety
    ///
    /// This method uses read and write locks on the internal metrics. Ensure proper
    /// synchronization when used in a multi-threaded context.
    ///
    /// # Example
    ///
    /// ```
    /// let mut seq_group = SequenceGroup::new(/* ... */);
    /// // ... after some token generation ...
    /// match seq_group.get_last_latency(Instant::now()) {
    ///     Ok(latency) => println!("Latency since last token: {:?}", latency),
    ///     Err(e) => eprintln!("Error: {:?}", e),
    /// }
    /// ```
    pub fn get_last_latency(&mut self, now: Instant) -> Result<Duration, SequenceError> {
        if self.is_prefill() {
            return Err(SequenceError::WhileInPrefix);
        }
        let latency = { now - self.metrics.read().unwrap().last_token_time };
        {
            self.metrics.write().unwrap().last_token_time = now;
        }
        Ok(latency)
    }

    /// Sets the first token time for request-level timings.
    ///
    /// This function attempts to set the time when the first token was generated
    /// for this request. It only sets the time if it hasn't been set before and
    /// if the first token has just been generated.
    ///
    /// # Arguments
    ///
    /// * `time` - The current time to potentially set as the first token time.
    ///
    /// # Behavior
    ///
    /// - If the first token time has already been set, this function does nothing.
    /// - If the output length of the first sequence is exactly 1 (indicating the first
    ///   token has just been generated), it sets the first token time.
    /// - In cases where a sequence group is swapped and recomputed, the time between
    ///   iterations is counted in the total processing time, rather than recalculating
    ///   the time to first token. This is because from the user's perspective, there
    ///   is simply a longer generation delay.
    ///
    /// # Note
    ///
    /// This function is currently marked as `#[allow(dead_code)]` as it may not be
    /// used in the current implementation but is kept for potential future use or
    /// for debugging purposes.
    #[allow(dead_code)]
    fn maybe_set_first_token_time(&mut self, time: Instant) {
        // NOTE: in a case where a sequence_group is swapped and
        // recomputed, the time between iterations is counted
        // in TPOT, rather than recalculating TTFT (since from the
        // POV of the user, there is simply a long generation delay.
        let initial_seq_len = self
            .sequences
            .iter()
            .next()
            .map(|(_, s)| s.read().unwrap().get_output_len())
            .unwrap_or_default();
        let mut metrics_guard = self.metrics.write().unwrap();
        let first_token_time = metrics_guard.first_token_time;
        if first_token_time.is_none() && initial_seq_len == 1 {
            metrics_guard.first_token_time = Some(time);
        }
    }

    /// Sets the first scheduled time and calculates the time spent in queue for request-level timings.
    ///
    /// This method updates the request metrics with the time when the request was first scheduled
    /// for processing and calculates how long the request spent waiting in the queue.
    ///
    /// # Arguments
    ///
    /// * `time` - The current time, typically when the request is first scheduled for processing.
    ///
    /// # Effects
    ///
    /// If the `first_scheduled_time` hasn't been set yet:
    /// - Sets `first_scheduled_time` to the provided `time`.
    /// - Calculates and sets `time_in_queue` as the duration between `arrival_time` and `time`.
    ///
    /// # Thread Safety
    ///
    /// This method uses a write lock on the metrics. Ensure proper synchronization when used in a multi-threaded context.
    ///
    /// # Note
    ///
    /// This method is idempotent; subsequent calls will not change the first scheduled time
    /// or time in queue if they have already been set.
    pub fn maybe_set_first_scheduled_time(&self, time: Instant) {
        let mut metrics_guard = self.metrics.write().unwrap();
        let (arrival_time, first_scheduled_time) = (
            metrics_guard.arrival_time,
            metrics_guard.first_scheduled_time,
        );
        if first_scheduled_time.is_none() {
            metrics_guard.first_scheduled_time = Some(time);
            metrics_guard.time_in_queue = Some(time - arrival_time);
        }
    }

    /// Sets the finished time for the sequence group.
    ///
    /// This method updates the `finished_time` in the group's metrics to mark when
    /// the sequence group completed processing.
    ///
    /// # Arguments
    ///
    /// * `time` - An `Instant` representing the completion time of the sequence group.
    ///
    /// # Thread Safety
    ///
    /// This method uses a write lock on the internal metrics. Ensure proper
    /// synchronization when used in a multi-threaded context.
    pub fn set_finished_time(&self, time: Instant) {
        self.metrics.write().unwrap().finished_time = Some(time);
    }

    /// Retrieves the arrival time of the `SequenceGroup`.
    ///
    /// This method returns the time when the sequence group was initially received
    /// or created.
    ///
    /// # Returns
    ///
    /// An `Instant` representing the arrival time of the sequence group.
    ///
    /// # Thread Safety
    ///
    /// This method uses a read lock on the internal metrics. It's safe to call
    /// concurrently, but be aware of potential contention in high-concurrency scenarios.
    pub fn arrival_time(&self) -> Instant {
        self.metrics.read().unwrap().arrival_time
    }

    /// Returns the maximum number of sequences that could be running in parallel for this request.
    ///
    /// This method determines the upper bound of concurrent sequences based on the generation parameters:
    ///
    /// - For beam search (when `best_of` > 1), it returns the `best_of` value, as this is the maximum
    ///   number of beam candidates that could be explored simultaneously.
    /// - For other sampling methods, it returns the current number of unfinished sequences, as this
    ///   represents the actual number of sequences that still need processing.
    ///
    /// # Returns
    ///
    /// * `usize` - The maximum number of sequences that could be running concurrently.
    ///
    /// # Note
    ///
    /// This method is useful for resource allocation and scheduling, as it provides an upper bound
    /// on the parallelism required for this sequence group.
    pub fn get_max_num_running_seqs(&self) -> usize {
        if self.next_token_chooser_params.best_of > 1 {
            // For beam search, maximally there will always be `best_of` beam
            // candidates running in the future.
            return self.next_token_chooser_params.best_of;
        }
        // At sampling stages, return the number of actual sequences
        // that are not finished yet.
        self.num_unfinished_sequences()
    }

    /// Retrieves sequences from the `SequenceGroup` based on their status.
    ///
    /// # Arguments
    ///
    /// * `status` - An optional `SequenceStatus` to filter the sequences.
    ///
    /// # Returns
    ///
    /// A vector of `Arc<RwLock<Sequence>>` containing the filtered sequences.
    ///
    /// # Details
    ///
    /// - If `status` is `Some(status)`, returns only sequences matching that status.
    /// - If `status` is `None`, returns all sequences in the group.
    ///
    /// # Examples
    ///
    /// ```
    /// let seq_group = SequenceGroup::new(/* ... */);
    ///
    /// // Get all sequences
    /// let all_seqs = seq_group.get_seqs(None);
    ///
    /// // Get only running sequences
    /// let running_seqs = seq_group.get_seqs(Some(SequenceStatus::Running));
    /// ```
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock on each sequence. Ensure proper
    /// synchronization when used in a multi-threaded context.
    pub fn get_seqs(&self, status: Option<SequenceStatus>) -> Vec<Arc<RwLock<Sequence>>> {
        match status {
            Some(status) => self
                .sequences
                .values()
                .filter_map(|seq| {
                    if seq.read().unwrap().sequence_status == status {
                        Some(seq.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            None => self.sequences.values().cloned().collect(),
        }
    }

    /// Retrieves sequence IDs from the `SequenceGroup`, optionally filtered by status.
    ///
    /// # Arguments
    ///
    /// * `status` - An optional `SequenceStatus` to filter the sequences.
    ///
    /// # Returns
    ///
    /// A vector of `u64` containing the IDs of the filtered sequences.
    ///
    /// # Examples
    ///
    /// ```
    /// let seq_group = SequenceGroup::new(/* ... */);
    ///
    /// // Get all sequence IDs
    /// let all_ids = seq_group.get_sequences_ids(None);
    ///
    /// // Get only running sequence IDs
    /// let running_ids = seq_group.get_sequences_ids(Some(SequenceStatus::Running));
    /// ```
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock on each sequence. Ensure proper
    /// synchronization when used in a multi-threaded context.
    pub fn get_sequences_ids(&self, status: Option<SequenceStatus>) -> Vec<u64> {
        match status {
            Some(status) => self
                .sequences
                .values()
                .filter_map(|seq| {
                    if seq.read().unwrap().sequence_status == status {
                        Some(seq.read().unwrap().sequence_id)
                    } else {
                        None
                    }
                })
                .collect(),
            None => self
                .sequences
                .values()
                .map(|s| s.read().unwrap().sequence_id)
                .collect(),
        }
    }

    /// Retrieves a reference to the first sequence in the group, optionally filtered by status.
    ///
    /// # Arguments
    ///
    /// * `status` - An optional `SequenceStatus` to filter the sequences.
    ///
    /// # Returns
    ///
    /// * `Option<&Arc<RwLock<Sequence>>>` - A reference to the first matching sequence, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// let seq_group = SequenceGroup::new(/* ... */);
    ///
    /// // Get the first sequence regardless of status
    /// let first_seq = seq_group.get_first_sequence(None);
    ///
    /// // Get the first running sequence
    /// let first_running_seq = seq_group.get_first_sequence(Some(SequenceStatus::Running));
    /// ```
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock on each sequence. Ensure proper
    /// synchronization when used in a multi-threaded context.
    pub fn get_first_sequence(
        &self,
        status: Option<SequenceStatus>,
    ) -> Option<&Arc<RwLock<Sequence>>> {
        match status {
            Some(status) => self
                .sequences
                .values()
                .find(|seq| seq.read().unwrap().get_sequence_status() == status),
            None => self.sequences.values().next(),
        }
    }

    /// Retrieves a shared reference to a `Sequence` with the specified `sequence_id`.
    ///
    /// This method searches through the sequences in the group and returns a reference
    /// to the `Arc<RwLock<Sequence>>` that matches the given `sequence_id`.
    ///
    /// # Arguments
    ///
    /// * `sequence_id` - The unique identifier of the sequence to retrieve.
    ///
    /// # Returns
    ///
    /// * `Option<&Arc<RwLock<Sequence>>>` - A reference to the matching sequence if found,
    ///   or `None` if no sequence with the given ID exists in the group.
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock on each sequence during the search.
    /// Ensure proper synchronization when used in a multi-threaded context.
    ///
    /// # Example
    ///
    /// ```
    /// let seq_group = SequenceGroup::new(/* ... */);
    /// let sequence_id = 42;
    /// if let Some(sequence) = seq_group.get_sequence_from_id(sequence_id) {
    ///     println!("Found sequence with ID: {}", sequence_id);
    /// } else {
    ///     println!("No sequence found with ID: {}", sequence_id);
    /// }
    /// ```
    pub fn get_sequence_from_id(&self, sequence_id: u64) -> Option<&Arc<RwLock<Sequence>>> {
        self.sequences
            .values()
            .find(|s| s.read().unwrap().sequence_id() == sequence_id)
    }

    /// Retrieves all unfinished sequences from the `SequenceGroup`.
    ///
    /// This method filters and returns a vector of all sequences that have not yet
    /// finished processing. A sequence is considered unfinished if its status is not
    /// in a terminal state (e.g., not FinishedStopped, FinishedLengthCapped, etc.).
    ///
    /// # Returns
    ///
    /// A `Vec<Arc<RwLock<Sequence>>>` containing cloned references to all unfinished sequences.
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock on each sequence. Ensure proper
    /// synchronization when used in a multi-threaded context.
    ///
    /// # Performance Considerations
    ///
    /// - This method clones `Arc` pointers, which is a relatively cheap operation.
    /// - However, it does iterate over all sequences in the group, which could be
    ///   expensive for very large sequence groups.
    ///
    /// # Example
    ///
    /// ```
    /// let seq_group = SequenceGroup::new(/* ... */);
    /// let unfinished_seqs = seq_group.get_unfinished_sequences();
    /// println!("Number of unfinished sequences: {}", unfinished_seqs.len());
    /// ```
    pub fn get_unfinished_sequences(&self) -> Vec<Arc<RwLock<Sequence>>> {
        self.sequences
            .values()
            .filter(|s| !s.read().unwrap().is_finished())
            .cloned()
            .collect()
    }

    /// Retrieves all finished sequences from the `SequenceGroup`.
    ///
    /// This method filters and returns a vector of all sequences that have completed processing.
    /// A sequence is considered finished if its status is in a terminal state
    /// (e.g., FinishedStopped, FinishedLengthCapped, etc.).
    ///
    /// # Returns
    ///
    /// A `Vec<Arc<RwLock<Sequence>>>` containing cloned references to all finished sequences.
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock on each sequence. Ensure proper
    /// synchronization when used in a multi-threaded context.
    ///
    /// # Performance Considerations
    ///
    /// - This method clones `Arc` pointers, which is a relatively cheap operation.
    /// - However, it does iterate over all sequences in the group, which could be
    ///   expensive for very large sequence groups.
    ///
    /// # Example
    ///
    /// ```
    /// let seq_group = SequenceGroup::new(/* ... */);
    /// let finished_seqs = seq_group.get_finished_sequences();
    /// println!("Number of finished sequences: {}", finished_seqs.len());
    /// ```
    pub fn get_finished_sequences(&self) -> Vec<Arc<RwLock<Sequence>>> {
        self.sequences
            .values()
            .filter(|s| s.read().unwrap().is_finished())
            .cloned()
            .collect()
    }

    /// Updates the number of computed tokens for all unfinished sequences in the group.
    ///
    /// This method iterates through all sequences in the group and updates their
    /// computed token count, but only for sequences that are not yet finished.
    ///
    /// # Arguments
    ///
    /// * `num_new_computed_tokens` - The number of new tokens that have been computed
    ///   in the current batch.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the update was successful for all sequences.
    /// * `Err(SequenceError)` if there was an error updating any sequence.
    ///
    /// # Errors
    ///
    /// This method can return an error if:
    /// - There's an issue acquiring read or write locks on any sequence.
    /// - The `update_num_computed_tokens` operation fails for any sequence.
    ///
    /// # Thread Safety
    ///
    /// This method uses read and write locks on each sequence. Ensure proper
    /// synchronization when used in a multi-threaded context.
    #[instrument(skip_all)]
    pub fn update_num_computed_tokens(
        &self,
        num_new_computed_tokens: usize,
    ) -> Result<(), SequenceError> {
        let _enter = self.span.enter();
        trace!("Updating number of computed tokens");
        for sequence in self.sequences.values() {
            let is_finished = { sequence.read_lock()?.is_finished() };
            if !is_finished {
                {
                    sequence
                        .write_lock()?
                        .sequence_data
                        .update_num_computed_tokens(num_new_computed_tokens)?;
                }
            }
        }
        Ok(())
    }

    /// Calculates the total number of uncomputed tokens across all unfinished sequences in the group.
    ///
    /// This method iterates through all sequences in the group, and for each unfinished sequence,
    /// it sums up the number of uncomputed tokens. This is useful for determining how much
    /// computation is left to be done for the entire sequence group.
    ///
    /// # Returns
    ///
    /// * `usize` - The total number of uncomputed tokens across all unfinished sequences.
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock on each sequence. Ensure proper synchronization
    /// when used in a multi-threaded context.
    ///
    /// # Performance Considerations
    ///
    /// - This method iterates over all sequences in the group, which could be expensive
    ///   for very large sequence groups.
    /// - It acquires and releases a read lock for each sequence, which may impact
    ///   performance in high-contention scenarios.
    ///
    /// # Example
    ///
    /// ```
    /// let seq_group = SequenceGroup::new(/* ... */);
    /// let uncomputed_tokens = seq_group.get_num_uncomputed_tokens();
    /// println!("Number of uncomputed tokens: {}", uncomputed_tokens);
    /// ```
    pub fn get_num_uncomputed_tokens(&self) -> usize {
        let mut num_uncomputed_tokens = 0;
        for sequence in self.sequences.values() {
            if !sequence.read().unwrap().is_finished() {
                num_uncomputed_tokens += sequence
                    .read()
                    .unwrap()
                    .sequence_data
                    .get_num_uncomputed_tokens();
            }
        }
        num_uncomputed_tokens
    }

    /// Returns the number of sequences in the group, optionally filtered by a specific status.
    ///
    /// # Arguments
    ///
    /// * `status` - An optional `SequenceStatus` to filter the sequences.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of sequences matching the given status, or the total number of sequences if no status is specified.
    ///
    /// # Examples
    ///
    /// ```
    /// let seq_group = SequenceGroup::new(/* ... */);
    ///
    /// // Get total number of sequences
    /// let total = seq_group.get_num_sequences(None);
    ///
    /// // Get number of running sequences
    /// let running = seq_group.get_num_sequences(Some(SequenceStatus::Running));
    /// ```
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock on each sequence. Ensure proper
    /// synchronization when used in a multi-threaded context.
    pub fn get_num_sequences(&self, status: Option<SequenceStatus>) -> usize {
        if let Some(status) = status {
            let mut len = 0;
            for sequence in self.sequences.values() {
                if sequence.read().unwrap().sequence_status == status {
                    len += 1;
                }
            }
            len
        } else {
            self.sequences.len()
        }
    }

    /// Get the total number of logical blocks needed to be allocated for this `SequenceGroup`.
    ///
    /// This function returns the number of logical token blocks for the first sequence
    /// in the group that matches the given status. Since all sequences in a `SequenceGroup`
    /// share the same initial prompt, checking one sequence is sufficient.
    ///
    /// # Arguments
    ///
    /// * `status` - The `SequenceStatus` to filter sequences by.
    ///
    /// # Returns
    ///
    /// * `Some(usize)` - The number of logical token blocks if a matching sequence is found.
    /// * `None` - If no sequence with the given status is found.
    ///
    /// # Note
    ///
    /// This method acquires a read lock on each sequence. Ensure proper
    /// synchronization when used in a multi-threaded context.
    pub fn get_num_total_logical_token_blocks(&self, status: SequenceStatus) -> Option<usize> {
        // NOTE: All `Sequence`s in `SequenceGroup` share the same initial prompt, therefore
        // it is sufficient to check how many logical token blocks are contained in the first `Sequence` with `status`
        for sequence in self.sequences.values() {
            if sequence.read().unwrap().sequence_status == status {
                return Some(
                    sequence
                        .read()
                        .unwrap()
                        .get_num_total_logical_token_blocks(),
                );
            }
        }
        None
    }

    /// Returns the number of unfinished sequences in the group.
    ///
    /// An unfinished sequence is one that has not yet reached a terminal state
    /// (e.g., not stopped, length capped, or aborted).
    ///
    /// # Returns
    ///
    /// * `usize` - The count of unfinished sequences.
    ///
    /// # Performance Note
    ///
    /// This method calls `get_unfinished_sequences()`, which iterates over all sequences.
    /// For frequent checks on large sequence groups, consider caching this value if possible.
    pub fn num_unfinished_sequences(&self) -> usize {
        self.get_unfinished_sequences().len()
    }

    /// Returns the number of finished sequences in the group.
    ///
    /// A finished sequence is one that has reached a terminal state
    /// (e.g., stopped, length capped, or aborted).
    ///
    /// # Returns
    ///
    /// * `usize` - The count of finished sequences.
    ///
    /// # Performance Note
    ///
    /// This method calls `get_finished_sequences()`, which iterates over all sequences.
    /// For frequent checks on large sequence groups, consider caching this value if possible.
    pub fn num_finished_sequences(&self) -> usize {
        self.get_finished_sequences().len()
    }

    /// Checks if the sequence group is in the prefill phase.
    ///
    /// This method determines if the sequence group is still in the prefill stage
    /// by checking the first sequence in the group. All sequences in a group
    /// should be in the same phase (either all in prefill or all in decode).
    ///
    /// # Returns
    ///
    /// * `true` if the sequence group is in the prefill phase.
    /// * `false` if the sequence group is in the decode phase or if there are no sequences.
    ///
    /// # Note
    ///
    /// This method assumes that all sequences in the group are in the same phase.
    /// It only checks the first sequence for efficiency.
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock on the first sequence. Ensure proper
    /// synchronization when used in a multi-threaded context.
    pub fn is_prefill(&self) -> bool {
        self.sequences
            .iter()
            .next()
            .map(|(_, s)| s.read().unwrap().is_prefill())
            .unwrap_or(false)
    }

    /// Finds a `Sequence` with the given `sequence_id` in this `SequenceGroup`.
    ///
    /// # Arguments
    ///
    /// * `sequence_id` - The unique identifier of the sequence to find.
    ///
    /// # Returns
    ///
    /// * `Some(Arc<RwLock<Sequence>>)` if a sequence with the given ID is found.
    /// * `None` if no sequence with the given ID exists in this group.
    pub fn find(&self, sequence_id: u64) -> Option<Arc<RwLock<Sequence>>> {
        self.sequences.get(&sequence_id).cloned()
    }

    /// Adds a new `Sequence` to this `SequenceGroup`.
    ///
    /// If a sequence with the same ID already exists in the group, this method does nothing.
    ///
    /// # Arguments
    ///
    /// * `sequence` - The new sequence to add, wrapped in an `Arc<RwLock<>>`.
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock on the sequence to get its ID.
    pub fn add(&mut self, sequence: Arc<RwLock<Sequence>>) {
        let sequence_id = { sequence.read().unwrap().sequence_id };
        if self.sequences.contains_key(&sequence_id) {
            return;
        }
        self.sequences.insert(sequence_id, sequence);
    }

    /// Removes a `Sequence` from this `SequenceGroup`.
    ///
    /// This method is idempotent - if no sequence with the given ID exists,
    /// this method does nothing.
    ///
    /// # Arguments
    ///
    /// * `sequence_id` - The unique identifier of the sequence to remove.
    pub fn remove(&mut self, sequence_id: u64) {
        self.sequences.remove(&sequence_id);
    }

    /// Checks if generation is finished for all `Sequence`s in this `SequenceGroup`.
    ///
    /// # Returns
    ///
    /// * `true` if all sequences in the group have finished processing.
    /// * `false` if any sequence in the group is still in progress.
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock on each sequence in the group.
    pub fn is_finished(&self) -> bool {
        self.sequences
            .values()
            .all(|s| s.read().unwrap().is_finished())
    }

    /// Returns the next token chooser parameters for this `SequenceGroup`.
    ///
    /// # Returns
    ///
    /// A clone of the `NextTokenChooserParameters` associated with this group.
    pub fn next_token_chooser_params(&self) -> NextTokenChooserParameters {
        self.next_token_chooser_params.clone()
    }

    /// Returns the stopping criteria parameters for this `SequenceGroup`.
    ///
    /// This method provides access to the stopping criteria used to determine
    /// when to halt the generation process for sequences in this group.
    ///
    /// # Returns
    ///
    /// A clone of the `StoppingCriteriaParameters` associated with this group.
    pub fn stopping_params(&self) -> StoppingCriteriaParameters {
        self.stopping_criteria.clone()
    }
}

impl std::fmt::Debug for SequenceGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequenceGroup")
            .field("request_id", &self.request_id)
            .field("sequences", &self.sequences)
            .field("metrics", &self.metrics)
            .field("prompt_logprobs", &self.prompt_logprobs)
            .field("next_token_chooser_params", &self.next_token_chooser_params)
            .field("stopping_criteria", &self.stopping_criteria)
            .finish()
    }
}

/// Metadata for a sequence group, used to create `AttentionMetadata`.
///
/// This struct encapsulates various parameters and data structures related to a group of sequences
/// being processed together, typically for batched inference in language models.
pub struct SequenceGroupMetadata {
    /// Unique identifier for the request associated with this sequence group.
    pub request_id: String,
    /// Indicates whether the current processing stage is for the initial prompt (true) or for token generation (false).
    pub is_prompt: bool,
    /// Parameters controlling the selection of the next token in the sequence.
    pub next_token_chooser_params: NextTokenChooserParameters,
    /// Criteria for determining when to stop sequence generation.
    pub stopping_criteria_params: StoppingCriteriaParameters,
    /// Mapping of sequence IDs to their corresponding block numbers in physical memory.
    /// This is used for efficient memory management of token sequences.
    pub block_tables: HashMap<u64, Vec<u32>>,
    /// Indicates whether sampling should be performed during token generation.
    /// Set to false for certain operations like chunked prefill where sampling isn't needed.
    pub do_sample: bool,
    /// Number of tokens to be processed per sequence in the current batch.
    /// This is used for chunked processing of long sequences.
    pub token_chunk_size: usize,
    /// Detailed data for each sequence in the group, keyed by sequence ID.
    pub sequence_data: HashMap<u64, SequenceData>,
    /// Processor for modifying logits during token generation, shared across the sequence group.
    pub logits_processor: Arc<RwLock<LogitsProcessor>>,
}

impl SequenceGroupMetadata {
    /// Constructor
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: String,
        is_prompt: bool,
        sequence_data: HashMap<u64, SequenceData>,
        next_token_chooser_params: NextTokenChooserParameters,
        stopping_criteria_params: StoppingCriteriaParameters,
        block_tables: HashMap<u64, Vec<u32>>,
        do_sample: bool,
        token_chunk_size: Option<usize>,
        logits_processor: Arc<RwLock<LogitsProcessor>>,
    ) -> Self {
        let token_chunk_size = if let Some(size) = token_chunk_size {
            size
        } else if is_prompt {
            sequence_data
                .values()
                .next()
                .map(|s| s.length())
                .unwrap_or(0)
        } else {
            1
        };

        Self {
            request_id,
            is_prompt,
            sequence_data,
            next_token_chooser_params,
            stopping_criteria_params,
            block_tables,
            do_sample,
            token_chunk_size,
            logits_processor,
        }
    }

    /// Getter for `request_id`
    pub fn request_id(&self) -> String {
        self.request_id.clone()
    }
}

impl std::fmt::Debug for SequenceGroupMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequenceGroupMetadata")
            .field("request_id", &self.request_id)
            .field("is_prompt", &self.is_prompt)
            .field("next_token_chooser_params", &self.next_token_chooser_params)
            .field("stopping_criteria_params", &self.stopping_criteria_params)
            .field("block_tables", &self.block_tables)
            .field("do_sample", &self.do_sample)
            .field("token_chunk_size", &self.token_chunk_size)
            .field("sequence_data", &self.sequence_data)
            .finish()
    }
}

/// Represents the output of a language model for a single sequence step.
///
/// This struct encapsulates the information produced by the model for a single token
/// generation step, including the generated token, its log probabilities, and metadata
/// for beam search and stopping conditions.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceOutput {
    /// The ID of the parent sequence. Used for tracking lineage in beam search.
    pub parent_sequence_id: u64,
    /// The ID of the token generated in this step.
    pub output_token: u32,
    /// A mapping of token IDs to their log probabilities.
    ///
    /// For each token ID, this map contains LogP(x_i+1 | x_0, ..., x_i),
    /// where x_i+1 is the potential next token and x_0, ..., x_i is the current sequence.
    pub logprob: HashMap<u32, LogProb>,
    /// Indicates whether this token is a stop token, signaling the end of generation.
    pub is_stop_token: bool,
}

/// Metrics for token generation in a sequence group.
///
/// This struct captures performance metrics related to the token generation process
/// for a group of sequences processed together.
#[derive(Clone, Debug, Default)]
pub struct SequenceGroupMetrics {
    /// Time taken to generate the batched output, in seconds.
    ///
    /// This field represents the total time spent generating tokens for all sequences
    /// in the group during a single batch processing step.
    ///
    /// `None` if the time measurement is not available or hasn't been set.
    pub time_to_generate: Option<f32>,
    /// Number of tokens generated in the current batch.
    ///
    /// This field represents the total count of new tokens produced across all
    /// sequences in the group during a single batch processing step.
    pub num_tokens_generated: usize,
}

/// Represents the output for a group of sequences after a single generation step.
///
/// This struct encapsulates the results of token generation for multiple sequences
/// processed together, typically in a batched inference operation. It includes
/// individual sequence outputs, optional tensor data, and performance metrics.
#[derive(Debug, Default)]
pub struct SequenceGroupOutput {
    /// Mapping of sequence IDs to their corresponding output for this generation step.
    pub outputs: HashMap<u64, SequenceOutput>,
    /// Optional tensor of probabilities for the sampled tokens.
    /// Shape: [num_sequences, vocab_size]
    pub sampled_token_probs: Option<Tensor>,
    /// Optional tensor of log probabilities for all tokens in the vocabulary.
    /// Shape: [num_sequences, vocab_size]
    pub logprobs: Option<Tensor>,
    /// Optional tensor of sampled token IDs.
    /// Shape: [num_sequences]
    pub sampled_token_ids: Option<Tensor>,
    /// Optional metrics from speculative decoding, if applicable.
    pub spec_decode_worker_metrics: Option<SpecDecodeWorkerMetrics>,
    /// Performance metrics for this generation step across all sequences in the group.
    pub sequence_group_metrics: SequenceGroupMetrics,
}

impl SequenceGroupOutput {
    // Creates an empty instance of `Self`
    pub fn empty() -> Self {
        Self {
            ..Default::default()
        }
    }

    /// Checks if the current instance is empty
    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
            && self.sampled_token_ids.is_none()
            && self.logprobs.is_none()
            && self.sampled_token_ids.is_none()
            && self.spec_decode_worker_metrics.is_none()
    }
}

#[derive(Debug)]
/// `SpecDecoderWorkerMetrics`
pub struct SpecDecodeWorkerMetrics {
    /// The empirical acceptance rate of the proposal method on a per-token basis.
    /// This is useful for evaluating how well the proposal method aligns with the
    /// scoring method.
    pub draft_acceptance_rate: f32,
    /// The empirical efficiency, measured as the number of tokens emitted by the
    /// system divided by the number of tokens that could be emitted by the system
    /// if the proposal method were perfect.
    pub system_efficiency: f32,
    /// The number of speculative tokens produced by the proposal method.
    pub draft_tokens: i32,
    /// The number of tokens emitted by the entire system.
    pub emitted_tokens: i32,
    /// The number of tokens accepted by the scoring model and verification
    /// routine, e.g. Llama2-70B and lossless rejection sampling.
    ///
    /// NOTE: Any token accepted by the verification routine is considered
    /// accepted (regardless of if the speculative prefix is also accepted). The
    /// user will usually see less accepted tokens. This metric is helpful when
    /// evaluating alignment of the proposal method with the scoring model.
    pub accepted_tokens: i32,
    /// The number of speculative tokens per sequence.
    pub num_spec_tokens: i32,
}

/// `ExecuteModelRequest` - The model execution request
#[derive(Clone, Debug)]
pub struct ExecuteModelRequest {
    /// The sequence groups metadata vector
    pub sequence_groups_metadata: Vec<Arc<SequenceGroupMetadata>>,
    /// Blocks to swap in. List of CPU -> GPU block number
    pub blocks_to_swap_in: HashMap<u32, u32>,
    /// Blocks to swap out. List of GPU -> CPU block number
    pub blocks_to_swap_out: HashMap<u32, u32>,
    /// Blocks to copy. Source to dest block
    pub blocks_to_copy: HashMap<u32, u32>,
    /// The number of requests in the running queue
    pub running_queue_size: usize,
}

impl ExecuteModelRequest {
    /// Constructor
    pub fn new(
        sequence_groups_metadata: Vec<Arc<SequenceGroupMetadata>>,
        blocks_to_swap_in: HashMap<u32, u32>,
        blocks_to_swap_out: HashMap<u32, u32>,
        blocks_to_copy: HashMap<u32, u32>,
        running_queue_size: usize,
    ) -> Self {
        Self {
            sequence_groups_metadata,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            running_queue_size,
        }
    }

    /// Creates a new empty instance. This is useful
    /// to communicate to the `ModelExecutor` service
    /// when there is no scheduled running sequences,
    /// and therefore we should just wait until
    /// new requests arrive
    pub fn empty() -> Self {
        Self {
            sequence_groups_metadata: vec![],
            blocks_to_copy: HashMap::default(),
            blocks_to_swap_in: HashMap::default(),
            blocks_to_swap_out: HashMap::default(),
            running_queue_size: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.sequence_groups_metadata.is_empty()
            && self.blocks_to_copy.is_empty()
            && self.blocks_to_swap_in.is_empty()
            && self.blocks_to_swap_out.is_empty()
            && self.running_queue_size == 0
    }
}

#[derive(Debug, Error)]
pub enum SequenceError {
    #[error("Invalid number of newly generated tokens for sequence")]
    InvalidNumberGeneratedTokens,
    #[error("Invalid last token generation while in prefix phase")]
    WhileInPrefix,
    #[error("Constructor error: `{0}`")]
    ConstructorError(String),
    #[error("Poison error: `{0}`")]
    PoisonError(String),
    #[error("Block error: `{0}`")]
    BlockError(#[from] BlockError),
    #[error("Missing sequence with id = `{0}`")]
    MissingSequence(u64),
    #[error("Failed to acquire read lock: `{0}`")]
    ReadLockError(String),
    #[error("Failed to acquire write lock: `{0}`")]
    WriteLockError(String),
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn sample_outputs() -> HashMap<u64, SequenceOutput> {
        (0..5_u64)
            .map(|i| {
                (
                    i,
                    SequenceOutput {
                        parent_sequence_id: 0,
                        output_token: i as u32,
                        logprob: HashMap::new(),
                        is_stop_token: false,
                    },
                )
            })
            .collect()
    }

    fn sampler_output(sample_outputs: HashMap<u64, SequenceOutput>) -> SequenceGroupOutput {
        SequenceGroupOutput {
            outputs: sample_outputs,
            sampled_token_ids: None,
            sampled_token_probs: None,
            logprobs: None,
            spec_decode_worker_metrics: None,
            sequence_group_metrics: SequenceGroupMetrics {
                time_to_generate: None,
                num_tokens_generated: 0,
            },
        }
    }

    /// Create a dummy prompt sequence and sequence group.
    pub(crate) fn create_dummy_prompt(
        request_id: u64,
        prompt_length: usize,
        block_size: Option<usize>,
        best_of: usize,
    ) -> (Arc<RwLock<Sequence>>, SequenceGroup) {
        let block_size = block_size.unwrap_or(prompt_length);

        // Create dummy prompt sequence with tokens 0...block_size-1
        // and prompt "0 ... block_size".
        let prompt_tokens: Vec<u32> = (0..(prompt_length as u32)).collect();
        let prompt_str = prompt_tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<String>>()
            .join(" ");
        let prompt = Sequence::new(request_id, prompt_str, prompt_tokens, block_size, false)
            .expect("Failed to create prompt sequence");
        let seq_group = SequenceGroup::new(
            request_id.to_string(),
            vec![prompt.clone()],
            Instant::now(),
            NextTokenChooserParameters {
                best_of,
                ..Default::default()
            },
            Default::default(),
            LogitsProcessor::new(0, None, None),
        )
        .expect("Failed to construct a new sequence group");

        let prompt = seq_group.sequences.values().next().unwrap().clone();
        (prompt, seq_group)
    }

    #[test]
    fn test_sampler_output_initialization() {
        let sample_outputs = sample_outputs();
        let sampler_output = sampler_output(sample_outputs.clone());
        assert_eq!(sampler_output.outputs.len(), sample_outputs.len());
        assert!(sampler_output.logprobs.is_none());
        assert!(sampler_output.spec_decode_worker_metrics.is_none());
        assert!(sampler_output.sampled_token_ids.is_none());
        assert!(sampler_output.sampled_token_probs.is_none());
    }

    // #[test]
    // fn test_sampler_output_eq() {
    //     let sample_outputs = sample_outputs();
    //     let sampler_output1 = SamplerOutput {
    //         outputs: sample_outputs.clone(),
    //         sampled_token_ids: None,
    //         sampled_token_probs: None,
    //         logprobs: None,
    //         spec_decode_worker_metrics: None,
    //     };
    //     let sampler_output2 = SamplerOutput {
    //         outputs: sample_outputs.clone(),
    //         sampled_token_ids: None,
    //         sampled_token_probs: None,
    //         logprobs: None,
    //         spec_decode_worker_metrics: None,
    //     };
    //     let sampler_output3 = SamplerOutput {
    //         outputs: sample_outputs[..sample_outputs.len() - 1].to_vec(),
    //         sampled_token_ids: None,
    //         sampled_token_probs: None,
    //         logprobs: None,
    //         spec_decode_worker_metrics: None,
    //     };

    //     assert_eq!(sampler_output1, sampler_output2);
    //     assert_ne!(sampler_output1, sampler_output3)
    // }

    #[test]
    fn test_sequence_data_prefill() {
        let mut sequence_data = SequenceData::new(vec![1, 2, 3, 4], vec![]);
        assert_eq!(sequence_data.get_num_uncomputed_tokens(), 4);
        assert_eq!(sequence_data.get_num_computed_tokens(), 0);

        // advance by `2`
        sequence_data
            .update_num_computed_tokens(2)
            .expect("Failed to update");
        assert_eq!(sequence_data.get_num_uncomputed_tokens(), 2);
        assert_eq!(sequence_data.get_num_computed_tokens(), 2);

        // advance by `1`
        sequence_data
            .update_num_computed_tokens(1)
            .expect("Failed to update");
        assert_eq!(sequence_data.get_num_uncomputed_tokens(), 1);
        assert_eq!(sequence_data.get_num_computed_tokens(), 3);

        // append tokens and reset, simulating recompute
        sequence_data.add_token_id(1, 0.0);
        sequence_data.reset_state_for_recompute();
        assert_eq!(sequence_data.get_num_uncomputed_tokens(), 5);
        assert_eq!(sequence_data.get_num_computed_tokens(), 0);
    }

    #[test]
    fn test_sequence_group_stage() {
        let (_, mut seq_group) = create_dummy_prompt(1, 12, None, 5);
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(6)
            .expect("Failed to update");
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(5)
            .expect("Failed to update");
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(1)
            .expect("Failed to update");
        assert!(!seq_group.is_prefill());

        let seqs = seq_group.get_seqs(None);
        assert_eq!(seqs.len(), 1);

        seq_group
            .sequences
            .values_mut()
            .enumerate()
            .for_each(|(i, v)| {
                if i == 0 {
                    v.write().unwrap().sequence_data.add_token_id(1, 0.0);
                }
            });
        seq_group
            .sequences
            .values_mut()
            .for_each(|v| v.write().unwrap().reset_state_for_recompute());
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(5)
            .expect("Failed to update");
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(7)
            .expect("Failed to update");
        assert!(seq_group.is_prefill());

        seq_group
            .update_num_computed_tokens(1)
            .expect("Failed to update");
        assert!(!seq_group.is_prefill())
    }
}
