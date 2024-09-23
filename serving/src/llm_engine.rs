use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use futures::StreamExt;
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::{
    mpsc::{error::SendError, UnboundedReceiver, UnboundedSender},
    oneshot::error::RecvError,
};
use tracing::{error, info, info_span, instrument, trace, Span};

use crate::{
    model_executor::ModelThreadDispatcher,
    policy::FcfsPolicy,
    scheduler::{Scheduler, SchedulerError, SchedulerOutputs},
    sequence::{
        ExecuteModelRequest, LogProb, RequestMetrics, Sequence, SequenceError, SequenceGroup,
        SequenceGroupMetadata, SequenceGroupOutput, SequenceOutput, SequenceStatus,
    },
    types::{ReadLock, WriteLock},
    validation::StoppingCriteriaParameters,
};

/// Time in milliseconds we wait until we schedule new received requests,
/// in case the `LlmEngine` was on halt.
const SCHEDULE_WAIT_PERIOD: u64 = 100;

/// `LlmEngine` - An asynchronous worker responsible for scheduling new requests
/// and communicating with the `ModelExecutor` service to send new requests
/// for continuously batched AI inference.
pub struct LlmEngine {
    /// Channel for sending newly generated AI outputs to Atoma's client.
    atoma_client_sender: UnboundedSender<Vec<GenerateRequestOutput>>,
    /// Dispatcher for communicating with the model executor's running thread,
    /// responsible for running prefill and decoding inference to produce AI-generated outputs.
    model_thread_dispatcher: ModelThreadDispatcher,
    /// Channel for receiving new requests from the running main `LlmService` instance.
    request_receiver: UnboundedReceiver<SequenceGroup>,
    /// Metadata of currently scheduled `SequenceGroup`s.
    sequence_groups_metadata: Vec<Arc<SequenceGroupMetadata>>,
    /// Current outputs from the scheduler.
    scheduler_outputs: SchedulerOutputs,
    /// Instance of the `Scheduler` with a First-Come-First-Serve policy.
    scheduler: Scheduler<FcfsPolicy>,
    /// Tokenizer for decoding sequences.
    tokenizer: Tokenizer,
    /// Tracing span for logging and monitoring.
    span: Span,
}

impl LlmEngine {
    /// Constructor
    pub fn new(
        atoma_client_sender: UnboundedSender<Vec<GenerateRequestOutput>>,
        model_thread_dispatcher: ModelThreadDispatcher,
        request_receiver: UnboundedReceiver<SequenceGroup>,
        scheduler: Scheduler<FcfsPolicy>,
        tokenizer: Tokenizer,
    ) -> Self {
        Self {
            atoma_client_sender,
            model_thread_dispatcher,
            sequence_groups_metadata: vec![],
            scheduler_outputs: SchedulerOutputs::create_empty(),
            scheduler,
            tokenizer,
            request_receiver,
            span: info_span!("llm-engine"),
        }
    }

    /// Main loop of the `LlmEngine`.
    ///
    /// This loop performs the following tasks:
    /// 1. Listens for incoming `SequenceGroup` requests and adds them to the `Scheduler`.
    /// 2. Waits for new outputs from the `ModelExecutor` service, processes these outputs,
    ///    updates the states of the associated `SequenceGroup`s, and re-schedules new requests.
    /// 3. Sends finished `SequenceGroup` outputs to the Atoma client service.
    ///
    /// The loop uses `tokio::select!` to concurrently handle incoming requests and model outputs.
    /// If there are no ongoing scheduled sequence groups, it waits for a short period before
    /// scheduling all received requests.
    #[instrument(skip(self))]
    pub async fn run(mut self) -> Result<(), EngineError> {
        let span = self.span.clone();
        let _enter = span.enter();

        loop {
            tokio::select! {
                Some(sequence_group) = self.request_receiver.recv() => {
                    trace!("Received new sequence group, with id = {}", sequence_group.request_id);
                    // 1. Adds the received `SequenceGroup` to the `Scheduler` instance.
                    self.scheduler.add_sequence_group(sequence_group);

                    // 2. If the current `LlmInstance` doesn't have any on-going
                    //    scheduled sequence groups, we wait some time and then
                    //    schedule all the received requests so far.
                    //    This includes the request added in 1.
                    if self.sequence_groups_metadata.is_empty() && self.scheduler_outputs.is_empty() {
                        tokio::time::sleep(Duration::from_millis(SCHEDULE_WAIT_PERIOD)).await;
                        self.step()?;
                    }
                },
                Some(outputs) = self.model_thread_dispatcher.responses.next() => {
                    self.handle_outputs(outputs.map_err(EngineError::RecvError)).await?;
                }
                else => {
                    continue;
                }
            }
        }
    }

    /// Handles newly AI generated `SequenceGroupOutput`'s.
    ///
    /// This method processes the outputs generated by the AI model, schedules new requests,
    /// and sends the finished outputs to the Atoma client service. It performs the following steps:
    /// 1. Processes the newly AI generated outputs.
    /// 2. Schedules new requests.
    /// 3. Sends the finished outputs to the Atoma client service.
    ///
    /// If an error occurs while processing the outputs, it logs the error and continues to
    /// schedule new requests to maintain the system's liveness.
    ///
    /// # Arguments
    ///
    /// * `outputs` - A `Result` containing a vector of `SequenceGroupOutput` on success,
    ///               or an `EngineError` on failure.
    ///
    /// # Returns
    ///
    /// * `Result<(), EngineError>` - Returns `Ok(())` if the outputs are handled successfully,
    ///                                or an `EngineError` if an error occurs.
    #[instrument(skip_all)]
    async fn handle_outputs(
        &mut self,
        outputs: Result<Vec<SequenceGroupOutput>, EngineError>,
    ) -> Result<(), EngineError> {
        let span = self.span.clone();
        let _enter = span.enter();

        match outputs {
            Ok(outputs) => {
                // 1. Processes the newly AI generated outputs
                let request_outputs = self.process_generated_outputs(outputs)?;

                // 2. Schedules new requests
                self.step()?;

                // 3. After scheduling new requests to the `ModelExecutor`
                //    we can send the finished outputs to the atoma client
                //    service.
                // NOTE: This is after scheduling new sequences above,
                //    we do so to optimize GPU utilization. This is
                //    supposed to be safe
                if !request_outputs.is_empty() {
                    self.atoma_client_sender.send(request_outputs)?;
                }
            }
            Err(e) => {
                error!("Invalid generated outputs with error: {e}");
                // NOTE: In order to maintain the system live, we need to keep calling
                // the `self.step()` method, even in possible scenarios of failure.
                self.step()?;
            }
        }
        Ok(())
    }

    /// Main scheduling method of `LlmEngine`.
    ///
    /// This method performs the following tasks:
    /// 1. Schedules new requests using the associated `Scheduler`.
    /// 2. Updates internal state with the new scheduling information.
    /// 3. If there are scheduled requests, creates and sends a new `ExecuteModelRequest`
    ///    to the `ModelExecutor`'s thread.
    ///
    /// # Returns
    /// - `Ok(())` if the scheduling and request sending are successful.

    #[instrument(skip_all)]
    pub fn step(&mut self) -> Result<(), EngineError> {
        let span = self.span.clone();
        let _enter = span.enter();

        trace!("`LlmEngine` new step..");
        // 1. Schedule new requests
        let (sequence_groups_metadata, scheduler_outputs) = self.scheduler.schedule()?;

        // 2. Update `self.scheduler_groups_metadata` and `scheduler_outputs`
        self.sequence_groups_metadata = sequence_groups_metadata.clone();
        self.scheduler_outputs = scheduler_outputs.clone();

        // 3. If the scheduled data is empty, it means that
        //     no new requests were received.
        if scheduler_outputs.is_empty() {
            return Ok(());
        }

        let execute_model_request = ExecuteModelRequest::new(
            sequence_groups_metadata,
            scheduler_outputs.blocks_to_swap_in,
            scheduler_outputs.blocks_to_swap_out,
            scheduler_outputs.blocks_to_copy,
            scheduler_outputs.running_queue_size,
        );

        // 4. Sends a new `ExecuteModelRequest` to the underlying `ModelExecutor`'s thread
        self.model_thread_dispatcher.send(execute_model_request);

        Ok(())
    }

    /// Processes newly generated AI outputs for sequence groups
    ///
    /// This function performs the following tasks:
    /// 1. Updates the state of each sequence in the scheduled sequence groups
    /// 2. Records metrics for sequence group processing times
    /// 3. Frees finished sequence groups
    /// 4. Collects and returns outputs for finished sequence groups
    ///
    /// # Arguments
    ///
    /// * `outputs` - A vector of `SequenceGroupOutput` containing the generated outputs
    ///
    /// # Returns
    ///
    /// * `Result<Vec<GenerateRequestOutput>, EngineError>` - A vector of `GenerateRequestOutput`
    ///   for finished sequence groups, or an error if processing fails
    #[instrument(skip_all)]
    fn process_generated_outputs(
        &mut self,
        outputs: Vec<SequenceGroupOutput>,
    ) -> Result<Vec<GenerateRequestOutput>, EngineError> {
        let now = Instant::now();

        for (output, (sequence_group_metadata, scheduled_sequence_group)) in outputs.iter().zip(
            self.sequence_groups_metadata
                .iter()
                .zip(self.scheduler_outputs.scheduled_sequence_groups.iter()),
        ) {
            // 1. Update the number of computed tokens for scheduled `SequenceGroup`
            scheduled_sequence_group
                .scheduled_group
                .update_num_computed_tokens(scheduled_sequence_group.token_chunk_size)?;

            let stopping_criteria_params =
                scheduled_sequence_group.scheduled_group.stopping_params();

            // 2. Iterate over each `Sequence`s of `ScheduledSequenceGroup` and update its current state
            // after the new LLM inference iteration has been performed
            for (sequence_id, sequence) in scheduled_sequence_group.scheduled_group.sequences.iter()
            {
                let sequence_output = if let Some(output) = output.outputs.get(sequence_id) {
                    output
                } else {
                    error!(
                        "Missing generated sequence output token for sequence with id = {}",
                        sequence_id
                    );
                    return Err(EngineError::MissingSequenceOutputToken(*sequence_id));
                };

                // 3. Updates the state of the current `Sequence`
                self.update_sequence(
                    sequence,
                    sequence_output,
                    sequence_group_metadata,
                    &stopping_criteria_params,
                )?;
            }

            // 4. Add a few metrics
            let metrics_guard = scheduled_sequence_group
                .scheduled_group
                .metrics
                .read()
                .unwrap();

            let arrival_time_histogram = metrics::histogram!("sequence-group-arrival-time");
            arrival_time_histogram.record(metrics_guard.arrival_time.elapsed().as_secs_f32());

            let last_token_time_histogram = metrics::histogram!("sequence-group-last-token-time");
            last_token_time_histogram.record(metrics_guard.last_token_time.elapsed().as_secs_f32());
        }

        // 5. Free all finished sequence groups
        self.scheduler.free_finished_sequence();

        // 6. Keep track of all the finished `SequenceGroup`s
        let mut request_outputs = Vec::new();
        for scheduled_sequence_group in self.scheduler_outputs.scheduled_sequence_groups.iter() {
            scheduled_sequence_group
                .scheduled_group
                .maybe_set_first_scheduled_time(now);

            if scheduled_sequence_group.scheduled_group.is_finished() {
                request_outputs.push(GenerateRequestOutput::from_sequence_group(
                    &scheduled_sequence_group.scheduled_group,
                ));
            }
        }
        for sequence_group in self.scheduler_outputs.ignored_seq_groups.iter() {
            sequence_group.maybe_set_first_scheduled_time(now);
        }

        Ok(request_outputs)
    }

    /// Updates the state of a `Sequence` after an LLM inference iteration
    ///
    /// This method handles both the decoding phase (when generating new tokens) and the prefill phase.
    ///
    /// # Arguments
    /// * `sequence` - The `Sequence` to update
    /// * `sequence_output` - The output from the LLM for this sequence
    /// * `sequence_group_metadata` - Metadata for the sequence group
    /// * `stopping_criteria_params` - Parameters for stopping criteria
    ///
    /// # Returns
    /// * `Result<(), EngineError>` - Ok if successful, or an error if something goes wrong
    ///
    /// # Behavior
    /// - In decoding phase (do_sample == true):
    ///   1. Updates sequence with new token, logprobs, and cumulative probability
    ///   2. Decodes and appends new token to output text
    ///   3. Checks for stopping conditions (stop token, EOS, length limits)
    ///   4. Updates sequence status if stopping condition is met
    /// - In prefill phase (do_sample == false):
    ///   1. Only updates the sequence's output log probabilities
    #[instrument(skip_all)]
    fn update_sequence(
        &self,
        sequence: &Arc<RwLock<Sequence>>,
        sequence_output: &SequenceOutput,
        sequence_group_metadata: &SequenceGroupMetadata,
        stopping_criteria_params: &StoppingCriteriaParameters,
    ) -> Result<(), EngineError> {
        let sequence_id = { sequence.read_lock()?.sequence_id() };
        // 1. Get the AI generated next output token id.
        let generated_token_id = sequence_output.output_token;
        let is_stop_token = sequence_output.is_stop_token;

        if sequence_group_metadata.do_sample {
            let mut sequence_guard_lock = sequence.write_lock()?;
            // NOTE: this means we are in decoding phase.
            // That is, we are generating new output tokens
            // and these should be added to the `Sequence`'s
            // state.

            // 2. Update the `Sequence`'s output log-probabilities.
            //
            // 3. Update the `Sequence`'s `SequenceData` cumulative probabilities,
            //    if we are in decoding phase.
            //
            // 4. Update the `Sequence`'s `SequenceData` output tokens,
            //    if we are in decoding phase.
            sequence_guard_lock
                .add_token_id(generated_token_id, sequence_output.logprob.clone())?;

            // 5. Decode the generated output token id.
            let token_ids = sequence_guard_lock.sequence_data.get_token_ids();
            let generated_text = self
                .tokenizer
                .decode(&token_ids, true)
                .map_err(|e| EngineError::TokenizerError(e.to_string()))?;

            // 6. Update the `output_text` with the newly generated token,
            //    if in decoding phase.
            let generated_token = if sequence_guard_lock.tokens.last().is_some() {
                let start = sequence_guard_lock.output_text.chars().count();
                generated_text.chars().skip(start).collect::<String>()
            } else {
                let start = sequence_guard_lock.prompt.chars().count();
                generated_text.chars().skip(start).collect()
            };

            sequence_guard_lock.output_text.push_str(&generated_token);

            // 7. Check if the last generated token is a stop token.
            //    If so, update the `Sequence`'s `SequenceState` and
            //    the `stop_reason`, as well.
            if stopping_criteria_params
                .stop_sequences
                .contains(&generated_token)
            {
                info!("Current sequence with id = {sequence_id} has finished execution due to stopping token = {generated_token}");
                {
                    sequence_guard_lock.stop_reason = Some(generated_token_id)
                }

                sequence_guard_lock.set_sequence_status(SequenceStatus::FinishedStopped)
            }

            // 8. Check if the current `Sequence` last generated token
            //    id equals to the `eos_token_id`, in which case the
            //    the `Sequence`'s status should become `FinishedStopped`.
            if is_stop_token && !stopping_criteria_params.ignore_eos_token {
                sequence_guard_lock.set_sequence_status(SequenceStatus::FinishedStopped)
            }

            // 9. Check if the `Sequence`'s length exceeds that of
            //     `SchedulerConfig`'s. If so, update the `Sequence`'s
            //     `SequenceStatus` to `FinishedLengthCapped`.
            let sequence_len = sequence_guard_lock.length();
            if sequence_len > self.scheduler.scheduler_config.max_model_len() {
                sequence_guard_lock.set_sequence_status(SequenceStatus::FinishedLengthCapped)
            }

            // 10. Check if the `Sequence`'s output length exceeds that of
            //     Request's `max_new_tokens`.
            let sequence_output_len = sequence_guard_lock.get_output_len();
            if sequence_output_len >= stopping_criteria_params.max_new_tokens as usize {
                sequence_guard_lock.set_sequence_status(SequenceStatus::FinishedLengthCapped)
            }

            // 11. Update the `Sequence`'s tokens vec.
            sequence_guard_lock.tokens.push(generated_token)
        } else {
            // NOTE: in this case, we are not sampling newly
            // generated tokens. That is, we are in prefill
            // phase (possibly while chunking)
            // without generating the next token. For this reason,
            // we do not have to add tokens to the current
            // `Sequence`'s state.

            // 2. Update the `Sequence`'s output log-probabilities.
            sequence
                .write_lock()?
                .output_logprobs
                .push(sequence_output.logprob.clone());
        }

        Ok(())
    }
}

/// `RequestOutput` - Output of running AI inference over a `SequenceGroup`
#[derive(Debug)]
pub struct GenerateRequestOutput {
    /// Request id
    pub request_id: String,
    /// The `String` prompt
    pub prompt: String,
    /// Inference outputs
    pub inference_outputs: Vec<InferenceOutput>,
    /// Prompt token ids
    pub prompt_token_ids: Vec<u32>,
    /// Is finished
    pub is_finished: bool,
    /// Metrics
    pub metrics: Arc<RwLock<RequestMetrics>>,
}

impl GenerateRequestOutput {
    /// Creates a new `Self` instance from a `SequenceGroup`
    pub fn from_sequence_group(sequence_group: &SequenceGroup) -> Self {
        debug!(
            "Creating `GenerateRequestOutput` from sequence group with id = {}",
            sequence_group.request_id
        );
        let mut sequences = sequence_group.sequences.values().collect::<Vec<_>>();

        let top_n_sequences = if sequences.len() == 1 {
            sequences
        } else {
            // Get top n sequences
            let n = sequence_group.next_token_chooser_params().n;
            sequences.sort_by(|s1, s2| {
                s1.read()
                    .unwrap()
                    .cumulative_logprob()
                    .partial_cmp(&s2.read().unwrap().cumulative_logprob())
                    .unwrap()
            });
            sequences[..n].to_vec()
        };

        let inference_outputs = top_n_sequences
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let s = s.read().unwrap();
                InferenceOutput {
                    index: i,
                    output_text: s.get_output_text(),
                    token_ids: s.get_token_ids(),
                    cumulative_logprob: s.cumulative_logprob(),
                    logprobs: s.output_logprobs.clone(),
                    finish_reason: s.get_sequence_status().finished_reason(),
                    stop_reason: s.stop_reason,
                }
            })
            .collect::<Vec<_>>();

        let is_finished = sequence_group.is_finished();
        if is_finished {
            sequence_group.set_finished_time(Instant::now());
        }
        Self {
            request_id: sequence_group.request_id.clone(),
            inference_outputs,
            prompt: sequence_group.prompt(),
            prompt_token_ids: sequence_group.prompt_token_ids(),
            is_finished,
            metrics: sequence_group.metrics.clone(),
        }
    }
}

/// `InferenceOutput` - Output of running AI inference on a given sequence group
#[derive(Clone, Debug)]
pub struct InferenceOutput {
    /// The index of the output in the request
    pub index: usize,
    /// The generated output text
    pub output_text: String,
    /// The token ids of the generated output text
    pub token_ids: Vec<u32>,
    /// The cumulative log probability of the generated
    /// output text
    pub cumulative_logprob: f32,
    /// The log probabilities of the top probability words at each
    /// position if the logprobs are requested
    pub logprobs: Vec<HashMap<u32, LogProb>>,
    /// The reason why the sequence is finished
    pub finish_reason: Option<String>,
    /// The stop token id that caused the completion
    /// to stop, None if the completion finished for some other reason
    /// including encountering the eos token
    pub stop_reason: Option<u32>,
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Scheduler error: `{0}`")]
    SchedulerError(#[from] SchedulerError),
    #[error("Sequence error: `{0}`")]
    SequenceError(#[from] SequenceError),
    #[error("Missing sequence output token, id = `{0}`")]
    MissingSequenceOutputToken(u64),
    #[error("Tokenizer error: `{0}`")]
    TokenizerError(String),
    #[error("Send error: `{0}`")]
    SendError(#[from] SendError<Vec<GenerateRequestOutput>>),
    #[error("Recv error: `{0}`")]
    RecvError(#[from] RecvError),
}
