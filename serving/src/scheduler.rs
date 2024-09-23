use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Debug,
    marker::PhantomData,
    sync::Arc,
    time::Instant,
};

use crate::{
    block_manager::{AllocationStatus, BlockSpaceManager, BlockSpaceManagerError},
    config::{CacheConfig, SchedulerConfig},
    policy::Policy,
    sequence::{SequenceData, SequenceError, SequenceGroup, SequenceGroupMetadata, SequenceStatus},
    types::{ReadLock, WriteLock},
};
use thiserror::Error;
use tracing::{debug, error, info, info_span, instrument, trace, warn, Span};

/// Preemption modes.
///
/// 1. `Swapping`: Swap out the blocks of the preempted sequences to CPU memory
///     and swap them back in when the sequences are resumed.
/// 2. `Recomputation`: Discard the blocks of the preempted sequences and
///     recompute them when the sequences are resumed, treating the sequences as
///     new prompts.
#[derive(Debug, PartialEq, Eq)]
pub enum PreemptionMode {
    Swap,
    Recomputation,
}

/// `SchedulingBudget` - The available slots for scheduling.
///
/// TODO: Right now, the budget is request_id-aware meaning it can ignore
///  budget update from the same request_id. It is because in normal scheduling
///  path, we update `Running` num_seqs ahead of time, meaning it could be
///  updated more than once when scheduling `Running` requests. Since this won't
///  happen if we only have chunked prefill scheduling, we can remove this
///  feature from the API when chunked prefill is enabled by default.
#[derive(Debug)]
struct SchedulingBudget {
    /// Maximum number of tokens that can be scheduled
    pub token_budget: usize,
    /// Maximum number of sequences that can be scheduled
    pub max_num_sequences: usize,
    /// Set of request IDs that have updated num_batched_tokens.
    request_ids_num_batched_tokens: HashSet<String>,
    /// Set of request IDs that have updated num_curr_seqs.
    request_ids_num_curr_seqs: HashSet<String>,
    /// Number of batched tokens currently used.
    num_batched_tokens: usize,
    /// Number of current scheduled sequences.
    num_curr_seqs: usize,
    /// Tracing span
    pub span: Span,
}

impl SchedulingBudget {
    /// Creates a new `SchedulingBudget` with the specified token budget and maximum number of sequences.
    pub fn new(token_budget: usize, max_num_sequences: usize) -> Self {
        Self {
            token_budget,
            max_num_sequences,
            request_ids_num_batched_tokens: HashSet::new(),
            request_ids_num_curr_seqs: HashSet::new(),
            num_batched_tokens: 0,
            num_curr_seqs: 0,
            span: info_span!("scheduling-budget"),
        }
    }

    /// Checks if it is possible to schedule number of tokens
    #[instrument(skip_all)]
    pub fn can_schedule(
        &self,
        num_new_tokens: usize,
        num_new_sequences: usize,
    ) -> Result<bool, SchedulerError> {
        let _enter = self.span.enter();
        if num_new_sequences == 0 || num_new_tokens == 0 {
            error!("Empty scheduling, either `num_new_sequences` == 0 or `num_new_tokens` == 0");
            return Err(SchedulerError::EmptyScheduling);
        }

        Ok(
            (self.num_batched_tokens + num_new_tokens <= self.token_budget)
                && (self.num_curr_seqs + num_new_sequences <= self.max_num_sequences),
        )
    }

    /// Computes the remaining number of budget tokens
    pub fn remaining_budget_tokens(&self) -> usize {
        self.token_budget - self.num_batched_tokens
    }

    /// Adds number of batched tokens
    #[instrument(skip_all)]
    pub fn add_num_batched_tokens(&mut self, request_id: String, num_batched_tokens: usize) {
        let _enter = self.span.enter();
        trace!("Adding number of batched tokens");
        // If request has already been batched, simply return
        if self.request_ids_num_batched_tokens.contains(&request_id) {
            return;
        }

        self.request_ids_num_batched_tokens.insert(request_id);
        self.num_batched_tokens += num_batched_tokens;
    }

    /// Subtracts number of batched tokens
    #[instrument(skip_all)]
    pub fn subtract_num_batched_tokens(&mut self, request_id: &str, num_batched_tokens: usize) {
        let _enter = self.span.enter();
        trace!("Subtracting number of batched tokens..");
        // Only performs an action, if request with `request_id` has been already batched
        if self.request_ids_num_batched_tokens.contains(request_id) {
            self.request_ids_num_batched_tokens.remove(request_id);
            self.num_batched_tokens -= num_batched_tokens;
        }
    }

    /// Adds number sequences
    #[instrument(skip_all)]
    pub fn add_number_sequences(&mut self, request_id: String, num_current_sequences: usize) {
        let _enter = self.span.enter();
        trace!("Adding number of sequences..");
        // If request has already been added, simply return
        if self.request_ids_num_curr_seqs.contains(&request_id) {
            return;
        }

        self.request_ids_num_curr_seqs.insert(request_id);
        self.num_curr_seqs += num_current_sequences;
    }

    /// Subtracts number sequences
    #[instrument(skip_all)]
    pub fn subtracts_number_sequences(&mut self, request_id: &str, num_current_sequences: usize) {
        let _enter = self.span.enter();
        trace!("Subtracting number of sequences..");
        // Only performs an action, if request with `request_id` has been already added
        if self.request_ids_num_curr_seqs.contains(request_id) {
            self.request_ids_num_curr_seqs.remove(request_id);
            self.num_curr_seqs -= num_current_sequences;
        }
    }

    /// Number of batched tokens
    pub fn num_batched_tokens(&self) -> usize {
        self.num_batched_tokens
    }

    /// Number of current sequences
    pub fn num_current_sequences(&self) -> usize {
        self.num_curr_seqs
    }
}

/// `SchedulerRunningOutputs` - The requests that are scheduled from a running queue.
///
/// Could contain prefill (prefill that's chunked) or decodes. If there's not
/// enough memory, it can be preempted (for recompute) or swapped out.
pub struct SchedulerRunningOutputs {
    // Selected sequences that are running and in a decoding phase.
    decode_seq_groups: Vec<ScheduledSequenceGroup>,
    // Selected sequences that are running and in a prefill phase.
    // i.e., it means the prefill has been chunked.
    prefill_seq_groups: Vec<ScheduledSequenceGroup>,
    // The preempted sequences.
    preempted: Vec<SequenceGroup>,
    // Sequences that are swapped out.
    swapped_out: Vec<SequenceGroup>,
    // The blocks to swap out.
    blocks_to_swap_out: HashMap<u32, u32>,
    // The blocks to copy.
    blocks_to_copy: HashMap<u32, u32>,
}

impl SchedulerRunningOutputs {
    /// Create an empty `Self` instance
    fn create_empty() -> Self {
        Self {
            decode_seq_groups: vec![],
            prefill_seq_groups: vec![],
            preempted: vec![],
            swapped_out: vec![],
            blocks_to_swap_out: HashMap::new(),
            blocks_to_copy: HashMap::new(),
        }
    }
}

/// The requests that are scheduled from a swap queue.
///
/// Could contain prefill (prefill that's chunked) or decodes.
pub struct SchedulerSwappedInOutputs {
    /// Selected sequences that are going to be swapped in and is in a decoding phase.
    decode_seq_groups: Vec<ScheduledSequenceGroup>,
    /// Selected sequences that are going to be swapped in and in a prefill
    /// phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: Vec<ScheduledSequenceGroup>,
    /// The blocks to swap in.
    blocks_to_swap_in: HashMap<u32, u32>,
    /// The blocks to copy.
    blocks_to_copy: HashMap<u32, u32>,
    /// Infeasible sequence groups.
    infeasible_seq_groups: Vec<SequenceGroup>,
}

impl SchedulerSwappedInOutputs {
    /// Create an empty `Self` instance
    fn create_empty() -> Self {
        Self {
            decode_seq_groups: vec![],
            prefill_seq_groups: vec![],
            blocks_to_swap_in: HashMap::new(),
            blocks_to_copy: HashMap::new(),
            infeasible_seq_groups: vec![],
        }
    }
}

/// `SchedulerPrefillOutputs` - The requests that are scheduled from a waiting queue.
///
/// Could contain a fresh prefill requests or preempted requests that need
/// to be recomputed from scratch.
#[derive(Debug)]
pub struct SchedulerPrefillOutputs {
    /// Selected sequences for prefill
    sequence_groups: Vec<ScheduledSequenceGroup>,
    /// Ignored sequence groups.
    ignored_sequence_groups: Vec<SequenceGroup>,
}

impl SchedulerPrefillOutputs {
    /// Create an `empty` `Self` instance
    fn create_empty() -> Self {
        Self {
            sequence_groups: vec![],
            ignored_sequence_groups: vec![],
        }
    }
}

/// `SchedulerOutputs` - The scheduling decision made from a scheduler.
#[derive(Clone, Debug)]
pub struct SchedulerOutputs {
    /// Scheduled sequence groups.
    pub scheduled_sequence_groups: Vec<ScheduledSequenceGroup>,
    /// Number of prefill groups scheduled.
    number_prefill_groups: usize,
    /// Total number of batched tokens.
    num_batched_tokens: usize,
    /// Blocks to swap in. List of CPU -> GPU block number.
    pub blocks_to_swap_in: HashMap<u32, u32>,
    /// Blocks to swap out. List of GPU -> CPU block number.
    pub blocks_to_swap_out: HashMap<u32, u32>,
    /// Blocks to copy. Source to dest block.
    pub blocks_to_copy: HashMap<u32, u32>,
    /// Ignored sequence groups
    pub ignored_seq_groups: Vec<SequenceGroup>,
    /// The number of requests in the running queue
    pub running_queue_size: usize,
    /// Number of preempted sequence groups
    preempted: usize,
    /// Tracing span
    span: Span,
}

impl SchedulerOutputs {
    /// Validate that `SchedulerOutputs` is well formed
    #[instrument(skip_all)]
    fn validate(&self) -> Result<(), SchedulerError> {
        let _enter = self.span.enter();
        if !self.blocks_to_swap_in.is_empty() && !self.blocks_to_swap_out.is_empty() {
            error!("Swap in and swap out should never happen at the same time.");
            return Err(SchedulerError::InvalidSchedulerOutput(
                "Swap in and swap out should never happen at the same time.".into(),
            ));
        }
        Ok(())
    }

    /// Creates a new empty instance
    pub fn create_empty() -> Self {
        Self {
            scheduled_sequence_groups: vec![],
            num_batched_tokens: 0,
            number_prefill_groups: 0,
            blocks_to_copy: HashMap::default(),
            blocks_to_swap_in: HashMap::default(),
            blocks_to_swap_out: HashMap::default(),
            ignored_seq_groups: vec![],
            running_queue_size: 0,
            preempted: 0,
            span: info_span!("scheduler-output"),
        }
    }

    /// Checks if the current instance is empty
    pub fn is_empty(&self) -> bool {
        self.scheduled_sequence_groups.is_empty()
            && self.blocks_to_swap_in.is_empty()
            && self.blocks_to_swap_out.is_empty()
            && self.blocks_to_copy.is_empty()
    }
}

/// `Scheduler` - Responsible for managing the scheduling and execution of inference requests
///
/// The Scheduler handles the lifecycle of multiple `SequenceGroup` requests, including:
/// - Queueing new requests
/// - Allocating GPU/CPU memory resources
/// - Scheduling prefill (initial prompt processing) and decoding steps
/// - Managing preemption and swapping of sequences between GPU and CPU
/// - Optimizing throughput and latency based on configured policies
///
/// It relies on the `BlockSpaceManager` to efficiently allocate and manage GPU/CPU memory blocks.
#[derive(Debug)]
pub struct Scheduler<P> {
    /// Cache configuration
    pub(crate) cache_config: CacheConfig,
    /// `Scheduler` configuration
    pub(crate) scheduler_config: SchedulerConfig,
    /// `BlockSpaceManager` to handle block resources efficiently
    block_manager: BlockSpaceManager,
    /// Queue of SequenceGroups waiting to be scheduled
    waiting: VecDeque<SequenceGroup>,
    /// Queue of SequenceGroups currently executing on the GPU
    running: VecDeque<SequenceGroup>,
    /// Queue of SequenceGroups that have been swapped out to CPU memory
    swapped: VecDeque<SequenceGroup>,
    /// Time at previous scheduling step
    previous_time: Instant,
    /// Tracks if a prompt was scheduled in the previous step, used for latency calculations
    previous_prompt: bool,
    /// Duration of the last prompt processing, used for scheduling heuristics
    last_prompt_latency: f32,
    /// Total number of times sequences have been preempted, used for logging/monitoring
    num_cumulative_preemption: usize,
    /// Generic parameter for the scheduling policy
    _phantom: PhantomData<P>,
    /// Tracing span
    span: Span,
}

impl<P> Scheduler<P> {
    /// Constructor
    pub fn new(
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, SchedulerError> {
        Ok(Self {
            block_manager: BlockSpaceManager::new(
                cache_config.block_size(),
                cache_config.num_cpu_blocks(),
                cache_config.num_gpu_blocks(),
                cache_config.sliding_window(),
            )?,
            cache_config,
            scheduler_config,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            swapped: VecDeque::new(),
            previous_time: Instant::now(),
            previous_prompt: false,
            last_prompt_latency: 0.0,
            num_cumulative_preemption: 0,
            span: info_span!("scheduler"),
            _phantom: PhantomData,
        })
    }

    /// Aborts a sequence group with the given ID.
    ///
    /// This method searches for the sequence group with the specified ID in all state queues
    /// (waiting, running, and swapped). If found:
    ///
    /// 1. It removes the sequence group from its current queue.
    /// 2. For any unfinished sequences in the group, it:
    ///    a. Sets their status to `FinishedAborted`.
    ///    b. Frees the associated resources.
    ///
    /// If no matching sequence group is found, this method does nothing.
    ///
    /// # Arguments
    ///
    /// * `request_id` - The ID of the sequence group to abort.
    ///
    /// # Returns
    ///
    /// A `Result` which is:
    /// - `Ok(())` if the operation was successful (even if no matching group was found).
    /// - `Err(SchedulerError)` if an error occurred during the process.
    ///
    /// # Errors
    ///
    /// This method can return a `SchedulerError` if there are issues with:
    /// - Acquiring locks on sequences
    /// - Freeing sequence resources
    #[instrument(skip_all)]
    pub fn abort_sequence_group(&mut self, request_id: String) -> Result<(), SchedulerError> {
        let _enter = self.span.enter();
        debug!("Aborting sequence group..");

        let mut queue_identifier = 'w';
        let waiting_length = self.waiting.len();
        let running_length = self.running.len();

        let mut sequence_ids_to_free = vec![];
        let mut index = 0;
        for sequence_group in self
            .waiting
            .iter()
            .chain(self.running.iter().chain(self.swapped.iter()))
        {
            if sequence_group.request_id == request_id {
                for sequence in sequence_group.sequences.values() {
                    let mut sequence_guard_lock = sequence.write_lock()?;
                    let (sequence_id, is_finished) = (
                        sequence_guard_lock.sequence_id(),
                        sequence_guard_lock.is_finished(),
                    );
                    debug!("Sequence ID: {}, is finished: {}", sequence_id, is_finished);
                    if is_finished {
                        continue;
                    }
                    sequence_guard_lock.set_sequence_status(SequenceStatus::FinishedAborted);
                    sequence_ids_to_free.push(sequence_id);
                }

                break;
            }
            index += 1;
        }

        for sequence_id in sequence_ids_to_free {
            self.free_sequence(sequence_id)?;
        }

        if index >= waiting_length && index < waiting_length + running_length {
            queue_identifier = 'r';
        } else if index > waiting_length + running_length {
            queue_identifier = 's';
        }

        if queue_identifier == 'w' {
            self.waiting.retain(|s| s.request_id != request_id);
        } else if queue_identifier == 'r' {
            self.running.retain(|s| s.request_id != request_id);
        } else {
            self.swapped.retain(|s| s.request_id == request_id);
        }

        Ok(())
    }

    /// Aborts multiple sequence groups at once.
    ///
    /// This method iterates through the provided request IDs and aborts each corresponding
    /// sequence group. It's a convenient way to abort multiple sequence groups in a single call.
    ///
    /// # Arguments
    ///
    /// * `request_ids` - An iterator of String values, where each string is a request ID
    ///                   corresponding to a sequence group to be aborted.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if all sequence groups were successfully aborted, or an error
    /// if any abortion failed.
    ///
    /// # Errors
    ///
    /// This method can return a `SchedulerError` if there are issues with:
    /// - Acquiring locks on sequences
    /// - Freeing sequence resources
    /// - Any other error that might occur during the abortion of individual sequence groups
    ///
    /// # Example
    ///
    /// ```
    /// let request_ids = vec!["req1".to_string(), "req2".to_string()];
    /// scheduler.abort_sequence_groups(request_ids.into_iter())?;
    /// ```
    pub fn abort_sequence_groups(
        &mut self,
        request_ids: impl Iterator<Item = String>,
    ) -> Result<(), SchedulerError> {
        let _enter = self.span.enter();
        for request_id in request_ids {
            debug!("Aborting sequence group: {}", request_id);
            self.abort_sequence_group(request_id)?;
        }

        Ok(())
    }

    /// Frees blocks associated with sequences in a given `SequenceGroup` and removes the group from its current queue.
    ///
    /// # Arguments
    ///
    /// * `request_id` - The ID of the `SequenceGroup` to free.
    /// * `sequences_ids` - A slice of sequence IDs within the group to free blocks for.
    /// * `sequence_status` - The current status of the `SequenceGroup`, determining which queue to remove it from.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, or a `SchedulerError` if there was an issue freeing the blocks.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - There is an issue freeing blocks in the `BlockManager`
    /// - An invalid `SequenceStatus` is provided
    /// Frees blocks associated with sequences in a given `SequenceGroup` and removes the group from its current queue.
    ///
    /// # Arguments
    ///
    /// * `request_id` - The ID of the `SequenceGroup` to free.
    /// * `sequences_ids` - A slice of sequence IDs within the group to free blocks for.
    /// * `sequence_status` - The current status of the `SequenceGroup`, determining which queue to remove it from.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, or a `SchedulerError` if there was an issue freeing the blocks.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - There is an issue freeing blocks in the `BlockManager`
    /// - An invalid `SequenceStatus` is provided
    #[allow(dead_code)]
    fn free_sequences(
        &mut self,
        request_id: String,
        sequences_ids: &[u64],
        sequence_status: SequenceStatus,
    ) -> Result<(), SchedulerError> {
        for sequence_id in sequences_ids {
            self.block_manager.free(*sequence_id)?
        }

        if sequence_status == SequenceStatus::Waiting {
            self.waiting.retain(|s| s.request_id != request_id);
        } else if sequence_status == SequenceStatus::Running {
            self.running.retain(|s| s.request_id != request_id);
        } else if sequence_status == SequenceStatus::Swapped {
            self.swapped.retain(|s| s.request_id != request_id);
        } else {
            unreachable!("Sequence status should only be one of values [Waiting, Running, Swapped]")
        }

        Ok(())
    }

    /// Frees blocks associated with a given sequence.
    ///
    /// This method releases the memory blocks allocated to a specific sequence,
    /// making them available for reuse by other sequences.
    ///
    /// # Arguments
    ///
    /// * `sequence_id` - The unique identifier of the sequence to free.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the operation was successful, or a `SchedulerError`
    /// if there was an issue freeing the blocks.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - There is an issue freeing blocks in the `BlockManager`
    /// - The specified `sequence_id` does not exist or is invalid
    fn free_sequence(&mut self, sequence_id: u64) -> Result<(), SchedulerError> {
        Ok(self.block_manager.free(sequence_id)?)
    }

    /// Returns the total number of unfinished sequences across all queues.
    ///
    /// This method counts the sequences in the waiting, running, and swapped queues.
    ///
    /// # Returns
    ///
    /// The total number of unfinished sequences.
    pub fn num_unfinished_sequeces(&self) -> usize {
        self.waiting.len() + self.running.len() + self.swapped.len()
    }
}

impl<P: Policy> Scheduler<P> {
    /// Checks if there are any unfinished sequences in the scheduler.
    ///
    /// This method returns true if any of the scheduler's queues (waiting, running, or swapped)
    /// contain sequence groups. It provides a quick way to determine if there is still work
    /// to be done by the scheduler.
    ///
    /// # Returns
    ///
    /// `true` if there are any unfinished sequences, `false` otherwise.
    pub fn has_unfinished_sequences(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty() || !self.swapped.is_empty()
    }

    /// Schedules sequence groups that are currently running.
    ///
    /// This method processes the running queue, which includes both decode and chunked prefill requests.
    /// It handles scheduling, preemption, and swapping of sequence groups based on available resources.
    ///
    /// # Arguments
    ///
    /// * `running_queue` - A queue containing running requests (e.g., decodes). This argument is not modified in-place.
    /// * `budget` - The scheduling budget, which is updated in-place when decodes are preempted.
    /// * `enable_chunking` - If true, allows sequence groups to be chunked. Only a portion of tokens will be scheduled
    ///                       if the budget's `num_batched_tokens` doesn't have enough capacity for all tokens.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing:
    /// - The remaining running queue (should always be empty after scheduling)
    /// - `SchedulerRunningOutputs` containing the scheduling results
    ///
    /// # Errors
    ///
    /// Returns a `SchedulerError` if any scheduling operations fail.
    ///
    /// # Implementation Details
    ///
    /// - Sorts the running queue by priority
    /// - Processes each sequence group:
    ///   - Attempts to append slots and schedule for execution
    ///   - If resources are insufficient, performs preemption or swapping
    /// - Handles both prefill and decoding computations
    /// - Updates the scheduling budget accordingly
    #[instrument(skip_all)]
    fn schedule_running(
        &mut self,
        running_queue: VecDeque<SequenceGroup>,
        budget: &mut SchedulingBudget,
        enable_chunking: bool,
    ) -> Result<(VecDeque<SequenceGroup>, SchedulerRunningOutputs), SchedulerError> {
        let _enter = self.span.enter();
        trace!("Schedule running..");
        // Blocks that need to be swapped or copied before model execution
        let mut blocks_to_swap_out = HashMap::<u32, u32>::new();
        let mut blocks_to_copy = HashMap::<u32, u32>::new();

        let mut decode_seq_groups = Vec::<ScheduledSequenceGroup>::new();
        let mut prefill_seq_groups = Vec::<ScheduledSequenceGroup>::new();
        let mut preempted = Vec::<SequenceGroup>::new();
        let mut swapped_out = Vec::<SequenceGroup>::new();

        // Preemption happens only when there is no available slot
        // to keep all sequences groups in `Running` state.
        // In this case, the policy is responsible for deciding which sequence
        // groups should preempt next
        let now = Instant::now();
        let mut running_queue = P::sort_by_priority(now, &running_queue);

        while let Some(mut sequence_group) = running_queue.pop_front() {
            let num_running_tokens = self.get_num_tokens(
                &sequence_group,
                SequenceStatus::Running,
                enable_chunking,
                budget,
            )?;

            // if no tokens are being processed, we break the loop
            if num_running_tokens == 0 {
                // Need to push the popped sequence group back to the front
                running_queue.push_front(sequence_group);
                break;
            }

            loop {
                if !self.can_append_slots(&sequence_group) {
                    budget.subtract_num_batched_tokens(
                        &sequence_group.request_id,
                        num_running_tokens,
                    );
                    let num_running_sequences = sequence_group.get_max_num_running_seqs();
                    budget.subtracts_number_sequences(
                        &sequence_group.request_id,
                        num_running_sequences,
                    );

                    if let Some(mut victim_sequence_group) = running_queue.pop_back() {
                        // Preempt the lowest-priority sequence groups first
                        // victim lies at the end of `runnning_queue`, as it is was last in, last out
                        let preempted_mode = self.preempt(
                            &mut victim_sequence_group,
                            &mut blocks_to_swap_out,
                            None,
                        )?;
                        if preempted_mode == PreemptionMode::Recomputation {
                            preempted.push(victim_sequence_group);
                        } else {
                            swapped_out.push(victim_sequence_group);
                        }
                    } else {
                        // No other sequence groups can be preempted.
                        // Preempt the current `SequenceGroup`
                        let preempted_mode =
                            self.preempt(&mut sequence_group, &mut blocks_to_swap_out, None)?;

                        if preempted_mode == PreemptionMode::Recomputation {
                            preempted.push(sequence_group.clone());
                        } else {
                            swapped_out.push(sequence_group.clone());
                        }

                        // As no other sequence groups can be preempted, we stop the loop
                        break;
                    }
                } else {
                    self.append_slots(&sequence_group, &mut blocks_to_copy)?;
                    let is_prefill = sequence_group.is_prefill();
                    if is_prefill {
                        // Prefill computation
                        prefill_seq_groups.push(ScheduledSequenceGroup {
                            scheduled_group: sequence_group.clone(),
                            token_chunk_size: num_running_tokens,
                        });
                    } else {
                        // Decoding computation (only decodes 1 token at a time)
                        decode_seq_groups.push(ScheduledSequenceGroup {
                            scheduled_group: sequence_group.clone(),
                            token_chunk_size: 1,
                        });
                    }
                    budget.add_num_batched_tokens(
                        sequence_group.request_id.clone(),
                        num_running_tokens,
                    );

                    // OPTIMIZATION: Note that `get_max_num_running_seqs` is
                    // expensive. For the default scheduling chase where
                    // `enable_chunking` is false, `num_seqs` are updated before running
                    // this method, so we don't have to update it again here.
                    if enable_chunking {
                        let num_running_seqs = sequence_group.get_max_num_running_seqs();
                        budget.add_number_sequences(
                            sequence_group.request_id.clone(),
                            num_running_seqs,
                        )
                    }
                    break;
                }
            }
        }

        let scheduler_running_outputs = SchedulerRunningOutputs {
            decode_seq_groups,
            prefill_seq_groups,
            preempted,
            swapped_out,
            blocks_to_swap_out,
            blocks_to_copy,
        };

        Ok((running_queue, scheduler_running_outputs))
    }

    /// Schedules sequence groups that were previously swapped out to CPU memory.
    ///
    /// This method attempts to bring swapped-out sequence groups back into GPU memory
    /// and schedule them for execution, subject to the available scheduling budget.
    ///
    /// # Arguments
    ///
    /// * `swapped_queue` - A queue of sequence groups that are currently swapped out to CPU memory.
    ///                     This queue is not modified in-place.
    /// * `budget` - The current scheduling budget, which is updated as sequences are scheduled.
    /// * `enable_chunking` - If true, allows scheduling partial prefill computations when the budget
    ///                       doesn't have enough capacity for all tokens in a sequence group.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing:
    /// - The remaining `swapped_queue` after scheduling attempts
    /// - A `SchedulerSwappedInOutputs` struct with the scheduling results
    ///
    /// # Behavior
    ///
    /// 1. Sorts the swapped queue based on the scheduling policy's priority.
    /// 2. Attempts to swap in and schedule each sequence group:
    ///    - Checks if the group can be swapped in (enough GPU memory available)
    ///    - Verifies if the scheduling budget allows for the group's execution
    ///    - If successful, swaps in the group and prepares it for execution
    /// 3. Updates the scheduling budget for each successfully scheduled group
    /// 4. Handles cases where groups cannot be scheduled due to resource constraints
    ///
    /// # Errors
    ///
    /// Returns a `SchedulerError` if any scheduling operations fail.
    #[instrument(skip_all)]
    fn schedule_swapped(
        &mut self,
        swapped_queue: VecDeque<SequenceGroup>,
        budget: &mut SchedulingBudget,
        enable_chunking: bool,
    ) -> Result<(VecDeque<SequenceGroup>, SchedulerSwappedInOutputs), SchedulerError> {
        let _enter = self.span.enter();
        trace!("Schedule swapped..");
        // Blocks that need to be swapped or copied before model execution.
        let mut blocks_to_swap_in = HashMap::<u32, u32>::new();
        let mut blocks_to_copy = HashMap::<u32, u32>::new();
        let mut decode_seq_groups = Vec::<ScheduledSequenceGroup>::new();
        let mut prefill_seq_groups = Vec::<ScheduledSequenceGroup>::new();

        let now = Instant::now();

        let mut swapped_queue = P::sort_by_priority(now, &swapped_queue);
        let mut infeasible_seq_groups = Vec::<SequenceGroup>::new();

        while let Some(mut sequence_group) = swapped_queue.pop_front() {
            // If the sequence group cannot be swapped in, stop.
            let allocation_status = self.block_manager.can_swap_in(&sequence_group)?;
            if allocation_status == AllocationStatus::Later {
                // push the sequence group back to `swapped_queue`
                swapped_queue.push_front(sequence_group);
                break;
            } else if allocation_status == AllocationStatus::Never {
                warn!("Failing the request {} because there is not enough KV cache blocks to run the entire sequence..", 
                        sequence_group.request_id);
                for (_, sequence) in sequence_group.sequences.iter_mut() {
                    sequence
                        .write_lock()?
                        .set_sequence_status(SequenceStatus::FinishedIgnored);
                }
                infeasible_seq_groups.push(sequence_group.clone());
                continue;
            }

            // The total number of sequences in the RUNNING state should not
            // exceed the maximum number of sequences.
            let num_new_sequences = sequence_group.get_max_num_running_seqs();
            let num_new_tokens = self.get_num_tokens(
                &sequence_group,
                SequenceStatus::Swapped,
                enable_chunking,
                budget,
            )?;

            if num_new_tokens == 0 || !budget.can_schedule(num_new_tokens, num_new_sequences)? {
                trace!("Either no new tokens to be swapped or no available budget to swap tokens");
                // push the sequence group back to `swapped_queue`
                swapped_queue.push_front(sequence_group);
                break;
            }

            self.swap_in(&mut sequence_group, &mut blocks_to_swap_in)?;
            self.append_slots(&sequence_group, &mut blocks_to_copy)?;
            let is_preffil = sequence_group.is_prefill();
            if is_preffil {
                prefill_seq_groups.push(ScheduledSequenceGroup {
                    scheduled_group: sequence_group.clone(),
                    token_chunk_size: num_new_tokens,
                })
            } else {
                decode_seq_groups.push(ScheduledSequenceGroup {
                    scheduled_group: sequence_group.clone(),
                    token_chunk_size: 1,
                })
            }

            budget.add_num_batched_tokens(sequence_group.request_id.clone(), num_new_tokens);
            budget.add_number_sequences(sequence_group.request_id.clone(), num_new_sequences);
        }

        Ok((
            swapped_queue,
            SchedulerSwappedInOutputs {
                decode_seq_groups,
                prefill_seq_groups,
                blocks_to_swap_in,
                blocks_to_copy,
                infeasible_seq_groups,
            },
        ))
    }

    /// Schedule sequence groups that are in the prefill stage.
    ///
    /// This function processes the waiting queue, which contains prefill requests (initial prompts)
    /// and preempted requests that need to be recomputed from the beginning.
    ///
    /// # Arguments
    ///
    /// * `waiting_queue` - A queue containing prefill requests. This argument is not modified in-place.
    /// * `budget` - The scheduling budget, which is updated in-place when requests are scheduled.
    /// * `enable_chunking` - If true, allows sequence groups to be chunked. Only a portion of tokens
    ///                       will be scheduled if the budget's capacity is insufficient for all tokens.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing:
    /// - The remaining `waiting_queue` after scheduling attempts
    /// - A `SchedulerPrefillOutputs` struct with the scheduling results
    ///
    /// # Behavior
    ///
    /// 1. Processes each sequence group in the waiting queue:
    ///    - Checks if the group can be allocated (enough GPU memory available)
    ///    - Verifies if the prompt length is within limits
    ///    - Ensures the scheduling budget allows for the group's execution
    /// 2. If a group can be scheduled:
    ///    - Allocates resources and sets the sequence status to Running
    ///    - Updates the scheduling budget
    /// 3. Handles cases where groups cannot be scheduled:
    ///    - Due to resource constraints (pushed back to waiting queue)
    ///    - Due to exceeding limits (marked as ignored)
    ///
    /// # Errors
    ///
    /// Returns a `SchedulerError` if:
    /// - There's an invalid number of waiting sequences in a group
    /// - The number of new tokens doesn't match the prompt length (when chunking is disabled)
    /// - Any scheduling operations fail
    #[instrument(skip_all)]
    fn schedule_prefills(
        &mut self,
        mut waiting_queue: VecDeque<SequenceGroup>,
        budget: &mut SchedulingBudget,
        enable_chunking: bool,
    ) -> Result<(VecDeque<SequenceGroup>, SchedulerPrefillOutputs), SchedulerError> {
        let _enter = self.span.enter();
        trace!("Schedulig prefills..");

        let mut ignored_sequence_groups = Vec::<SequenceGroup>::new();
        let mut sequence_groups = Vec::<ScheduledSequenceGroup>::new();

        // We don't sort `waiting_queue` because we assume it is sorted. We also require
        // ownership of `waiting_queue` so that we don't change it in place, in this method.

        while !waiting_queue.is_empty() && self.passed_delay(Instant::now()) {
            // DON'T PANIC: at this point, we are guaranteed that `waiting_queue` is non-empty
            let mut sequence_group = waiting_queue.pop_front().unwrap();

            // To be used below
            let can_allocate = self.block_manager.can_allocate(&sequence_group);
            let num_new_tokens = self.get_num_tokens(
                &sequence_group,
                SequenceStatus::Waiting,
                enable_chunking,
                budget,
            )?;

            let mut waiting_sequences = sequence_group
                .sequences
                .iter()
                .filter_map(|(_, s)| {
                    if s.read().unwrap().get_sequence_status() == SequenceStatus::Waiting {
                        Some(s)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            if waiting_sequences.len() != 1 {
                error!("Waiting sequence group should have only one prompt sequence, it has {} for request = {}.", waiting_sequences.len(), sequence_group.request_id);
                return Err(SchedulerError::InvalidNumberWaitingSequence {
                    request_id: sequence_group.request_id.clone(),
                    num_sequences: waiting_sequences.len(),
                });
            }

            if !enable_chunking {
                // DON'T PANIC: by previous error check, we are guaranteed that `waiting_sequences` is non-empty
                let num_prompt_tokens = waiting_sequences.first().unwrap().read().unwrap().length();
                if num_prompt_tokens != num_new_tokens {
                    error!("Invalid number of new tokens, got `{num_new_tokens}`, but it should be `{num_prompt_tokens}`");
                    return Err(SchedulerError::InvalidNumberOfNewTokens {
                        num_prompt_tokens,
                        num_new_tokens,
                    });
                }
            }

            let prompt_limit = self.get_prompt_limit();
            if num_new_tokens > prompt_limit {
                warn!(
                    "Input prompt ({} tokens) is too long and exceeds limits of {}",
                    num_new_tokens, prompt_limit
                );
                for (_, sequence) in sequence_group.sequences.iter_mut() {
                    sequence
                        .write_lock()?
                        .set_sequence_status(SequenceStatus::FinishedIgnored)
                }
                ignored_sequence_groups.push(sequence_group.clone());
                continue;
            }

            // If the sequence cannot be allocated, just stop
            if can_allocate == AllocationStatus::Later {
                waiting_queue.push_front(sequence_group);
                break;
            } else if can_allocate == AllocationStatus::Never {
                warn!("Input prompt ({num_new_tokens} tokens) is too long and exceeds the capacity of `block_manager`");
                for sequence in waiting_sequences.iter_mut() {
                    sequence
                        .write_lock()?
                        .set_sequence_status(SequenceStatus::FinishedIgnored);
                }
                ignored_sequence_groups.push(sequence_group.clone());
                continue;
            }

            let num_new_sequences = sequence_group.get_max_num_running_seqs();
            if num_new_sequences == 0 || !budget.can_schedule(num_new_tokens, num_new_sequences)? {
                // Push the sequence group back to the `waiting_queue`
                waiting_queue.push_front(sequence_group);
                break;
            }

            // At this point, we can schedule this request
            self.allocate_and_set_running(&mut sequence_group)?;
            sequence_groups.push(ScheduledSequenceGroup {
                scheduled_group: sequence_group.clone(),
                token_chunk_size: num_new_tokens,
            });

            budget.add_num_batched_tokens(sequence_group.request_id.clone(), num_new_tokens);
            budget.add_number_sequences(sequence_group.request_id.clone(), num_new_sequences);
        }

        if !sequence_groups.is_empty() {
            self.previous_prompt = true;
        }

        Ok((
            waiting_queue,
            SchedulerPrefillOutputs {
                sequence_groups,
                ignored_sequence_groups,
            },
        ))
    }

    /// Schedule queued requests using the default policy.
    ///
    /// This method implements a scheduling policy designed to optimize throughput:
    /// 1. It first attempts to batch as many prefill requests as possible.
    /// 2. If no prefills are scheduled, it then schedules decode requests.
    /// 3. If there's pressure on GPU memory, decode requests may be swapped out or preempted.
    ///
    /// # Algorithm
    /// 1. Initialize a budget based on max tokens and sequences.
    /// 2. Account for currently running sequences in the budget.
    /// 3. If no requests are swapped:
    ///    - Schedule prefill requests first.
    /// 4. If no prefills were scheduled:
    ///    - Schedule running (decode) requests.
    ///    - If no preemptions occurred, attempt to swap in requests.
    /// 5. Update internal queues (waiting, running, swapped) based on scheduling results.
    /// 6. Collect and return scheduling results.
    ///
    /// # Returns
    /// Returns a `Result<SchedulerOutputs, SchedulerError>` containing:
    /// - Scheduled sequence groups
    /// - Number of batched tokens
    /// - Number of prefill groups
    /// - Block swap and copy information
    /// - Ignored sequence groups
    /// - Running queue size
    /// - Number of preempted requests
    ///
    /// # Errors
    /// Returns a `SchedulerError` if:
    /// - The number of batched tokens exceeds the configured maximum.
    /// - The number of sequences exceeds the configured maximum.
    /// - Chunked prefills are detected (which are not allowed in this policy).
    ///
    /// # Performance Considerations
    /// - Prioritizes prefill requests over decode requests for better throughput.
    /// - Implements preemption and swapping to manage GPU memory pressure.
    /// - Maintains ordering of preempted requests for fairness.
    #[instrument(skip_all)]
    fn schedule_default(&mut self) -> Result<SchedulerOutputs, SchedulerError> {
        let _enter = self.span.enter();
        trace!("Scheduling default..");
        // Include running requests to the budget.
        let mut budget = SchedulingBudget::new(
            self.scheduler_config.max_num_batched_tokens(),
            self.scheduler_config.max_num_sequences(),
        );

        // Make sure we include num running seqs before scheduling prefill
        for sequence_group in self.running.iter() {
            budget.add_number_sequences(
                sequence_group.request_id.clone(),
                sequence_group.get_max_num_running_seqs(),
            );
        }

        let mut remaining_running = self.running.clone();
        let mut remaining_waiting = self.waiting.clone();
        let mut remaining_swapped = self.swapped.clone();

        let mut prefills = SchedulerPrefillOutputs::create_empty();
        let mut running_scheduled = SchedulerRunningOutputs::create_empty();
        let mut swapped_in = SchedulerSwappedInOutputs::create_empty();

        // If any requests are swapped, prioritized swapped requests
        if self.swapped.is_empty() {
            // NOTE: we don't mutate `self.waiting` in place, instead we clone the `waiting` queue
            (remaining_waiting, prefills) =
                self.schedule_prefills(remaining_waiting, &mut budget, false)?;
        }

        // Don't schedule decodes if prefills are scheduled.
        // NOTE: If `schedule_prefills` doesn't enable chunking, `self.running`
        // only contains decode requests, not chunked prefills.
        if prefills.sequence_groups.is_empty() {
            // NOTE: we don't mutate `self.running` in place, instead we clone the `running` queue
            (remaining_running, running_scheduled) =
                self.schedule_running(remaining_running, &mut budget, false)?;

            // If any sequence group is preempted, do not swap in any sequence
            // group, because it means there's no slot for new running requests
            if running_scheduled.preempted.len() + running_scheduled.swapped_out.len() == 0 {
                (remaining_swapped, swapped_in) =
                    self.schedule_swapped(remaining_swapped, &mut budget, false)?
            }
        }

        if budget.num_batched_tokens > self.scheduler_config.max_num_batched_tokens() {
            error!("Number of budget batched tokens exceeds the configured number of max batched tokens");
            return Err(SchedulerError::InvalidNumberBudgetTokens(
                    "Number of budget batched tokens exceeds the configured number of max batched tokens".into()
                ));
        }

        if budget.num_current_sequences() > self.scheduler_config.max_num_sequences() {
            error!("Number of budget sequences exceed the configured number of max number of sequences");
            return Err(SchedulerError::InvalidNumberBudgetSequences(
                "Number of budget sequences exceed the configured number of max number of sequences".into()
            ));
        }

        // To be used later for method output
        let preempted = running_scheduled.preempted.len() + running_scheduled.swapped_out.len();

        // Update waiting requests
        self.waiting = remaining_waiting;
        // NOTE: need to reverse order of preempted sequence groups to preserve order once you push these
        // to the left on the `self.waiting` queue.
        // NOTE: Preempted running scheduled means there was not enough block space to be run on the
        // current inference loop, so these requests should have priority regarding newly received
        // requests.
        running_scheduled
            .preempted
            .iter()
            .rev()
            .for_each(|s| self.waiting.push_front(s.clone()));
        // Update new running requests
        self.running = remaining_running;
        // NOTE: newly prefill requests get appended first, then decoding ones
        self.running.extend(
            prefills
                .sequence_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );
        self.running.extend(
            running_scheduled
                .decode_seq_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );
        self.running.extend(
            swapped_in
                .decode_seq_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );

        // Update swapped requests
        self.swapped = remaining_swapped;
        self.swapped.extend(running_scheduled.swapped_out);

        // There should be no prefill from running queue because this policy
        // doesn't allow chunked prefills.
        if !running_scheduled.prefill_seq_groups.is_empty() {
            error!("Chunked prefills are not allowed for running schedules, there should be none but we received {}", running_scheduled.prefill_seq_groups.len());
            return Err(SchedulerError::ChunkedPrefillsNotAllowed(format!(
                "Chunked prefills are not allowed for running schedules, there should be none but we received {}",
                running_scheduled.prefill_seq_groups.len()
            )));
        }

        if !swapped_in.prefill_seq_groups.is_empty() {
            error!(
                "Chunked prefills are not allowed for swapped in schedules, there should be none but we received {}", running_scheduled.prefill_seq_groups.len()
            );
            return Err(SchedulerError::ChunkedPrefillsNotAllowed(
                format!("Chunked prefills are not allowed for swapped in schedules, there should be none but we received {}", swapped_in.prefill_seq_groups.len()),
            ));
        }

        let number_prefill_groups = prefills.sequence_groups.len();

        let scheduled_sequence_groups: Vec<ScheduledSequenceGroup> = prefills
            .sequence_groups
            .into_iter()
            .chain(
                running_scheduled
                    .decode_seq_groups
                    .into_iter()
                    .chain(swapped_in.decode_seq_groups.into_iter()),
            )
            .collect();

        let blocks_to_copy = running_scheduled
            .blocks_to_copy
            .into_iter()
            .chain(swapped_in.blocks_to_copy.into_iter())
            .collect();

        let ignored_seq_groups = prefills
            .ignored_sequence_groups
            .into_iter()
            .chain(swapped_in.infeasible_seq_groups.into_iter())
            .collect();

        // NOTE: `SchedulerOutputs` should only be used by read operations, as
        // contrary to original Python vllm implementation, `SchedulerOutputs` is passed
        // by value, and not by reference
        Ok(SchedulerOutputs {
            scheduled_sequence_groups,
            num_batched_tokens: budget.num_batched_tokens,
            number_prefill_groups,
            blocks_to_swap_in: swapped_in.blocks_to_swap_in,
            blocks_to_swap_out: running_scheduled.blocks_to_swap_out,
            blocks_to_copy,
            ignored_seq_groups,
            running_queue_size: self.running.len(),
            preempted,
            span: info_span!("scheduler-outputs"),
        })
    }

    /// Schedule queued requests using a chunked prefill approach.
    ///
    /// This method implements an optimized scheduling policy that allows batching
    /// prefill (prompt processing) and decode (token generation) requests together.
    /// The chunked prefill approach can improve GPU utilization and reduce inter-token
    /// latency by preventing decode requests from being blocked by long prefill requests.
    ///
    /// # Algorithm
    /// 1. Initialize a scheduling budget based on max tokens and sequences.
    /// 2. Schedule as many decoding requests as possible from the running queue.
    /// 3. If no preemptions occurred, attempt to schedule swapped-out requests.
    /// 4. Schedule new prefill requests, potentially in chunks.
    /// 5. Update internal queues (waiting, running, swapped) based on scheduling results.
    /// 6. Collect and return scheduling results.
    ///
    /// # Key Concepts
    /// - Budget: Tracks available resources for scheduling (tokens and sequences).
    /// - Preemption: Interrupting running requests when resources are constrained.
    /// - Swapping: Moving requests between GPU and CPU memory to manage resources.
    ///
    /// # Queue Updates
    /// - Waiting: Updated with remaining and preempted requests.
    /// - Running: Updated with newly scheduled prefill and decode requests.
    /// - Swapped: Updated with requests that couldn't fit in GPU memory.
    ///
    /// # Output Ordering
    /// Scheduled sequence groups are ordered as follows:
    /// 1. New prefill requests
    /// 2. Chunked prefill requests from running queue
    /// 3. Prefill requests from swapped-in queue
    /// 4. Decode requests from running queue
    /// 5. Decode requests from swapped-in queue
    ///
    /// # Errors
    /// Returns a `SchedulerError` if:
    /// - The number of batched tokens exceeds the configured maximum.
    /// - The number of sequences exceeds the configured maximum.
    #[instrument(skip_all)]
    fn schedule_chunked_prefill(&mut self) -> Result<SchedulerOutputs, SchedulerError> {
        let _enter = self.span.enter();
        trace!("Scheduling chunked prefill..");
        let mut budget = SchedulingBudget::new(
            self.scheduler_config.max_num_batched_tokens(),
            self.scheduler_config.max_num_sequences(),
        );

        let mut remaining_swapped = self.swapped.clone();
        let mut swapped_in = SchedulerSwappedInOutputs::create_empty();

        // Decoding should be always scheduled first by fcfs
        let (remaining_running, running_scheduled) =
            self.schedule_running(self.running.clone(), &mut budget, true)?;

        // Schedule swapped out requests.
        // If preemption happens, it means we don't have space for swap in
        if running_scheduled.preempted.len() + running_scheduled.swapped_out.len() == 0 {
            (remaining_swapped, swapped_in) =
                self.schedule_swapped(remaining_swapped, &mut budget, true)?;
        }

        // Schedule new prefills.
        let (remaining_waiting, prefills) =
            self.schedule_prefills(self.waiting.clone(), &mut budget, true)?;

        if budget.num_batched_tokens() > self.scheduler_config.max_num_batched_tokens() {
            error!("Number of budget batched tokens exceeds the configured number of max batched tokens");
            return Err(SchedulerError::InvalidNumberBudgetTokens(
                    "Number of budget batched tokens exceeds the configured number of max batched tokens".into()
                ));
        }

        if budget.num_current_sequences() > self.scheduler_config.max_num_sequences() {
            error!("Number of budget sequences exceed the configured number of max number of sequences");
            return Err(SchedulerError::InvalidNumberBudgetSequences(
                "Number of budget sequences exceed the configured number of max number of sequences".into()
            ));
        }

        // To be used later, on the output
        let preempted = running_scheduled.preempted.len() + running_scheduled.swapped_out.len();

        // Update waiting queue
        self.waiting = remaining_waiting;
        // NOTE: need to reverse order of preempted sequence groups to preserve order once you push these
        // to the left on the `self.waiting` queue.
        // NOTE: Preempted running scheduled means there was not enough block space to be run on the
        // current inference loop, so these requests should have priority regarding newly received
        // requests.
        running_scheduled
            .preempted
            .iter()
            .rev()
            .for_each(|s| self.waiting.push_front(s.clone()));

        // Update new running requests
        self.running = remaining_running;
        // NOTE: newly prefill requests get appended first, then decoding ones
        self.running.extend(
            prefills
                .sequence_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );
        self.running.extend(
            running_scheduled
                .decode_seq_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );
        self.running.extend(
            swapped_in
                .decode_seq_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );
        self.running.extend(
            swapped_in
                .prefill_seq_groups
                .iter()
                .map(|s| s.scheduled_group.clone()),
        );

        // Update swapped requests
        self.swapped = remaining_swapped;
        self.swapped.extend(running_scheduled.swapped_out);

        let number_prefill_groups = prefills.sequence_groups.len()
            + swapped_in.prefill_seq_groups.len()
            + running_scheduled.prefill_seq_groups.len();
        let scheduled_sequence_groups = prefills
            .sequence_groups
            .into_iter()
            .chain(
                running_scheduled.prefill_seq_groups.into_iter().chain(
                    swapped_in.prefill_seq_groups.into_iter().chain(
                        running_scheduled
                            .decode_seq_groups
                            .into_iter()
                            .chain(swapped_in.decode_seq_groups.into_iter()),
                    ),
                ),
            )
            .collect();
        let blocks_to_copy = running_scheduled
            .blocks_to_copy
            .into_iter()
            .chain(swapped_in.blocks_to_copy)
            .collect();
        let ignored_seq_groups = prefills.ignored_sequence_groups;

        Ok(SchedulerOutputs {
            scheduled_sequence_groups,
            num_batched_tokens: budget.num_batched_tokens(),
            number_prefill_groups,
            blocks_to_swap_in: swapped_in.blocks_to_swap_in,
            blocks_to_swap_out: running_scheduled.blocks_to_swap_out,
            blocks_to_copy,
            ignored_seq_groups,
            running_queue_size: self.running.len(),
            preempted,
            span: info_span!("scheduler-outputs"),
        })
    }

    /// Schedule queued requests based on the configured scheduling policy.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing:
    /// - `Ok(SchedulerOutputs)`: The scheduling results if successful.
    /// - `Err(SchedulerError)`: An error if scheduling fails.
    ///
    /// # Behavior
    ///
    /// This method chooses between two scheduling algorithms:
    ///
    /// 1. If chunked prefill is enabled (via `scheduler_config.enable_chunked_prefill()`):
    ///    - Calls `self.schedule_chunked_prefill()`, which allows batching prefill and decode requests together.
    ///    - This can improve GPU utilization for long prompts by processing them in chunks.
    ///
    /// 2. Otherwise:
    ///    - Calls `self.schedule_default()`, which prioritizes completing full prefills before scheduling decodes.
    ///    - This approach may be more suitable for shorter prompts or when strict ordering is required.
    ///
    /// # Notes
    ///
    /// - The choice between chunked and default scheduling can significantly impact performance and latency.
    /// - Chunked prefill is generally more efficient for longer prompts or when dealing with a mix of long and short requests.
    /// - The default scheduling may be preferable for simpler workloads or when you need to ensure all prefills complete before any decoding starts.
    #[instrument(skip_all)]
    fn schedule_(&mut self) -> Result<SchedulerOutputs, SchedulerError> {
        if self.scheduler_config.enable_chunked_prefill() {
            self.schedule_chunked_prefill()
        } else {
            self.schedule_default()
        }
    }

    /// Schedule queued requests and prepare metadata for execution.
    ///
    /// This method processes the internal state of the `Scheduler` to determine which
    /// sequence groups should be executed next. It prepares detailed metadata for each
    /// scheduled sequence group, including block allocations and sampling parameters.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing:
    /// - A vector of `Arc<SequenceGroupMetadata>`: Metadata for each scheduled sequence group,
    ///   ready for model execution.
    /// - `SchedulerOutputs`: Detailed scheduling results, including block management information.
    ///
    /// # Errors
    ///
    /// Returns a `SchedulerError` if:
    /// - The internal scheduling process fails.
    /// - There's an invalid state in prefill sequences.
    /// - Block table information is missing for a sequence.
    ///
    /// # Implementation Details
    ///
    /// 1. Calls the internal scheduling method to determine which sequence groups to run.
    /// 2. For each scheduled sequence group:
    ///    - Prepares sequence data and block table information.
    ///    - Determines if sampling should occur (based on prefill status and token counts).
    ///    - Creates a `SequenceGroupMetadata` object with all necessary execution information.
    /// 3. Updates block access times in the block manager.
    ///
    /// # Note
    ///
    /// This method is the main entry point for the scheduling process and should be called
    /// each time new work needs to be scheduled for the model.
    #[instrument(skip_all)]
    pub fn schedule(
        &mut self,
    ) -> Result<(Vec<Arc<SequenceGroupMetadata>>, SchedulerOutputs), SchedulerError> {
        let _enter = self.span.enter();
        trace!("Scheduling..");
        let scheduler_outputs = self.schedule_()?;
        let now = Instant::now();

        // Create input data structures
        let mut sequence_groups_metadata = Vec::new();

        for scheduled_sequence_group in scheduler_outputs.scheduled_sequence_groups.iter() {
            let sequence_group = scheduled_sequence_group.scheduled_group.clone();
            let token_chunk_size = scheduled_sequence_group.token_chunk_size;
            sequence_group.maybe_set_first_scheduled_time(now);

            // Mapping from sequence id to `SequenceData`
            let mut sequence_data = HashMap::<u64, SequenceData>::new();
            // Mapping from sequence id to `PhysicalBlock` number
            let mut block_tables = HashMap::<u64, Vec<u32>>::new();

            for sequence in sequence_group.sequences.iter().filter_map(|(_, s)| {
                if s.read().unwrap().get_sequence_status() == SequenceStatus::Running {
                    Some(s)
                } else {
                    None
                }
            }) {
                let sequence_guard_lock = sequence.read_lock()?;
                let sequence_id = sequence_guard_lock.sequence_id();
                sequence_data.insert(sequence_id, sequence_guard_lock.sequence_data.clone());
                if let Some(block_table_ids) = self.block_manager.get_block_table_ids(&sequence_id)
                {
                    block_tables.insert(sequence_id, block_table_ids);
                    self.block_manager
                        .access_all_blocks_in_sequence(&sequence_id, now)?;
                } else {
                    error!(
                        "Missing block table for sequence with id = {}",
                        sequence_guard_lock.sequence_id()
                    );
                }
            }

            let mut do_sample = true;
            if sequence_group.is_prefill() {
                if sequence_group.sequences.len() != 1 {
                    error!("Prefill mode has only one sequence");
                    return Err(SchedulerError::InvalidPrefillSequences(
                        "Prefill mode has only one sequence".into(),
                    ));
                }

                // DON'T PANIC: checked previously that `sequence_group.sequences.len() != 1`
                let sequence = sequence_group.sequences.values().next().unwrap();

                // In the next iteration, all prompt tokens are not computed.
                // It means the prefill is chunked, and we don't need sampling.
                // NOTE: We use get_len instead of get_prompt_len because when
                // a sequence is preempted, prefill includes previous generated
                // output tokens.
                let sequence_guard_lock = sequence.read_lock()?;
                if token_chunk_size + sequence_guard_lock.sequence_data.get_num_computed_tokens()
                    < sequence_guard_lock.sequence_data.length()
                {
                    do_sample = false;
                }
            }

            // It assumes the scheduled_seq_groups is ordered by
            // prefill < decoding.
            let is_prompt = sequence_group.is_prefill();
            let sequence_group_metadata = Arc::new(SequenceGroupMetadata::new(
                sequence_group.request_id.clone(),
                is_prompt,
                sequence_data,
                sequence_group.next_token_chooser_params(),
                sequence_group.stopping_params(),
                block_tables,
                do_sample,
                Some(token_chunk_size),
                sequence_group.logits_processor.clone(),
            ));
            sequence_groups_metadata.push(sequence_group_metadata);
        }

        // TODO: remove this code if not necessary
        // // Now that the batch has been created, we can assume all blocks in the
        // // batch will have been computed before the next scheduling invocation.
        // // This is because the engine assumes that a failure in model execution
        // //  will crash the vLLM instance / will not retry.
        // for scheduled_seq_group in scheduler_outputs.scheduled_sequence_groups.iter() {
        //     self.block_manager.mark_blocks_as_computed(
        //         scheduled_seq_group.scheduled_group)
        // }

        Ok((sequence_groups_metadata, scheduler_outputs))
    }
}

impl<P: Debug> Scheduler<P> {
    /// Calculates the number of new tokens to compute for a given sequence group.
    ///
    /// This function determines how many new tokens should be processed for a sequence group
    /// based on its current status and the available scheduling budget. It supports token
    /// chunking for efficient processing of long sequences.
    ///
    /// # Arguments
    ///
    /// * `sequence_group` - The sequence group to evaluate.
    /// * `sequence_status` - The status of sequences to consider (e.g., Running, Waiting).
    /// * `enable_chunking` - If true, allows processing a subset of available tokens to fit within budget.
    /// * `budget` - The current scheduling budget, used to limit token processing when chunking is enabled.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing:
    /// - `Ok(usize)`: The number of new tokens to compute.
    /// - `Err(SchedulerError)`: If there are no new tokens to schedule.
    ///
    /// # Behavior
    ///
    /// - Sums up new tokens across all sequences in the group with the specified status.
    /// - If chunking is enabled and there's only one sequence, limits tokens to fit the budget.
    /// - For multiple sequences (e.g., beam search), chunking is not applied.
    /// - Returns an error if no new tokens are available to schedule.
    ///
    /// # Note
    ///
    /// This function is crucial for balancing processing efficiency and resource utilization,
    /// especially for long sequences or when dealing with limited computational budgets.
    #[instrument(skip_all)]
    fn get_num_tokens(
        &self,
        sequence_group: &SequenceGroup,
        sequence_status: SequenceStatus,
        enable_chunking: bool,
        budget: &mut SchedulingBudget,
    ) -> Result<usize, SchedulerError> {
        let trace!(
            "Get number of tokens for sequence group with id = {}",
            sequence_group.request_id
        );
        let mut num_new_tokens = 0;
        let mut num_sequences_in_status = 0;

        for (_, seq) in sequence_group.sequences.iter() {
            let sequence_guard_lock = seq.read_lock()?;
            if sequence_guard_lock.get_sequence_status() == sequence_status {
                num_new_tokens += sequence_guard_lock.get_num_new_tokens();
                num_sequences_in_status += 1;
            }
        }

        if num_new_tokens == 0 {
            error!("No new tokens to be scheduled..");
            return Err(SchedulerError::ZeroNewTokensToSchedule);
        }

        // Chunk if a running request cannot fit in.
        // If the number of seqs > 1, it means it is doing beam search in a
        // decode phase. Do not chunk in that case.
        if enable_chunking && num_sequences_in_status == 1 {
            num_new_tokens = num_new_tokens.min(budget.remaining_budget_tokens());
        }

        Ok(num_new_tokens)
    }

    /// Checks if there is sufficient space in the KV cache to continue generation for the given sequence group.
    ///
    /// This method delegates to the `BlockManager` to determine if there are enough
    /// free blocks available to accommodate the next token for all running sequences
    /// in the group.
    ///
    /// # Arguments
    ///
    /// * `sequence_group` - The `SequenceGroup` to check for space availability.
    ///
    /// # Returns
    ///
    /// Returns `true` if there is enough space to append slots for all running
    /// sequences in the group, `false` otherwise.
    ///
    /// # Note
    ///
    /// This method is crucial for preventing out-of-memory errors and ensuring
    /// efficient use of the KV cache. It's typically called before attempting
    /// to generate the next token for a sequence group.
    #[instrument(skip_all)]
    fn can_append_slots(&self, sequence_group: &SequenceGroup) -> bool {
        let _enter = self.span.enter();
        trace!(
            "Can append slots for sequence group with id = {}",
            sequence_group.request_id
        );
        self.block_manager.can_append_slots(sequence_group)
    }

    /// Appends new slots to the running sequences in the given sequence group.
    ///
    /// This method allocates new KV cache blocks for each running sequence in the group
    /// and updates the `blocks_to_copy` map with any copy-on-write operations needed.
    ///
    /// # Arguments
    ///
    /// * `sequence_group` - The sequence group containing the sequences to append slots to.
    /// * `blocks_to_copy` - A mutable map that will be updated with any new copy-on-write operations.
    ///                      Keys are source block indices, values are destination block indices.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if slots were successfully appended, or a `SchedulerError` if an error occurred.
    ///
    /// # Effects
    ///
    /// - Allocates new KV cache blocks for running sequences
    /// - Updates `blocks_to_copy` with any necessary copy-on-write operations
    /// - Logs information about the operation
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - There's an issue accessing sequence data
    /// - The block manager fails to append slots
    #[instrument(skip_all)]
    fn append_slots(
        &mut self,
        sequence_group: &SequenceGroup,
        blocks_to_copy: &mut HashMap<u32, u32>,
    ) -> Result<(), SchedulerError> {
        let _enter = self.span.enter();
        trace!(
            "Appending slot to sequence group with id = {}",
            sequence_group.request_id
        );
        let running_sequences = sequence_group.sequences.iter().filter_map(|(_, s)| {
            if s.read().unwrap().get_sequence_status() == SequenceStatus::Running {
                Some(s)
            } else {
                None
            }
        });
        for sequence in running_sequences {
            let cows = self.block_manager.append_slots(sequence.read_lock()?)?;
            if let Some(cow) = cows {
                blocks_to_copy.insert(cow.0, cow.1);
            } else {
                warn!("No Copy on Write new blocks to append, for sequence with id = {} in sequence group with id = {}", 
                    sequence.read_lock()?.sequence_id(), sequence_group.request_id);
            }
        }
        Ok(())
    }

    /// Adds a new `SequenceGroup` to the end of the `waiting` queue.
    ///
    /// This method is used to enqueue new sequence groups for processing by the scheduler.
    /// The sequence group is added to the end of the waiting queue, maintaining a
    /// first-in-first-out (FIFO) order for newly added requests.
    ///
    /// # Arguments
    ///
    /// * `sequence_group` - The `SequenceGroup` to be added to the waiting queue.
    ///
    /// # Effects
    ///
    /// - The provided `sequence_group` is appended to the end of the `self.waiting` queue.
    /// - The total number of unfinished sequences in the scheduler increases.
    ///
    /// # Example
    ///
    /// ```
    /// let mut scheduler = Scheduler::new(/* ... */);
    /// let new_sequence_group = SequenceGroup::new(/* ... */);
    /// scheduler.add_sequence_group(new_sequence_group);
    /// ```
    ///
    /// # Note
    ///
    /// This method does not immediately schedule the added sequence group for processing.
    /// The actual scheduling occurs when the `schedule` method is called on the scheduler.
    #[instrument(skip_all)]
    pub fn add_sequence_group(&mut self, sequence_group: SequenceGroup) {
        let _enter = self.span.enter();
        trace!(
            "Adding sequence group with id = {}",
            sequence_group.request_id
        );
        self.waiting.push_back(sequence_group)
    }

    /// Preempts a sequence group, either by recomputation or swapping.
    ///
    /// # Arguments
    ///
    /// * `sequence_group` - The sequence group to preempt.
    /// * `blocks_to_swap_out` - A map to track blocks that need to be swapped out.
    /// * `preemption_mode` - Optional preemption mode. If None, the mode is determined automatically.
    ///
    /// # Returns
    ///
    /// The preemption mode that was used.
    ///
    /// # Details
    ///
    /// If preemption mode is not specified, it is determined as follows:
    /// - Recomputation is used by default for single-sequence groups, as it has lower overhead.
    /// - Swapping is used for multi-sequence groups (e.g., beam search), as recomputation is not currently supported.
    ///
    /// # Notes
    ///
    /// - FIXME: The current policy implicitly prioritizes multi-sequence groups over single-sequence groups,
    ///   as swapped sequences are prioritized over waiting sequences.
    /// - TODO: Implement recomputation support for multi-sequence groups.
    ///
    /// # Warnings
    ///
    /// Logs a warning every 50 preemptions about potential performance impact and suggests solutions.
    #[instrument(skip_all)]
    fn preempt(
        &mut self,
        sequence_group: &mut SequenceGroup,
        blocks_to_swap_out: &mut HashMap<u32, u32>,
        preemption_mode: Option<PreemptionMode>,
    ) -> Result<PreemptionMode, SchedulerError> {
        let _enter = self.span.enter();
        trace!(
            "Preempting sequence group with id = {}",
            sequence_group.request_id
        );
        // If preemption mode is not specified, we determine the mode as follows:
        // We use recomputation by default since it incurs lower overhead than
        // swapping. However, when the sequence group has multiple sequences
        // (e.g., beam search), recomputation is not currently supported. In
        // such a case, we use swapping instead.
        // FIXME: This makes our scheduling policy a bit bizarre.
        // As swapped sequences are prioritized over waiting sequences,
        // sequence groups with multiple sequences are implicitly prioritized
        // over sequence groups with a single sequence.
        // TODO: Support recomputation for sequence groups with multiple
        // sequences. This may require a more sophisticated CUDA kernel.
        let preemption_mode = if preemption_mode.is_none() {
            if sequence_group.get_max_num_running_seqs() == 1 {
                PreemptionMode::Recomputation
            } else {
                PreemptionMode::Swap
            }
        } else {
            preemption_mode.unwrap()
        };

        if self.num_cumulative_preemption % 50 == 0 {
            warn!("Sequence group with id = {} is preempted by {:?} mode because there is not enough KV cache space. 
                    This can affect the end-to-end performance. Increase `gpu_memory_utilization` or `tensor_parallel_size` 
                    to provide more KV cache memory. `total_num_cumulative_preemption = {}` ", 
                    sequence_group.request_id, preemption_mode, self.num_cumulative_preemption + 1);
        }
        self.num_cumulative_preemption += 1;

        if preemption_mode == PreemptionMode::Recomputation {
            self.preempt_by_recompute(sequence_group)?;
        } else if preemption_mode == PreemptionMode::Swap {
            self.preempt_by_swap(sequence_group, blocks_to_swap_out)?;
        } else {
            unreachable!("Preemption mode not supported");
        }

        Ok(preemption_mode)
    }

    /// Preempts a `SequenceGroup` by resetting it for recomputation.
    ///
    /// This method handles preemption by resetting the state of the sequence group
    /// to allow for recomputation from the beginning. It is typically used when
    /// there are not enough resources to continue processing the sequence group.
    ///
    /// # Arguments
    ///
    /// * `sequence_group` - A mutable reference to the `SequenceGroup` to be preempted.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if preemption is successful, or a `SchedulerError` if an error occurs.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - There is not exactly one running sequence in the group (only single sequences can be recomputed).
    /// - There are issues accessing or modifying sequence data.
    ///
    /// # Effects
    ///
    /// - Sets the status of the running sequence to `Waiting`.
    /// - Frees the resources associated with the sequence.
    /// - Resets the sequence state to allow for recomputation.
    ///
    /// # Notes
    ///
    /// This method is part of the preemption strategy and should be used carefully
    /// as it affects the execution flow of the sequence group.
    #[instrument(skip_all)]
    fn preempt_by_recompute(
        &mut self,
        sequence_group: &mut SequenceGroup,
    ) -> Result<(), SchedulerError> {
        let _enter = self.span.enter();
        trace!(
            "Preemption by recomputation for sequence group with id = {}",
            sequence_group.request_id
        );
        let sequences = sequence_group
            .sequences
            .iter_mut()
            .filter_map(|(_, s)| {
                if s.read().unwrap().get_sequence_status() == SequenceStatus::Running {
                    Some(s)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if sequences.len() != 1 {
            error!("Number of sequences in `SequenceGroup` for preempt by recompute should be 1, but is {}", sequences.len());
            return Err(SchedulerError::InvalidNumberSequencesForRecompute(
                sequences.len(),
            ));
        }

        for sequence in sequences {
            let mut sequence_guard_lock = sequence.write_lock()?;
            sequence_guard_lock.set_sequence_status(SequenceStatus::Waiting);

            self.free_sequence(sequence_guard_lock.sequence_id())?;
            sequence_guard_lock.reset_state_for_recompute();
        }

        Ok(())
    }

    /// Preempts a `SequenceGroup` by swapping it out of GPU memory.
    ///
    /// This method handles preemption by moving the sequence group's data from GPU
    /// memory to CPU memory, allowing other sequences to use the freed GPU resources.
    ///
    /// # Arguments
    ///
    /// * `sequence_group` - A mutable reference to the `SequenceGroup` to be preempted.
    /// * `blocks_to_swap_out` - A mutable reference to a HashMap that will be updated
    ///   with the blocks that need to be swapped out. Keys are the GPU block IDs,
    ///   and values are the corresponding CPU block IDs.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if preemption is successful, or a `SchedulerError` if an error occurs.
    ///
    /// # Effects
    ///
    /// - Calls `self.swap_out()` to move the sequence group's data to CPU memory.
    /// - Updates `blocks_to_swap_out` with the blocks that need to be transferred.
    /// - Changes the status of affected sequences to `Swapped`.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - There is not enough CPU memory to accommodate the swapped-out data.
    /// - The swap-out operation fails for any reason.
    ///
    /// # Note
    ///
    /// This method is part of the preemption strategy and should be used when GPU
    /// resources need to be freed for higher-priority tasks. It allows for efficient
    /// management of limited GPU memory by temporarily moving less critical data to CPU.
    #[instrument(skip_all)]
    fn preempt_by_swap(
        &mut self,
        sequence_group: &mut SequenceGroup,
        blocks_to_swap_out: &mut HashMap<u32, u32>,
    ) -> Result<(), SchedulerError> {
        let _enter = self.span.enter();
        trace!(
            "Preemption by swap for sequence group with id = {}..",
            sequence_group.request_id
        );

        self.swap_out(sequence_group, blocks_to_swap_out)?;

        Ok(())
    }

    /// Swaps out GPU blocks to CPU blocks for a given sequence group.
    ///
    /// This method is used to free up GPU memory by moving data for a sequence group
    /// from GPU to CPU memory. It's typically called when GPU memory is constrained
    /// and lower-priority sequences need to be temporarily moved to make room for
    /// higher-priority work.
    ///
    /// # Arguments
    ///
    /// * `sequence_group` - The sequence group to swap out.
    /// * `blocks_to_swap_out` - A mutable map that will be updated with the block mappings
    ///                          from GPU to CPU. Keys are GPU block IDs, values are CPU block IDs.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the swap out was successful, or a `SchedulerError` if there
    /// was an issue (e.g., not enough CPU swap space).
    ///
    /// # Effects
    ///
    /// - Updates the `blocks_to_swap_out` map with new GPU to CPU block mappings.
    /// - Changes the status of affected sequences in the group from `Running` to `Swapped`.
    /// - Frees up GPU memory by moving data to CPU memory.
    ///
    /// # Errors
    ///
    /// Returns `SchedulerError::NotEnoughBlockSpaceForSwapOut` if there isn't sufficient
    /// CPU swap space available to perform the operation.
    #[instrument(skip_all)]
    fn swap_out(
        &mut self,
        sequence_group: &mut SequenceGroup,
        blocks_to_swap_out: &mut HashMap<u32, u32>,
    ) -> Result<(), SchedulerError> {
        let _enter = self.span.enter();
        trace!(
            "Swapping out for sequence group with id = {}",
            sequence_group.request_id
        );

        if !self.block_manager.can_swap_out(sequence_group)? {
            error!("Aborted due to the lack of CPU swap space. Please increase the swap space to avoid this error.");
            return Err(SchedulerError::NotEnoughBlockSpaceForSwapOut);
        }

        let mapping = self.block_manager.swap_out(sequence_group)?;
        blocks_to_swap_out.extend(mapping.iter());
        sequence_group.sequences.iter_mut().for_each(|(_, s)| {
            let mut sequence_guard_lock = s.write().unwrap();
            let sequence_status = sequence_guard_lock.get_sequence_status();
            if sequence_status == SequenceStatus::Running {
                sequence_guard_lock.set_sequence_status(SequenceStatus::Swapped)
            }
        });

        Ok(())
    }

    /// Swaps in blocks from CPU memory to GPU memory for a given sequence group.
    ///
    /// This method moves data from CPU memory back to GPU memory for sequences that were
    /// previously swapped out. It updates the block mappings and sequence statuses accordingly.
    ///
    /// # Arguments
    ///
    /// * `sequence_group` - The sequence group to swap in.
    /// * `blocks_to_swap_in` - A mutable map that will be updated with the block mappings
    ///                         from CPU to GPU. Keys are CPU block IDs, values are GPU block IDs.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the swap in was successful, or a `SchedulerError` if there
    /// was an issue (e.g., not enough GPU memory).
    ///
    /// # Effects
    ///
    /// - Updates the `blocks_to_swap_in` map with new CPU to GPU block mappings.
    /// - Changes the status of affected sequences in the group from `Swapped` to `Running`.
    /// - Moves data from CPU memory to GPU memory.
    ///
    /// # Errors
    ///
    /// May return a `SchedulerError` if:
    /// - There's insufficient GPU memory to accommodate the swapped-in data.
    /// - The block manager encounters an error during the swap-in process.
    #[instrument(skip_all)]
    fn swap_in(
        &mut self,
        sequence_group: &mut SequenceGroup,
        blocks_to_swap_in: &mut HashMap<u32, u32>,
    ) -> Result<(), SchedulerError> {
        let _enter = self.span.enter();
        trace!(
            "Swapping in for sequence group with id = {}",
            sequence_group.request_id
        );
        let mapping = self.block_manager.swap_in(sequence_group)?;
        blocks_to_swap_in.extend(mapping.iter());
        sequence_group.sequences.iter_mut().for_each(|(_, s)| {
            let mut sequence_guard_lock = s.write().unwrap();
            let sequence_status = sequence_guard_lock.get_sequence_status();
            if sequence_status == SequenceStatus::Swapped {
                sequence_guard_lock.set_sequence_status(SequenceStatus::Running)
            }
        });

        Ok(())
    }

    /// Determines if enough time has passed to schedule the next prompt.
    ///
    /// This function implements a delay mechanism to potentially improve batching efficiency
    /// by allowing the waiting queue to accumulate more requests before scheduling.
    ///
    /// # Arguments
    ///
    /// * `now` - The current timestamp.
    ///
    /// # Returns
    ///
    /// * `true` if enough time has passed to schedule the next prompt, `false` otherwise.
    ///
    /// # Behavior
    ///
    /// 1. Updates the last prompt latency if the previous operation was a prompt.
    /// 2. Always updates the previous time and resets the prompt flag.
    /// 3. If delay factor is set and there are waiting requests:
    ///    - Calculates the earliest arrival time of waiting requests.
    ///    - Returns true if either:
    ///      a) The time since the earliest arrival exceeds the delay factor * last prompt latency.
    ///      b) There are no currently running requests.
    /// 4. If delay factor is not set or there are no waiting requests, always returns true.
    fn passed_delay(&mut self, now: Instant) -> bool {
        let _enter = self.span.enter();
        trace!("Checking if enough time has passed to schedule the next prompt");
        if self.previous_prompt {
            self.last_prompt_latency = (now - self.previous_time).as_secs_f32();
        }

        self.previous_time = now;
        self.previous_prompt = false;

        // Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor() > 0.0 && !self.waiting.is_empty() {
            // DON'T PANIC: at this point, we are guaranteed that `self.waiting` is non-empty
            let earliest_arrival_time =
                self.waiting.iter().map(|s| s.arrival_time()).min().unwrap();
            (now - earliest_arrival_time).as_secs_f32()
                > self.scheduler_config.delay_factor() * self.last_prompt_latency
                || self.running.is_empty()
        } else {
            true
        }
    }

    /// Determines the maximum allowed length for prompts based on the scheduler configuration.
    ///
    /// # Returns
    ///
    /// - If chunked prefill is enabled: Returns `max_model_len`.
    /// - If chunked prefill is disabled: Returns the minimum of `max_model_len` and `max_num_batched_tokens`.
    ///
    /// # Behavior
    ///
    /// This function helps enforce limits on prompt lengths to ensure efficient scheduling:
    ///
    /// - With chunked prefill: Allows longer prompts up to the model's maximum length, as they can be processed in chunks.
    /// - Without chunked prefill: Restricts prompts to fit within a single batch, balancing between model capacity and scheduling efficiency.
    ///
    /// # Note
    ///
    /// The returned limit affects how prompts are handled during scheduling, potentially leading to truncation or rejection of overly long prompts.
    fn get_prompt_limit(&self) -> usize {
        if self.scheduler_config.enable_chunked_prefill() {
            self.scheduler_config.max_model_len()
        } else {
            self.scheduler_config
                .max_model_len()
                .min(self.scheduler_config.max_num_batched_tokens())
        }
    }

    /// Allocates blocks to a `SequenceGroup` and sets its sequences' status to `Running`.
    ///
    /// This function performs two main tasks:
    /// 1. Allocates memory blocks for the given sequence group using the block manager.
    /// 2. Updates the status of all waiting sequences in the group to running.
    ///
    /// # Arguments
    ///
    /// * `sequence_group` - A mutable reference to the `SequenceGroup` to be allocated and updated.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the allocation and status update are successful, or a `SchedulerError` if there's an issue with block allocation.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The block manager fails to allocate blocks for the sequence group.
    ///
    /// # Side Effects
    ///
    /// - Modifies the internal state of the block manager by allocating blocks.
    /// - Updates the status of sequences within the given `sequence_group`.
    ///
    /// # Example
    ///
    /// ```
    /// let mut scheduler = Scheduler::new(/* ... */);
    /// let mut sequence_group = SequenceGroup::new(/* ... */);
    ///
    /// match scheduler.allocate_and_set_running(&mut sequence_group) {
    ///     Ok(()) => println!("Allocation successful and sequences set to running"),
    ///     Err(e) => eprintln!("Failed to allocate: {}", e),
    /// }
    /// ```
    fn allocate_and_set_running(
        &mut self,
        sequence_group: &mut SequenceGroup,
    ) -> Result<(), SchedulerError> {
        let _enter = self.span.enter();
        trace!(
            "Allocating blocks for sequence group with id = {}",
            sequence_group.request_id
        );
        self.block_manager.allocate(sequence_group)?;
        sequence_group.sequences.iter_mut().for_each(|(_, s)| {
            let mut sequence_guard_lock = s.write().unwrap();
            let sequence_status = sequence_guard_lock.get_sequence_status();
            if sequence_status == SequenceStatus::Waiting {
                sequence_guard_lock.set_sequence_status(SequenceStatus::Running)
            }
        });
        Ok(())
    }

    /// Removes finished sequences from the running queue.
    ///
    /// This method filters out any sequence groups that have completed their
    /// generation (i.e., are marked as finished) from the `running` queue.
    /// This helps to free up resources and maintain an accurate list of
    /// actively running sequences.
    ///
    /// # Effects
    ///
    /// - Modifies `self.running` to only contain unfinished sequence groups.
    /// - Does not directly free block table resources; this should be handled
    ///   separately by the block manager.
    ///
    /// # Note
    ///
    /// This method should be called periodically to clean up the running queue,
    /// typically after each generation step or when checking for completed sequences.
    #[instrument(skip(self))]
    pub fn free_finished_sequence(&mut self) {
        let _enter = self.span.enter();
        trace!("Freeing finished sequence");
        self.running = self
            .running
            .iter()
            .filter_map(|s| {
                if !s.is_finished() {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
    }
}

/// A `SequenceGroup` that has been scheduled
#[derive(Clone, Debug)]
pub struct ScheduledSequenceGroup {
    /// The `SequenceGroup` that has been scheduled
    pub scheduled_group: SequenceGroup,
    /// The number of tokens to process in the next iteration
    ///
    /// This value is:
    /// - 1 for decoding (generating a single new token)
    /// - Equal to the number of prompt tokens for a full prefill
    /// - Smaller than the total prompt tokens if prefill is chunked
    pub token_chunk_size: usize,
}

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("Block space manager error: `{0}`")]
    BlockSpaceManagerError(#[from] BlockSpaceManagerError),
    #[error("Empty scheduling")]
    EmptyScheduling,
    #[error("Zero number of new tokens to schedule")]
    ZeroNewTokensToSchedule,
    #[error("Invalid number of sequences for recompute: `{0}`")]
    InvalidNumberSequencesForRecompute(usize),
    #[error("Not enough block space for swap out")]
    NotEnoughBlockSpaceForSwapOut,
    #[error("Invalid number of waiting sequences for request `{request_id}`: `{num_sequences}`")]
    InvalidNumberWaitingSequence {
        request_id: String,
        num_sequences: usize,
    },
    #[error("Invalid number of new tokens, got `{num_new_tokens}`, but it should be `{num_prompt_tokens}`")]
    InvalidNumberOfNewTokens {
        num_prompt_tokens: usize,
        num_new_tokens: usize,
    },
    #[error("Invalid scheduler output: `{0}`")]
    InvalidSchedulerOutput(String),
    #[error("Invalid number of budget tokens: `{0}`")]
    InvalidNumberBudgetTokens(String),
    #[error("Invalid number of sequences: `{0}`")]
    InvalidNumberBudgetSequences(String),
    #[error("Chunked prefills not allowed: `{0}`")]
    ChunkedPrefillsNotAllowed(String),
    #[error("Invalid prefill sequence: `{0}`")]
    InvalidPrefillSequences(String),
    #[error("Sequence error: `{0}`")]
    SequenceError(#[from] SequenceError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        policy::FcfsPolicy,
        sequence::{tests::create_dummy_prompt, LogProb},
    };
    use std::time::Duration;

    fn get_sequence_groups(scheduler_outputs: &SchedulerOutputs) -> Vec<SequenceGroup> {
        scheduler_outputs
            .scheduled_sequence_groups
            .iter()
            .map(|s| s.scheduled_group.clone())
            .collect()
    }

    fn schedule_and_update_computed_tokens(
        scheduler: &mut Scheduler<FcfsPolicy>,
    ) -> (Vec<Arc<SequenceGroupMetadata>>, SchedulerOutputs) {
        let (metadatas, mut outputs) = scheduler.schedule().expect("Failed to schedule");
        for (s, meta) in outputs
            .scheduled_sequence_groups
            .iter_mut()
            .zip(metadatas.iter())
        {
            s.scheduled_group
                .update_num_computed_tokens(meta.token_chunk_size)
                .expect("Failed to updated number of computed tokens");
        }
        (metadatas, outputs)
    }

    fn add_new_token(scheduler: &mut Scheduler<FcfsPolicy>, token_id: u32) {
        scheduler.running.iter_mut().for_each(|s| {
            for (_, sequence) in s.sequences.iter() {
                {
                    sequence
                        .write()
                        .unwrap()
                        .add_token_id(
                            token_id,
                            HashMap::from_iter([(token_id, LogProb::new(0.5, None, None))]),
                        )
                        .expect("Failed to add token id");
                }
            }
        });
    }

    fn add_new_token_to_output(out: &SchedulerOutputs, token_id: u32) {
        let sequence_groups = get_sequence_groups(out);
        for sequence_group in sequence_groups {
            for sequence in sequence_group.sequences.values() {
                sequence
                    .write()
                    .unwrap()
                    .add_token_id(
                        token_id,
                        HashMap::from_iter([(token_id, LogProb::new(0.5, None, None))]),
                    )
                    .expect("Failed to add token id");
            }
        }
    }

    fn add_new_token_to_sequence_group(
        token_chunk_size: usize,
        sequence_group: &mut SequenceGroup,
        token_id: u32,
    ) {
        sequence_group
            .update_num_computed_tokens(token_chunk_size)
            .expect("Failed ot update number compute tokens");
        for sequence in sequence_group.sequences.values() {
            sequence
                .write()
                .unwrap()
                .add_token_id(
                    token_id,
                    HashMap::from_iter([(token_id, LogProb::new(0.5, None, None))]),
                )
                .expect("Failed to add new token to sequence group")
        }
    }

    fn add_token_budget(
        budget: &mut SchedulingBudget,
        num_batched_tokens: usize,
        num_current_sequences: usize,
    ) {
        let (_, mock_seq_group) = create_dummy_prompt(10, 60, None, 1);
        budget.add_num_batched_tokens(mock_seq_group.request_id.clone(), num_batched_tokens);
        budget.add_number_sequences(mock_seq_group.request_id, num_current_sequences);
    }

    #[test]
    fn test_scheduler_add_sequence_group() {
        const BLOCK_SIZE: usize = 4;
        const GPU_MEMORY_UTILIZATION: f32 = 1.0;
        const NUM_CPU_BLOCKS: usize = 4;
        const NUM_GPU_BLOCKS: usize = 4;
        const SWAP_SPACE: usize = 1;

        const MAX_NUM_BATCHED_TOKENS: usize = 100;
        const MAX_NUM_SEQUENCES: usize = 64;
        const MAX_MODEL_LEN: usize = 1;
        let scheduler_config = SchedulerConfig::new(
            MAX_NUM_BATCHED_TOKENS,
            MAX_NUM_SEQUENCES,
            MAX_MODEL_LEN,
            0.0,
            false,
            0,
        )
        .expect("Failed to generate `SchedulerConfig`");

        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            GPU_MEMORY_UTILIZATION,
            SWAP_SPACE,
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate `CacheConfig`");

        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to generate `Scheduler`");

        // adds multiple sequence groups to `Scheduler` instance
        let num_sequence_group: usize = 4;
        for i in 0..num_sequence_group {
            let (_, sequence_group) = create_dummy_prompt(i as u64, BLOCK_SIZE, None, 1);
            scheduler.add_sequence_group(sequence_group);
            assert_eq!(scheduler.num_unfinished_sequeces(), i + 1);
        }
    }

    #[test]
    fn test_scheduler_abort_sequence_group() {
        const BLOCK_SIZE: usize = 4;
        const GPU_MEMORY_UTILIZATION: f32 = 1.0;
        const NUM_CPU_BLOCKS: usize = 4;
        const NUM_GPU_BLOCKS: usize = 4;
        const SWAP_SPACE: usize = 1;

        const MAX_NUM_BATCHED_TOKENS: usize = 100;
        const MAX_NUM_SEQUENCES: usize = 64;
        const MAX_MODEL_LEN: usize = 1;
        let scheduler_config = SchedulerConfig::new(
            MAX_NUM_BATCHED_TOKENS,
            MAX_NUM_SEQUENCES,
            MAX_MODEL_LEN,
            0.0,
            false,
            0,
        )
        .expect("Failed to generate `SchedulerConfig`");

        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            GPU_MEMORY_UTILIZATION,
            SWAP_SPACE,
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate `CacheConfig`");

        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to generate `Scheduler`");

        // adds multiple sequence groups to `Scheduler` instance
        let num_sequence_group: usize = 4;
        let mut requests_ids = HashSet::new();
        for i in 0..num_sequence_group {
            let (_, sequence_group) = create_dummy_prompt(i as u64, BLOCK_SIZE, None, 1);
            scheduler.add_sequence_group(sequence_group);
            requests_ids.insert(format!("{i}"));
        }

        assert_eq!(scheduler.num_unfinished_sequeces(), num_sequence_group);
        scheduler
            .abort_sequence_groups(requests_ids.into_iter())
            .expect("Failed to abort sequence groups");
        assert_eq!(scheduler.num_unfinished_sequeces(), 0);
    }

    #[test]
    fn test_scheduler_schedule_simple() {
        const BLOCK_SIZE: usize = 4;
        const GPU_MEMORY_UTILIZATION: f32 = 1.0;
        const NUM_CPU_BLOCKS: usize = 8;
        const NUM_GPU_BLOCKS: usize = 8;
        const SWAP_SPACE: usize = 1;

        const MAX_NUM_BATCHED_TOKENS: usize = 100;
        const MAX_NUM_SEQUENCES: usize = 4;
        const MAX_MODEL_LEN: usize = 16;
        let scheduler_config = SchedulerConfig::new(
            MAX_NUM_BATCHED_TOKENS,
            MAX_NUM_SEQUENCES,
            MAX_MODEL_LEN,
            0.0,
            false,
            0,
        )
        .expect("Failed to generate `SchedulerConfig`");

        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            GPU_MEMORY_UTILIZATION,
            SWAP_SPACE,
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate `CacheConfig`");

        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to generate `Scheduler`");

        let num_sequence_groups = 4;
        let mut running = vec![];

        for i in 0..num_sequence_groups {
            let (_, sequence_group) = create_dummy_prompt(i as u64, BLOCK_SIZE, None, 1);
            scheduler.add_sequence_group(sequence_group.clone());
            running.push(sequence_group);
        }

        // Schedule sequence groups prompts.
        let num_tokens = BLOCK_SIZE * num_sequence_groups;
        let (sequence_groups_metadata, outputs) =
            schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&outputs);
        assert_eq!(
            sequence_groups
                .iter()
                .map(|s| s.request_id.clone())
                .collect::<HashSet<_>>(),
            running
                .iter()
                .map(|s| s.request_id.clone())
                .collect::<HashSet<_>>()
        );
        for sequence_group in sequence_groups.iter() {
            let running_sequence_group = running
                .iter()
                .find(|s| s.request_id == sequence_group.request_id)
                .unwrap()
                .clone();
            assert_eq!(
                running_sequence_group.next_token_chooser_params(),
                sequence_group.next_token_chooser_params()
            );
            assert_eq!(
                running_sequence_group.sequences.keys().collect::<Vec<_>>(),
                sequence_group.sequences.keys().collect::<Vec<_>>()
            );
        }
        assert_eq!(outputs.num_batched_tokens, num_tokens);
        assert!(
            outputs.blocks_to_copy.is_empty()
                && outputs.blocks_to_swap_in.is_empty()
                && outputs.blocks_to_swap_out.is_empty()
        );
        assert_eq!(sequence_groups_metadata.len(), num_sequence_groups);

        // add a new token for each running `SequenceGroup`'s internal `Sequence`
        add_new_token(&mut scheduler, 1);

        // Schedule seq groups generation.
        let (sequence_groups_metadata, outputs) =
            schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&outputs);
        assert_eq!(
            sequence_groups
                .iter()
                .map(|s| s.request_id.clone())
                .collect::<HashSet<_>>(),
            running
                .iter()
                .map(|s| s.request_id.clone())
                .collect::<HashSet<_>>()
        );
        for sequence_group in sequence_groups.iter() {
            let running_sequence_group = running
                .iter()
                .find(|s| s.request_id == sequence_group.request_id)
                .unwrap()
                .clone();
            assert_eq!(
                running_sequence_group.next_token_chooser_params(),
                sequence_group.next_token_chooser_params()
            );
            assert_eq!(
                running_sequence_group.stopping_params(),
                sequence_group.stopping_params(),
            );
            assert_eq!(
                running_sequence_group.sequences.keys().collect::<Vec<_>>(),
                sequence_group.sequences.keys().collect::<Vec<_>>()
            );
        }
        assert_eq!(outputs.num_batched_tokens, num_sequence_groups);
        assert!(
            outputs.blocks_to_copy.is_empty()
                && outputs.blocks_to_swap_in.is_empty()
                && outputs.blocks_to_swap_out.is_empty()
        );
        assert_eq!(sequence_groups_metadata.len(), num_sequence_groups);

        add_new_token(&mut scheduler, 1)
    }

    #[test]
    /// Verify running batched tokens are not applied to prefill requests.
    fn test_scheduler_prefill_prioritized() {
        const BLOCK_SIZE: usize = 4;
        const GPU_MEMORY_UTILIZATION: f32 = 1.0;
        const NUM_CPU_BLOCKS: usize = 2;
        const NUM_GPU_BLOCKS: usize = 2;
        const SWAP_SPACE: usize = 1;

        const MAX_NUM_BATCHED_TOKENS: usize = 30;
        const MAX_NUM_SEQUENCES: usize = 2;
        const MAX_MODEL_LEN: usize = 30;
        let scheduler_config = SchedulerConfig::new(
            MAX_NUM_BATCHED_TOKENS,
            MAX_NUM_SEQUENCES,
            MAX_MODEL_LEN,
            0.0,
            false,
            0,
        )
        .expect("Failed to generate `SchedulerConfig`");

        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            GPU_MEMORY_UTILIZATION,
            SWAP_SPACE,
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate `CacheConfig`");

        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to generate `Scheduler`");

        // Add seq groups to scheduler
        let (_, sequence_group_a) = create_dummy_prompt(1, 1, None, 1);
        scheduler.add_sequence_group(sequence_group_a.clone());

        // Schedule seq groups prompts
        let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let out_sequence_groups = get_sequence_groups(&out);
        assert_eq!(out_sequence_groups.len(), 1);
        assert_eq!(
            out_sequence_groups[0].request_id,
            sequence_group_a.request_id
        );
        assert_eq!(out_sequence_groups[0].sequences.values().len(), 1);
        let sequence = out_sequence_groups[0]
            .sequences
            .values()
            .next()
            .unwrap()
            .read()
            .unwrap();
        let sequence_a = sequence_group_a
            .sequences
            .values()
            .next()
            .unwrap()
            .read()
            .unwrap();

        assert_eq!(sequence.sequence_data, sequence_a.sequence_data);
        assert_eq!(
            sequence.get_num_new_tokens(),
            sequence_a.get_num_new_tokens()
        );
        assert_eq!(sequence.get_last_token_id(), sequence_a.get_last_token_id());
        assert_eq!(
            sequence.get_num_new_tokens(),
            sequence_a.get_num_new_tokens()
        );
        assert_eq!(
            sequence.get_num_total_logical_token_blocks(),
            sequence_a.get_num_total_logical_token_blocks()
        );
        assert_eq!(sequence.get_token_ids(), sequence_a.get_token_ids());
        assert_eq!(
            sequence.get_sequence_status(),
            sequence_a.get_sequence_status()
        );

        // Add a new prefill request B
        let (_, sequence_group_b) = create_dummy_prompt(2, 30, None, 1);
        scheduler.add_sequence_group(sequence_group_b.clone());

        // Verify prefill requests are prioritized. Since max_batched_num_tokens
        // is 1, new prefill request has to be scheduled first.
        let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let out_sequence_groups = get_sequence_groups(&out);
        assert_eq!(out_sequence_groups.len(), 1);
        assert_eq!(
            out_sequence_groups[0].request_id,
            sequence_group_b.request_id
        );
        assert_eq!(out_sequence_groups[0].sequences.values().len(), 1);
        let sequence = out_sequence_groups[0]
            .sequences
            .values()
            .next()
            .unwrap()
            .read()
            .unwrap();
        let sequence_b = sequence_group_b
            .sequences
            .values()
            .next()
            .unwrap()
            .read()
            .unwrap();

        assert_eq!(sequence.sequence_data, sequence_b.sequence_data);
        assert_eq!(
            sequence.get_num_new_tokens(),
            sequence_b.get_num_new_tokens()
        );
        assert_eq!(sequence.get_last_token_id(), sequence_b.get_last_token_id());
        assert_eq!(
            sequence.get_num_new_tokens(),
            sequence_b.get_num_new_tokens()
        );
        assert_eq!(
            sequence.get_num_total_logical_token_blocks(),
            sequence_b.get_num_total_logical_token_blocks()
        );
        assert_eq!(sequence.get_token_ids(), sequence_b.get_token_ids());
        assert_eq!(
            sequence.get_sequence_status(),
            sequence_b.get_sequence_status()
        );
    }

    #[test]
    fn test_scheduler_schedule_preempt_abort() {
        const BLOCK_SIZE: usize = 4;
        const GPU_MEMORY_UTILIZATION: f32 = 1.0;
        const NUM_CPU_BLOCKS: usize = 2;
        const NUM_GPU_BLOCKS: usize = 2;
        const SWAP_SPACE: usize = 1;

        const MAX_NUM_BATCHED_TOKENS: usize = 64;
        const MAX_NUM_SEQUENCES: usize = 2;
        const MAX_MODEL_LEN: usize = 16;
        let scheduler_config = SchedulerConfig::new(
            MAX_NUM_BATCHED_TOKENS,
            MAX_NUM_SEQUENCES,
            MAX_MODEL_LEN,
            0.0,
            false,
            0,
        )
        .expect("Failed to generate `SchedulerConfig`");

        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            GPU_MEMORY_UTILIZATION,
            SWAP_SPACE,
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate `CacheConfig`");

        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to generate `Scheduler`");

        // Add seq groups to scheduler
        let (_, sequence_group_a) = create_dummy_prompt(1, BLOCK_SIZE, None, 1);
        let (_, sequence_group_b) = create_dummy_prompt(2, BLOCK_SIZE, None, 1);

        scheduler.add_sequence_group(sequence_group_a.clone());
        scheduler.add_sequence_group(sequence_group_b.clone());

        // Schedule seq groups prompts
        let (sequence_group_metadata, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&out);
        assert_eq!(sequence_groups.len(), 2);
        assert_eq!(
            sequence_groups[0].request_id,
            sequence_group_a.request_id.clone()
        );
        assert_eq!(sequence_groups[0].sequences.values().len(), 1);

        {
            let sequence = sequence_groups[0]
                .sequences
                .values()
                .next()
                .unwrap()
                .read()
                .unwrap();
            let sequence_a = sequence_group_a
                .sequences
                .values()
                .next()
                .unwrap()
                .read()
                .unwrap();

            assert_eq!(sequence.sequence_data, sequence_a.sequence_data);
            assert_eq!(
                sequence.get_num_new_tokens(),
                sequence_a.get_num_new_tokens()
            );
            assert_eq!(sequence.get_last_token_id(), sequence_a.get_last_token_id());
            assert_eq!(
                sequence.get_num_new_tokens(),
                sequence_a.get_num_new_tokens()
            );
            assert_eq!(
                sequence.get_num_total_logical_token_blocks(),
                sequence_a.get_num_total_logical_token_blocks()
            );
            assert_eq!(sequence.get_token_ids(), sequence_a.get_token_ids());
            assert_eq!(
                sequence.get_sequence_status(),
                sequence_a.get_sequence_status()
            );

            assert_eq!(
                sequence_groups[1].request_id,
                sequence_group_b.request_id.clone()
            );
            assert_eq!(sequence_groups[1].sequences.values().len(), 1);
        }

        assert_eq!(
            sequence_groups[1].request_id,
            sequence_group_b.request_id.clone()
        );
        assert_eq!(sequence_groups[1].sequences.values().len(), 1);

        {
            let sequence = sequence_groups[1]
                .sequences
                .values()
                .next()
                .unwrap()
                .read()
                .unwrap();
            let sequence_b = sequence_group_b
                .sequences
                .values()
                .next()
                .unwrap()
                .read()
                .unwrap();

            assert_eq!(sequence.sequence_data, sequence_b.sequence_data);
            assert_eq!(
                sequence.get_num_new_tokens(),
                sequence_b.get_num_new_tokens()
            );
            assert_eq!(sequence.get_last_token_id(), sequence_b.get_last_token_id());
            assert_eq!(
                sequence.get_num_new_tokens(),
                sequence_b.get_num_new_tokens()
            );
            assert_eq!(
                sequence.get_num_total_logical_token_blocks(),
                sequence_b.get_num_total_logical_token_blocks()
            );
            assert_eq!(sequence.get_token_ids(), sequence_b.get_token_ids());
            assert_eq!(
                sequence.get_sequence_status(),
                sequence_b.get_sequence_status()
            );

            assert_eq!(out.num_batched_tokens, BLOCK_SIZE * 2);
            assert!(
                out.blocks_to_copy.is_empty()
                    && out.blocks_to_swap_in.is_empty()
                    && out.blocks_to_swap_out.is_empty()
            );
            assert_eq!(sequence_group_metadata.len(), 2);
            assert_eq!(scheduler.num_unfinished_sequeces(), 2);
        }

        // Append "generated" tokens, allowing the sequence to mark prompt tokens as
        // processed
        add_new_token(&mut scheduler, 1);

        // Schedule sequence groups generation and preempt sequence b
        let (sequence_group_metadata, out) = schedule_and_update_computed_tokens(&mut scheduler);

        let sequence_groups = get_sequence_groups(&out);
        assert_eq!(sequence_groups.len(), 1);
        assert_eq!(out.preempted, 1);
        assert_eq!(sequence_group_metadata.len(), 1);
        assert_eq!(scheduler.num_unfinished_sequeces(), 2);
        assert_eq!(out.num_batched_tokens, 1);
        assert!(
            out.blocks_to_copy.is_empty()
                && out.blocks_to_swap_in.is_empty()
                && out.blocks_to_swap_out.is_empty()
        );
        assert_eq!(scheduler.waiting.len(), 1);
        assert_eq!(scheduler.running.len(), 1);
        assert_eq!(scheduler.swapped.len(), 0);

        // Abort sequence a and reschedule sequence b with recomputation
        scheduler
            .abort_sequence_group("1".to_string())
            .expect("Failed to abort sequence group");
        let (sequence_group_metadata, out) = schedule_and_update_computed_tokens(&mut scheduler);

        let sequences_groups = get_sequence_groups(&out);

        assert_eq!(sequences_groups.len(), 1);
        assert_eq!(sequences_groups[0].request_id, sequence_group_b.request_id);
        assert_eq!(scheduler.waiting.len(), 0);
        assert_eq!(scheduler.running.len(), 1);
        assert_eq!(scheduler.swapped.len(), 0);

        assert_eq!(sequence_group_metadata.len(), 1);
    }

    #[test]
    fn test_scheduler_max_seqs() {
        const BLOCK_SIZE: usize = 4;
        const NUM_SEQ_GROUP: usize = 4;
        const MAX_SEQ_GROUP: usize = 2;
        const MAX_MODEL_LEN: usize = 16;
        const NUM_CPU_BLOCKS: usize = 8;
        const NUM_GPU_BLOCKS: usize = 8;
        let scheduler_config =
            SchedulerConfig::new(64, MAX_SEQ_GROUP, MAX_MODEL_LEN, 0.0, false, 0)
                .expect("Failed to get schedule config");
        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            1.0,
            1,
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to generate cache config");

        let mut scheduler =
            Scheduler::new(cache_config, scheduler_config).expect("Failed to generate scheduler");
        let mut all_sequence_groups = vec![];

        // Add sequence groups to the scheduler
        for i in 0..NUM_SEQ_GROUP {
            let (_, seq_group) = create_dummy_prompt(i as u64, BLOCK_SIZE, None, 1);
            all_sequence_groups.push(seq_group);
        }

        // Append sequence group 1
        scheduler.add_sequence_group(all_sequence_groups[0].clone());

        // Schedule sequence group prompts
        let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&out);
        assert_eq!(sequence_groups.len(), 1);
        assert_eq!(
            sequence_groups[0].request_id,
            all_sequence_groups[0].request_id
        );

        add_new_token_to_output(&out, 1);

        // Schedule sequence groups generation
        let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&out);
        assert_eq!(sequence_groups.len(), 1);
        assert_eq!(
            sequence_groups[0].request_id,
            all_sequence_groups[0].request_id
        );

        // Append two more sequence groups
        scheduler.add_sequence_group(all_sequence_groups[1].clone());
        scheduler.add_sequence_group(all_sequence_groups[2].clone());

        // Schedule sequence group prompts
        // Only 1 sequence group should be scheduled since `max_seq_group` is 2
        // and one is prompting
        let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
        let sequence_groups = get_sequence_groups(&out);
        assert_eq!(sequence_groups.len(), 1);
        assert_eq!(
            sequence_groups[0].request_id,
            all_sequence_groups[1].request_id
        )
    }

    #[test]
    fn test_scheduler_delay_factor() {
        const BLOCK_SIZE: usize = 4;
        const NUM_CPU_BLOCKS: usize = 8;
        const NUM_GPU_BLOCKS: usize = 8;
        let scheduler_config = SchedulerConfig::new(100, 64, 16, 0.5, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(
            BLOCK_SIZE,
            1.0,
            1,
            None,
            None,
            NUM_CPU_BLOCKS,
            NUM_GPU_BLOCKS,
        )
        .expect("Failed to get cache config");
        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to get scheduler");

        // schedule first prompt
        let (_, sequence_group) = create_dummy_prompt(0, BLOCK_SIZE, None, 1);
        scheduler.add_sequence_group(sequence_group.clone());
        let (sequence_group_meta, out) = schedule_and_update_computed_tokens(&mut scheduler);
        assert!(out.number_prefill_groups > 0);
        assert_eq!(sequence_group_meta[0].request_id(), "0".to_string());

        add_new_token_to_output(&out, 1);

        // wait for a second before scheduling next prompt
        std::thread::sleep(Duration::from_secs(1));

        let (_, sequence_group) = create_dummy_prompt(1, BLOCK_SIZE, None, 1);
        scheduler.add_sequence_group(sequence_group.clone());

        // second prompt should NOT be scheduled
        let (sequence_group_meta, out) = schedule_and_update_computed_tokens(&mut scheduler);
        assert_eq!(out.number_prefill_groups, 0);
        assert_eq!(sequence_group_meta[0].request_id(), "0".to_string());

        add_new_token(&mut scheduler, 1);

        // wait for more than 0.5 seconds and try again
        std::thread::sleep(Duration::from_millis(600));
        let (sequence_group_meta, out) = schedule_and_update_computed_tokens(&mut scheduler);
        assert!(out.number_prefill_groups > 0);
        assert_eq!(sequence_group_meta[0].request_id(), "1".to_string());

        add_new_token(&mut scheduler, 1);
    }

    // #[test]
    // fn test_swapped_out_prioritized() {
    //     const BLOCK_SIZE: usize = 4;
    //     let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
    //         .expect("Failed to get scheduler config");
    //     let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, "auto".into(), None, None, 8, 8)
    //         .expect("Failed to get cache config");
    //     let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
    //         .expect("Failed to get scheduler");

    //     // best_of = 2 * 3 == 6 sequences
    //     for i in 0..3 {
    //         let (_, sequence_group) = create_dummy_prompt(i, 60, None, false, 2);
    //         scheduler.add_sequence_group(sequence_group);
    //     }

    //     let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);

    //     // prefill is scheduled now
    //     assert_eq!(out.scheduled_sequence_groups.len(), 3);
    //     add_new_token_to_output(&out, 1);

    //     // The last request should be swapped out
    //     let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
    //     assert_eq!(out.scheduled_sequence_groups.len(), 2);
    //     assert_eq!(out.num_batched_tokens, 2);
    //     assert!(!out.blocks_to_swap_out.is_empty());
    //     assert!(out.blocks_to_swap_in.is_empty());

    //     // Add 1 more task. Swap should be prioritized over prefill
    //     let (_, sequence_group) = create_dummy_prompt(2, 60, None, false, 2);
    //     scheduler.add_sequence_group(sequence_group);
    //     let (_, out) = schedule_and_update_computed_tokens(&mut scheduler);
    //     add_new_token_to_output(&out, 1);

    //     // assert_eq!(out.scheduled_sequence_groups.len(), 1);
    //     // assert_eq!(out.num_batched_tokens, 3);
    //     // assert!(out.blocks_to_swap_in.is_empty());
    //     // assert!(out.blocks_to_swap_out.is_empty())
    // }

    #[test]
    /// Test prompt longer than max_prompt_len is aborted
    fn test_prefill_schedule_max_prompt_len() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 30, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to get scheduler");

        let (_, seq_group) = create_dummy_prompt(0, 60, None, 1);
        let waiting = VecDeque::from_iter([seq_group]);
        let mut budget = SchedulingBudget::new(10000, 10000);

        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting, &mut budget, false)
            .expect("Failed to schedule prefills");

        assert_eq!(output.ignored_sequence_groups.len(), 1);
        assert_eq!(output.sequence_groups.len(), 0);
        assert_eq!(budget.num_batched_tokens, 0);
        assert_eq!(budget.num_curr_seqs, 0);
        assert_eq!(remaining_waiting.len(), 0);
    }

    #[test]
    /// Test token budget respected.
    fn test_prefill_schedule_token_budget() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");

        let mut waiting: VecDeque<SequenceGroup> = VecDeque::new();
        let mut budget = SchedulingBudget::new(0, 10_000);

        for i in 0..2 {
            let (_, sequence_group) = create_dummy_prompt(i, 60, None, 1);
            waiting.push_back(sequence_group);
        }

        // 0 token budget == nothing is scheduled
        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting.clone(), &mut budget, false)
            .expect("Failed to run schedule prefills");
        assert_eq!(output.ignored_sequence_groups.len(), 0);
        assert_eq!(output.sequence_groups.len(), 0);
        assert_eq!(budget.num_batched_tokens, 0);
        assert_eq!(budget.num_curr_seqs, 0);
        assert_eq!(remaining_waiting.len(), 2);

        // 60 token budget == 1 request scheduled
        let mut budget = SchedulingBudget::new(60, 10_000);
        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting, &mut budget, false)
            .expect("Failed to run schedule prefills");
        assert_eq!(output.ignored_sequence_groups.len(), 0);
        assert_eq!(output.sequence_groups.len(), 1);
        assert_eq!(budget.num_batched_tokens, 60);
        assert_eq!(budget.num_curr_seqs, 1);
        assert_eq!(remaining_waiting.len(), 1);

        // Test when current_batched_tokens is respected
        let mut scheduler = Scheduler::<FcfsPolicy>::new(cache_config, scheduler_config)
            .expect("Failed to get scheduler");
        let mut waiting = VecDeque::new();
        let mut budget = SchedulingBudget::new(60, 10_000);
        add_token_budget(&mut budget, 30, 0);
        let (_, sequence_group) = create_dummy_prompt(1, 60, None, 1);
        // Cannot schedule a prompt that doesn't fit the budget.
        waiting.push_back(sequence_group);
        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting.clone(), &mut budget, false)
            .expect("Failed to schedule prefills");

        assert_eq!(output.ignored_sequence_groups.len(), 0);
        assert_eq!(output.sequence_groups.len(), 0);
        assert_eq!(budget.num_batched_tokens, 30);
        assert_eq!(budget.num_curr_seqs, 0);
        assert_eq!(remaining_waiting.len(), 1);

        let mut budget = SchedulingBudget::new(90, 10_000);
        add_token_budget(&mut budget, 30, 0);

        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting, &mut budget, false)
            .expect("Failed to schedule prefills");
        assert_eq!(output.sequence_groups.len(), 1);
        assert_eq!(budget.num_batched_tokens, 90);
        assert_eq!(budget.num_curr_seqs, 1);
        assert_eq!(remaining_waiting.len(), 0);
    }

    #[test]
    /// Test max seq respected
    fn test_prefill_schedule_max_seqs() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");
        let mut budget = SchedulingBudget::new(10_000, 2);

        let mut waiting = VecDeque::new();
        for i in 0..3 {
            let (_, sequence_group) = create_dummy_prompt(i, 60, None, 1);
            waiting.push_back(sequence_group);
        }

        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting.clone(), &mut budget, false)
            .expect("Failed to run schedule prefills");

        assert_eq!(output.ignored_sequence_groups.len(), 0);
        assert_eq!(output.sequence_groups.len(), 2);
        assert_eq!(budget.num_batched_tokens, 120);
        assert_eq!(budget.num_curr_seqs, 2);
        assert_eq!(remaining_waiting.len(), 1);

        // Verify curr_num_seqs is respected
        let mut waiting = VecDeque::new();
        let mut budget = SchedulingBudget::new(10_000, 2);
        add_token_budget(&mut budget, 0, 2);

        let (_, sequence_group) = create_dummy_prompt(2, 60, None, 1);
        waiting.push_back(sequence_group);

        let (remaining_waiting, output) = scheduler
            .schedule_prefills(waiting.clone(), &mut budget, false)
            .expect("Failed to run schedule prefills");
        assert_eq!(output.ignored_sequence_groups.len(), 0);
        assert_eq!(output.sequence_groups.len(), 0);
        assert_eq!(budget.num_batched_tokens, 0);
        assert_eq!(budget.num_curr_seqs, 2);
        assert_eq!(remaining_waiting.len(), 1);
    }

    #[test]
    /// Test sequence cannot be scheduled due to block manager has no capacity
    fn test_prefill_schedule_no_block_manager_capacity() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");
        let mut budget = SchedulingBudget::new(10_000, 10_000);

        let mut waiting = VecDeque::new();
        for i in 0..3 {
            let (_, sequence_group) = create_dummy_prompt(i, 60, None, 1);
            waiting.push_back(sequence_group);
        }
        let (remaining_waiting, output) = scheduler
            .mock_schedule_prefill(waiting, &mut budget, false, AllocationStatus::Later)
            .expect("Failed to schedule prefill");

        assert_eq!(output.ignored_sequence_groups.len(), 0);
        assert_eq!(output.sequence_groups.len(), 0);
        assert_eq!(budget.num_batched_tokens, 0);
        assert_eq!(budget.num_curr_seqs, 0);
        assert_eq!(remaining_waiting.len(), 3);

        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");
        let mut budget = SchedulingBudget::new(10_000, 10_000);

        let mut waiting = VecDeque::new();
        for i in 0..3 {
            let (_, sequence_group) = create_dummy_prompt(i, 60, None, 1);
            waiting.push_back(sequence_group);
        }

        let (remaining_waiting, output) = scheduler
            .mock_schedule_prefill(waiting, &mut budget, false, AllocationStatus::Never)
            .expect("Failed to schedule prefill");

        assert_eq!(output.ignored_sequence_groups.len(), 3);
        assert_eq!(output.sequence_groups.len(), 0);
        assert_eq!(budget.num_batched_tokens, 0);
        assert_eq!(budget.num_curr_seqs, 0);
        assert_eq!(remaining_waiting.len(), 0);
    }

    #[test]
    /// Test sequence cannot be scheduled due to block manager has no capacity
    fn test_decode_schedule_preempted() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");

        let mut running = VecDeque::new();
        for i in 0..3 {
            let (_, mut sequence_group) = create_dummy_prompt(i, 60, None, 1);
            scheduler
                .allocate_and_set_running(&mut sequence_group)
                .expect("Failed to allocate and set running");
            add_new_token_to_sequence_group(60, &mut sequence_group, 1);
            running.push_back(sequence_group);
        }

        // 1 cannot be scheduled, and the lowest priority (request 2)
        // should be preempted. 1 will also be preempted.
        let mut budget = SchedulingBudget::new(10_000, 10_000);
        let (remaining_running, output) = scheduler
            .mock_schedule_running(running, &mut budget, false, "1", None)
            .expect("Failed to run mock schedule running");

        assert_eq!(remaining_running.len(), 0);

        assert_eq!(output.decode_seq_groups.len(), 1);
        assert_eq!(output.prefill_seq_groups.len(), 0);
        assert_eq!(
            output.decode_seq_groups[0].scheduled_group.request_id,
            "0".to_string()
        );
        assert_eq!(output.preempted.len(), 2);
        // Verify budget is updated
        assert_eq!(budget.num_batched_tokens, 1);

        // NOTE: When enable_chunk is false, num_seqs budget is not updated.
        // Both should be preempted, not swapped
        assert!(output.blocks_to_swap_out.is_empty());
        assert!(output.blocks_to_copy.is_empty());
    }

    #[test]
    /// Test best_of > 1 swap out blocks
    fn test_decode_swap_beam_search() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");

        let mut budget = SchedulingBudget::new(10_000, 10_000);

        let mut running = VecDeque::new();
        for i in 0..3 {
            let (_, mut sequence_group) = create_dummy_prompt(i, 60, None, 2);
            scheduler
                .allocate_and_set_running(&mut sequence_group)
                .expect("Failed to allocate and set running");
            add_new_token_to_sequence_group(60, &mut sequence_group, 1);
            running.push_back(sequence_group.clone());
            budget.add_number_sequences(
                sequence_group.request_id.clone(),
                sequence_group.get_max_num_running_seqs(),
            );
            budget.add_num_batched_tokens(
                sequence_group.request_id.clone(),
                sequence_group.get_num_sequences(Some(SequenceStatus::Running)),
            );
        }

        let (remaining_running, output) = scheduler
            .mock_schedule_running(running, &mut budget, false, "2", None)
            .expect("Failed to run mock schedule running");
        // output.blocks_to_swap_out.extend([(5, 7)]);

        assert_eq!(remaining_running.len(), 0);
        assert_eq!(output.decode_seq_groups.len(), 2);
        assert_eq!(output.prefill_seq_groups.len(), 0);
        assert_eq!(
            output.decode_seq_groups[0]
                .scheduled_group
                .request_id
                .as_str(),
            "0"
        );
        assert_eq!(
            output.decode_seq_groups[1]
                .scheduled_group
                .request_id
                .as_str(),
            "1"
        );
        // assert_eq!(output.preempted.len(), 0);
        assert_eq!(output.swapped_out.len(), 1);
        // Budget should reflect preempted requests
        assert_eq!(budget.num_batched_tokens, 2);
        assert_eq!(budget.num_curr_seqs, 4);
        assert!(output.blocks_to_copy.is_empty());
    }

    #[test]
    fn test_schedule_decode_blocks_to_copy_update() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");

        let mut running = VecDeque::new();

        let (_, mut sequence_group) = create_dummy_prompt(1, 60, None, 2);
        scheduler
            .allocate_and_set_running(&mut sequence_group)
            .expect("Failed to allocate and set running");
        add_new_token_to_sequence_group(60, &mut sequence_group, 1);
        running.push_back(sequence_group);

        let mut budget = SchedulingBudget::new(10_000, 10_000);
        let (remaining_running, output) = scheduler
            .mock_schedule_running(running, &mut budget, false, "", Some((2, 3)))
            .expect("Failed to schedule running");

        // The last request should be swapped out.
        assert_eq!(remaining_running.len(), 0);
        assert_eq!(output.decode_seq_groups.len(), 1);
        assert_eq!(output.prefill_seq_groups.len(), 0);
        assert_eq!(output.preempted.len(), 0);
        assert_eq!(output.swapped_out.len(), 0);
        // Nothing is preempted.
        assert_eq!(output.blocks_to_swap_out, HashMap::from_iter([]));
        // Since append_slot returns the source -> dist mapping, it should
        // applied.
        assert_eq!(output.blocks_to_copy, HashMap::from_iter([(2, 3)]));
    }

    #[test]
    fn test_schedule_swapped_simple() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");

        let mut swapped = VecDeque::new();
        let mut blocks_to_swap_out = HashMap::new();

        let (_, mut sequence_group) = create_dummy_prompt(1, 60, None, 2);
        scheduler
            .allocate_and_set_running(&mut sequence_group)
            .expect("Failed to allocate and set running");
        add_new_token_to_sequence_group(60, &mut sequence_group, 1);

        scheduler
            .swap_out(&mut sequence_group, &mut blocks_to_swap_out)
            .expect("Failed to swap out");
        swapped.push_back(sequence_group);

        let mut budget = SchedulingBudget::new(10_000, 10_000);
        let (remaining_swapped, output) = scheduler
            .schedule_swapped(swapped, &mut budget, false)
            .expect("Failed to schedule swapped");

        assert_eq!(remaining_swapped.len(), 0);
        assert_eq!(budget.num_batched_tokens, 1);
        assert_eq!(budget.num_curr_seqs, 2);
        assert_eq!(output.decode_seq_groups.len(), 1);
        assert_eq!(output.prefill_seq_groups.len(), 0);

        // swap in is the reverse of swap out
        let mut blocks_to_swap_in_rev = HashMap::new();
        for (swap_in, swap_out) in output.blocks_to_swap_in {
            blocks_to_swap_in_rev.insert(swap_out, swap_in);
        }
        assert_eq!(blocks_to_swap_out, blocks_to_swap_in_rev)
    }

    #[test]
    fn test_schedule_swapped_max_token_budget() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");

        let mut swapped = VecDeque::new();
        let mut blocks_to_swap_out = HashMap::new();

        for i in 0..2 {
            let (_, mut sequence_group) = create_dummy_prompt(i, 60, None, 2);
            scheduler
                .allocate_and_set_running(&mut sequence_group)
                .expect("Failed to allocate and set running");
            add_new_token_to_sequence_group(60, &mut sequence_group, 1);
            scheduler
                .swap_out(&mut sequence_group, &mut blocks_to_swap_out)
                .expect("Failed to swap out");
            swapped.push_back(sequence_group);
        }

        let mut budget = SchedulingBudget::new(1, 10_000);
        let (remaining_swapped, output) = scheduler
            .schedule_swapped(swapped.clone(), &mut budget, false)
            .expect("Failed to schedule swapped");

        assert_eq!(remaining_swapped.len(), 1);
        assert_eq!(budget.num_batched_tokens, 1);
        assert_eq!(budget.num_curr_seqs, 2);
        assert_eq!(output.decode_seq_groups.len(), 1);
        assert_eq!(output.prefill_seq_groups.len(), 0);

        // Verify that num_batched_tokens are respected
        let mut budget = SchedulingBudget::new(1, 10_000);
        add_token_budget(&mut budget, 1, 0);
        let (remaining_swapped, output) = scheduler
            .schedule_swapped(remaining_swapped, &mut budget, false)
            .expect("Failed to schedule swapped");

        assert_eq!(remaining_swapped.len(), 1);
        assert_eq!(budget.num_batched_tokens, 1);
        assert_eq!(budget.num_curr_seqs, 0);
        assert_eq!(output.decode_seq_groups.len(), 0);
        assert_eq!(output.prefill_seq_groups.len(), 0);
    }

    #[test]
    fn test_schedule_swapped_max_seqs() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");

        let mut swapped = VecDeque::new();
        let mut blocks_to_swap_out = HashMap::new();

        for i in 0..4 {
            let (_, mut sequence_group) = create_dummy_prompt(i, 60, None, 1);
            scheduler
                .allocate_and_set_running(&mut sequence_group)
                .expect("Failed to allocate and set running");
            add_new_token_to_sequence_group(60, &mut sequence_group, 1);
            scheduler
                .swap_out(&mut sequence_group, &mut blocks_to_swap_out)
                .expect("Failed to swap out");
            swapped.push_back(sequence_group);
        }

        let mut budget = SchedulingBudget::new(10_000, 2);
        let (remaining_swapped, output) = scheduler
            .schedule_swapped(swapped, &mut budget, false)
            .expect("Failed to schedule swapped");

        assert_eq!(remaining_swapped.len(), 2);
        assert_eq!(budget.num_batched_tokens, 2);
        assert_eq!(budget.num_curr_seqs, 2);
        assert_eq!(output.decode_seq_groups.len(), 2);
        assert_eq!(output.prefill_seq_groups.len(), 0);

        // Verify that `num_curr_seqs` is respected
        let (remaining_swapped, output) = scheduler
            .schedule_swapped(remaining_swapped, &mut budget, false)
            .expect("Failed to schedule swapped");

        assert_eq!(remaining_swapped.len(), 2);
        assert_eq!(budget.num_batched_tokens, 2);
        assert_eq!(budget.num_curr_seqs, 2);
        assert_eq!(output.decode_seq_groups.len(), 0);
        assert_eq!(output.prefill_seq_groups.len(), 0);
    }

    #[test]
    fn test_schedule_swapped_cannot_swap_in() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");

        let mut swapped = VecDeque::new();
        let mut blocks_to_swap_out = HashMap::new();

        for i in 0..2 {
            let (_, mut sequence_group) = create_dummy_prompt(i, 60, None, 2);
            scheduler
                .allocate_and_set_running(&mut sequence_group)
                .expect("Failed to allocate and set running");
            add_new_token_to_sequence_group(60, &mut sequence_group, 1);
            scheduler
                .swap_out(&mut sequence_group, &mut blocks_to_swap_out)
                .expect("Failed to swap out");
            swapped.push_back(sequence_group);
        }

        let mut budget = SchedulingBudget::new(10_000, 10_000);
        let (remaining_swapped, output) = scheduler
            .mock_schedule_swapped(
                &mut budget,
                swapped,
                false,
                Some(AllocationStatus::Later),
                None,
            )
            .expect("Failed to schedule swapped");

        assert_eq!(remaining_swapped.len(), 2);
        assert_eq!(budget.num_batched_tokens, 0);
        assert_eq!(budget.num_curr_seqs, 0);
        assert_eq!(output.decode_seq_groups.len(), 0);
        assert_eq!(output.prefill_seq_groups.len(), 0);
    }

    #[test]
    fn test_infeasible_swap() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");

        let mut swapped = VecDeque::new();
        let mut blocks_to_swap_out = HashMap::new();

        for i in 0..2 {
            let (_, mut sequence_group) = create_dummy_prompt(i, 60, None, 2);
            scheduler
                .allocate_and_set_running(&mut sequence_group)
                .expect("Failed to allocate and set running");
            add_new_token_to_sequence_group(60, &mut sequence_group, 1);
            scheduler
                .swap_out(&mut sequence_group, &mut blocks_to_swap_out)
                .expect("Failed to swap out");
            swapped.push_back(sequence_group);
        }

        let mut budget = SchedulingBudget::new(10_000, 10_000);
        let (remaining_swapped, output) = scheduler
            .mock_schedule_swapped(
                &mut budget,
                swapped,
                false,
                Some(AllocationStatus::Never),
                None,
            )
            .expect("Failed to schedule swapped");

        assert_eq!(remaining_swapped.len(), 0);
        assert_eq!(output.infeasible_seq_groups.len(), 2);
        assert_eq!(budget.num_batched_tokens, 0);
        assert_eq!(budget.num_curr_seqs, 0);
        assert_eq!(output.decode_seq_groups.len(), 0);
        assert_eq!(output.prefill_seq_groups.len(), 0);
    }

    #[test]
    fn test_schedule_swapped_blocks_to_copy() {
        const BLOCK_SIZE: usize = 4;
        let scheduler_config = SchedulerConfig::new(1000, 1000, 1000, 0.0, false, 0)
            .expect("Failed to get scheduler config");
        let cache_config = CacheConfig::new(BLOCK_SIZE, 1.0, 1, None, None, 8, 8)
            .expect("Failed to get cache config");
        let mut scheduler =
            Scheduler::<FcfsPolicy>::new(cache_config.clone(), scheduler_config.clone())
                .expect("Failed to get scheduler");

        let mut swapped = VecDeque::new();
        let mut blocks_to_swap_out = HashMap::new();

        let (_, mut sequence_group) = create_dummy_prompt(1, 60, None, 2);
        scheduler
            .allocate_and_set_running(&mut sequence_group)
            .expect("Failed to allocate and set running");
        add_new_token_to_sequence_group(60, &mut sequence_group, 1);
        scheduler
            .swap_out(&mut sequence_group, &mut blocks_to_swap_out)
            .expect("Failed to swap out");
        swapped.push_back(sequence_group);

        // The last request should be swapped out

        let mut budget = SchedulingBudget::new(10_000, 10_000);
        let (remaining_swapped, output) = scheduler
            .mock_schedule_swapped(&mut budget, swapped, false, None, Some((2, 3)))
            .expect("Failed to schedule swapped");

        assert_eq!(remaining_swapped.len(), 0);
        assert_eq!(output.decode_seq_groups.len(), 1);
        assert_eq!(output.prefill_seq_groups.len(), 0);
        assert_eq!(output.blocks_to_copy, HashMap::from_iter([(2, 3)]))
    }

    #[test]
    fn test_scheduling_budget() {
        const TOKEN_BUDGET: usize = 4;
        const MAX_SEQS: usize = 4;

        let mut budget = SchedulingBudget::new(TOKEN_BUDGET, MAX_SEQS);
        assert!(budget.can_schedule(1, 1).expect("Failed to schedule"));
        assert!(budget.can_schedule(4, 4).expect("Failed to schedule"));
        assert!(!budget.can_schedule(1, 5).expect("Failed to schedule"));
        assert!(!budget.can_schedule(5, 1).expect("Failed to schedule"));
        assert!(!budget.can_schedule(5, 5).expect("Failed to schedule"));
        assert_eq!(budget.remaining_budget_tokens(), TOKEN_BUDGET);

        // Verify add/subtract num batched tokens.
        let (_, seq_group) = create_dummy_prompt(1, 3, None, 1);
        budget.add_num_batched_tokens(seq_group.request_id.clone(), 2);
        assert_eq!(budget.remaining_budget_tokens(), 2);
        assert_eq!(budget.num_batched_tokens, 2);
        assert!(budget.can_schedule(2, 1).expect("Failed to schedule"));
        assert!(!budget.can_schedule(3, 1).expect("Failed to schedule"));

        // Verify adding another seq group is no-op.
        budget.add_num_batched_tokens(seq_group.request_id.clone(), 2);
        assert_eq!(budget.remaining_budget_tokens(), 2);
        assert_eq!(budget.num_batched_tokens, 2);
        budget.subtract_num_batched_tokens(&seq_group.request_id, 2);
        assert_eq!(budget.remaining_budget_tokens(), 4);
        assert_eq!(budget.num_batched_tokens, 0);
        budget.subtract_num_batched_tokens(&seq_group.request_id, 2);
        assert_eq!(budget.remaining_budget_tokens(), 4);
        assert_eq!(budget.num_batched_tokens, 0);

        // Verify add/subtract max seqs.
        let (_, seq_group) = create_dummy_prompt(1, 3, None, 1);
        budget.add_number_sequences(seq_group.request_id.clone(), 2);
        assert!(budget.can_schedule(1, 2).expect("Failed to can schedule"));
        assert!(!budget.can_schedule(1, 3).expect("Failed to can schedule"));
        assert_eq!(budget.num_curr_seqs, 2);

        // Verify adding another seq group is no-op.
        budget.add_number_sequences(seq_group.request_id.clone(), 2);
        assert_eq!(budget.num_curr_seqs, 2);
        budget.subtracts_number_sequences(&seq_group.request_id, 2);
        assert_eq!(budget.num_curr_seqs, 0);
        budget.subtracts_number_sequences(&seq_group.request_id, 2);
        assert_eq!(budget.num_curr_seqs, 0);
    }

    impl Scheduler<FcfsPolicy> {
        fn mock_schedule_prefill(
            &mut self,
            mut waiting_queue: VecDeque<SequenceGroup>,
            budget: &mut SchedulingBudget,
            enable_chunking: bool,
            allocation: AllocationStatus,
        ) -> Result<(VecDeque<SequenceGroup>, SchedulerPrefillOutputs), SchedulerError> {
            let mut ignored_sequence_groups = Vec::<SequenceGroup>::new();
            let mut sequence_groups = Vec::<ScheduledSequenceGroup>::new();

            // We don't sort `waiting_queue` because we assume it is sorted. We also require
            // ownership of `waiting_queue` so that we don't change it in place, in this method.

            while !waiting_queue.is_empty() && self.passed_delay(Instant::now()) {
                // DON'T PANIC: at this point, we are guaranteed that `waiting_queue` is non-empty
                let mut sequence_group = waiting_queue.pop_front().unwrap();

                // To be used below
                let can_allocate: AllocationStatus = allocation.clone();
                let num_new_tokens = self.get_num_tokens(
                    &sequence_group,
                    SequenceStatus::Waiting,
                    enable_chunking,
                    budget,
                )?;

                let mut waiting_sequences = sequence_group
                    .sequences
                    .iter()
                    .filter_map(|(_, s)| {
                        if s.read().unwrap().get_sequence_status() == SequenceStatus::Waiting {
                            Some(s)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                if waiting_sequences.len() != 1 {
                    error!("Waiting sequence group should have only one prompt sequence, it has {} for request = {}.", waiting_sequences.len(), sequence_group.request_id);
                    return Err(SchedulerError::InvalidNumberWaitingSequence {
                        request_id: sequence_group.request_id.clone(),
                        num_sequences: waiting_sequences.len(),
                    });
                }

                if !enable_chunking {
                    // DON'T PANIC: by previous error check, we are guaranteed that `waiting_sequences` is non-empty
                    let num_prompt_tokens =
                        waiting_sequences.first().unwrap().read().unwrap().length();
                    if num_prompt_tokens != num_new_tokens {
                        error!("Invalid number of new tokens, got `{num_new_tokens}`, but it should be `{num_prompt_tokens}`");
                        return Err(SchedulerError::InvalidNumberOfNewTokens {
                            num_prompt_tokens,
                            num_new_tokens,
                        });
                    }
                }

                let prompt_limit = self.get_prompt_limit();
                if num_new_tokens > prompt_limit {
                    warn!(
                        "Input prompt ({} tokens) is too long and exceeds limits of {}",
                        num_new_tokens, prompt_limit
                    );
                    for (_, sequence) in sequence_group.sequences.iter_mut() {
                        sequence
                            .write()
                            .unwrap()
                            .set_sequence_status(SequenceStatus::FinishedIgnored)
                    }
                    ignored_sequence_groups.push(sequence_group.clone());
                    continue;
                }

                // If the sequence cannot be allocated, just stop
                if can_allocate == AllocationStatus::Later {
                    waiting_queue.push_front(sequence_group);
                    break;
                } else if can_allocate == AllocationStatus::Never {
                    warn!("Input prompt ({num_new_tokens} tokens) is too long and exceeds the capacity of `block_manager`");
                    for sequence in waiting_sequences.iter_mut() {
                        sequence
                            .write()
                            .unwrap()
                            .set_sequence_status(SequenceStatus::FinishedIgnored);
                    }
                    ignored_sequence_groups.push(sequence_group.clone());
                    continue;
                }

                let num_new_sequences = sequence_group.get_max_num_running_seqs();
                if num_new_sequences == 0
                    || !budget.can_schedule(num_new_tokens, num_new_sequences)?
                {
                    // Push the sequence group back to the `waiting_queue`
                    waiting_queue.push_front(sequence_group);
                    break;
                }

                // At this point, we can schedule this request
                self.allocate_and_set_running(&mut sequence_group)?;
                sequence_groups.push(ScheduledSequenceGroup {
                    scheduled_group: sequence_group.clone(),
                    token_chunk_size: num_new_tokens,
                });

                budget.add_num_batched_tokens(sequence_group.request_id.clone(), num_new_tokens);
                budget.add_number_sequences(sequence_group.request_id.clone(), num_new_sequences);
            }

            if !sequence_groups.is_empty() {
                self.previous_prompt = true;
            }

            Ok((
                waiting_queue,
                SchedulerPrefillOutputs {
                    sequence_groups,
                    ignored_sequence_groups,
                },
            ))
        }

        fn mock_schedule_running(
            &mut self,
            running_queue: VecDeque<SequenceGroup>,
            budget: &mut SchedulingBudget,
            enable_chunking: bool,
            id: &str,
            append_slots: Option<(u32, u32)>,
        ) -> Result<(VecDeque<SequenceGroup>, SchedulerRunningOutputs), SchedulerError> {
            // Blocks that need to be swapped or copied before model execution
            let mut blocks_to_swap_out = HashMap::<u32, u32>::new();
            let mut blocks_to_copy = HashMap::<u32, u32>::new();

            let mut decode_seq_groups = Vec::<ScheduledSequenceGroup>::new();
            let mut prefill_seq_groups = Vec::<ScheduledSequenceGroup>::new();
            let mut preempted = Vec::<SequenceGroup>::new();
            let mut swapped_out = Vec::<SequenceGroup>::new();

            // Preemption happens only when there is no available slot
            // to keep all sequences groups in `Running` state.
            // In this case, the policy is responsible for deciding which sequence
            // groups should preempt next
            let now = Instant::now();
            let mut running_queue = FcfsPolicy::sort_by_priority(now, &running_queue);

            while let Some(mut sequence_group) = running_queue.pop_front() {
                let num_running_tokens = self.get_num_tokens(
                    &sequence_group,
                    SequenceStatus::Running,
                    enable_chunking,
                    budget,
                )?;

                // if no tokens are being processed, we break the loop
                if num_running_tokens == 0 {
                    break;
                }

                fn cannot_append_second_group(sequence_group: &SequenceGroup, id: &str) -> bool {
                    sequence_group.request_id != id
                }

                loop {
                    if !cannot_append_second_group(&sequence_group, id) {
                        budget.subtract_num_batched_tokens(
                            &sequence_group.request_id,
                            num_running_tokens,
                        );
                        let num_running_sequences = sequence_group.get_max_num_running_seqs();
                        budget.subtracts_number_sequences(
                            &sequence_group.request_id,
                            num_running_sequences,
                        );

                        if let Some(mut victim_sequence_group) = running_queue.pop_back() {
                            // Preempt the lowest-priority sequence groups first
                            // victim lies at the end of `runnning_queue`, as it is was last in, last out
                            let preempted_mode = self.preempt(
                                &mut victim_sequence_group,
                                &mut blocks_to_swap_out,
                                None,
                            )?;
                            if preempted_mode == PreemptionMode::Recomputation {
                                preempted.push(victim_sequence_group);
                            } else {
                                swapped_out.push(victim_sequence_group);
                            }
                        } else {
                            // No other sequence groups can be preempted.
                            // Preempt the current `SequenceGroup`
                            let preempted_mode =
                                self.preempt(&mut sequence_group, &mut blocks_to_swap_out, None)?;

                            if preempted_mode == PreemptionMode::Recomputation {
                                preempted.push(sequence_group.clone());
                            } else {
                                swapped_out.push(sequence_group.clone());
                            }

                            // As no other sequence groups can be preempted, we stop the loop
                            break;
                        }
                    } else {
                        // MOCK
                        if let Some(value) = append_slots {
                            self.mock_append_slots(&sequence_group, &mut blocks_to_copy, value)
                                .unwrap();
                        } else {
                            self.append_slots(&sequence_group, &mut blocks_to_copy)?;
                        }
                        let is_prefill = sequence_group.is_prefill();
                        if is_prefill {
                            // Prefill computation
                            prefill_seq_groups.push(ScheduledSequenceGroup {
                                scheduled_group: sequence_group.clone(),
                                token_chunk_size: num_running_tokens,
                            });
                        } else {
                            // Decoding computation (only decodes 1 token at a time)
                            decode_seq_groups.push(ScheduledSequenceGroup {
                                scheduled_group: sequence_group.clone(),
                                token_chunk_size: 1,
                            });
                        }
                        budget.add_num_batched_tokens(
                            sequence_group.request_id.clone(),
                            num_running_tokens,
                        );

                        // OPTIMIZATION: Note that `get_max_num_running_seqs` is
                        // expensive. For the default scheduling chase where
                        // `enable_chunking` is false, `num_seqs` are updated before running
                        // this method, so we don't have to update it again here.
                        if enable_chunking {
                            let num_running_seqs = sequence_group.get_max_num_running_seqs();
                            budget.add_number_sequences(
                                sequence_group.request_id.clone(),
                                num_running_seqs,
                            )
                        }
                        break;
                    }
                }
            }

            let scheduler_running_outputs = SchedulerRunningOutputs {
                decode_seq_groups,
                prefill_seq_groups,
                preempted,
                swapped_out,
                blocks_to_swap_out,
                blocks_to_copy,
            };

            Ok((running_queue, scheduler_running_outputs))
        }

        fn mock_append_slots(
            &mut self,
            sequence_group: &SequenceGroup,
            blocks_to_copy: &mut HashMap<u32, u32>,
            value: (u32, u32),
        ) -> Result<(), SchedulerError> {
            info!(
                "Appending slot to sequence group with id = {}",
                sequence_group.request_id
            );
            let running_sequences = sequence_group.sequences.iter().filter_map(|(_, s)| {
                if s.read().unwrap().get_sequence_status() == SequenceStatus::Running {
                    Some(s)
                } else {
                    None
                }
            });
            for _ in running_sequences {
                blocks_to_copy.insert(value.0, value.1);
            }
            Ok(())
        }

        fn mock_schedule_swapped(
            &mut self,
            budget: &mut SchedulingBudget,
            swapped_queue: VecDeque<SequenceGroup>,
            enable_chunking: bool,
            allocation_status: Option<AllocationStatus>,
            append_slots: Option<(u32, u32)>,
        ) -> Result<(VecDeque<SequenceGroup>, SchedulerSwappedInOutputs), SchedulerError> {
            info!("Schedule swapped..");
            // Blocks that need to be swapped or copied before model execution.
            let mut blocks_to_swap_in = HashMap::<u32, u32>::new();
            let mut blocks_to_copy = HashMap::<u32, u32>::new();
            let mut decode_seq_groups = Vec::<ScheduledSequenceGroup>::new();
            let mut prefill_seq_groups = Vec::<ScheduledSequenceGroup>::new();

            let now = Instant::now();

            let mut swapped_queue = FcfsPolicy::sort_by_priority(now, &swapped_queue);
            let mut infeasible_seq_groups = Vec::<SequenceGroup>::new();

            while let Some(mut sequence_group) = swapped_queue.pop_front() {
                // If the sequence group cannot be swapped in, stop.
                let allocation_status = if let Some(allocation_status) = allocation_status.clone() {
                    allocation_status
                } else {
                    self.block_manager.can_allocate(&sequence_group)
                };

                if allocation_status == AllocationStatus::Later {
                    // push the sequence group back to `swapped_queue`
                    swapped_queue.push_front(sequence_group);
                    break;
                } else if allocation_status == AllocationStatus::Never {
                    warn!("Failing the request {} because there is not enough KV cache blocks to run the entire sequence..", 
                            sequence_group.request_id);
                    for (_, sequence) in sequence_group.sequences.iter_mut() {
                        sequence
                            .write()
                            .unwrap()
                            .set_sequence_status(SequenceStatus::FinishedIgnored);
                    }
                    infeasible_seq_groups.push(sequence_group.clone());
                    continue;
                }

                // The total number of sequences in the RUNNING state should not
                // exceed the maximum number of sequences.
                let num_new_sequences = sequence_group.get_max_num_running_seqs();
                let num_new_tokens = self.get_num_tokens(
                    &sequence_group,
                    SequenceStatus::Swapped,
                    enable_chunking,
                    budget,
                )?;

                if num_new_tokens == 0 || !budget.can_schedule(num_new_tokens, num_new_sequences)? {
                    info!(
                        "Either no new tokens to be swapped or no available budget to swap tokens"
                    );
                    // push the sequence group back to `swapped_queue`
                    swapped_queue.push_front(sequence_group);
                    break;
                }

                self.swap_in(&mut sequence_group, &mut blocks_to_swap_in)?;
                if let Some(value) = append_slots {
                    self.mock_append_slots(&sequence_group, &mut blocks_to_copy, value)
                        .unwrap();
                } else {
                    self.append_slots(&sequence_group, &mut blocks_to_copy)?;
                }

                let is_preffil = sequence_group.is_prefill();
                if is_preffil {
                    prefill_seq_groups.push(ScheduledSequenceGroup {
                        scheduled_group: sequence_group.clone(),
                        token_chunk_size: num_new_tokens,
                    })
                } else {
                    decode_seq_groups.push(ScheduledSequenceGroup {
                        scheduled_group: sequence_group.clone(),
                        token_chunk_size: 1,
                    })
                }

                budget.add_num_batched_tokens(sequence_group.request_id.clone(), num_new_tokens);
                budget.add_number_sequences(sequence_group.request_id.clone(), num_new_sequences);
            }

            Ok((
                swapped_queue,
                SchedulerSwappedInOutputs {
                    decode_seq_groups,
                    prefill_seq_groups,
                    blocks_to_swap_in,
                    blocks_to_copy,
                    infeasible_seq_groups,
                },
            ))
        }
    }
}
