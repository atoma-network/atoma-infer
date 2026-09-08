//! Scheduler tests, including the port of the vLLM-derived scheduler tests.
//!
//! Port ledger against the previous engine's scheduler, `crates/backends/vllm/src/scheduler.rs`
//! until that engine was deleted:
//!
//! - Ported with semantics intact: `add_sequence_group`, `schedule_simple`, `max_seqs`,
//!   `prefill_schedule_max_prompt_len`, `prefill_schedule_max_seqs`,
//!   `prefill_schedule_no_block_manager_capacity`, `scheduling_budget` (in `budget.rs`),
//!   `finished_sequence_groups_return_their_blocks`,
//!   `sequential_requests_do_not_exhaust_the_block_pool`,
//!   `abort_removes_the_sequence_group_from_every_queue` and `scheduler_abort_sequence_group`
//!   (abort is the client dropping its egress receiver). Where the old tests counted free blocks
//!   after a finish, these count `available` blocks: a finished request's full blocks stay resident
//!   as evictable cache, which the old block manager had no notion of.
//! - Rewritten, divergence documented on the test: `prefill_prioritized` (decode-first, mixed
//!   batches), `prefill_schedule_token_budget` (a prompt over the remaining budget is chunked, not
//!   refused), `scheduler_schedule_preempt_abort` and `decode_schedule_preempted` (preemption is
//!   from the tail of the running queue and the victim computes again; nothing is swapped).
//! - Dropped, with the machinery they tested: every `schedule_swapped_*`, `infeasible_swap`,
//!   `decode_swap_beam_search` and `schedule_decode_blocks_to_copy_update` (swap and copy-on-write
//!   do not exist; preemption recomputes), `scheduler_delay_factor` (no delay factor), and
//!   `abort_of_unknown_request` (there is no abort by id; a dropped egress receiver is the cancel).

mod budget;
mod drain;
mod intake;
mod longest_prefix;
mod lost_clients;
mod preemption;
mod prefix_cache;
mod priority;
mod stop_criteria;

use crate::kv::{BlockPool, HashAlgorithm, PaddingReservation};
use crate::request::{
    egress, EgressReceiver, EgressSender, FinishReason, NewRequest, Priority, RequestEvent,
    SamplingParams, StopCriteria,
};
use crate::scheduler::{AdmissionPolicy, Scheduled, Scheduler, SchedulerConfig};
use crate::test_support::{requests, tokens};
use crate::types::RequestSlot;

const BLOCK_SIZE: usize = 4;
const WINDOW: usize = 8;
const MAX_REQUESTS: usize = 64;
/// High enough that no test trips the backlog sweep unless it sets its own limit.
const MAX_CLIENT_BACKLOG: usize = 1024;
/// The end-of-sequence token; prompts count up from one and never reach it.
const EOS: u32 = 99;

/// A scheduler over `blocks` blocks whose max model length is exactly what the pool holds.
fn scheduler(blocks: u32, token_budget: usize, max_batch: usize) -> Scheduler {
    Scheduler::new(
        config(blocks, token_budget, max_batch),
        BlockPool::new(blocks),
    )
    .expect("the pool holds one maximum-length request")
}

fn config(blocks: u32, token_budget: usize, max_batch: usize) -> SchedulerConfig {
    SchedulerConfig {
        token_budget: tokens(token_budget),
        max_batch: requests(max_batch),
        max_model_len: tokens(blocks as usize * BLOCK_SIZE),
        block_size: tokens(BLOCK_SIZE),
        window: requests(WINDOW),
        admission: AdmissionPolicy::Fcfs,
        max_requests: requests(MAX_REQUESTS),
        max_client_backlog: tokens(MAX_CLIENT_BACKLOG),
        eos_token_ids: vec![EOS],
        hash_algorithm: HashAlgorithm::Sha256V1,
    }
}

#[test]
fn the_slot_count_covers_every_live_request_and_every_padding_dummy() {
    // Sixty-four requests, and the four dummies a maximum batch of four reserves.
    assert_eq!(config(16, 100, 4).slot_count(), 68);
    assert_eq!(
        config(16, 100, 1).slot_count(),
        65,
        "a maximum batch of one still reserves the row a dummy run fills"
    );

    // Every slot the scheduler hands out is inside the count: the dummies take theirs at
    // startup, and the slab hands the rest out until it is full.
    let config = config(16, 100, 4);
    let mut pool = BlockPool::new(16);
    let reservation = PaddingReservation::reserve(&mut pool, config.max_batch).expect("reserves");
    let dummy_count = reservation.dummy_count();
    reservation.release(&mut pool);
    assert_eq!(
        dummy_count, 4,
        "one per entry of the maximum batch, so a dummy run fills every row"
    );

    let mut scheduler = scheduler(16, 100, 4);
    let mut clients = Vec::new();
    let mut highest = 0;
    for _ in 0..MAX_REQUESTS {
        let (egress, client) = egress();
        clients.push(client);
        let slot = scheduler
            .intake(new_request(3, 4, egress))
            .expect("the prompt fits the model");
        highest = highest.max(slot.index());
    }
    assert_eq!(highest, MAX_REQUESTS - 1, "the slab filled from zero");
    assert!(
        highest + dummy_count < config.slot_count(),
        "the dummies' slots fit above a full slab"
    );
}

/// A scheduler that retires a client leaving more than `max_backlog` events unread.
fn backlog_scheduler(blocks: u32, max_backlog: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig {
            max_client_backlog: tokens(max_backlog),
            ..config(blocks, 100, 4)
        },
        BlockPool::new(blocks),
    )
    .expect("the pool holds one maximum-length request")
}

/// A scheduler admitting by longest prefix match over a window of `window` candidates.
fn lpm_scheduler(blocks: u32, max_batch: usize, window: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig {
            admission: AdmissionPolicy::LongestPrefixMatch,
            window: requests(window),
            ..config(blocks, 1000, max_batch)
        },
        BlockPool::new(blocks),
    )
    .expect("the pool holds one maximum-length request")
}

/// A scheduler admitting by priority over a window of `window` candidates.
fn priority_scheduler(blocks: u32, max_batch: usize, window: usize) -> Scheduler {
    Scheduler::new(
        SchedulerConfig {
            admission: AdmissionPolicy::Priority,
            window: requests(window),
            ..config(blocks, 1000, max_batch)
        },
        BlockPool::new(blocks),
    )
    .expect("the pool holds one maximum-length request")
}

/// The requests a test submits, with their clients' receivers kept alive.
#[derive(Default)]
struct Clients {
    receivers: Vec<EgressReceiver>,
}

impl Clients {
    /// Submits a `prompt_len`-token prompt allowed `max_new_tokens` generated tokens, asking for
    /// no priority. Each submission gets its own token range, so requests never share a prefix
    /// unless a test submits an explicit prompt through [`Clients::submit_prompt`].
    fn submit(
        &mut self,
        scheduler: &mut Scheduler,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> RequestSlot {
        self.submit_at_priority(scheduler, prompt_len, max_new_tokens, Priority::default())
    }

    /// The same, submitted at `priority`.
    fn submit_at_priority(
        &mut self,
        scheduler: &mut Scheduler,
        prompt_len: usize,
        max_new_tokens: usize,
        priority: Priority,
    ) -> RequestSlot {
        let base = u32::try_from(1000 * (self.receivers.len() + 1)).unwrap();
        let prompt = (base + 1..=base + u32::try_from(prompt_len).unwrap()).collect();
        self.intake(scheduler, prompt, max_new_tokens, priority)
    }

    /// Submits `prompt` exactly, for tests about shared prefixes.
    fn submit_prompt(
        &mut self,
        scheduler: &mut Scheduler,
        prompt: Vec<u32>,
        max_new_tokens: usize,
    ) -> RequestSlot {
        self.intake(scheduler, prompt, max_new_tokens, Priority::default())
    }

    /// Takes `prompt` in at `priority`, keeping its client's receiver alive.
    fn intake(
        &mut self,
        scheduler: &mut Scheduler,
        prompt: Vec<u32>,
        max_new_tokens: usize,
        priority: Priority,
    ) -> RequestSlot {
        let (sender, receiver) = egress();
        self.receivers.push(receiver);
        scheduler
            .intake(NewRequest {
                prompt,
                priority,
                ..new_request(1, max_new_tokens, sender)
            })
            .expect("the prompt fits the model")
    }
}

fn new_request(prompt_len: usize, max_new_tokens: usize, egress: EgressSender) -> NewRequest {
    NewRequest {
        prompt: (1..=u32::try_from(prompt_len).unwrap()).collect(),
        sampling: SamplingParams::default(),
        stop: StopCriteria {
            max_new_tokens: tokens(max_new_tokens),
            ignore_eos: false,
        },
        priority: Priority::default(),
        egress,
    }
}

/// Schedules one pass and applies it with token `1` sampled for every sampling entry.
fn step(scheduler: &mut Scheduler) -> Scheduled {
    step_sampling(scheduler, 1)
}

/// Schedules one pass and applies it with `token` sampled for every sampling entry.
fn step_sampling(scheduler: &mut Scheduler, token: u32) -> Scheduled {
    let scheduled = scheduler.schedule();
    let sampled = vec![token; scheduled.sampling_entries().count()];
    scheduler.apply(&scheduled, &sampled);
    scheduled
}

/// Every event a client has received so far.
fn events(receiver: &EgressReceiver) -> Vec<RequestEvent> {
    receiver.try_iter().collect()
}

fn finish_reason(receiver: &EgressReceiver) -> Option<FinishReason> {
    events(receiver).into_iter().find_map(|event| match event {
        RequestEvent::Finished { reason, .. } => Some(reason),
        RequestEvent::Token { .. } => None,
    })
}

/// A prompt of exactly `blocks` full blocks, the same for every caller.
fn shared_prompt(blocks: usize) -> Vec<u32> {
    (1..=u32::try_from(blocks * BLOCK_SIZE).unwrap()).collect()
}

fn slots(scheduled: &Scheduled) -> Vec<RequestSlot> {
    scheduled.entries.iter().map(|entry| entry.slot).collect()
}
