//! Building a step command from request state, with zero device reads.
//!
//! Every length, token and block id the executor needs is host-native in the request slab, so
//! the command is a pure function of a scheduling pass over that state. Padding and dispatch
//! happen here, on the engine thread: the live batch is decided, and when a captured graph
//! serves it the padding dummies fill the batch up to the bucket. The executor acts on the
//! command and re-derives nothing.

use crate::dispatch::{DispatchDecision, Dispatcher};
use crate::scheduler::{Entry, Scheduled, Scheduler};
use crate::step::{CommandEntry, StepCommand};
use crate::types::{RequestSlot, SequenceIndex};

/// The command for `scheduled`: its entries read from the scheduler's request state, the
/// dispatch decision for its live batch, and — when a graph serves it — enough padding dummies
/// to reach the bucket the graph was captured for.
///
/// # Panics
///
/// Panics when `scheduled` is empty, when an entry's block table does not cover its sequence
/// length, or when the bucket needs more dummies than were reserved. Each is a bug in the
/// scheduler or the configuration, not a runtime state: the caller skips empty passes, the
/// scheduler grows every table before it schedules an entry, and the engine refuses a bucket
/// ladder its reservation cannot pad to.
#[must_use]
pub(crate) fn build_command(
    scheduler: &Scheduler,
    scheduled: &Scheduled,
    dispatcher: &mut Dispatcher,
) -> StepCommand {
    let live_batch = scheduled
        .live_batch()
        .expect("an empty pass issues no command");
    let dispatch = dispatcher.dispatch(live_batch);
    let padding_count = match dispatch {
        DispatchDecision::FullReplay(key) | DispatchDecision::SegmentedReplay(key) => {
            key.padded_token_count().get() - live_batch.token_count.get()
        }
        DispatchDecision::Eager(_) => 0,
    };
    assert!(
        padding_count <= scheduler.padding().len(),
        "the bucket needs {padding_count} dummies but {} are reserved",
        scheduler.padding().len()
    );
    let block_size = scheduler.config().block_size.get();
    let mut entries: Vec<CommandEntry> =
        Vec::with_capacity(scheduled.entries.len() + padding_count);
    entries.extend(
        scheduled
            .entries
            .iter()
            .map(|entry| live_entry(scheduler, entry, block_size)),
    );
    entries.extend(
        scheduler.padding()[..padding_count]
            .iter()
            .map(|&slot| padding_entry(scheduler, slot)),
    );
    StepCommand {
        step: scheduled.step,
        entries,
        padding_count,
        dispatch,
    }
}

/// A live request's entry: the `query_len` tokens it computes this step, over the sequence's own
/// blocks.
///
/// The input tokens are the slice from `context_len` to the length the sequence reaches after the
/// step, so a decode carries one token and a prefill chunk carries only that chunk. Sampling
/// parameters are attached only when the entry samples this step.
///
/// # Panics
///
/// Panics when the slot is not live, or when its block table does not cover the length the
/// sequence reaches this step. Both are scheduler bugs, not runtime states: a slot stays live
/// until its result is applied, and the scheduler grows every table before it schedules an entry.
fn live_entry(scheduler: &Scheduler, entry: &Entry, block_size: usize) -> CommandEntry {
    let request = scheduler
        .request(entry.slot)
        .expect("a scheduled slot is live until its result is applied");
    let sequence = &request.sequences()[entry.sequence.get() as usize];
    let sequence_len = entry.sequence_len();
    assert!(
        sequence.block_table().len() * block_size >= sequence_len,
        "block table of {} blocks does not cover {sequence_len} tokens",
        sequence.block_table().len()
    );
    CommandEntry {
        request: request.id(),
        slot: entry.slot,
        sequence: entry.sequence,
        context_len: entry.context_len,
        input_tokens: sequence.tokens()[entry.context_len..sequence_len].to_vec(),
        block_table: sequence.block_table().to_vec(),
        sampling: entry.samples.then(|| request.sampling()),
    }
}

/// A dummy's entry: one token over its own block, context empty, never sampling.
fn padding_entry(scheduler: &Scheduler, slot: RequestSlot) -> CommandEntry {
    let dummy = scheduler.request(slot).expect("dummies live forever");
    debug_assert!(dummy.is_padding());
    let sequence = &dummy.sequences()[0];
    CommandEntry {
        request: dummy.id(),
        slot,
        sequence: SequenceIndex::new(0),
        context_len: 0,
        input_tokens: sequence.tokens().to_vec(),
        block_table: sequence.block_table().to_vec(),
        sampling: None,
    }
}

#[cfg(test)]
mod tests {
    use super::build_command;
    use crate::attention::SupportLevel;
    use crate::dispatch::{
        BucketLadder, DispatchConfig, DispatchDecision, Dispatcher, EagerReason,
    };
    use crate::kv::{BlockPool, HashAlgorithm, PaddingReservation};
    use crate::request::{
        egress, EgressReceiver, NewRequest, Priority, SamplingParams, StopCriteria,
    };
    use crate::scheduler::{AdmissionPolicy, Scheduled, Scheduler, SchedulerConfig};
    use crate::test_support::{contract, requests, tokens};
    use crate::types::{BlockId, RequestSlot, StepId};

    const BLOCK_SIZE: usize = 4;
    const MAX_BATCH: usize = 8;

    /// A scheduler with the padding dummies for a maximum batch of eight in its slab, and the
    /// block ids they hold.
    fn scheduler(token_budget: usize) -> (Scheduler, Vec<BlockId>) {
        let mut pool = BlockPool::new(24);
        let reservation = PaddingReservation::reserve(&mut pool, requests(MAX_BATCH)).unwrap();
        let dummy_blocks = reservation.block_ids();
        let scheduler = Scheduler::with_padding(
            SchedulerConfig {
                token_budget: tokens(token_budget),
                max_batch: requests(MAX_BATCH),
                max_model_len: tokens(32),
                block_size: tokens(BLOCK_SIZE),
                window: requests(8),
                admission: AdmissionPolicy::Fcfs,
                max_requests: requests(8),
                max_client_backlog: tokens(1024),
                eos_token_ids: Vec::new(),
                hash_algorithm: HashAlgorithm::Sha256V1,
            },
            pool,
            reservation,
        )
        .unwrap();
        (scheduler, dummy_blocks)
    }

    fn eager_dispatcher() -> Dispatcher {
        Dispatcher::new(
            &DispatchConfig {
                bucket_ladder: BucketLadder::new(Vec::new()).unwrap(),
                captured_max_requests: requests(MAX_BATCH),
            },
            &contract(SupportLevel::Never, &[]),
        )
    }

    fn replay_dispatcher() -> Dispatcher {
        Dispatcher::new(
            &DispatchConfig {
                bucket_ladder: BucketLadder::new(vec![1, 2, 4, 8]).unwrap(),
                captured_max_requests: requests(MAX_BATCH),
            },
            &contract(SupportLevel::Always, &[]),
        )
    }

    fn submit(
        scheduler: &mut Scheduler,
        prompt: Vec<u32>,
        temperature: f32,
    ) -> (RequestSlot, EgressReceiver) {
        let (sender, receiver) = egress();
        let slot = scheduler
            .intake(NewRequest {
                prompt,
                sampling: SamplingParams {
                    temperature,
                    ..SamplingParams::default()
                },
                stop: StopCriteria {
                    max_new_tokens: tokens(8),
                    ignore_eos: false,
                },
                priority: Priority::default(),
                egress: sender,
            })
            .unwrap();
        (slot, receiver)
    }

    fn apply_ones(scheduler: &mut Scheduler, scheduled: &Scheduled) {
        let sampled = vec![1; scheduled.sampling_entries().count()];
        scheduler.apply(scheduled, &sampled);
    }

    #[test]
    fn the_command_mirrors_the_pass_with_host_native_lengths_tokens_and_tables() {
        let (mut scheduler, _) = scheduler(100);
        let mut dispatcher = eager_dispatcher();
        let (first, _a) = submit(&mut scheduler, vec![10, 11, 12, 13, 14], 0.5);
        let (second, _b) = submit(&mut scheduler, vec![20, 21], 0.9);

        let scheduled = scheduler.schedule();
        let command = build_command(&scheduler, &scheduled, &mut dispatcher);
        assert_eq!(command.step, scheduled.step);
        assert_eq!(command.entries.len(), 2);
        assert_eq!(command.padding_count, 0);
        assert!(matches!(command.dispatch, DispatchDecision::Eager(_)));

        let entry = &command.entries[0];
        assert_eq!(entry.slot, first);
        assert_eq!(entry.context_len, 0);
        assert_eq!(entry.input_tokens, [10, 11, 12, 13, 14]);
        assert_eq!(entry.sequence_len(), 5);
        assert_eq!(entry.block_table.len(), 2, "five tokens need two blocks");
        assert_eq!(
            entry.block_table,
            scheduler.request(first).unwrap().sequences()[0].block_table()
        );
        assert_eq!(entry.sampling.map(|s| s.temperature), Some(0.5));

        let entry = &command.entries[1];
        assert_eq!(entry.slot, second);
        assert_eq!(entry.input_tokens, [20, 21]);
        assert_eq!(entry.block_table.len(), 1);
        assert_eq!(entry.sampling.map(|s| s.temperature), Some(0.9));
        assert_eq!(command.sampling_count(), 2);
        apply_ones(&mut scheduler, &scheduled);

        // Decoding: the one input token is the token just sampled, at the context's end.
        let scheduled = scheduler.schedule();
        let command = build_command(&scheduler, &scheduled, &mut dispatcher);
        assert_eq!(command.entries[0].context_len, 5);
        assert_eq!(command.entries[0].input_tokens, [1]);
        assert_eq!(command.entries[0].sequence_len(), 6);
        assert_eq!(command.entries[1].context_len, 2);
        assert_eq!(command.entries[1].input_tokens, [1]);
        apply_ones(&mut scheduler, &scheduled);
    }

    #[test]
    fn a_non_final_prefill_chunk_carries_its_slice_and_no_sampling() {
        let (mut scheduler, _) = scheduler(3);
        let mut dispatcher = eager_dispatcher();
        let (_, _client) = submit(&mut scheduler, vec![10, 11, 12, 13, 14], 0.5);

        let scheduled = scheduler.schedule();
        let command = build_command(&scheduler, &scheduled, &mut dispatcher);
        assert_eq!(command.entries[0].input_tokens, [10, 11, 12]);
        assert_eq!(command.entries[0].sampling, None);
        assert_eq!(command.sampling_count(), 0);
        apply_ones(&mut scheduler, &scheduled);

        let scheduled = scheduler.schedule();
        let command = build_command(&scheduler, &scheduled, &mut dispatcher);
        assert_eq!(command.entries[0].context_len, 3);
        assert_eq!(command.entries[0].input_tokens, [13, 14]);
        assert!(command.entries[0].samples());
        apply_ones(&mut scheduler, &scheduled);
    }

    /// Three decoding requests pad to the bucket of four with one dummy: its own block, one
    /// padding token, an empty context and no sampling, after every live entry.
    #[test]
    fn a_replayed_batch_is_padded_to_its_bucket_with_reserved_dummies() {
        let (mut scheduler, dummy_blocks) = scheduler(100);
        let mut dispatcher = replay_dispatcher();
        let clients: Vec<_> = (0..3)
            .map(|i| submit(&mut scheduler, vec![10 + i, 20 + i], 0.5))
            .collect();
        let prefill = scheduler.schedule();
        let command = build_command(&scheduler, &prefill, &mut dispatcher);
        assert!(
            matches!(
                command.dispatch,
                DispatchDecision::Eager(EagerReason::NotUniformDecode { .. })
            ),
            "prefill is not uniform decode"
        );
        assert_eq!(command.padding_count, 0);
        apply_ones(&mut scheduler, &prefill);

        let decode = scheduler.schedule();
        let command = build_command(&scheduler, &decode, &mut dispatcher);
        let DispatchDecision::FullReplay(key) = command.dispatch else {
            panic!("three uniform decodes replay the bucket of four");
        };
        assert_eq!(key.padded_token_count(), tokens(4));
        assert_eq!(key.request_count(), requests(3));
        assert_eq!(command.padding_count, 1);
        assert_eq!(command.entries.len(), 4);
        assert_eq!(command.live_entries().len(), 3);
        assert_eq!(
            command.token_count(),
            Some(tokens(4)),
            "padded to the bucket"
        );
        assert_eq!(command.sampling_count(), 3, "dummies never sample");

        let dummy = &command.entries[3];
        assert_eq!(dummy.slot, scheduler.padding()[0]);
        assert_eq!(dummy.context_len, 0);
        assert_eq!(dummy.input_tokens, [0]);
        assert_eq!(dummy.block_table, [dummy_blocks[0]]);
        assert_eq!(dummy.sampling, None);
        assert!(clients.iter().all(|(slot, _)| *slot != dummy.slot));
        apply_ones(&mut scheduler, &decode);
    }

    #[test]
    fn dummies_occupy_slots_for_the_process_lifetime_and_never_enter_admission() {
        let (mut scheduler, _) = scheduler(100);
        assert_eq!(scheduler.padding().len(), MAX_BATCH);
        assert_eq!(
            scheduler.live_request_count(),
            0,
            "dummies are not live requests"
        );
        assert!(scheduler.has_room());
        for slot in scheduler.padding() {
            let dummy = scheduler.request(*slot).unwrap();
            assert!(dummy.is_padding());
            assert_eq!(dummy.sequences()[0].block_table().len(), 1);
        }

        let scheduled = scheduler.schedule();
        assert!(scheduled.is_empty(), "no dummy is ever admitted");
        assert!(scheduler.running().is_empty());
        for slot in scheduler.padding().to_vec() {
            assert!(
                scheduler.request(slot).is_some(),
                "still there after a pass"
            );
        }
    }

    #[test]
    fn an_eager_batch_inserts_no_dummies() {
        let (mut scheduler, _) = scheduler(100);
        let mut dispatcher = replay_dispatcher();
        let clients: Vec<_> = (0..MAX_BATCH)
            .map(|i| submit(&mut scheduler, vec![10 + u32::try_from(i).unwrap()], 0.5))
            .collect();
        let _ = &clients;
        let prefill = scheduler.schedule();
        apply_ones(&mut scheduler, &prefill);
        let decode = scheduler.schedule();
        let command = build_command(&scheduler, &decode, &mut dispatcher);
        let DispatchDecision::FullReplay(key) = command.dispatch else {
            panic!("eight uniform decodes are exactly the bucket of eight");
        };
        assert_eq!(key.padded_token_count(), tokens(8));
        assert_eq!(command.padding_count, 0, "already at the bucket");
        apply_ones(&mut scheduler, &decode);
    }

    #[test]
    #[should_panic(expected = "an empty pass issues no command")]
    fn an_empty_pass_is_never_built_into_a_command() {
        let (scheduler, _) = scheduler(100);
        let scheduled = Scheduled {
            step: StepId::new(1),
            entries: Vec::new(),
            preempted: Vec::new(),
        };
        let _ = build_command(&scheduler, &scheduled, &mut eager_dispatcher());
    }
}
