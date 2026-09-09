//! The executor: one pinned thread per rank, owning a device and acting on the engine thread's
//! step commands.
//!
//! The engine thread decides everything about a step on the host and hands the executor a step
//! command over a ring; the executor runs the model forward for it, which samples the tokens the
//! command asks for, and hands a step result back. It re-derives nothing the command already
//! settled.
//!
//! For a keyed step, the arrays the host writes anew every step go up in one copy: the model's
//! five inputs, the sampler's two per-step arrays and its live-row count, packed into one block,
//! written into a staging entry the staging ring hands out, and copied to the device in front of
//! the step's work. The staging entry's fence, signaled behind that copy, is what says when the
//! host may write the staging entry again, and the staging ring's depth bounds how many of those
//! copies can be in flight at once. The step's work itself is a replay: at startup each rank
//! captures every bucket of the bucket ladder into a graph over the padding dummies' blocks and
//! reports what that cost, and a keyed batch replays the graph its bucket selects.

pub mod batch;
pub mod config;
pub mod decode;
#[cfg(feature = "cuda")]
pub mod device;
pub mod executor;
pub mod forward;
pub mod logits;
pub mod model;
pub(crate) mod pinned;
pub mod readback;
pub mod sampling;

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;
    use std::time::Duration;

    use atoma_core::attention::{
        BackendDeclaration, CaptureContract, ModelDeclaration, SupportLevel,
    };
    use atoma_core::dispatch::{
        BucketLadder, DispatchConfig, DispatchDecision, Dispatcher, EagerReason, LiveBatch,
    };
    use atoma_core::engine::{EngineConfig, EngineHandle};
    use atoma_core::kv::HashAlgorithm;
    use atoma_core::request::{
        egress, EgressReceiver, NewRequest, Priority, SamplingParams, StopCriteria, PADDING_TOKEN,
    };
    use atoma_core::scheduler::{AdmissionPolicy, SchedulerConfig};
    use atoma_core::step::{CommandEntry, StepCommand};
    use atoma_core::types::{
        BlockId, RequestCount, RequestId, RequestSlot, SequenceIndex, StepId, TokenCount,
    };
    use parking_lot::Mutex;
    use thiserror::Error;

    use crate::batch::BatchLayout;
    use crate::forward::Forward;

    pub(crate) const BLOCK_SIZE: TokenCount = TokenCount::new(4).expect("nonzero");
    const MAX_BATCH: usize = 4;
    const BLOCKS: u32 = 16;
    const EOS: u32 = 99;

    /// How long a test waits on a thread before calling it wedged: generous, since a loaded
    /// machine is slow rather than broken.
    pub(crate) const WAIT: Duration = Duration::from_secs(30);

    /// An idle deadline longer than any test, so a test that finishes proves the thread was
    /// woken rather than that it passed at its deadline.
    const LONG_DEADLINE: Duration = Duration::from_mins(5);

    fn tokens(value: usize) -> TokenCount {
        TokenCount::new(value).expect("test token counts are nonzero")
    }

    fn requests(value: usize) -> RequestCount {
        RequestCount::new(value).expect("test request counts are nonzero")
    }

    /// A small engine: four-token blocks, batches of four, a slab of eight.
    pub(crate) fn engine_config() -> EngineConfig {
        EngineConfig {
            scheduler: SchedulerConfig {
                token_budget: tokens(64),
                max_batch: requests(MAX_BATCH),
                max_model_len: tokens(32),
                block_size: BLOCK_SIZE,
                window: requests(8),
                admission: AdmissionPolicy::Fcfs,
                max_requests: requests(8),
                max_client_backlog: tokens(1024),
                eos_token_ids: vec![EOS],
                hash_algorithm: HashAlgorithm::Sha256V1,
            },
            dispatch: DispatchConfig {
                bucket_ladder: BucketLadder::new(vec![1, 2, 4]).expect("nonempty"),
                captured_max_requests: requests(MAX_BATCH),
            },
            block_count: BLOCKS,
            ingress_capacity: requests(8),
            idle_deadline: LONG_DEADLINE,
            step_deadline: WAIT,
        }
    }

    /// One backend that captures anything and a model that breaks nothing.
    pub(crate) fn contract() -> CaptureContract {
        CaptureContract::resolve(
            &[BackendDeclaration::new(
                "test-backend",
                SupportLevel::Always,
            )],
            &ModelDeclaration::new("test-model"),
        )
    }

    /// Submits a greedy request of `prompt_len` tokens asking for `max_new_tokens`.
    pub(crate) fn submit(
        handle: &EngineHandle,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> EgressReceiver {
        let (sender, receiver) = egress();
        let request = NewRequest {
            prompt: (1..=u32::try_from(prompt_len).expect("fits u32")).collect(),
            sampling: SamplingParams::default(),
            stop: StopCriteria {
                max_new_tokens: tokens(max_new_tokens),
                ignore_eos: false,
            },
            priority: Priority::default(),
            egress: sender,
        };
        handle.ingress.try_send(request).expect("ingress has room");
        receiver
    }

    #[derive(Debug, Clone, PartialEq, Eq, Error)]
    #[error("the fake forward was told to fail on its command number {command}")]
    pub(crate) struct FakeForwardError {
        pub(crate) command: usize,
    }

    /// A forward that samples one chosen token for every selected row; it can be told to fail
    /// on its n-th command, and it keeps every layout it ran.
    ///
    /// Commands are counted rather than matched by step id: the scheduler mints a step id on
    /// every pass, empty ones included, so the first served step's id depends on whether the
    /// engine passed before the request arrived.
    pub(crate) struct FakeForward {
        token: u32,
        fail_on_command: Option<usize>,
        seen: usize,
        sampled: Vec<u32>,
        served: Arc<Mutex<Vec<BatchLayout>>>,
    }

    impl FakeForward {
        pub(crate) fn constant(token: u32) -> Self {
            Self {
                token,
                fail_on_command: None,
                seen: 0,
                sampled: Vec::new(),
                served: Arc::default(),
            }
        }

        /// Fails on the `command`-th command it is given, counting from one.
        pub(crate) fn failing_on_command(mut self, command: usize) -> Self {
            self.fail_on_command = Some(command);
            self
        }

        /// Every layout run so far, shared with whoever holds the clone.
        pub(crate) fn served(&self) -> Arc<Mutex<Vec<BatchLayout>>> {
            Arc::clone(&self.served)
        }
    }

    impl Forward for FakeForward {
        type Error = FakeForwardError;

        fn forward(&mut self, layout: &BatchLayout) -> Result<&[u32], FakeForwardError> {
            self.seen += 1;
            if self.fail_on_command == Some(self.seen) {
                return Err(FakeForwardError { command: self.seen });
            }
            self.served.lock().push(layout.clone());
            self.sampled.clear();
            self.sampled.resize(layout.selected.len(), self.token);
            Ok(&self.sampled)
        }
    }

    /// An entry for `request` in the slot of the same number, sampling under the default
    /// parameters when `samples`.
    pub(crate) fn entry(
        request: u64,
        context_len: usize,
        input_tokens: Vec<u32>,
        blocks: &[u32],
        samples: bool,
    ) -> CommandEntry {
        let slot = u32::try_from(request).expect("test request numbers fit u32");
        CommandEntry {
            request: RequestId::new(request),
            slot: RequestSlot::new(slot),
            sequence: SequenceIndex::new(0),
            context_len,
            input_tokens,
            block_table: blocks.iter().map(|&block| BlockId::new(block)).collect(),
            sampling: samples.then(SamplingParams::default),
        }
    }

    /// A one-token decode entry for `request` in `slot`, sampling under `params`.
    pub(crate) fn sampling_entry(request: u64, slot: u32, params: SamplingParams) -> CommandEntry {
        CommandEntry {
            slot: RequestSlot::new(slot),
            sampling: Some(params),
            ..entry(request, 0, vec![1], &[10], true)
        }
    }

    /// A padding dummy's entry: one token over its own block, never sampling.
    pub(crate) fn dummy(request: u64, block: u32) -> CommandEntry {
        entry(request, 0, vec![PADDING_TOKEN], &[block], false)
    }

    /// A step command over `live` entries as the engine would issue it: the dispatcher's
    /// decision for them under [`engine_config`], and as many dummies as their bucket needs,
    /// each over its own block.
    pub(crate) fn keyed_command(live: Vec<CommandEntry>) -> StepCommand {
        let config = engine_config();
        let mut dispatcher = Dispatcher::new(&config.dispatch, &contract());
        let token_count: usize = live.iter().map(CommandEntry::query_len).sum();
        let dispatch = dispatcher.dispatch(LiveBatch {
            token_count: tokens(token_count),
            request_count: requests(live.len()),
            uniform_decode: live.iter().all(|entry| entry.query_len() == 1),
        });
        let padding_count = match dispatch {
            DispatchDecision::FullReplay(key) | DispatchDecision::SegmentedReplay(key) => {
                key.padded_token_count().get() - token_count
            }
            DispatchDecision::Eager(_) => 0,
        };
        let mut entries = live;
        let first_dummy = entries.len() as u64 + 1;
        let first_block = 100;
        entries.extend((0..padding_count).map(|index| {
            dummy(
                first_dummy + index as u64,
                first_block + u32::try_from(index).expect("a padding count fits u32"),
            )
        }));
        StepCommand {
            step: StepId::new(1),
            entries,
            padding_count,
            dispatch,
        }
    }

    /// An eager step command over `entries`, the last `padding_count` of which are dummies.
    pub(crate) fn command(entries: Vec<CommandEntry>, padding_count: usize) -> StepCommand {
        StepCommand {
            step: StepId::new(1),
            entries,
            padding_count,
            dispatch: DispatchDecision::Eager(EagerReason::RequestsAboveCapturedMaximum {
                request_count: RequestCount::new(1).expect("nonzero"),
                captured_maximum: RequestCount::new(1).expect("nonzero"),
            }),
        }
    }
}
