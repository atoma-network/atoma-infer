//! Decode parity, the capture of the bucket ladder and its replay, on a device.
//!
//! Builds the decode step over runtime tensors beside the candle forward on the same weights and
//! KV cache, captures every bucket of the bucket ladder into a graph over the padding dummies'
//! blocks — no graph may allocate or free memory — then runs decode steps of varying ids,
//! lengths and block tables through three paths and compares them: the eager step against
//! candle on their logits, where the argmax of every live row must agree and the largest
//! absolute difference is reported against a bound, and the replay of the bucket's graph against
//! the eager step on the token it samples and the cache it writes, which must be the eager
//! step's argmax and the eager step's writes bit for bit. Every row also goes through candle
//! alone, so the run measures what candle's own logits do when nothing changes but the live
//! batch it is computed in; that spread is the floor the step is read against, and it is printed
//! beside the step's.
//!
//! The replay runs first at every step, over a snapshot of the cache, and must write each live
//! row's slot and nothing else in the blocks it may touch; the eager step then runs the same
//! command and its writes must be the replay's bit for bit. Run the other way round, a replay
//! that skipped its copy-in or its model step would read the eager step's inputs and outputs as
//! its own and pass.
//!
//! A replay reads the token of a row the device has sampled for before from the device, through
//! the graph's gather, so the keyed command a replay runs carries a decoy token in the host's
//! copy for every such row: a graph whose model step ran ahead of its gather would decode the
//! decoy, and its token would not be the eager step's. The batch sizes of the first steps run
//! through every bucket of the bucket ladder, so every graph is replayed.
//!
//! Free device memory is read through the runtime context around every eager step and every
//! replay, and across a soak of replays, and it must not change: the session captures in
//! relaxed mode, where an allocation from the capturing thread is legal, so the recording alone
//! does not prove a step allocates nothing, and a lazy allocation that stays is what the
//! free-memory check catches. That the reading responds is shown first, with a scratch
//! allocation it must see.
//!
//! Needs a device, the CUDA toolkit and a Llama checkpoint loadable in bf16; run through
//! `scripts/decode-parity.sh`. Under NCCL the decode step stays on candle and there is nothing
//! to compare.

#![cfg(all(feature = "cuda", not(feature = "nccl")))]
// The evidence block is this test's product; it goes to stdout on purpose.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::cmp::Ordering;
use std::env;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

use atoma_core::attention::{CaptureContract, ModelDeclaration};
use atoma_core::dispatch::{
    BucketLadder, DispatchConfig, DispatchDecision, Dispatcher, EagerReason, LiveBatch,
};
use atoma_core::request::{SamplingParams, PADDING_TOKEN};
use atoma_core::step::{CommandEntry, StepCommand};
use atoma_core::types::{
    BlockId, RequestCount, RequestId, RequestSlot, SequenceIndex, StepId, TokenCount,
};
use atoma_engine::batch::BatchLayout;
use atoma_engine::config::{DeviceOrdinal, Dtype, ModelConfig, ModelId, PromptTemplate};
use atoma_engine::decode::declaration;
use atoma_engine::decode::graphs::CaptureReport;
use atoma_engine::decode::ring::StagingDepth;
use atoma_engine::device::capture::{capture_bucket_ladder, Captured};
use atoma_engine::device::decode::{DecodeStep, DecodeStepPlan};
use atoma_engine::device::forward::{Allocated, CudaForward};
use atoma_engine::device::sampler::DeviceSampler;
use atoma_engine::device::{Checkpoint, KvCache, KvGeometry, RankDevice, Weights};
use atoma_engine::forward::Forward;
use atoma_engine::model::{fetch, llama_config};
use atoma_engine::readback::Readback;
use atoma_runtime::arena::BucketIdx;
use atoma_runtime::context::{DeviceBytes, RuntimeContext};
use atoma_runtime::session::Allocation;
use candle_core::{DType, Tensor};
use cudarc::driver::result::{free_sync, malloc_sync};
use cudarc::driver::{sys, CudaContext};

const DEFAULT_MODEL: &str = "NousResearch/Meta-Llama-3.1-8B-Instruct";
const BLOCK_SIZE: usize = 16;
const BLOCK_COUNT: usize = 512;
const MAX_MODEL_LEN: usize = 512;
/// As many sequences as the maximum batch, which is what the largest bucket the step serves
/// holds, so every bucket is reached.
const SEQUENCES: usize = MAX_BATCH;
const STEPS: usize = 32;
const LADDER: [usize; 4] = [1, 2, 4, 8];
const MAX_BATCH: usize = 8;
/// The request slots the sampler holds: one per sequence and one per padding dummy, as the
/// engine sizes them.
const SLOTS: usize = SEQUENCES + MAX_BATCH;
/// What the free-memory reading is shown to respond to, ahead of anything it is asked to hold
/// still across: larger than any driver chunk.
const SCRATCH_BYTES: usize = 64 * 1024 * 1024;
/// Replays of one keyed step run back to back, with free memory read before and after.
const SOAK_REPLAYS: usize = 128;
/// The largest absolute difference on the f32 logits accepted unless `PARITY_MAX_ABS_DIFF`
/// says otherwise; the measured value is printed either way. Above what candle's own logits move
/// by when only the live batch they are computed in changes, which the run measures: on an A100,
/// 0.375 for both Llama 3.1 8B and 3.2 1B, against 0.579 for the step.
const DEFAULT_MAX_ABS_DIFF: f32 = 0.75;
/// The largest absolute difference accepted between the key and value rows the step writes into
/// the cache and candle's writes of the same slots, unless `PARITY_KV_MAX_ABS_DIFF` says
/// otherwise; the measured value is printed either way. Both paths write bf16 from their own
/// projections, and the step rotates keys in f32 where candle rotates in bf16: on an A100, 0.75
/// for Llama 3.1 8B.
const DEFAULT_KV_MAX_ABS_DIFF: f32 = 1.0;
/// Prompt tokens are drawn below this id: Llama 3's special tokens sit at the top of the
/// vocabulary, and a prompt of those is not a prompt.
const TOKEN_ID_CEILING: usize = 120_000;

fn tokens(value: usize) -> TokenCount {
    TokenCount::new(value).expect("nonzero")
}

fn requests(value: usize) -> RequestCount {
    RequestCount::new(value).expect("nonzero")
}

/// A small deterministic generator, so a run is reproducible from its seed alone.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next()).expect("fits") % bound
    }
}

/// One sequence under test: its tokens so far, the blocks it owns, how many tokens the cache
/// holds, and where its last token is.
struct Sequence {
    tokens: Vec<u32>,
    blocks: Vec<u32>,
    context_len: usize,
    last_token: LastToken,
}

/// Where a sequence's last token is: on the host alone until a replay samples for the sequence,
/// then on the device too, where the next replay's gather reads it and the host's copy is not.
#[derive(Clone, Copy)]
enum LastToken {
    OnHost,
    OnDevice,
}

impl Sequence {
    fn entry(&self, index: usize, input: Vec<u32>) -> CommandEntry {
        CommandEntry {
            request: RequestId::new(index as u64 + 1),
            slot: RequestSlot::new(u32::try_from(index).expect("fits")),
            sequence: SequenceIndex::new(0),
            context_len: self.context_len,
            input_tokens: input,
            block_table: self
                .blocks
                .iter()
                .map(|&block| BlockId::new(block))
                .collect(),
            sampling: Some(SamplingParams::default()),
        }
    }

    /// The token this sequence decodes next: the first one the cache does not hold.
    fn next_token(&self) -> u32 {
        self.tokens[self.context_len]
    }

    /// The token the host's copy carries for a replay: a decoy once the device holds the
    /// sequence's last token, since the graph's gather is what supplies it then, and the true
    /// token before.
    fn replay_input(&self, vocab: usize) -> u32 {
        let token = self.next_token();
        let LastToken::OnDevice = self.last_token else {
            return token;
        };
        if usize::try_from(token).expect("fits") + 1 < vocab {
            token + 1
        } else {
            token - 1
        }
    }
}

fn dummy(index: usize, block: u32) -> CommandEntry {
    CommandEntry {
        request: RequestId::new(1000 + index as u64),
        slot: RequestSlot::new(u32::try_from(SEQUENCES + index).expect("fits")),
        sequence: SequenceIndex::new(0),
        context_len: 0,
        input_tokens: vec![PADDING_TOKEN],
        block_table: vec![BlockId::new(block)],
        sampling: None,
    }
}

fn dispatch_config() -> DispatchConfig {
    DispatchConfig {
        bucket_ladder: BucketLadder::new(LADDER.to_vec()).expect("nonzero buckets"),
        captured_max_requests: requests(MAX_BATCH),
    }
}

fn eager() -> DispatchDecision {
    DispatchDecision::Eager(EagerReason::NotUniformDecode {
        token_count: tokens(1),
        request_count: requests(1),
    })
}

fn lay_out(command: &StepCommand) -> BatchLayout {
    BatchLayout::lay_out(command, tokens(BLOCK_SIZE)).expect("the command lays out")
}

/// The padding dummies' blocks: the last of the pool, one per dummy of the maximum batch, which
/// is what the capture fills every bucket's dummy run from. A live step's padding rows all sit
/// on the last of them.
fn dummy_blocks() -> Vec<BlockId> {
    (BLOCK_COUNT - MAX_BATCH..BLOCK_COUNT)
        .map(|block| BlockId::new(u32::try_from(block).expect("fits")))
        .collect()
}

/// One decode step over the live sequences, three ways: the keyed command the engine would
/// issue, padded to its bucket with dummies over `dummy_block`, for the eager step; the same
/// command with a decoy token in every row the device holds the token of, for the replay; and
/// the command marked eager, for candle.
struct Commands {
    keyed: StepCommand,
    replayed: StepCommand,
    on_candle: StepCommand,
}

fn decode_commands(
    live: &[(usize, &Sequence)],
    dispatcher: &mut Dispatcher,
    dummy_block: u32,
    vocab: usize,
    step: u64,
) -> Commands {
    let dispatch = dispatcher.dispatch(LiveBatch {
        token_count: tokens(live.len()),
        request_count: requests(live.len()),
        uniform_decode: true,
    });
    let DispatchDecision::FullReplay(key) = dispatch else {
        panic!("a uniform decode of {} is keyed: {dispatch:?}", live.len());
    };
    let padding_count = key.padded_token_count().get() - live.len();
    let entries = |input: &dyn Fn(&Sequence) -> u32| -> Vec<CommandEntry> {
        let mut entries: Vec<CommandEntry> = live
            .iter()
            .map(|&(index, sequence)| sequence.entry(index, vec![input(sequence)]))
            .collect();
        entries.extend((0..padding_count).map(|index| dummy(index, dummy_block)));
        entries
    };
    let keyed = StepCommand {
        step: StepId::new(step),
        entries: entries(&Sequence::next_token),
        padding_count,
        dispatch,
    };
    let replayed = StepCommand {
        step: StepId::new(step),
        entries: entries(&|sequence| sequence.replay_input(vocab)),
        padding_count,
        dispatch,
    };
    let on_candle = StepCommand {
        step: StepId::new(step),
        entries: keyed.entries.clone(),
        padding_count,
        dispatch: eager(),
    };
    Commands {
        keyed,
        replayed,
        on_candle,
    }
}

/// Everything the Allocation phase produced for the device under test.
struct Rig {
    context: RuntimeContext,
    allocation: Allocation,
    allocated: Allocated,
    decode_step: DecodeStep,
    /// What every recording of the bucket ladder holds the sample of, and what a replay samples
    /// through.
    sampler: DeviceSampler,
    /// The harness reads logits through a readback of its own; the sampler's brings tokens.
    readback: Readback<f32>,
    vocab: usize,
}

/// Opens device zero, loads `model` in bf16 and builds both forwards over it.
fn open(model: &ModelConfig) -> Rig {
    let files = fetch(model).expect("the checkpoint fetches");
    let config = llama_config(&files.config).expect("the config reads");
    let context = RuntimeContext::new(0).expect("device 0 opens");
    let allocation = Allocation::new(&context).expect("the session opens");
    let device = RankDevice::open(&allocation, DeviceOrdinal::new(0)).expect("candle opens");
    let checkpoint = Checkpoint {
        files: &files,
        config: &config,
        dtype: model.dtype.into(),
    };
    let weights = Weights::load(&allocation, &device, checkpoint).expect("the weights load");
    let geometry =
        KvGeometry::new(&config, BLOCK_COUNT, tokens(BLOCK_SIZE), 1).expect("the geometry");
    let kv_cache = KvCache::allocate(&allocation, &device, &config, geometry, model.dtype.into())
        .expect("the cache allocates");
    let readback = Readback::new(
        &allocation,
        device.stream().context(),
        MAX_BATCH,
        config.vocab_size,
    )
    .expect("the readback pins");
    let sampler = DeviceSampler::new(
        &allocation,
        device.stream(),
        SLOTS,
        requests(MAX_BATCH),
        config.vocab_size,
    )
    .expect("the sampler allocates");
    let plan = DecodeStepPlan {
        dispatch: dispatch_config(),
        max_batch: requests(MAX_BATCH),
        max_model_len: tokens(MAX_MODEL_LEN),
        block_size: tokens(BLOCK_SIZE),
        dtype: model.dtype,
        staging_depth: StagingDepth::default(),
    };
    let decode_step = DecodeStep::build(&allocation, &device, &weights, &kv_cache, &plan)
        .expect("the decode step builds");
    Rig {
        context,
        allocation,
        allocated: Allocated {
            device,
            weights,
            kv_cache,
            // The sampler joins once the bucket ladder is captured over it.
            sampler: None,
            vocab: config.vocab_size,
        },
        decode_step,
        sampler,
        readback,
        vocab: config.vocab_size,
    }
}

/// What one bucket's graph holds: its nodes, and how many of them allocate or free memory.
struct GraphNodes {
    nodes: usize,
    memory_nodes: usize,
}

/// Prints what capturing the bucket ladder cost and what each graph holds, and holds every
/// graph to allocating or freeing nothing.
fn report_capture(captured: &Captured) -> Vec<GraphNodes> {
    let Captured {
        session,
        graphs,
        report,
    } = captured;
    let nodes: Vec<GraphNodes> = LADDER
        .iter()
        .enumerate()
        .map(|(index, &rows)| {
            let graph = graphs
                .graph(BucketIdx(index))
                .expect("every bucket of the bucket ladder was captured");
            let recorded = session.entry(graph).graph();
            let nodes = recorded.node_count().expect("the graph reports its nodes");
            let memory_nodes = recorded
                .memory_node_count()
                .expect("the graph reports its node types");
            println!(
                "capture: the bucket of {rows} recorded as a graph of {nodes} nodes, \
                 {memory_nodes} of them allocating or freeing memory"
            );
            assert_eq!(
                memory_nodes, 0,
                "the bucket of {rows}'s graph allocates or frees memory"
            );
            GraphNodes {
                nodes,
                memory_nodes,
            }
        })
        .collect();
    for cost in &report.graphs {
        println!(
            "capture: bucket {} of {} rows took {:?} and used {} bytes; {} bytes free after",
            cost.bucket.0, cost.rows, cost.elapsed, cost.used, cost.free
        );
    }
    println!(
        "capture: {} graphs took {:?} and used {} bytes in all; {} bytes free after",
        report.graphs.len(),
        report.elapsed,
        report.used,
        report.free
    );
    match report.graph_memory() {
        Ok(memory) => println!(
            "capture: graph memory fits {} bytes fixed plus {} bytes a graph",
            memory.fixed(),
            memory.marginal()
        ),
        Err(error) => println!("capture: {error}"),
    }
    nodes
}

/// Shows the free-memory reading responds: a scratch allocation of `SCRATCH_BYTES` straight
/// from the driver, past any pool, must drop it by at least that much. A reading that held at
/// one value would pass every check that asks it to hold still, so it is asked to move first.
fn free_memory_responds(context: &RuntimeContext) -> DeviceBytes {
    let idle = free_memory(context);
    // SAFETY: a plain device allocation on the context the reading just bound to this thread,
    // freed below before anything else is asked of the device.
    let scratch = unsafe { malloc_sync(SCRATCH_BYTES) }.expect("the scratch allocates");
    let held = free_memory(context);
    // SAFETY: the allocation above, freed once.
    unsafe { free_sync(scratch) }.expect("the scratch frees");
    assert!(
        idle.saturating_sub(held).get() >= SCRATCH_BYTES,
        "the free-memory reading did not see {SCRATCH_BYTES} bytes allocated: {idle} free before, \
         {held} after"
    );
    println!(
        "free memory: {idle} bytes idle, {held} with {SCRATCH_BYTES} bytes of scratch held; the \
         reading responds"
    );
    free_memory(context)
}

fn free_memory(context: &RuntimeContext) -> DeviceBytes {
    context
        .free_memory()
        .expect("the device reports its free memory")
}

/// Prefills every sequence through candle over its own blocks, and appends the token the
/// prefill's largest logit names.
fn prefill(forward: &mut CudaForward, readback: &mut Readback<f32>, sequences: &mut [Sequence]) {
    for (index, sequence) in sequences.iter_mut().enumerate() {
        let command = StepCommand {
            step: StepId::new(100 + index as u64),
            entries: vec![sequence.entry(index, sequence.tokens.clone())],
            padding_count: 0,
            dispatch: eager(),
        };
        let logits = forward
            .forward_logits(&lay_out(&command), readback)
            .expect("the prefill runs on candle");
        let next = argmax(logits.row(0).expect("one row"));
        sequence.context_len = sequence.tokens.len();
        sequence.tokens.push(next);
    }
}

/// What the comparison of every decode step found.
#[derive(Default)]
struct Parity {
    rows: usize,
    argmax_disagreements: usize,
    /// Rows whose argmaxes differ on ids candle's own logits hold at one value: bf16 cannot
    /// separate them, so the reference has no order to disagree with.
    ties: usize,
    max_abs_diff: f32,
    sum_abs_diff: f64,
    /// The same rows through candle alone against candle in the live batch.
    candle_max_abs_diff: f32,
    candle_sum_abs_diff: f64,
    /// The key and value rows the step wrote into the cache against candle's writes of the same
    /// slots.
    kv_max_abs_diff: f32,
    /// Rows the replay sampled a token for that is not the eager step's argmax, and rows where
    /// the two ids hold one f32 value in the eager step's logits, so the graph's argmax had no
    /// order to keep.
    replay_disagreements: usize,
    replay_ties: usize,
    /// How many steps each bucket of the bucket ladder was replayed at.
    replays_per_bucket: [usize; LADDER.len()],
}

impl Parity {
    fn mean_abs_diff(&self) -> f64 {
        self.sum_abs_diff / self.row_count()
    }

    fn candle_mean_abs_diff(&self) -> f64 {
        self.candle_sum_abs_diff / self.row_count()
    }

    fn row_count(&self) -> f64 {
        f64::from(u32::try_from(self.rows).expect("a row count fits"))
    }
}

/// Both forwards over the sequences under test, and what comparing them has found so far.
struct Harness {
    context: RuntimeContext,
    forward: CudaForward,
    readback: Readback<f32>,
    sequences: Vec<Sequence>,
    dispatcher: Dispatcher,
    parity: Parity,
    /// Every layer's cache, the handles candle holds, for reading slots back.
    cache: Vec<Tensor>,
    /// The device's stream-ordered allocator, watched around every keyed step.
    pool: Option<sys::CUmemoryPool>,
    vocab: usize,
}

/// Runs each live entry through candle on its own, one row to a step, so every row has the
/// reference computed in a second shape. Candle's logits move with the live batch, so a pair of
/// ids it orders one way batched and the other way alone is a pair it cannot order at all.
fn candle_alone(
    forward: &mut CudaForward,
    readback: &mut Readback<f32>,
    live: &[(usize, &Sequence)],
    step: usize,
) -> Vec<Vec<f32>> {
    live.iter()
        .map(|&(index, sequence)| {
            let command = StepCommand {
                step: StepId::new(500 + step as u64),
                entries: vec![sequence.entry(index, vec![sequence.next_token()])],
                padding_count: 0,
                dispatch: eager(),
            };
            let logits = forward
                .forward_logits(&lay_out(&command), readback)
                .expect("the one-entry step runs on candle");
            logits.row(0).expect("row").to_vec()
        })
        .collect()
}

/// Runs one decode step over `chosen` three ways and compares them: the replay of the bucket's
/// graph first, which must write each live row's slot and nothing else, then the eager decode
/// step, whose cache writes must be the replay's bit for bit and whose logits are compared
/// against candle row by row, with the token the replay sampled held to the eager step's argmax.
/// Then advances each chosen sequence by that token. Every chosen sequence also decodes alone
/// on candle, which measures candle against itself over the same row; the live batch runs on
/// candle last, so what the cache holds at the end of the step is what it held before this
/// measurement was taken.
fn compare_step(harness: &mut Harness, chosen: &[usize], step: usize) {
    let Harness {
        context,
        forward,
        readback,
        sequences,
        dispatcher,
        parity,
        cache,
        pool,
        vocab,
    } = harness;
    let dummy_block = u32::try_from(BLOCK_COUNT - 1).expect("fits");
    let live: Vec<(usize, &Sequence)> = chosen
        .iter()
        .map(|&index| (index, &sequences[index]))
        .collect();
    let commands = decode_commands(&live, dispatcher, dummy_block, *vocab, 200 + step as u64);
    let keyed = lay_out(&commands.keyed);
    let written: Vec<usize> = keyed.slot_mapping[..chosen.len()]
        .iter()
        .map(|&slot| usize::try_from(slot).expect("a live row's slot"))
        .collect();
    let kv_width = kv_width(cache);
    let before = snapshot(cache, &written);
    let replayed = lay_out(&commands.replayed);
    let sampled: Vec<u32> = holding_free_memory(context, *pool, step, "the replay", || {
        forward
            .forward(&replayed)
            .expect("the keyed batch replays its bucket's graph")
            .to_vec()
    });
    let after_replay = snapshot(cache, &written);
    check_step_writes(
        &before,
        &after_replay,
        &written,
        kv_width,
        &format!("step {step}, the replay"),
    );

    let tensor_logits: Vec<Vec<f32>> =
        holding_free_memory(context, *pool, step, "the eager decode step", || {
            let logits = forward
                .forward_logits(&keyed, readback)
                .expect("the keyed batch runs on the decode step");
            (0..logits.rows())
                .map(|row| logits.row(row).expect("row").to_vec())
                .collect()
        });
    let after_step = snapshot(cache, &written);
    assert!(
        identical(&after_replay, &after_step),
        "step {step}: the eager step's cache writes are not the replay's, bit for bit"
    );
    let bucket = LADDER
        .iter()
        .position(|&rows| rows == keyed.entry_count())
        .expect("a keyed batch is padded to a bucket of the bucket ladder");
    parity.replays_per_bucket[bucket] += 1;

    let alone = candle_alone(forward, readback, &live, step);
    let candle_logits = forward
        .forward_logits(&lay_out(&commands.on_candle), readback)
        .expect("the eager step runs on candle");
    let after_candle = snapshot(cache, &written);
    let kv_diff = widest_slot_diff(&after_step, &after_candle, &written, kv_width);
    parity.kv_max_abs_diff = parity.kv_max_abs_diff.max(kv_diff);
    assert_eq!(tensor_logits.len(), chosen.len(), "one row per live entry");
    assert_eq!(candle_logits.rows(), chosen.len());
    assert_eq!(sampled.len(), chosen.len(), "one token per live entry");
    for (row, tensor_row) in tensor_logits.iter().enumerate() {
        let candle_row = candle_logits.row(row).expect("row");
        record_argmax(parity, step, row, tensor_row, (candle_row, &alone[row]));
        record_replay(parity, step, row, tensor_row, sampled[row]);
        let diff = widest(tensor_row, candle_row);
        parity.max_abs_diff = parity.max_abs_diff.max(diff);
        parity.sum_abs_diff += f64::from(diff);
        let candle_diff = widest(&alone[row], candle_row);
        parity.candle_max_abs_diff = parity.candle_max_abs_diff.max(candle_diff);
        parity.candle_sum_abs_diff += f64::from(candle_diff);
        parity.rows += 1;
    }
    for (&index, next) in chosen.iter().zip(sampled) {
        let sequence = &mut sequences[index];
        sequence.context_len += 1;
        sequence.tokens.push(next);
        sequence.last_token = LastToken::OnDevice;
    }
}

/// Runs `work` with free device memory read through the runtime context before and after it,
/// and the stream-ordered allocator's high-water mark watched across it, and holds both still:
/// `what` allocated nothing that stayed and took nothing from the pool.
fn holding_free_memory<T>(
    context: &RuntimeContext,
    pool: Option<sys::CUmemoryPool>,
    step: usize,
    what: &str,
    work: impl FnOnce() -> T,
) -> T {
    let free_before = free_memory(context);
    let used_before = pool.map(pool_watch);
    let result = work();
    let free_after = free_memory(context);
    assert_eq!(
        free_before,
        free_after,
        "step {step}: {what} left {} bytes allocated",
        free_before.get().abs_diff(free_after.get())
    );
    if let (Some(pool), Some(used_before)) = (pool, used_before) {
        let high = pool_high(pool);
        assert_eq!(
            high,
            used_before,
            "step {step}: {what} took {} bytes from the stream-ordered allocator",
            high.abs_diff(used_before)
        );
    }
    result
}

/// Compares the two forwards' argmax on one row, counting a disagreement or a tie. `reference`
/// is candle's row twice: computed in the live batch, and computed alone.
fn record_argmax(
    parity: &mut Parity,
    step: usize,
    row: usize,
    tensor_row: &[f32],
    reference: (&[f32], &[f32]),
) {
    let (candle_row, alone_row) = reference;
    let (ours, theirs) = (argmax(tensor_row), argmax(candle_row));
    if ours == theirs {
        return;
    }
    // The reference orders the two ids only when it holds the same order in both shapes it was
    // computed in. Candle reads its logits back in bf16 and its logits move with the live batch
    // by far more than one bf16 step, so a pair it holds at one value, or ranks one way batched
    // and the other way alone, is a pair it cannot order: the row is a tie, not a disagreement.
    let (ours_at, theirs_at) = (at(ours), at(theirs));
    let batched = candle_row[ours_at].total_cmp(&candle_row[theirs_at]);
    let alone = alone_row[ours_at].total_cmp(&alone_row[theirs_at]);
    let tied = batched != alone || batched == Ordering::Equal;
    if tied {
        parity.ties += 1;
    } else {
        parity.argmax_disagreements += 1;
    }
    eprintln!(
        "step {step} row {row}: {} — ids {ours}/{theirs}; step {:.6}/{:.6}, candle batched \
         {:.6}/{:.6}, candle alone {:.6}/{:.6}",
        if tied { "tie" } else { "disagreement" },
        tensor_row[ours_at],
        tensor_row[theirs_at],
        candle_row[ours_at],
        candle_row[theirs_at],
        alone_row[ours_at],
        alone_row[theirs_at]
    );
}

/// Compares the token the replay sampled on one row with the eager step's argmax over the same
/// row, counting a disagreement, or a tie where the eager step's f32 logits hold the two ids at
/// one value and the graph's greedy sample had no order to keep.
fn record_replay(parity: &mut Parity, step: usize, row: usize, tensor_row: &[f32], sampled: u32) {
    let eager = argmax(tensor_row);
    if sampled == eager {
        return;
    }
    let (sampled_at, eager_at) = (at(sampled), at(eager));
    let tied = tensor_row[sampled_at].to_bits() == tensor_row[eager_at].to_bits();
    if tied {
        parity.replay_ties += 1;
    } else {
        parity.replay_disagreements += 1;
    }
    eprintln!(
        "step {step} row {row}: replay {} — sampled {sampled} (logit {:.4}), eager argmax {eager} \
         (logit {:.4})",
        if tied { "tie" } else { "disagreement" },
        tensor_row[sampled_at],
        tensor_row[eager_at]
    );
}

/// The bound the variable `name` sets, or `default` when it is unset or not a number.
fn bound_from_env(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|bound| bound.parse().ok())
        .unwrap_or(default)
}

/// `SEQUENCES` sequences of random tokens, each over its own run of blocks below the dummies'.
fn seed_sequences(random: &mut Lcg, vocab: usize) -> Vec<Sequence> {
    let blocks_each = MAX_MODEL_LEN.div_ceil(BLOCK_SIZE);
    assert!(
        SEQUENCES * blocks_each <= BLOCK_COUNT - MAX_BATCH,
        "the sequences' blocks stay below the padding dummies'"
    );
    (0..SEQUENCES)
        .map(|index| Sequence {
            tokens: (0..8 + random.below(40))
                .map(|_| u32::try_from(random.below(vocab.min(TOKEN_ID_CEILING))).expect("fits"))
                .collect(),
            blocks: (0..blocks_each)
                .map(|block| u32::try_from(index * blocks_each + block).expect("fits"))
                .collect(),
            context_len: 0,
            last_token: LastToken::OnHost,
        })
        .collect()
}

/// How many sequences step `step` decodes: the first steps walk the batch sizes up through every
/// bucket of the bucket ladder, so every graph is replayed, and the rest are drawn at random.
fn batch_size(step: usize, random: &mut Lcg) -> usize {
    if step < SEQUENCES {
        step + 1
    } else {
        1 + random.below(SEQUENCES)
    }
}

/// Replays one keyed step of every sequence `SOAK_REPLAYS` times back to back, with free device
/// memory read through the runtime context before and after: a replay that allocates anything
/// that stays would show as a drop across the soak. The sequences are not advanced, so every
/// replay decodes the same position and writes the same slots.
fn soak(harness: &mut Harness) -> SoakReadings {
    let Harness {
        context,
        forward,
        sequences,
        dispatcher,
        vocab,
        ..
    } = harness;
    let dummy_block = u32::try_from(BLOCK_COUNT - 1).expect("fits");
    let live: Vec<(usize, &Sequence)> = sequences.iter().enumerate().collect();
    let commands = decode_commands(&live, dispatcher, dummy_block, *vocab, 900);
    let replayed = lay_out(&commands.replayed);
    let before = free_memory(context);
    for _ in 0..SOAK_REPLAYS {
        forward
            .forward(&replayed)
            .expect("the keyed batch replays its bucket's graph");
    }
    let after = free_memory(context);
    SoakReadings { before, after }
}

/// Free device memory read before and after the soak.
#[derive(Clone, Copy)]
struct SoakReadings {
    before: DeviceBytes,
    after: DeviceBytes,
}

/// The bounds the run is held to: `PARITY_MAX_ABS_DIFF` on the f32 logits, the eager step
/// against candle, and `PARITY_KV_MAX_ABS_DIFF` on the key and value rows the step writes
/// against candle's writes of the same slots, each the default where the variable is unset.
struct Bounds {
    logits: f32,
    kv: f32,
}

impl Bounds {
    fn from_env() -> Self {
        Self {
            logits: bound_from_env("PARITY_MAX_ABS_DIFF", DEFAULT_MAX_ABS_DIFF),
            kv: bound_from_env("PARITY_KV_MAX_ABS_DIFF", DEFAULT_KV_MAX_ABS_DIFF),
        }
    }
}

/// What capturing the bucket ladder left to report: what each graph holds, what the capture
/// cost, and the free memory read once the reading was shown to respond.
struct CaptureEvidence {
    graphs: Vec<GraphNodes>,
    report: CaptureReport,
    free_after: DeviceBytes,
}

/// The checkpoint under test: `PARITY_MODEL`, or Llama 3.1 8B Instruct, in bf16.
fn model_under_test() -> ModelConfig {
    ModelConfig {
        id: env::var("PARITY_MODEL").map_or_else(|_| ModelId::new(DEFAULT_MODEL), ModelId::new),
        revision: "main".to_owned(),
        cache_dir: None,
        dtype: Dtype::Bf16,
        prompt_template: PromptTemplate::Llama3,
    }
}

/// Opens the device, captures the bucket ladder over the rig's dummies' blocks, and builds the
/// forward that serves from it beside the sequences under test, prefilled through candle: the
/// harness, and what the capture left to report.
fn build_harness(model: &ModelConfig, random: &mut Lcg) -> (Harness, CaptureEvidence) {
    let Rig {
        context,
        allocation,
        mut allocated,
        mut decode_step,
        mut sampler,
        readback,
        vocab,
    } = open(model);
    let contract = CaptureContract::resolve(&[declaration()], &ModelDeclaration::new("llama"));
    let dispatcher = Dispatcher::new(&dispatch_config(), &contract);

    let captured = capture_bucket_ladder(
        &context,
        allocation,
        &mut decode_step,
        &mut sampler,
        &dummy_blocks(),
    )
    .expect("every bucket of the bucket ladder captures");
    let graphs = report_capture(&captured);
    let Captured {
        session,
        graphs: graph_set,
        report,
    } = captured;
    let free_after = free_memory_responds(&context);
    let cache: Vec<Tensor> = allocated.kv_cache.layers().to_vec();
    let pool = default_pool(allocated.device.stream().context());
    if pool.is_none() {
        println!("pool: the device has no stream-ordered allocator to watch");
    }
    allocated.sampler = Some(sampler);
    let forward =
        CudaForward::new(allocated, decode_step, graph_set, session).expect("the forward builds");

    let sequences = seed_sequences(random, vocab);
    let mut harness = Harness {
        context,
        forward,
        readback,
        sequences,
        dispatcher,
        parity: Parity::default(),
        cache,
        pool,
        vocab,
    };
    prefill(
        &mut harness.forward,
        &mut harness.readback,
        &mut harness.sequences,
    );
    (
        harness,
        CaptureEvidence {
            graphs,
            report,
            free_after,
        },
    )
}

/// Prints the evidence block: what every comparison found, what the capture cost, and what the
/// soak read.
fn print_evidence(
    model: &ModelConfig,
    parity: &Parity,
    capture: &CaptureEvidence,
    bounds: &Bounds,
    soak: SoakReadings,
) {
    println!("=============== decode parity evidence ===============");
    println!("model:                {}", model.id);
    println!("decode steps:         {STEPS}");
    println!("rows compared:        {}", parity.rows);
    println!("argmax disagreements: {}", parity.argmax_disagreements);
    println!("argmax ties:          {}", parity.ties);
    println!("max |logit diff|:     {:.6}", parity.max_abs_diff);
    println!("mean |logit diff|:    {:.6}", parity.mean_abs_diff());
    println!("bound:                {}", bounds.logits);
    println!("candle alone against candle in the live batch:");
    println!("  max |logit diff|:   {:.6}", parity.candle_max_abs_diff);
    println!("  mean |logit diff|:  {:.6}", parity.candle_mean_abs_diff());
    println!("cache writes, the step against candle over the same slots:");
    println!("  max |k/v diff|:     {:.6}", parity.kv_max_abs_diff);
    println!("  bound:              {}", bounds.kv);
    let report = &capture.report;
    println!("bucket ladder capture, {} graphs:", capture.graphs.len());
    for ((rows, graph), cost) in LADDER.iter().zip(&capture.graphs).zip(&report.graphs) {
        println!(
            "  bucket {rows:>2}: {} nodes, {} memory nodes, {:?}, {} bytes",
            graph.nodes, graph.memory_nodes, cost.elapsed, cost.used
        );
    }
    println!(
        "  in all:    {:?}, {} bytes; {} bytes free after",
        report.elapsed, report.used, report.free
    );
    match report.graph_memory() {
        Ok(memory) => println!(
            "  fit:       {} bytes fixed + {} bytes a graph",
            memory.fixed(),
            memory.marginal()
        ),
        Err(error) => println!("  fit:       {error}"),
    }
    println!("replay against the eager step:");
    println!("  disagreements:      {}", parity.replay_disagreements);
    println!("  ties:               {}", parity.replay_ties);
    for (rows, replays) in LADDER.iter().zip(parity.replays_per_bucket) {
        println!("  bucket {rows:>2} replayed: {replays} steps");
    }
    println!(
        "free memory: {} bytes after the capture; {SOAK_REPLAYS} replays: {} bytes before, {} \
         after",
        capture.free_after, soak.before, soak.after
    );
}

/// Holds the run to its bounds: every argmax and every replayed token agrees where the
/// reference orders the ids, every bucket was replayed, and the soak allocated nothing that
/// stayed.
fn assert_evidence(parity: &Parity, bounds: &Bounds, soak: SoakReadings) {
    assert_eq!(
        parity.argmax_disagreements, 0,
        "every live row's argmax agrees on ids candle orders the same way batched and alone"
    );
    assert!(
        parity.max_abs_diff <= bounds.logits,
        "the largest logit difference {} is above the bound {}",
        parity.max_abs_diff,
        bounds.logits
    );
    assert!(
        parity.kv_max_abs_diff <= bounds.kv,
        "the largest cache-write difference {} is above the bound {}",
        parity.kv_max_abs_diff,
        bounds.kv
    );
    assert_eq!(
        parity.replay_disagreements, 0,
        "every replay samples the eager step's argmax on ids the eager logits separate"
    );
    assert!(
        parity.replays_per_bucket.iter().all(|&replays| replays > 0),
        "every bucket of the bucket ladder was replayed: {:?}",
        parity.replays_per_bucket
    );
    assert_eq!(
        soak.before,
        soak.after,
        "{SOAK_REPLAYS} replays left {} bytes allocated",
        soak.before.get().abs_diff(soak.after.get())
    );
}

#[test]
#[ignore = "needs a device, the CUDA toolkit and a Llama checkpoint; run scripts/decode-parity.sh"]
fn the_two_forwards_agree_on_every_decode_and_every_bucket_captures_and_replays() {
    let model = model_under_test();
    let bounds = Bounds::from_env();
    let mut random = Lcg(0x5EED_2026_0903);
    let (mut harness, capture) = build_harness(&model, &mut random);

    for step in 0..STEPS {
        let mut chosen: Vec<usize> = (0..SEQUENCES).collect();
        for index in (1..SEQUENCES).rev() {
            chosen.swap(index, random.below(index + 1));
        }
        chosen.truncate(batch_size(step, &mut random));
        chosen.sort_unstable();
        compare_step(&mut harness, &chosen, step);
    }
    let soak = soak(&mut harness);

    print_evidence(&model, &harness.parity, &capture, &bounds, soak);
    assert_evidence(&harness.parity, &bounds, soak);
}

/// Whether two snapshots hold the same values bit for bit, `-0.0` and `NaN` included.
fn identical(a: &Snapshot, b: &Snapshot) -> bool {
    let same = |x: &[f32], y: &[f32]| {
        x.len() == y.len() && x.iter().zip(y).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    a.rows.len() == b.rows.len()
        && a.rows
            .iter()
            .zip(&b.rows)
            .all(|(x, y)| x.len() == y.len() && x.iter().zip(y).all(|(x, y)| same(x, y)))
        && a.dummy.len() == b.dummy.len()
        && a.dummy.iter().zip(&b.dummy).all(|(x, y)| same(x, y))
}

/// Elements one slot holds for K or V: every key-value head's row.
fn kv_width(cache: &[Tensor]) -> usize {
    let dims = cache[0].dims();
    dims[3] * dims[4]
}

/// The K then V rows of one block of every layer, as f32 on the host: `[block_size, kv_width]`
/// each, row-major.
fn block_of(cache: &[Tensor], block: usize) -> Vec<Vec<f32>> {
    cache
        .iter()
        .map(|layer| {
            layer
                .narrow(1, block, 1)
                .expect("the block lies in the cache")
                .to_dtype(DType::F32)
                .expect("bf16 reads as f32")
                .flatten_all()
                .expect("flattens")
                .to_vec1::<f32>()
                .expect("copies to the host")
        })
        .collect()
}

/// The blocks a keyed step may write, read back: each live row's block, and the dummies'.
struct Snapshot {
    rows: Vec<Vec<Vec<f32>>>,
    dummy: Vec<Vec<f32>>,
}

fn snapshot(cache: &[Tensor], written: &[usize]) -> Snapshot {
    Snapshot {
        rows: written
            .iter()
            .map(|&slot| block_of(cache, slot / BLOCK_SIZE))
            .collect(),
        dummy: block_of(cache, BLOCK_COUNT - 1),
    }
}

/// The K and V ranges of the slot at `offset` in one block's values.
fn slot_ranges(offset: usize, kv_width: usize) -> [Range<usize>; 2] {
    let k = offset * kv_width..(offset + 1) * kv_width;
    let v_base = BLOCK_SIZE * kv_width;
    [k.clone(), v_base + k.start..v_base + k.end]
}

/// Holds `what`, one step over the cache, to writing each live row's own slot, and the dummies'
/// slot of the dummy block, and nothing else in those blocks. The eager step and candle's run of
/// the same batch overwrite the rows' slots afterwards, so a write that landed anywhere else
/// would otherwise go unseen: later steps would read the same wrong cache through every path.
fn check_step_writes(
    before: &Snapshot,
    after: &Snapshot,
    written: &[usize],
    kv_width: usize,
    what: &str,
) {
    for (row, (before, after)) in before.rows.iter().zip(&after.rows).enumerate() {
        let ranges = slot_ranges(written[row] % BLOCK_SIZE, kv_width);
        for (layer, (before, after)) in before.iter().zip(after).enumerate() {
            assert!(
                ranges
                    .iter()
                    .any(|range| before[range.clone()] != after[range.clone()]),
                "{what}, row {row} layer {layer}: no write to the row's slot"
            );
            untouched_outside(
                before,
                after,
                &ranges,
                &format!("{what}, row {row} layer {layer}"),
            );
        }
    }
    let ranges = slot_ranges(0, kv_width);
    for (layer, (before, after)) in before.dummy.iter().zip(&after.dummy).enumerate() {
        untouched_outside(
            before,
            after,
            &ranges,
            &format!("{what}, dummy block layer {layer}"),
        );
    }
}

/// Every value outside `ranges` is bit-identical between `before` and `after`.
fn untouched_outside(before: &[f32], after: &[f32], ranges: &[Range<usize>], at: &str) {
    for (index, (before, after)) in before.iter().zip(after).enumerate() {
        if ranges.iter().any(|range| range.contains(&index)) {
            continue;
        }
        assert!(
            before.to_bits() == after.to_bits(),
            "{at}: a write outside the slot, at value {index}"
        );
    }
}

/// The largest absolute difference between the slots the step wrote and candle's writes of the
/// same slots.
fn widest_slot_diff(step: &Snapshot, candle: &Snapshot, written: &[usize], kv_width: usize) -> f32 {
    let mut widest_diff = 0.0f32;
    for (row, (step, candle)) in step.rows.iter().zip(&candle.rows).enumerate() {
        for range in slot_ranges(written[row] % BLOCK_SIZE, kv_width) {
            for (step, candle) in step.iter().zip(candle) {
                widest_diff = widest_diff.max(widest(&step[range.clone()], &candle[range.clone()]));
            }
        }
    }
    widest_diff
}

/// The device's default stream-ordered allocator, or `None` where the driver has none.
fn default_pool(context: &Arc<CudaContext>) -> Option<sys::CUmemoryPool> {
    let mut pool: sys::CUmemoryPool = ptr::null_mut();
    // SAFETY: a driver query for the context's device; the out-pointer lives for the call.
    unsafe { sys::cuDeviceGetDefaultMemPool(&raw mut pool, context.cu_device()) }
        .result()
        .ok()
        .map(|()| pool)
}

/// Resets the pool's used-memory high-water mark and returns its usage now: a step that takes
/// nothing from the pool leaves the mark at this value.
fn pool_watch(pool: sys::CUmemoryPool) -> u64 {
    let mut zero = 0u64;
    // SAFETY: `pool` is the device's default pool, and the attribute takes a u64.
    unsafe {
        sys::cuMemPoolSetAttribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
            (&raw mut zero).cast(),
        )
    }
    .result()
    .expect("the high-water mark resets");
    pool_attribute(
        pool,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
    )
}

/// The pool's used-memory high-water mark since the last [`pool_watch`].
fn pool_high(pool: sys::CUmemoryPool) -> u64 {
    pool_attribute(
        pool,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
    )
}

fn pool_attribute(pool: sys::CUmemoryPool, attribute: sys::CUmemPool_attribute) -> u64 {
    let mut value = 0u64;
    // SAFETY: `pool` is the device's default pool, and both usage attributes are u64s.
    unsafe { sys::cuMemPoolGetAttribute(pool, attribute, (&raw mut value).cast()) }
        .result()
        .expect("the pool reports its usage");
    value
}

/// The largest absolute difference between two logits rows.
fn widest(row: &[f32], other: &[f32]) -> f32 {
    row.iter()
        .zip(other)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

/// A token id as an index into a logits row.
fn at(token: u32) -> usize {
    usize::try_from(token).expect("a token id indexes its row")
}

fn argmax(row: &[f32]) -> u32 {
    let (index, _) = row.iter().enumerate().fold(
        (0, f32::NEG_INFINITY),
        |(best, best_value), (index, &value)| {
            if value > best_value {
                (index, value)
            } else {
                (best, best_value)
            }
        },
    );
    u32::try_from(index).expect("fits")
}
