//! Decode parity and the capture check on a device.
//!
//! Builds the decode step over runtime tensors beside the candle forward on the same weights and
//! KV cache, records the step under capture over a dummy run to show the driver accepts it, then
//! runs decode steps of varying ids, lengths and block tables through both forwards and compares
//! their logits: the argmax of every live row must agree, and the largest absolute difference is
//! reported against a bound. Every row also goes through candle alone, so the run measures what
//! candle's own logits do when nothing changes but the live batch it is computed in; that spread
//! is the floor the step is read against, and it is printed beside the step's.
//!
//! Around every keyed step the device's free memory is read, and it must not change: the
//! session captures in relaxed mode, where an allocation from the capturing thread is legal, so
//! the recording alone does not prove the step allocates nothing, and a lazy allocation that
//! stays is what the free-memory check catches.
//!
//! Needs a device, the CUDA toolkit and a Llama checkpoint loadable in bf16; run through
//! `scripts/decode-parity.sh`. Under NCCL the decode step stays on candle and there is nothing
//! to compare.

#![cfg(all(feature = "cuda", not(feature = "nccl")))]
// The evidence block is this test's product; it goes to stdout on purpose.
#![allow(clippy::print_stdout, clippy::print_stderr)]

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
use atoma_engine::decode::graphs::GraphSet;
use atoma_engine::decode::ring::StagingDepth;
use atoma_engine::decode::staging::DummyRun;
use atoma_engine::device::decode::{DecodeStep, DecodeStepPlan};
use atoma_engine::device::forward::{Allocated, CudaForward};
use atoma_engine::device::{Checkpoint, KvCache, KvGeometry, RankDevice, Weights};
use atoma_engine::model::{fetch, llama_config};
use atoma_engine::readback::Readback;
use atoma_runtime::arena::BucketIdx;
use atoma_runtime::context::RuntimeContext;
use atoma_runtime::session::{Allocation, BakedBuffers, GraphIdx, Replay};
use candle_core::{DType, Tensor};
use cudarc::driver::result::mem_get_info;
use cudarc::driver::{sys, CudaContext};

const DEFAULT_MODEL: &str = "NousResearch/Meta-Llama-3.1-8B-Instruct";
const BLOCK_SIZE: usize = 16;
const BLOCK_COUNT: usize = 512;
const MAX_MODEL_LEN: usize = 512;
const SEQUENCES: usize = 4;
const STEPS: usize = 32;
const LADDER: [usize; 4] = [1, 2, 4, 8];
const MAX_BATCH: usize = 8;
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

/// One sequence under test: its tokens so far, the blocks it owns, and how many tokens the
/// cache holds.
struct Sequence {
    tokens: Vec<u32>,
    blocks: Vec<u32>,
    context_len: usize,
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
}

fn dummy(index: usize, block: u32) -> CommandEntry {
    CommandEntry {
        request: RequestId::new(1000 + index as u64),
        slot: RequestSlot::new(u32::try_from(1000 + index).expect("fits")),
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

/// A decode step over the `live` sequences: the keyed command the engine would issue, padded to
/// its bucket with dummies over `dummy_block`, and the same command marked eager.
fn decode_commands(
    live: &[(usize, &Sequence)],
    dispatcher: &mut Dispatcher,
    dummy_block: u32,
    step: u64,
) -> (StepCommand, StepCommand) {
    let dispatch = dispatcher.dispatch(LiveBatch {
        token_count: tokens(live.len()),
        request_count: requests(live.len()),
        uniform_decode: true,
    });
    let DispatchDecision::FullReplay(key) = dispatch else {
        panic!("a uniform decode of {} is keyed: {dispatch:?}", live.len());
    };
    let padding_count = key.padded_token_count().get() - live.len();
    let mut entries: Vec<CommandEntry> = live
        .iter()
        .map(|&(index, sequence)| sequence.entry(index, vec![sequence.next_token()]))
        .collect();
    entries.extend((0..padding_count).map(|index| dummy(index, dummy_block)));
    let keyed = StepCommand {
        step: StepId::new(step),
        entries: entries.clone(),
        padding_count,
        dispatch,
    };
    let on_candle = StepCommand {
        step: StepId::new(step),
        entries,
        padding_count,
        dispatch: eager(),
    };
    (keyed, on_candle)
}

/// Everything the Allocation phase produced for the device under test.
struct Rig {
    allocation: Allocation,
    allocated: Allocated,
    decode_step: DecodeStep,
    /// The harness reads logits, and samples nothing; the readback is its own.
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
        allocation,
        allocated: Allocated {
            device,
            weights,
            kv_cache,
            // The harness compares logits and samples nothing, so it holds no sampler.
            sampler: None,
            vocab: config.vocab_size,
        },
        decode_step,
        readback,
        vocab: config.vocab_size,
    }
}

/// The capture check: the bucket-of-one step, over a dummy run of one padding row on the dummy
/// block, warms up and records without the driver invalidating it. Returns the Replay phase, the
/// graph, its node count, and how many of its nodes allocate or free memory.
fn record_bucket_of_one(
    allocation: Allocation,
    decode_step: &mut DecodeStep,
    dummy_block: u32,
) -> (Replay, GraphIdx, usize, usize) {
    let run = DummyRun::new(BucketIdx(0), vec![BlockId::new(dummy_block)]);
    let entry = decode_step.stage_dummy(&run).expect("the dummy run stages");
    let mut capture = allocation.into_capture();
    capture
        .warm_up(
            &mut decode_step
                .copy_in(entry, run.bucket())
                .expect("bucket 0 is served"),
        )
        .expect("the copy-in runs eagerly");
    capture
        .warm_up(&mut decode_step.descriptor(run.bucket()).expect("bucket 0"))
        .expect("the step warms up");
    let graph = capture
        .record(
            &mut decode_step.descriptor(run.bucket()).expect("bucket 0"),
            BakedBuffers::default(),
        )
        .expect("the step records without invalidating the capture");
    let recorded = capture.entry(graph).graph();
    let nodes = recorded.node_count().expect("the graph reports its nodes");
    let memory_nodes = recorded
        .memory_node_count()
        .expect("the graph reports its node types");
    (capture.into_replay(), graph, nodes, memory_nodes)
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
    forward: CudaForward,
    readback: Readback<f32>,
    sequences: Vec<Sequence>,
    dispatcher: Dispatcher,
    parity: Parity,
    /// Every layer's cache, the handles candle holds, for reading slots back.
    cache: Vec<Tensor>,
    /// The device's stream-ordered allocator, watched around every keyed step.
    pool: Option<sys::CUmemoryPool>,
}

/// Runs one decode step over `chosen` through both forwards and compares them row by row, then
/// advances each chosen sequence by the token candle sampled. Every chosen sequence also decodes
/// alone on candle, which measures candle against itself over the same row; the live batch runs
/// last, so what the cache holds at the end of the step is what it held before this measurement
/// was taken.
fn compare_step(harness: &mut Harness, chosen: &[usize], step: usize) {
    let Harness {
        forward,
        readback,
        sequences,
        dispatcher,
        parity,
        cache,
        pool,
    } = harness;
    let dummy_block = u32::try_from(BLOCK_COUNT - 1).expect("fits");
    let live: Vec<(usize, &Sequence)> = chosen
        .iter()
        .map(|&index| (index, &sequences[index]))
        .collect();
    let (keyed, on_candle) = decode_commands(&live, dispatcher, dummy_block, 200 + step as u64);
    let keyed = lay_out(&keyed);
    let written: Vec<usize> = keyed.slot_mapping[..chosen.len()]
        .iter()
        .map(|&slot| usize::try_from(slot).expect("a live row's slot"))
        .collect();
    let kv_width = kv_width(cache);
    let before = snapshot(cache, &written);
    let (free_before, _) = mem_get_info().expect("the device reports its free memory");
    let used_before = (*pool).map(pool_watch);
    let tensor_logits: Vec<Vec<f32>> = {
        let logits = forward
            .forward_logits(&keyed, readback)
            .expect("the keyed batch runs on the decode step");
        (0..logits.rows())
            .map(|row| logits.row(row).expect("row").to_vec())
            .collect()
    };
    let (free_after, _) = mem_get_info().expect("the device reports its free memory");
    assert_eq!(
        free_before,
        free_after,
        "step {step}: the decode step left {} bytes allocated",
        free_before.abs_diff(free_after)
    );
    if let (Some(pool), Some(used_before)) = (*pool, used_before) {
        let high = pool_high(pool);
        assert_eq!(
            high,
            used_before,
            "step {step}: the decode step took {} bytes from the stream-ordered allocator",
            high.abs_diff(used_before)
        );
    }
    let after_step = snapshot(cache, &written);
    check_step_writes(&before, &after_step, &written, kv_width, step);
    let alone: Vec<Vec<f32>> = live
        .iter()
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
        .collect();
    let candle_logits = forward
        .forward_logits(&lay_out(&on_candle), readback)
        .expect("the eager step runs on candle");
    let after_candle = snapshot(cache, &written);
    let kv_diff = widest_slot_diff(&after_step, &after_candle, &written, kv_width);
    parity.kv_max_abs_diff = parity.kv_max_abs_diff.max(kv_diff);
    assert_eq!(tensor_logits.len(), chosen.len(), "one row per live entry");
    assert_eq!(candle_logits.rows(), chosen.len());
    let mut sampled = Vec::with_capacity(chosen.len());
    for (row, tensor_row) in tensor_logits.iter().enumerate() {
        let candle_row = candle_logits.row(row).expect("row");
        let theirs = record_argmax(parity, step, row, tensor_row, candle_row);
        let diff = widest(tensor_row, candle_row);
        parity.max_abs_diff = parity.max_abs_diff.max(diff);
        parity.sum_abs_diff += f64::from(diff);
        let candle_diff = widest(&alone[row], candle_row);
        parity.candle_max_abs_diff = parity.candle_max_abs_diff.max(candle_diff);
        parity.candle_sum_abs_diff += f64::from(candle_diff);
        parity.rows += 1;
        sampled.push(theirs);
    }
    for (&index, next) in chosen.iter().zip(sampled) {
        let sequence = &mut sequences[index];
        sequence.context_len += 1;
        sequence.tokens.push(next);
    }
}

/// Compares the two forwards' argmax on one row, counting a disagreement or a tie, and returns
/// candle's, which is what the sequence advances by.
fn record_argmax(
    parity: &mut Parity,
    step: usize,
    row: usize,
    tensor_row: &[f32],
    candle_row: &[f32],
) -> u32 {
    let (ours, theirs) = (argmax(tensor_row), argmax(candle_row));
    if ours == theirs {
        return theirs;
    }
    // Both forwards' logits for both ids. Candle reads its logits back in bf16: when it holds
    // the two ids at one value it cannot order them, and its argmax takes the lower id, so the
    // row is a tie rather than a disagreement. The tie is exact equality of what candle holds.
    let (ours_at, theirs_at) = (at(ours), at(theirs));
    let tied = candle_row[ours_at].to_bits() == candle_row[theirs_at].to_bits();
    if tied {
        parity.ties += 1;
    } else {
        parity.argmax_disagreements += 1;
    }
    eprintln!(
        "step {step} row {row}: {} — decode step argmax {ours} (step {:.4}, candle {:.4}), \
         candle argmax {theirs} (step {:.4}, candle {:.4})",
        if tied { "tie" } else { "disagreement" },
        tensor_row[ours_at],
        candle_row[ours_at],
        tensor_row[theirs_at],
        candle_row[theirs_at]
    );
    theirs
}

/// The bound the variable `name` sets, or `default` when it is unset or not a number.
fn bound_from_env(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|bound| bound.parse().ok())
        .unwrap_or(default)
}

/// `SEQUENCES` sequences of random tokens, each over its own run of blocks.
fn seed_sequences(random: &mut Lcg, vocab: usize) -> Vec<Sequence> {
    let blocks_each = MAX_MODEL_LEN.div_ceil(BLOCK_SIZE);
    (0..SEQUENCES)
        .map(|index| Sequence {
            tokens: (0..8 + random.below(40))
                .map(|_| u32::try_from(random.below(vocab.min(TOKEN_ID_CEILING))).expect("fits"))
                .collect(),
            blocks: (0..blocks_each)
                .map(|block| u32::try_from(index * blocks_each + block).expect("fits"))
                .collect(),
            context_len: 0,
        })
        .collect()
}

#[test]
#[ignore = "needs a device, the CUDA toolkit and a Llama checkpoint; run scripts/decode-parity.sh"]
fn the_two_forwards_agree_on_every_decode_and_the_step_records_under_capture() {
    let model = ModelConfig {
        id: env::var("PARITY_MODEL").map_or_else(|_| ModelId::new(DEFAULT_MODEL), ModelId::new),
        revision: "main".to_owned(),
        cache_dir: None,
        dtype: Dtype::Bf16,
        prompt_template: PromptTemplate::Llama3,
    };
    let bound = bound_from_env("PARITY_MAX_ABS_DIFF", DEFAULT_MAX_ABS_DIFF);
    let kv_bound = bound_from_env("PARITY_KV_MAX_ABS_DIFF", DEFAULT_KV_MAX_ABS_DIFF);
    let Rig {
        allocation,
        allocated,
        mut decode_step,
        readback,
        vocab,
    } = open(&model);
    let contract = CaptureContract::resolve(&[declaration()], &ModelDeclaration::new("llama"));
    let dispatcher = Dispatcher::new(&dispatch_config(), &contract);
    let dummy_block = u32::try_from(BLOCK_COUNT - 1).expect("fits");

    let (session, graph, nodes, memory_nodes) =
        record_bucket_of_one(allocation, &mut decode_step, dummy_block);
    println!(
        "capture: the bucket-of-one step recorded as a graph of {nodes} nodes, {memory_nodes} of \
         them allocating or freeing memory"
    );
    assert_eq!(
        memory_nodes, 0,
        "the recorded graph allocates or frees memory"
    );
    let cache: Vec<Tensor> = allocated.kv_cache.layers().to_vec();
    let pool = default_pool(allocated.device.stream().context());
    if pool.is_none() {
        println!("pool: the device has no stream-ordered allocator to watch");
    }
    // The harness reads logits through the eager step and replays nothing; the one graph it
    // recorded is the bucket-of-one's, and no other bucket has one.
    let forward = CudaForward::new(allocated, decode_step, GraphSet::new(vec![graph]), session)
        .expect("the forward builds");

    let mut random = Lcg(0x5EED_2026_0903);
    let sequences = seed_sequences(&mut random, vocab);
    let mut harness = Harness {
        forward,
        readback,
        sequences,
        dispatcher,
        parity: Parity::default(),
        cache,
        pool,
    };
    prefill(
        &mut harness.forward,
        &mut harness.readback,
        &mut harness.sequences,
    );

    for step in 0..STEPS {
        let mut chosen: Vec<usize> = (0..SEQUENCES).collect();
        for index in (1..SEQUENCES).rev() {
            chosen.swap(index, random.below(index + 1));
        }
        chosen.truncate(1 + random.below(SEQUENCES));
        chosen.sort_unstable();
        compare_step(&mut harness, &chosen, step);
    }
    let parity = harness.parity;

    println!("=============== decode parity evidence ===============");
    println!("model:                {}", model.id);
    println!("decode steps:         {STEPS}");
    println!("rows compared:        {}", parity.rows);
    println!("argmax disagreements: {}", parity.argmax_disagreements);
    println!("argmax ties:          {}", parity.ties);
    println!("max |logit diff|:     {:.6}", parity.max_abs_diff);
    println!("mean |logit diff|:    {:.6}", parity.mean_abs_diff());
    println!("bound:                {bound}");
    println!("candle alone against candle in the live batch:");
    println!("  max |logit diff|:   {:.6}", parity.candle_max_abs_diff);
    println!("  mean |logit diff|:  {:.6}", parity.candle_mean_abs_diff());
    println!("capture graph nodes:  {nodes}");
    println!("capture memory nodes: {memory_nodes}");
    println!("cache writes, the step against candle over the same slots:");
    println!("  max |k/v diff|:     {:.6}", parity.kv_max_abs_diff);
    println!("  bound:              {kv_bound}");
    assert_eq!(
        parity.argmax_disagreements, 0,
        "every live row's argmax agrees on ids candle's logits separate"
    );
    assert!(
        parity.max_abs_diff <= bound,
        "the largest logit difference {} is above the bound {bound}",
        parity.max_abs_diff
    );
    assert!(
        parity.kv_max_abs_diff <= kv_bound,
        "the largest cache-write difference {} is above the bound {kv_bound}",
        parity.kv_max_abs_diff
    );
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

/// Holds the decode step to writing each live row's own slot, and the dummies' slot of the dummy
/// block, and nothing else in those blocks. Candle's run of the same batch overwrites the rows'
/// slots afterwards, so a write that landed anywhere else would otherwise go unseen: later steps
/// would read the same wrong cache through both forwards.
fn check_step_writes(
    before: &Snapshot,
    after: &Snapshot,
    written: &[usize],
    kv_width: usize,
    step: usize,
) {
    for (row, (before, after)) in before.rows.iter().zip(&after.rows).enumerate() {
        let ranges = slot_ranges(written[row] % BLOCK_SIZE, kv_width);
        for (layer, (before, after)) in before.iter().zip(after).enumerate() {
            assert!(
                ranges
                    .iter()
                    .any(|range| before[range.clone()] != after[range.clone()]),
                "step {step} row {row} layer {layer}: the decode step did not write its slot"
            );
            untouched_outside(
                before,
                after,
                &ranges,
                &format!("step {step} row {row} layer {layer}"),
            );
        }
    }
    let ranges = slot_ranges(0, kv_width);
    for (layer, (before, after)) in before.dummy.iter().zip(&after.dummy).enumerate() {
        untouched_outside(
            before,
            after,
            &ranges,
            &format!("step {step} dummy block layer {layer}"),
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
            "{at}: the decode step wrote outside its slot, at value {index}"
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
