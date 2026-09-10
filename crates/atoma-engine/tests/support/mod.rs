//! The device rig the decode parity harness and the bucket-ladder gates share: one device
//! opened, a Llama checkpoint loaded in bf16, the decode step over runtime tensors built beside
//! the candle forward on the same weights and KV cache, every bucket of a bucket ladder captured
//! over the padding dummies' blocks, and sequences of random tokens prefilled through candle to
//! decode from. What differs between the two harnesses is stated in a [`RigPlan`]: the model,
//! the bucket ladder and the maximum batch, the KV geometry, how many sequences run, and how
//! the arena is placed. What each measures over the rig is its own.
//!
//! The step commands a harness issues are built here too: the keyed command the engine would
//! issue, padded to its bucket with dummies, the same command with a decoy token in every row
//! the device holds the token of, so a replay that ran its model step ahead of its gather would
//! decode the decoy, and the command marked eager for candle. So are the readings a harness
//! holds still: free device memory through the runtime context, the stream-ordered allocator's
//! high-water mark, and snapshots of the cache blocks a step may write.
//!
//! Compiled into each harness that declares it; whichever helpers a harness leaves unused are
//! the other harness's.

#![allow(
    dead_code,
    reason = "two harnesses share this module and each uses a subset of it"
)]
// A harness's evidence block goes to stdout on purpose.
#![allow(clippy::print_stdout, clippy::print_stderr)]

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
use atoma_engine::model::{fetch, llama_config};
use atoma_engine::readback::Readback;
#[cfg(feature = "test-support")]
use atoma_runtime::arena::RoleTable;
use atoma_runtime::arena::{ArenaLayout, BucketIdx};
use atoma_runtime::context::{DeviceBytes, RuntimeContext};
use atoma_runtime::session::Allocation;
use candle_core::{DType, Tensor};
use cudarc::driver::result::{free_sync, malloc_sync};
use cudarc::driver::{sys, CudaContext};

/// What the free-memory reading is shown to respond to, ahead of anything it is asked to hold
/// still across: larger than any driver chunk.
pub const SCRATCH_BYTES: usize = 64 * 1024 * 1024;
/// Prompt tokens are drawn below this id: Llama 3's special tokens sit at the top of the
/// vocabulary, and a prompt of those is not a prompt.
pub const TOKEN_ID_CEILING: usize = 120_000;
/// A sequence's prompt is this many tokens plus fewer than [`PROMPT_SPREAD`] more.
pub const SHORTEST_PROMPT: usize = 8;
pub const PROMPT_SPREAD: usize = 40;

pub fn tokens(value: usize) -> TokenCount {
    TokenCount::new(value).expect("nonzero")
}

pub fn requests(value: usize) -> RequestCount {
    RequestCount::new(value).expect("nonzero")
}

/// The checkpoint `id` in bf16 under the Llama 3 prompt template, fetched from the hub's cache.
pub fn model_config(id: ModelId) -> ModelConfig {
    ModelConfig {
        id,
        revision: "main".to_owned(),
        cache_dir: None,
        dtype: Dtype::Bf16,
        prompt_template: PromptTemplate::Llama3,
    }
}

/// What a harness's rig is built from.
#[derive(Debug, Clone)]
pub struct RigPlan {
    pub model: ModelConfig,
    /// The bucket ladder the step captures, in tokens per bucket.
    pub ladder: Vec<usize>,
    /// The largest batch a step holds, which is also the largest bucket the step serves.
    pub max_batch: usize,
    pub block_size: usize,
    pub block_count: usize,
    pub max_model_len: usize,
    /// How many sequences decode: at most the maximum batch, and every one owns its own run of
    /// blocks below the padding dummies'.
    pub sequences: usize,
    pub arena_layout: ArenaLayout,
    /// A role table in place of the model's own, for a gate that declares a lifetime short.
    #[cfg(feature = "test-support")]
    pub roles: Option<RoleTable>,
}

impl RigPlan {
    /// The request slots the sampler holds: one per sequence and one per padding dummy, as the
    /// engine sizes them.
    pub fn slots(&self) -> usize {
        self.sequences + self.max_batch
    }

    pub fn dispatch_config(&self) -> DispatchConfig {
        DispatchConfig {
            bucket_ladder: BucketLadder::new(self.ladder.clone()).expect("nonzero buckets"),
            captured_max_requests: requests(self.max_batch),
        }
    }

    /// The padding dummies' blocks: the last of the pool, one per dummy of the maximum batch,
    /// which is what the capture fills every bucket's dummy run from.
    pub fn dummy_blocks(&self) -> Vec<BlockId> {
        (self.block_count - self.max_batch..self.block_count)
            .map(|block| BlockId::new(u32::try_from(block).expect("fits")))
            .collect()
    }

    /// The block a live step's padding rows all sit on: the last of the dummies'.
    pub fn dummy_block(&self) -> u32 {
        u32::try_from(self.block_count - 1).expect("fits")
    }

    /// Where `rows` rows sit in the bucket ladder: the index of the bucket of exactly that
    /// many rows, which a keyed batch is padded to.
    pub fn bucket_of(&self, rows: usize) -> usize {
        self.ladder
            .iter()
            .position(|&bucket| bucket == rows)
            .expect("a keyed batch is padded to a bucket of the bucket ladder")
    }
}

/// A small deterministic generator, so a run is reproducible from its seed alone.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    pub fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next()).expect("fits") % bound
    }
}

/// One sequence under test: its tokens so far, the blocks it owns, how many tokens the cache
/// holds, and where its last token is.
pub struct Sequence {
    pub tokens: Vec<u32>,
    pub blocks: Vec<u32>,
    pub context_len: usize,
    pub last_token: LastToken,
}

/// Where a sequence's last token is: on the host alone until a replay samples for the sequence,
/// then on the device too, where the next replay's gather reads it and the host's copy is not.
#[derive(Clone, Copy)]
pub enum LastToken {
    OnHost,
    OnDevice,
}

impl Sequence {
    pub fn entry(&self, index: usize, input: Vec<u32>) -> CommandEntry {
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
    pub fn next_token(&self) -> u32 {
        self.tokens[self.context_len]
    }

    /// The token the host's copy carries for a replay: a decoy once the device holds the
    /// sequence's last token, since the graph's gather is what supplies it then, and the true
    /// token before.
    pub fn replay_input(&self, vocab: usize) -> u32 {
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

    /// Advances the sequence by `next`, the token the device sampled for it, which the device
    /// now holds for the next replay's gather.
    pub fn advance(&mut self, next: u32) {
        self.context_len += 1;
        self.tokens.push(next);
        self.last_token = LastToken::OnDevice;
    }
}

/// The padding dummy `index` over `block`, in the slot after the `sequences` sequences'.
pub fn dummy(index: usize, block: u32, sequences: usize) -> CommandEntry {
    CommandEntry {
        request: RequestId::new(1000 + index as u64),
        slot: RequestSlot::new(u32::try_from(sequences + index).expect("fits")),
        sequence: SequenceIndex::new(0),
        context_len: 0,
        input_tokens: vec![PADDING_TOKEN],
        block_table: vec![BlockId::new(block)],
        sampling: None,
    }
}

pub fn eager() -> DispatchDecision {
    DispatchDecision::Eager(EagerReason::NotUniformDecode {
        token_count: tokens(1),
        request_count: requests(1),
    })
}

pub fn lay_out(command: &StepCommand, block_size: usize) -> BatchLayout {
    BatchLayout::lay_out(command, tokens(block_size)).expect("the command lays out")
}

/// One decode step over the live sequences, three ways: the keyed command the engine would
/// issue, padded to its bucket with dummies over the dummy block, for the eager step; the same
/// command with a decoy token in every row the device holds the token of, for the replay; and
/// the command marked eager, for candle.
pub struct Commands {
    pub keyed: StepCommand,
    pub replayed: StepCommand,
    pub on_candle: StepCommand,
}

/// The three commands of step `step` over `live`, the sequences decoding and their indices,
/// under `plan`'s dummy block and slots.
pub fn decode_commands(
    live: &[(usize, &Sequence)],
    dispatcher: &mut Dispatcher,
    plan: &RigPlan,
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
        entries.extend(
            (0..padding_count).map(|index| dummy(index, plan.dummy_block(), plan.sequences)),
        );
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
pub struct Rig {
    pub plan: RigPlan,
    pub context: RuntimeContext,
    pub allocation: Allocation,
    pub allocated: Allocated,
    pub decode_step: DecodeStep,
    /// What every recording of the bucket ladder holds the sample of, and what a replay samples
    /// through.
    pub sampler: DeviceSampler,
    /// The harness reads logits through a readback of its own; the sampler's brings tokens.
    pub readback: Readback<f32>,
    pub vocab: usize,
}

/// Opens device zero, loads the plan's model in bf16 and builds both forwards over it.
pub fn open(plan: RigPlan) -> Rig {
    let files = fetch(&plan.model).expect("the checkpoint fetches");
    let config = llama_config(&files.config).expect("the config reads");
    let context = RuntimeContext::new(0).expect("device 0 opens");
    let allocation = Allocation::new(&context).expect("the session opens");
    let device = RankDevice::open(&allocation, DeviceOrdinal::new(0)).expect("candle opens");
    let checkpoint = Checkpoint {
        files: &files,
        config: &config,
        dtype: plan.model.dtype.into(),
    };
    let weights = Weights::load(&allocation, &device, checkpoint).expect("the weights load");
    let geometry = KvGeometry::new(&config, plan.block_count, tokens(plan.block_size), 1)
        .expect("the geometry");
    let kv_cache = KvCache::allocate(
        &allocation,
        &device,
        &config,
        geometry,
        plan.model.dtype.into(),
    )
    .expect("the cache allocates");
    let readback = Readback::new(
        &allocation,
        device.stream().context(),
        plan.max_batch,
        config.vocab_size,
    )
    .expect("the readback pins");
    let sampler = DeviceSampler::new(
        &allocation,
        device.stream(),
        plan.slots(),
        requests(plan.max_batch),
        config.vocab_size,
    )
    .expect("the sampler allocates");
    let step_plan = DecodeStepPlan {
        dispatch: plan.dispatch_config(),
        max_batch: requests(plan.max_batch),
        max_model_len: tokens(plan.max_model_len),
        block_size: tokens(plan.block_size),
        dtype: plan.model.dtype,
        staging_depth: StagingDepth::default(),
        arena_layout: plan.arena_layout,
        #[cfg(feature = "test-support")]
        roles: plan.roles.clone(),
    };
    let decode_step = DecodeStep::build(&allocation, &device, &weights, &kv_cache, &step_plan)
        .expect("the decode step builds");
    Rig {
        plan,
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
pub struct GraphNodes {
    pub nodes: usize,
    pub memory_nodes: usize,
}

/// Prints what capturing the bucket ladder cost and what each graph holds, and holds every
/// graph to allocating or freeing nothing.
pub fn report_capture(captured: &Captured, ladder: &[usize]) -> Vec<GraphNodes> {
    let Captured {
        session,
        graphs,
        report,
    } = captured;
    let nodes: Vec<GraphNodes> = ladder
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
pub fn free_memory_responds(context: &RuntimeContext) -> DeviceBytes {
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

pub fn free_memory(context: &RuntimeContext) -> DeviceBytes {
    context
        .free_memory()
        .expect("the device reports its free memory")
}

/// Prefills every sequence through candle over its own blocks, and appends the token the
/// prefill's largest logit names.
pub fn prefill(
    forward: &mut CudaForward,
    readback: &mut Readback<f32>,
    sequences: &mut [Sequence],
    block_size: usize,
) {
    for (index, sequence) in sequences.iter_mut().enumerate() {
        let command = StepCommand {
            step: StepId::new(100 + index as u64),
            entries: vec![sequence.entry(index, sequence.tokens.clone())],
            padding_count: 0,
            dispatch: eager(),
        };
        let logits = forward
            .forward_logits(&lay_out(&command, block_size), readback)
            .expect("the prefill runs on candle");
        let next = argmax(logits.row(0).expect("one row"));
        sequence.context_len = sequence.tokens.len();
        sequence.tokens.push(next);
    }
}

/// The forward over the sequences under test, captured and prefilled, and what a harness
/// reads around every step it runs.
pub struct Harness {
    pub plan: RigPlan,
    pub context: RuntimeContext,
    pub forward: CudaForward,
    pub readback: Readback<f32>,
    pub sequences: Vec<Sequence>,
    pub dispatcher: Dispatcher,
    /// Every layer's cache, the handles candle holds, for reading slots back.
    pub cache: Vec<Tensor>,
    /// The device's stream-ordered allocator, watched around every keyed step.
    pub pool: Option<sys::CUmemoryPool>,
    pub vocab: usize,
}

/// The sequences `chosen` names, each with its index.
pub fn live<'a>(sequences: &'a [Sequence], chosen: &[usize]) -> Vec<(usize, &'a Sequence)> {
    chosen
        .iter()
        .map(|&index| (index, &sequences[index]))
        .collect()
}

/// What capturing the bucket ladder left to report: what each graph holds, what the capture
/// cost, and the free memory read once the reading was shown to respond.
pub struct CaptureEvidence {
    pub graphs: Vec<GraphNodes>,
    pub report: CaptureReport,
    pub free_after: DeviceBytes,
}

/// Opens the device, captures the bucket ladder over the rig's dummies' blocks, and builds the
/// forward that serves from it beside the sequences under test, prefilled through candle: the
/// harness, and what the capture left to report.
pub fn build_harness(plan: RigPlan, random: &mut Lcg) -> (Harness, CaptureEvidence) {
    let Rig {
        plan,
        context,
        allocation,
        mut allocated,
        mut decode_step,
        mut sampler,
        readback,
        vocab,
    } = open(plan);
    let contract = CaptureContract::resolve(&[declaration()], &ModelDeclaration::new("llama"));
    let dispatcher = Dispatcher::new(&plan.dispatch_config(), &contract);

    let captured = capture_bucket_ladder(
        &context,
        allocation,
        &mut decode_step,
        &mut sampler,
        &plan.dummy_blocks(),
    )
    .expect("every bucket of the bucket ladder captures");
    let graphs = report_capture(&captured, &plan.ladder);
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

    let sequences = seed_sequences(&plan, random, vocab);
    let mut harness = Harness {
        plan,
        context,
        forward,
        readback,
        sequences,
        dispatcher,
        cache,
        pool,
        vocab,
    };
    prefill(
        &mut harness.forward,
        &mut harness.readback,
        &mut harness.sequences,
        harness.plan.block_size,
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

/// Runs `work` with free device memory read through the runtime context before and after it,
/// and the stream-ordered allocator's high-water mark watched across it, and holds both still:
/// `what` allocated nothing that stayed and took nothing from the pool.
pub fn holding_free_memory<T>(
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

/// The plan's sequences of random tokens, each over its own run of blocks below the dummies'.
pub fn seed_sequences(plan: &RigPlan, random: &mut Lcg, vocab: usize) -> Vec<Sequence> {
    let blocks_each = plan.max_model_len.div_ceil(plan.block_size);
    assert!(
        plan.sequences * blocks_each <= plan.block_count - plan.max_batch,
        "the sequences' blocks stay below the padding dummies'"
    );
    (0..plan.sequences)
        .map(|index| Sequence {
            tokens: (0..SHORTEST_PROMPT + random.below(PROMPT_SPREAD))
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

/// The indices below `count` in a random order: what a step's sequences are drawn from the
/// front of.
pub fn shuffled(count: usize, random: &mut Lcg) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..count).collect();
    for index in (1..count).rev() {
        indices.swap(index, random.below(index + 1));
    }
    indices
}

/// Whether two snapshots hold the same values bit for bit, `-0.0` and `NaN` included.
pub fn identical(a: &Snapshot, b: &Snapshot) -> bool {
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
pub fn kv_width(cache: &[Tensor]) -> usize {
    let dims = cache[0].dims();
    dims[3] * dims[4]
}

/// The K then V rows of one block of every layer, as f32 on the host: `[block_size, kv_width]`
/// each, row-major.
pub fn block_of(cache: &[Tensor], block: usize) -> Vec<Vec<f32>> {
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
pub struct Snapshot {
    pub rows: Vec<Vec<Vec<f32>>>,
    pub dummy: Vec<Vec<f32>>,
    /// The block size the snapshot was taken over, which is what a slot's offset in its block
    /// is read under.
    pub block_size: usize,
}

pub fn snapshot(cache: &[Tensor], written: &[usize], plan: &RigPlan) -> Snapshot {
    Snapshot {
        rows: written
            .iter()
            .map(|&slot| block_of(cache, slot / plan.block_size))
            .collect(),
        dummy: block_of(cache, plan.block_count - 1),
        block_size: plan.block_size,
    }
}

/// The K and V ranges of the slot at `offset` in one block's values, under `block_size`.
pub fn slot_ranges(offset: usize, kv_width: usize, block_size: usize) -> [Range<usize>; 2] {
    let k = offset * kv_width..(offset + 1) * kv_width;
    let v_base = block_size * kv_width;
    [k.clone(), v_base + k.start..v_base + k.end]
}

/// Holds `what`, one step over the cache, to writing each live row's own slot, and the dummies'
/// slot of the dummy block, and nothing else in those blocks. The eager step and candle's run of
/// the same batch overwrite the rows' slots afterwards, so a write that landed anywhere else
/// would otherwise go unseen: later steps would read the same wrong cache through every path.
pub fn check_step_writes(
    before: &Snapshot,
    after: &Snapshot,
    written: &[usize],
    kv_width: usize,
    what: &str,
) {
    let block_size = before.block_size;
    for (row, (before, after)) in before.rows.iter().zip(&after.rows).enumerate() {
        let ranges = slot_ranges(written[row] % block_size, kv_width, block_size);
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
    let ranges = slot_ranges(0, kv_width, block_size);
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
pub fn untouched_outside(before: &[f32], after: &[f32], ranges: &[Range<usize>], at: &str) {
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
pub fn widest_slot_diff(
    step: &Snapshot,
    candle: &Snapshot,
    written: &[usize],
    kv_width: usize,
) -> f32 {
    let block_size = step.block_size;
    let mut widest_diff = 0.0f32;
    for (row, (step, candle)) in step.rows.iter().zip(&candle.rows).enumerate() {
        for range in slot_ranges(written[row] % block_size, kv_width, block_size) {
            for (step, candle) in step.iter().zip(candle) {
                widest_diff = widest_diff.max(widest(&step[range.clone()], &candle[range.clone()]));
            }
        }
    }
    widest_diff
}

/// The device's default stream-ordered allocator, or `None` where the driver has none.
pub fn default_pool(context: &Arc<CudaContext>) -> Option<sys::CUmemoryPool> {
    let mut pool: sys::CUmemoryPool = ptr::null_mut();
    // SAFETY: a driver query for the context's device; the out-pointer lives for the call.
    unsafe { sys::cuDeviceGetDefaultMemPool(&raw mut pool, context.cu_device()) }
        .result()
        .ok()
        .map(|()| pool)
}

/// Resets the pool's used-memory high-water mark and returns its usage now: a step that takes
/// nothing from the pool leaves the mark at this value.
pub fn pool_watch(pool: sys::CUmemoryPool) -> u64 {
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
pub fn pool_high(pool: sys::CUmemoryPool) -> u64 {
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
pub fn widest(row: &[f32], other: &[f32]) -> f32 {
    row.iter()
        .zip(other)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

/// A token id as an index into a logits row.
pub fn at(token: u32) -> usize {
    usize::try_from(token).expect("a token id indexes its row")
}

pub fn argmax(row: &[f32]) -> u32 {
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
