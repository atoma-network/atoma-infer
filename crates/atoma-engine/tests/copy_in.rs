//! The per-step copy-in on a device: what a staging entry carries is what the device block
//! holds, when a staging entry's fence is passed, and what an acquire costs when the host runs
//! ahead.
//!
//! No checkpoint and no model: the inputs are built over a context, a stream and a session of
//! their own, so what is measured is the staging and the copy and nothing else. A step's seven
//! arrays — the five the model step reads and the two the sampler reads — are staged into one
//! staging entry's pinned block and copied in through it; the device block is read back and every
//! array compared at the bucket's packed offsets, and past the bucket's packed length the block
//! is still as its allocation zeroed it. A second step, differing in all seven, is staged into
//! the other staging entry before either copy-in runs, so each staging entry holds a step of its
//! own when the first copy reads: each readback shows what its own staging entry was staged
//! with, and it is the pair of them that says a copy-in does not reach one fixed pinned block —
//! a `stage` naming block zero instead of the staging entry's leaves the second step in front of
//! the first readback, and a `copy_in` naming it leaves the first step in front of the second.
//! A dummy run then goes through the first staging entry again, taken back through the
//! non-blocking half of the staging ring's protocol. Both copies have been waited on by then, so
//! what that shows is a real fence answering a query and reading passed — which the staging
//! ring's own tests cannot show over their fake fence — and not the order of the signal.
//!
//! What none of the three copies pins is which of the two pinned blocks a staging entry names:
//! swap the blocks wherever a staging entry indexes one and every comparison still passes. The
//! dummy run pins less still, since the staging ring comes round to the first staging entry for
//! it, so a `stage_dummy` naming block zero stages the block it would have anyway.
//!
//! The order is the second test: it asks the fence while the copy that reads the staging entry
//! is still in flight. One staging entry, a block table wide enough that its copy takes tens of
//! microseconds on the fastest link a host has to a device, and [`FENCE_QUERIES`] queries, each
//! in the few microseconds after its own copy-in with nothing waited on in between. Every one of
//! them must read not passed, which is true only of a signal enqueued behind the copy.
//!
//! The third test times `acquire` over [`ACQUIRES`] staged copy-ins with no other wait, which is
//! the host running ahead of the device by the whole staging depth, and holds its p99 to a bound.
//!
//! All three open the device, and the third measures a latency on it, so they are run one at
//! a time. Run through `scripts/copy-in.sh`. Under NCCL the decode step stays on candle, so
//! nothing keys a batch and nothing copies in.

#![cfg(all(feature = "cuda", not(feature = "nccl")))]
// The evidence block is this test's product; it goes to stdout on purpose.
#![allow(clippy::print_stdout)]

use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use atoma_core::attention::{CaptureContract, ModelDeclaration};
use atoma_core::dispatch::{BucketLadder, DispatchConfig, DispatchDecision, Dispatcher, LiveBatch};
use atoma_core::request::{SamplingParams, PADDING_TOKEN};
use atoma_core::step::{CommandEntry, StepCommand};
use atoma_core::types::{
    BlockId, RequestCount, RequestId, RequestSlot, SequenceIndex, StepId, TokenCount,
};
use atoma_engine::batch::BatchLayout;
use atoma_engine::decode::batch::{Checked, DecodeBatch, DecodeBuckets};
use atoma_engine::decode::declaration;
use atoma_engine::decode::inputs::DecodeInputs;
use atoma_engine::decode::ring::{StagingDepth, StagingEntry};
use atoma_engine::decode::staging::{DummyRun, StagingShape};
use atoma_runtime::arena::BucketIdx;
use atoma_runtime::context::RuntimeContext;
use atoma_runtime::session::{Allocation, Replay};
use cudarc::driver::CudaStream;

/// The bucket ladder under test.
const LADDER: [usize; 3] = [1, 2, 4];
/// The largest bucket of the ladder, and the maximum batch that keeps every bucket usable.
const LARGEST_BUCKET: usize = 4;
/// Rows of the bucket both steps and the dummy run fill.
const ROWS: usize = 2;
/// That bucket: the ladder's second.
const BUCKET: BucketIdx = BucketIdx(1);
/// Columns of the block table, wide enough that the largest bucket's table is longer than the
/// alignment, so the buckets pack to different lengths.
const WIDTH: usize = 64;
const BLOCK_SIZE: TokenCount = TokenCount::new(16).expect("nonzero");
/// Staging entries in the staging ring: two, so one block's copy is in flight while the host
/// writes the other.
const DEPTH: StagingDepth = StagingDepth::new(2).expect("nonzero");
/// Positions the rotary tables would cover: above every position staged here.
const MAX_POSITION: usize = 512;

// Bucket 2's packed offsets at 64 columns: token ids, positions and key lengths take 8 bytes
// each and the slot mapping 16, every one of them padded to the 256-byte alignment; the block
// table's two rows take 512; the sampler's two arrays follow, 8 bytes each and padded the same
// way.
const TOKEN_IDS_AT: usize = 0;
const POSITIONS_AT: usize = 256;
const KEY_LENGTHS_AT: usize = 512;
const SLOT_MAPPING_AT: usize = 768;
const BLOCK_TABLE_AT: usize = 1024;
const ROW_SLOTS_AT: usize = 1536;
const GATHER_SLOTS_AT: usize = 1792;
/// Bucket 2's packed length: the gather slots' offset plus their aligned length, which is what
/// one of its copies carries and where the device block is left as its allocation zeroed it.
const PACKED_BYTES: usize = 2048;
/// The device block: the largest bucket's packed length, whose block table takes 1024 bytes, so
/// its sampler arrays sit 512 further on than bucket 2's.
const BLOCK_BYTES: usize = 2560;

/// Columns of the block table the fence test stages at: bucket 2's two rows of them come to
/// 16 MiB, so one copy takes tens of microseconds on the fastest link a host has to a device and
/// hundreds on the links most have, against the two or three microseconds the host spends
/// between enqueuing the copy and asking the fence.
const WIDE_WIDTH: usize = 2 * 1024 * 1024;
/// What one of those copies carries: bucket 2's four single-row arrays and the sampler's two,
/// each padded to the alignment, and its two table rows.
const WIDE_COPY_BYTES: usize = 4 * 256 + ROWS * WIDE_WIDTH * 4 + 2 * 256;
/// Staging entries the fence test runs at: one, so the staging entry a copy is reading is the
/// staging entry the next query asks about.
const FENCE_DEPTH: StagingDepth = StagingDepth::new(1).expect("nonzero");
/// The copy-ins the fence test asks the fence after: a fixed count, so nothing spins, and eight
/// of them, so a fence signaled ahead of its copy has to win eight races between the host
/// reaching the query and the driver retiring the record, rather than one. Every one of the eight
/// must read not passed: a copy this size cannot land inside the microseconds the host spends
/// getting to the query, so a single passed query is the signal in the wrong place.
const FENCE_QUERIES: usize = 8;

/// Acquires timed with nothing else waiting, which is more than the thousand the guarantee is
/// read over.
const ACQUIRES: usize = 1024;
/// The p99 an `acquire` must stay under, in microseconds, unless `COPY_IN_ACQUIRE_P99_MICROS`
/// says otherwise.
///
/// Measured on an A100-SXM4-40GB, at [`DEPTH`], over five runs of [`ACQUIRES`] each: a p99 of
/// 3.1 to 3.9 microseconds from an unoptimized build and 6.0 to 6.2 from an optimized one, the
/// optimized build the higher of the two. The max is not bounded and ranged from 13 to 32
/// microseconds over the same runs. The bound stays at 20, three times the highest p99
/// measured, so that a host slower to its device than this one does not redden the test.
const DEFAULT_ACQUIRE_P99_MICROS: f64 = 20.0;

/// The device, the session the copy-ins run on, and the inputs under test.
struct Rig {
    inputs: DecodeInputs,
    session: Replay,
    /// The stream the device block was allocated on, and the one the readback runs on.
    stream: Arc<CudaStream>,
}

impl Rig {
    /// A rig over a block table of [`WIDTH`] columns and a staging ring of [`DEPTH`]: the two
    /// staging entries the readbacks go through, and the depth the acquires are timed at.
    fn open() -> Self {
        Self::over(shape(WIDTH), DEPTH)
    }

    /// A rig over a block table of [`WIDE_WIDTH`] columns and the one staging entry the fence
    /// test asks about: the copy is long enough to be caught in flight, and the staging entry it
    /// reads is the one the next query asks about.
    fn wide() -> Self {
        Self::over(shape(WIDE_WIDTH), FENCE_DEPTH)
    }

    /// Opens device zero and builds the inputs for [`LADDER`]'s buckets at `shape` over a staging
    /// ring of `depth`, then leaves the Allocation phase: only copy-ins remain.
    fn over(shape: StagingShape, depth: StagingDepth) -> Self {
        let context = RuntimeContext::new(0).expect("device 0 opens");
        let allocation = Allocation::new(&context).expect("the session opens");
        let stream = context.cuda().default_stream();
        let inputs = DecodeInputs::new(&allocation, &stream, shape, &buckets(), depth)
            .expect("the staging ring pins its blocks and the device block allocates");
        // The block was allocated on this stream and the copy-ins run on the capture stream, so
        // the two are joined here, in the Allocation phase, as the decode step joins them.
        stream.synchronize().expect("the allocations land");
        Self {
            inputs,
            session: allocation.into_capture().into_replay(),
            stream,
        }
    }

    /// Copies `bucket`'s packed length in from `entry`'s pinned block, waits for the copy, and
    /// reads the whole device block back.
    fn copy_in(&self, entry: StagingEntry, bucket: BucketIdx) -> Vec<u8> {
        let mut copy_in = self
            .inputs
            .copy_in(entry, bucket)
            .expect("the bucket is one the inputs stage for");
        self.session
            .run(&mut copy_in)
            .expect("the copy-in enqueues");
        self.session.synchronize().expect("the copy lands");
        let block = self
            .stream
            .clone_dtoh(self.inputs.device_block())
            .expect("the device block reads back");
        // The readback is enqueued on this stream and copies into plain host memory, so it is
        // waited on before the bytes are read.
        self.stream.synchronize().expect("the readback lands");
        block
    }
}

/// What the seven arrays must read as on the device once a step's copy has landed: one value
/// per row of the bucket, and each row's blocks.
struct Staged {
    token_ids: [u32; ROWS],
    positions: [i32; ROWS],
    key_lengths: [i32; ROWS],
    slot_mapping: [i64; ROWS],
    /// Each row's blocks in order; the rest of the row is zero to the full width.
    blocks: [Vec<i32>; ROWS],
    row_slots: [i32; ROWS],
    gather_slots: [i32; ROWS],
}

impl Staged {
    /// Asserts `block` — the device block, read back — holds every one of the seven arrays at
    /// bucket 2's packed offsets.
    fn holds(&self, block: &[u8]) {
        assert_eq!(
            block.len(),
            BLOCK_BYTES,
            "the device block holds the largest bucket's packed length"
        );
        let held = |at: usize, expected: &[u8], input: &str| {
            assert_eq!(
                &block[at..at + expected.len()],
                expected,
                "the {input} copied in"
            );
        };
        held(TOKEN_IDS_AT, &words(&self.token_ids), "token ids");
        held(POSITIONS_AT, &words(&self.positions), "positions");
        held(KEY_LENGTHS_AT, &words(&self.key_lengths), "key lengths");
        held(SLOT_MAPPING_AT, &words(&self.slot_mapping), "slot mapping");
        held(BLOCK_TABLE_AT, &words(&self.table()), "block table");
        held(ROW_SLOTS_AT, &words(&self.row_slots), "row slots");
        held(GATHER_SLOTS_AT, &words(&self.gather_slots), "gather slots");
        assert!(
            block[PACKED_BYTES..].iter().all(|&byte| byte == 0),
            "past bucket 2's packed length the device block is as its allocation zeroed it: the \
             copy carried the bucket's length and stopped"
        );
    }

    /// The block table as it is staged: each row's blocks, then zero to the full width.
    fn table(&self) -> Vec<i32> {
        self.blocks
            .iter()
            .flat_map(|row| {
                let mut cells = vec![0; WIDTH];
                cells[..row.len()].copy_from_slice(row);
                cells
            })
            .collect()
    }
}

/// A value one of the seven arrays holds, in the native-endian bytes the device reads it as.
trait Word: Copy {
    fn bytes(self) -> Vec<u8>;
}

impl Word for u32 {
    fn bytes(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }
}

impl Word for i32 {
    fn bytes(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }
}

impl Word for i64 {
    fn bytes(self) -> Vec<u8> {
        self.to_ne_bytes().to_vec()
    }
}

/// `values` as the array holding them reads on the device: one native-endian word each.
fn words<T: Word>(values: &[T]) -> Vec<u8> {
    values.iter().copied().flat_map(Word::bytes).collect()
}

fn dispatch_config() -> DispatchConfig {
    DispatchConfig {
        bucket_ladder: BucketLadder::new(LADDER.to_vec()).expect("nonzero buckets"),
        captured_max_requests: RequestCount::new(LARGEST_BUCKET).expect("nonzero"),
    }
}

fn buckets() -> DecodeBuckets {
    DecodeBuckets::usable(
        &dispatch_config(),
        RequestCount::new(LARGEST_BUCKET).expect("nonzero"),
    )
}

/// The staging shape of [`LADDER`]'s largest bucket at `width` columns.
fn shape(width: usize) -> StagingShape {
    StagingShape {
        max_tokens: buckets().largest(),
        block_table_width: width,
        max_position: MAX_POSITION,
        block_size: BLOCK_SIZE,
    }
}

/// One decoding entry: `token` computed at `context_len` over `blocks`, sampling.
fn entry(request: u64, context_len: usize, token: u32, blocks: &[u32]) -> CommandEntry {
    CommandEntry {
        request: RequestId::new(request),
        slot: RequestSlot::new(u32::try_from(request).expect("a test request number fits")),
        sequence: SequenceIndex::new(0),
        context_len,
        input_tokens: vec![token],
        block_table: blocks.iter().map(|&block| BlockId::new(block)).collect(),
        sampling: Some(SamplingParams::default()),
    }
}

/// The keyed batch of `entries` at `width` columns, laid out and held to its bucket as the
/// executor holds it.
fn keyed(entries: Vec<CommandEntry>, width: usize) -> (BatchLayout, DecodeBatch) {
    let contract = CaptureContract::resolve(&[declaration()], &ModelDeclaration::new("llama"));
    let mut dispatcher = Dispatcher::new(&dispatch_config(), &contract);
    let dispatch = dispatcher.dispatch(LiveBatch {
        token_count: TokenCount::new(entries.len()).expect("nonzero"),
        request_count: RequestCount::new(entries.len()).expect("nonzero"),
        uniform_decode: true,
    });
    let DispatchDecision::FullReplay(key) = dispatch else {
        panic!(
            "a uniform decode of {} rows is keyed: {dispatch:?}",
            entries.len()
        );
    };
    let command = StepCommand {
        step: StepId::new(1),
        entries,
        padding_count: 0,
        dispatch,
    };
    let layout = BatchLayout::lay_out(&command, BLOCK_SIZE).expect("the command lays out");
    let checked = DecodeBatch::check(&layout, key, &buckets(), width).expect("the batch is keyed");
    let Checked::Step(batch) = checked else {
        panic!("the decode step serves a two-row uniform decode: {checked:?}");
    };
    (layout, batch)
}

/// The first step under test, and what its seven arrays must read as: a token at position 3 over
/// one block, and a token at position 20 over the second of two.
/// Every value but the block table's padding is nonzero, so a device block still holding what
/// its allocation zeroed cannot pass for this one.
fn first_step() -> (BatchLayout, DecodeBatch, Staged) {
    let (layout, batch) = keyed(first_rows(), WIDTH);
    let staged = Staged {
        token_ids: [9, 7],
        positions: [3, 20],
        // The key length is the context plus this step's token.
        key_lengths: [4, 21],
        // A slot is its block's id times the block size, plus the token's offset in the block:
        // position 3 sits in the first block, position 20 in the second.
        slot_mapping: [10 * 16 + 3, 21 * 16 + 4],
        blocks: [vec![10], vec![20, 21]],
        row_slots: [5, 6],
        gather_slots: [7, 8],
    };
    (layout, batch, staged)
}

/// The two rows the first step and the fence test both stage.
fn first_rows() -> Vec<CommandEntry> {
    vec![entry(1, 3, 9, &[10]), entry(2, 20, 7, &[20, 21])]
}

/// The step the fence test copies in: the first step's two rows keyed at [`WIDE_WIDTH`], whose
/// block table is the whole of what makes the copy long. Nothing reads this one back, so what
/// the rows hold does not matter.
fn wide_step() -> (BatchLayout, DecodeBatch) {
    keyed(first_rows(), WIDE_WIDTH)
}

/// The second step, whose every array differs from the first's, so what it reads back cannot be
/// what the first copy left in the device block.
fn second_step() -> (BatchLayout, DecodeBatch, Staged) {
    let (layout, batch) = keyed(
        vec![entry(1, 5, 4242, &[12]), entry(2, 33, 777, &[30, 31, 32])],
        WIDTH,
    );
    let staged = Staged {
        token_ids: [4242, 777],
        positions: [5, 33],
        key_lengths: [6, 34],
        // Position 33 sits in the third block, one token in.
        slot_mapping: [12 * 16 + 5, 32 * 16 + 1],
        blocks: [vec![12], vec![30, 31, 32]],
        row_slots: [13, 14],
        gather_slots: [15, 16],
    };
    (layout, batch, staged)
}

/// The dummy run under test, and what its seven arrays must read as: every row a padding row
/// over its own block. Its positions and its zero-filled table cells are read against a device
/// block the second step left nonzero there.
fn dummy_run() -> (DummyRun, Staged) {
    let run = DummyRun::new(BUCKET, vec![BlockId::new(15), BlockId::new(3)]);
    let staged = Staged {
        token_ids: [PADDING_TOKEN, PADDING_TOKEN],
        positions: [0, 0],
        key_lengths: [1, 1],
        // A padding row's token goes to its block's first KV slot.
        slot_mapping: [15 * 16, 3 * 16],
        blocks: [vec![15], vec![3]],
        // No row samples and every row keeps the token the host staged.
        row_slots: [-1, -1],
        gather_slots: [-1, -1],
    };
    (run, staged)
}

/// The bound the variable `name` sets, in microseconds, or `default` when it is unset.
///
/// # Panics
///
/// Panics when `name` is set to anything but a positive number: an override a rig operator
/// mistyped must be heard, not measured against the default, and neither `nan` nor `inf` is a
/// bound a p99 can be held to.
fn bound_from_env(name: &str, default: f64) -> f64 {
    let bound = match env::var(name) {
        Ok(bound) => bound,
        Err(env::VarError::NotPresent) => return default,
        Err(error) => panic!("{name} cannot be read: {error}"),
    };
    let micros: f64 = bound
        .parse()
        .unwrap_or_else(|error| panic!("{name} is set to {bound:?}, which is no number: {error}"));
    assert!(
        micros.is_finite() && micros > 0.0,
        "{name} is set to {bound:?}, which is no positive number of microseconds"
    );
    micros
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/copy-in.sh"]
fn what_each_staging_entry_copies_in_is_what_the_device_block_holds() {
    let mut rig = Rig::open();
    let (first_layout, first_batch, first) = first_step();
    let (second_layout, second_batch, second) = second_step();
    let (run, dummy) = dummy_run();

    // Both steps are staged before either is copied in, so both staging entries hold a step of
    // their own when the first copy reads. It is the pair of readbacks below that says the two
    // copies did not reach one fixed pinned block: a `stage` naming block zero instead of the
    // staging entry's leaves the second step in front of the first readback, and a `copy_in`
    // naming it leaves the first step in front of the second, so either one reddens a `holds`.
    // Which of the two pinned blocks a staging entry names is left open: the blocks swapped
    // wherever a staging entry indexes one keep every comparison green, and so does a
    // `stage_dummy` naming block zero, since the staging ring comes round to the first staging
    // entry for the dummy run.
    let first_entry = rig.inputs.acquire().expect("the first staging entry");
    assert_eq!(
        first_entry.index(),
        0,
        "the first acquire takes the first block"
    );
    let arrays = rig
        .inputs
        .stage(&first_entry, &first_layout, &first_batch)
        .expect("the first step stages");
    // The sampler decides these two per step; what one copy carries is what is under test here,
    // so the test writes them itself, as the bucket's rows hold them.
    arrays.row_slots.copy_from_slice(&first.row_slots);
    arrays.gather_slots.copy_from_slice(&first.gather_slots);

    // Nothing has signaled either fence yet, so this acquire waits on nothing.
    let second_entry = rig.inputs.acquire().expect("the second staging entry");
    assert_eq!(
        second_entry.index(),
        1,
        "the second acquire takes the other block"
    );
    let arrays = rig
        .inputs
        .stage(&second_entry, &second_layout, &second_batch)
        .expect("the second step stages");
    arrays.row_slots.copy_from_slice(&second.row_slots);
    arrays.gather_slots.copy_from_slice(&second.gather_slots);

    first.holds(&rig.copy_in(first_entry, first_batch.bucket));
    second.holds(&rig.copy_in(second_entry, second_batch.bucket));

    // The first staging entry taken back through the non-blocking half of the protocol. Both
    // copies have been waited on, so what this shows is a real fence answering `cuEventQuery`
    // and reading passed rather than failing: the fake fence of the staging ring's own tests
    // cannot show that. The order of the signal is the fence test's, below.
    let entry = rig
        .inputs
        .try_acquire()
        .expect("the fence answers")
        .expect("the first staging entry's fence reads passed once its copy is waited on");
    assert_eq!(
        entry.index(),
        0,
        "the staging ring comes round to the first block"
    );
    rig.inputs
        .stage_dummy(&entry, &run)
        .expect("the dummy run stages");
    dummy.holds(&rig.copy_in(entry, run.bucket()));
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/copy-in.sh"]
fn a_staging_entry_does_not_come_free_while_the_copy_reading_it_is_in_flight() {
    let mut rig = Rig::wide();
    let (layout, batch) = wide_step();
    let mut in_flight = 0_usize;

    for _ in 0..FENCE_QUERIES {
        // Nothing an earlier turn enqueued is left on the stream, so the only thing this turn's
        // fence can be waiting for is this turn's copy.
        rig.session.synchronize().expect("the stream drains");
        let entry = rig.inputs.acquire().expect("the one staging entry");
        rig.inputs
            .stage(&entry, &layout, &batch)
            .expect("the step stages");
        let mut copy_in = rig
            .inputs
            .copy_in(entry, batch.bucket)
            .expect("the bucket is one the inputs stage for");
        rig.session.run(&mut copy_in).expect("the copy-in enqueues");
        // Asked in the few microseconds it takes to reach here, with nothing waited on in
        // between: the copy has megabytes left to move, so a fence signaled behind it cannot be
        // passed. A staging entry handed back anyway is dropped, and the staging ring offers the
        // same one again next turn.
        if rig
            .inputs
            .try_acquire()
            .expect("the fence answers")
            .is_none()
        {
            in_flight += 1;
        }
    }
    rig.session.synchronize().expect("the last copy lands");

    println!("=============== copy-in fence evidence ===============");
    println!("copy bytes:    {WIDE_COPY_BYTES}");
    println!("queries:       {FENCE_QUERIES}");
    println!("in flight:     {in_flight}");
    println!("======================================================");
    let passed = FENCE_QUERIES - in_flight;
    assert_eq!(
        in_flight, FENCE_QUERIES,
        "the staging entry's fence read passed in {passed} of {FENCE_QUERIES} queries, each \
         taken while a {WIDE_COPY_BYTES}-byte copy was still reading its pinned block: a copy \
         that size cannot land inside the microseconds the host spends reaching the query, so \
         the fence is signaled ahead of the copy rather than behind it, or the copy is not \
         asynchronous"
    );
}

#[test]
#[ignore = "needs a device and the CUDA toolkit; run scripts/copy-in.sh"]
fn acquiring_a_staging_entry_stays_under_its_bound_while_the_host_runs_ahead() {
    let mut rig = Rig::open();
    let (layout, batch, _) = first_step();
    let bound = bound_from_env("COPY_IN_ACQUIRE_P99_MICROS", DEFAULT_ACQUIRE_P99_MICROS);
    let depth = DEPTH.get();
    let mut acquires: Vec<Duration> = Vec::with_capacity(ACQUIRES);

    // Nothing waits on the device between one copy-in and the next, so once the staging ring
    // has come round every acquire asks a fence whose copy may still be in flight.
    for _ in 0..ACQUIRES {
        let started = Instant::now();
        let entry = rig.inputs.acquire().expect("a staging entry comes free");
        acquires.push(started.elapsed());
        rig.inputs
            .stage(&entry, &layout, &batch)
            .expect("the step stages");
        let mut copy_in = rig
            .inputs
            .copy_in(entry, batch.bucket)
            .expect("the bucket is one the inputs stage for");
        rig.session.run(&mut copy_in).expect("the copy-in enqueues");
    }
    rig.session.synchronize().expect("the last copies land");

    acquires.sort_unstable();
    let micros = |taken: Duration| taken.as_secs_f64() * 1e6;
    let max = micros(*acquires.last().expect("every acquire was timed"));
    let p99 = micros(acquires[ACQUIRES * 99 / 100]);
    println!("=============== copy-in acquire evidence ===============");
    println!("acquires:      {ACQUIRES}");
    println!("staging depth: {depth}");
    println!("max:           {max:.3} us");
    println!("p99:           {p99:.3} us");
    println!("bound:         {bound:.3} us");
    println!("========================================================");
    assert!(
        p99 <= bound,
        "the p99 acquire took {p99:.3} us, above the {bound:.3} us bound; the host waited on \
         the device for a staging entry to come free"
    );
}
