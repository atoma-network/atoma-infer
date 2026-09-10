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
//! The rig — the device, the two forwards, the capture, the sequences and the commands — is the
//! one the bucket-ladder gates run over too, in `support`.
//!
//! Needs a device, the CUDA toolkit and a Llama checkpoint loadable in bf16; run through
//! `scripts/decode-parity.sh`. Under NCCL the decode step stays on candle and there is nothing
//! to compare.

#![cfg(all(feature = "cuda", not(feature = "nccl")))]
// The evidence block is this test's product; it goes to stdout on purpose.
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod support;

use std::cmp::Ordering;
use std::env;

use atoma_core::step::StepCommand;
use atoma_core::types::StepId;
use atoma_engine::config::{ModelConfig, ModelId};
use atoma_engine::device::forward::CudaForward;
use atoma_engine::forward::Forward;
use atoma_engine::readback::Readback;
use atoma_runtime::arena::ArenaLayout;
use atoma_runtime::context::DeviceBytes;

use support::{
    argmax, at, build_harness, check_step_writes, decode_commands, eager, free_memory,
    holding_free_memory, identical, kv_width, lay_out, live_sequences, model_config, shuffled,
    snapshot, widest, widest_slot_diff, CaptureEvidence, Harness, Lcg, RigPlan, Sequence,
};

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
                .forward_logits(&lay_out(&command, BLOCK_SIZE), readback)
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
fn compare_step(harness: &mut Harness, parity: &mut Parity, chosen: &[usize], step: usize) {
    let Harness {
        plan,
        context,
        forward,
        readback,
        sequences,
        dispatcher,
        cache,
        pool,
        vocab,
    } = harness;
    let live = live_sequences(sequences, chosen);
    let commands = decode_commands(&live, dispatcher, plan, *vocab, 200 + step as u64);
    let keyed = lay_out(&commands.keyed, BLOCK_SIZE);
    let written: Vec<usize> = keyed.slot_mapping[..chosen.len()]
        .iter()
        .map(|&slot| usize::try_from(slot).expect("a live row's slot"))
        .collect();
    let kv_width = kv_width(cache);
    let before = snapshot(cache, &written, plan);
    let replayed = lay_out(&commands.replayed, BLOCK_SIZE);
    let sampled: Vec<u32> = holding_free_memory(context, *pool, step, "the replay", || {
        forward
            .forward(&replayed)
            .expect("the keyed batch replays its bucket's graph")
            .to_vec()
    });
    let after_replay = snapshot(cache, &written, plan);
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
    let after_step = snapshot(cache, &written, plan);
    assert!(
        identical(&after_replay, &after_step),
        "step {step}: the eager step's cache writes are not the replay's, bit for bit"
    );
    parity.replays_per_bucket[plan.bucket_of(keyed.entry_count())] += 1;

    let alone = candle_alone(forward, readback, &live, step);
    let candle_logits = forward
        .forward_logits(&lay_out(&commands.on_candle, BLOCK_SIZE), readback)
        .expect("the eager step runs on candle");
    let after_candle = snapshot(cache, &written, plan);
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
        sequences[index].advance(next);
    }
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
        plan,
        context,
        forward,
        sequences,
        dispatcher,
        vocab,
        ..
    } = harness;
    let live: Vec<(usize, &Sequence)> = sequences.iter().enumerate().collect();
    let commands = decode_commands(&live, dispatcher, plan, *vocab, 900);
    let replayed = lay_out(&commands.replayed, BLOCK_SIZE);
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

/// The checkpoint under test: `PARITY_MODEL`, or Llama 3.1 8B Instruct, in bf16.
fn model_under_test() -> ModelConfig {
    model_config(
        env::var("PARITY_MODEL").map_or_else(|_| ModelId::new(DEFAULT_MODEL), ModelId::new),
    )
}

/// The rig the parity run is measured over: the four-bucket bucket ladder at a maximum batch of
/// eight, as many sequences, and the arena placed greedily, as serving places it.
fn rig_plan(model: ModelConfig) -> RigPlan {
    RigPlan {
        model,
        ladder: LADDER.to_vec(),
        max_batch: MAX_BATCH,
        block_size: BLOCK_SIZE,
        block_count: BLOCK_COUNT,
        max_model_len: MAX_MODEL_LEN,
        sequences: SEQUENCES,
        arena_layout: ArenaLayout::Greedy,
        #[cfg(feature = "test-support")]
        roles: None,
    }
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
    let (mut harness, capture) = build_harness(rig_plan(model.clone()), &mut random);
    let mut parity = Parity::default();

    for step in 0..STEPS {
        let mut chosen = shuffled(SEQUENCES, &mut random);
        chosen.truncate(batch_size(step, &mut random));
        chosen.sort_unstable();
        compare_step(&mut harness, &mut parity, &chosen, step);
    }
    let soak = soak(&mut harness);

    print_evidence(&model, &parity, &capture, &bounds, soak);
    assert_evidence(&parity, &bounds, soak);
}
