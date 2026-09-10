//! The correctness gates over the full bucket ladder, on a device.
//!
//! Every bucket the decode step serves — the Hopper bucket ladder at or below the maximum batch,
//! nineteen buckets at the default of 128 — is captured into a graph over the padding dummies'
//! blocks, and the capture matrix is reported: what each graph holds, what it cost, and that no
//! graph allocates or frees memory. Then decode steps of varying token ids, sequence lengths and
//! block tables run through the replay of each bucket's graph and through the same decode step
//! eagerly, op by op over the same addresses, and the replay's logits must be the eager step's
//! bit for bit: at least 32 steps in all, every bucket replayed at least twice. After every
//! replay every baked address is read again and held to where it was baked, and again after each
//! of a thousand replays run back to back, across which free device memory must not move.
//!
//! The eager step is the reference, never candle: candle is not batch-invariant, so it has no
//! bit-exact bar to hold, where the eager keyed step is the same kernels in the same order over
//! the same addresses. The replay runs first at every step, so a replay that skipped its copy-in
//! or its model step could not read the eager step's outputs as its own; the eager step then
//! runs the same command. The tokens each replay samples are what the sequences advance by, so
//! the device and the host agree on every row's input, and a decoy token in the host's copy of
//! every row the device holds the token of shows the graph's gather is what supplies it.
//!
//! `LADDER_GATE_MODEL` names the checkpoint, Llama 3.2 1B Instruct unless set;
//! `LADDER_GATE_MAX_BATCH` the maximum batch the bucket ladder is cut at, 128 unless set and 512
//! for the whole Hopper ladder; `LADDER_GATE_STEPS` the identity steps, never fewer than twice
//! the bucket count. Needs a device, the CUDA toolkit and the checkpoint; run through
//! `scripts/ladder-gates.sh`. Under NCCL the decode step stays on candle and there is nothing to
//! gate.

#![cfg(all(feature = "cuda", not(feature = "nccl")))]
// The evidence block is this test's product; it goes to stdout on purpose.
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod support;

use std::env;

use atoma_core::dispatch::{BucketLadder, DispatchConfig, Platform};
use atoma_engine::config::{ModelConfig, ModelId};
use atoma_engine::decode::batch::DecodeBuckets;
use atoma_engine::forward::Forward;
use atoma_engine::logits::Logits;
use atoma_runtime::arena::ArenaLayout;
use atoma_runtime::context::DeviceBytes;

use support::{
    decode_commands, free_memory, holding_free_memory, lay_out, model_config, requests,
    CaptureEvidence, Harness, Lcg, RigPlan, Sequence, PROMPT_SPREAD, SHORTEST_PROMPT,
};

const DEFAULT_MODEL: &str = "unsloth/Llama-3.2-1B-Instruct";
const DEFAULT_MAX_BATCH: usize = 128;
/// Identity steps unless `LADDER_GATE_STEPS` says otherwise; a run makes at least twice the
/// bucket count either way, so every bucket is replayed twice.
const DEFAULT_STEPS: usize = 32;
const BLOCK_SIZE: usize = 16;
/// The longest a sequence grows: its prompt, the identity steps, and the soak's one position.
const MAX_MODEL_LEN: usize = 256;
/// Replays of one keyed step run back to back, with free memory read before and after and the
/// baked addresses read again after each.
const SOAK_REPLAYS: usize = 1000;

/// What the gates run over, read from `LADDER_GATE_*`.
struct Settings {
    model: ModelConfig,
    max_batch: usize,
    steps: usize,
}

impl Settings {
    fn from_env() -> Self {
        let model = env::var("LADDER_GATE_MODEL")
            .map_or_else(|_| ModelId::new(DEFAULT_MODEL), ModelId::new);
        Self {
            model: model_config(model),
            max_batch: count_from_env("LADDER_GATE_MAX_BATCH", DEFAULT_MAX_BATCH),
            steps: count_from_env("LADDER_GATE_STEPS", DEFAULT_STEPS),
        }
    }
}

/// The count the variable `name` sets, or `default` when it is unset.
///
/// # Panics
///
/// Panics when the variable is set to something that is not a count.
fn count_from_env(name: &str, default: usize) -> usize {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} is set to {value:?}, which is not a count"))
    })
}

/// The buckets the step serves at `max_batch`: the Hopper bucket ladder's distinct entries at
/// or below it, in order, as the engine reads them.
fn usable_ladder(max_batch: usize) -> Vec<usize> {
    let config = DispatchConfig {
        bucket_ladder: BucketLadder::default_for(Platform::Hopper),
        captured_max_requests: requests(max_batch),
    };
    DecodeBuckets::usable(&config, requests(max_batch))
        .tokens()
        .to_vec()
}

/// The rig the gates run over: the usable bucket ladder at the maximum batch, as many
/// sequences, each over its own blocks up to the longest it grows, and the arena placed
/// greedily, as serving places it.
fn rig_plan(settings: &Settings) -> RigPlan {
    let blocks_each = MAX_MODEL_LEN / BLOCK_SIZE;
    RigPlan {
        model: settings.model.clone(),
        ladder: usable_ladder(settings.max_batch),
        max_batch: settings.max_batch,
        block_size: BLOCK_SIZE,
        block_count: settings.max_batch * blocks_each + settings.max_batch,
        max_model_len: MAX_MODEL_LEN,
        sequences: settings.max_batch,
        arena_layout: ArenaLayout::Greedy,
        #[cfg(feature = "test-support")]
        roles: None,
    }
}

/// What holding every replay to the eager step found.
struct Identity {
    steps: usize,
    rows: usize,
    /// Rows whose replayed logits are not the eager step's bit for bit.
    rows_apart: usize,
    /// The first value that differed.
    first_apart: Option<Apart>,
    /// How many steps each bucket of the bucket ladder was replayed at.
    replays_per_bucket: Vec<usize>,
    /// How many times the baked addresses were read again after a replay and found where they
    /// were baked.
    baked_checks: usize,
}

/// One value the replay and the eager step disagree on: where it was and what each held.
struct Apart {
    step: usize,
    row: usize,
    index: usize,
    replayed: f32,
    eager: f32,
}

impl Identity {
    fn new(buckets: usize) -> Self {
        Self {
            steps: 0,
            rows: 0,
            rows_apart: 0,
            first_apart: None,
            replays_per_bucket: vec![0; buckets],
            baked_checks: 0,
        }
    }
}

/// The rows of `logits`, copied off the readback.
fn rows_of(logits: &Logits<'_>) -> Vec<Vec<f32>> {
    (0..logits.rows())
        .map(|row| logits.row(row).expect("row").to_vec())
        .collect()
}

/// How many sequences step `step` decodes: the first steps walk the bucket ladder up and then
/// down, so every bucket is replayed twice, and the rest are bucket sizes drawn at random.
fn batch_size(step: usize, ladder: &[usize], random: &mut Lcg) -> usize {
    let buckets = ladder.len();
    if step < buckets {
        ladder[step]
    } else if step < 2 * buckets {
        ladder[2 * buckets - 1 - step]
    } else {
        ladder[random.below(buckets)]
    }
}

/// Runs one decode step over `chosen` twice — the replay of its bucket's graph, then the same
/// decode step eagerly — and holds the replay's logits to the eager step's bit for bit and
/// every baked address to where it was baked, with free device memory and the pool held still
/// around each; then advances each chosen sequence by the token the replay sampled.
fn identity_step(harness: &mut Harness, identity: &mut Identity, chosen: &[usize], step: usize) {
    let Harness {
        plan,
        context,
        forward,
        readback,
        sequences,
        dispatcher,
        pool,
        vocab,
        ..
    } = harness;
    let live = support::live(sequences, chosen);
    let commands = decode_commands(&live, dispatcher, plan, *vocab, 200 + step as u64);
    let keyed = lay_out(&commands.keyed, BLOCK_SIZE);
    let replayed = lay_out(&commands.replayed, BLOCK_SIZE);
    let (replayed_rows, tokens): (Vec<Vec<f32>>, Vec<u32>) =
        holding_free_memory(context, *pool, step, "the replay", || {
            let replayed = forward
                .replay_logits(&replayed, readback)
                .expect("the keyed batch replays its bucket's graph");
            (rows_of(&replayed.logits), replayed.tokens.to_vec())
        });
    forward
        .baked_unmoved()
        .unwrap_or_else(|error| panic!("step {step}, after the replay: {error}"));
    identity.baked_checks += 1;
    let eager_rows: Vec<Vec<f32>> =
        holding_free_memory(context, *pool, step, "the eager decode step", || {
            let logits = forward
                .forward_logits(&keyed, readback)
                .expect("the keyed batch runs on the decode step");
            rows_of(&logits)
        });
    assert_eq!(replayed_rows.len(), chosen.len(), "one row per live entry");
    assert_eq!(eager_rows.len(), chosen.len(), "one row per live entry");
    assert_eq!(tokens.len(), chosen.len(), "one token per live entry");
    for (row, (replayed, eager)) in replayed_rows.iter().zip(&eager_rows).enumerate() {
        assert_eq!(replayed.len(), eager.len(), "a vocabulary wide");
        identity.rows += 1;
        let apart = replayed
            .iter()
            .zip(eager)
            .position(|(replayed, eager)| replayed.to_bits() != eager.to_bits());
        let Some(index) = apart else {
            continue;
        };
        identity.rows_apart += 1;
        eprintln!(
            "step {step} row {row}: the replay's logits are not the eager step's at index \
             {index}: {} against {}",
            replayed[index], eager[index]
        );
        identity.first_apart.get_or_insert(Apart {
            step,
            row,
            index,
            replayed: replayed[index],
            eager: eager[index],
        });
    }
    identity.replays_per_bucket[plan.bucket_of(keyed.entry_count())] += 1;
    identity.steps += 1;
    for (&index, next) in chosen.iter().zip(tokens) {
        sequences[index].advance(next);
    }
}

/// Free device memory read before and after the soak, and how many times the baked addresses
/// were read again across it.
#[derive(Clone, Copy)]
struct Soak {
    before: DeviceBytes,
    after: DeviceBytes,
    baked_checks: usize,
}

/// Replays one keyed step of every sequence `SOAK_REPLAYS` times back to back, with free device
/// memory read through the runtime context before and after, and every baked address read again
/// after each replay: a replay that allocates anything that stays would show as a drop across
/// the soak, and an address that moved would be named at the replay it moved after. The
/// sequences are not advanced, so every replay decodes the same position and writes the same
/// slots.
fn soak(harness: &mut Harness) -> Soak {
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
    for replay in 0..SOAK_REPLAYS {
        forward
            .forward(&replayed)
            .expect("the keyed batch replays its bucket's graph");
        forward
            .baked_unmoved()
            .unwrap_or_else(|error| panic!("soak replay {replay}: {error}"));
    }
    let after = free_memory(context);
    Soak {
        before,
        after,
        baked_checks: SOAK_REPLAYS,
    }
}

/// Prints the evidence block: the capture matrix, what holding every replay to the eager step
/// found, and what the soak read.
fn print_evidence(plan: &RigPlan, capture: &CaptureEvidence, identity: &Identity, soak: Soak) {
    println!("=============== bucket ladder gate evidence ===============");
    println!("model:                {}", plan.model.id);
    println!(
        "bucket ladder:        {:?} ({} buckets at a maximum batch of {})",
        plan.ladder,
        plan.ladder.len(),
        plan.max_batch
    );
    let report = &capture.report;
    println!("capture matrix, {} graphs:", capture.graphs.len());
    for ((rows, graph), cost) in plan.ladder.iter().zip(&capture.graphs).zip(&report.graphs) {
        println!(
            "  bucket {rows:>3}: {} nodes, {} memory nodes, {:?}, {} bytes",
            graph.nodes, graph.memory_nodes, cost.elapsed, cost.used
        );
    }
    println!(
        "  in all:     {:?}, {} bytes; {} bytes free after",
        report.elapsed, report.used, report.free
    );
    match report.graph_memory() {
        Ok(memory) => println!(
            "  fit:        {} bytes fixed + {} bytes a graph",
            memory.fixed(),
            memory.marginal()
        ),
        Err(error) => println!("  fit:        {error}"),
    }
    println!("replay against the eager step, bit for bit:");
    println!("  identity steps:     {}", identity.steps);
    println!("  rows compared:      {}", identity.rows);
    println!("  rows apart:         {}", identity.rows_apart);
    if let Some(apart) = &identity.first_apart {
        println!(
            "  first apart:        step {} row {} index {}: replay {} against eager {}",
            apart.step, apart.row, apart.index, apart.replayed, apart.eager
        );
    }
    for (rows, replays) in plan.ladder.iter().zip(&identity.replays_per_bucket) {
        println!("  bucket {rows:>3} replayed: {replays} steps");
    }
    println!(
        "baked addresses:      read again after {} identity replays and {} soak replays; every \
         one where it was baked",
        identity.baked_checks, soak.baked_checks
    );
    println!(
        "free memory:          {} bytes after the capture; {SOAK_REPLAYS} replays: {} bytes \
         before, {} after",
        capture.free_after, soak.before, soak.after
    );
}

/// Holds the run to the gates: every bucket captured with no memory node, every replay's logits
/// the eager step's, every bucket replayed twice over at least 32 steps, the baked addresses
/// read again after every replay, and the soak leaving free memory where it found it.
fn assert_evidence(plan: &RigPlan, capture: &CaptureEvidence, identity: &Identity, soak: Soak) {
    assert_eq!(
        capture.graphs.len(),
        plan.ladder.len(),
        "one graph per bucket of the bucket ladder"
    );
    assert!(
        capture.graphs.iter().all(|graph| graph.memory_nodes == 0),
        "no graph allocates or frees memory"
    );
    assert!(
        identity.steps >= DEFAULT_STEPS && identity.steps >= 2 * plan.ladder.len(),
        "at least {DEFAULT_STEPS} identity steps and two per bucket: {}",
        identity.steps
    );
    assert!(
        identity
            .replays_per_bucket
            .iter()
            .all(|&replays| replays >= 2),
        "every bucket of the bucket ladder was replayed at least twice: {:?}",
        identity.replays_per_bucket
    );
    assert_eq!(
        identity.rows_apart, 0,
        "every replay's logits are the eager step's bit for bit"
    );
    assert_eq!(
        identity.baked_checks + soak.baked_checks,
        identity.steps + SOAK_REPLAYS,
        "the baked addresses were read again after every replay"
    );
    assert_eq!(
        soak.before,
        soak.after,
        "{SOAK_REPLAYS} replays left {} bytes allocated",
        soak.before.get().abs_diff(soak.after.get())
    );
}

#[test]
#[ignore = "needs a device, the CUDA toolkit and a Llama checkpoint; run scripts/ladder-gates.sh"]
fn every_bucket_captures_and_every_replay_is_the_eager_step_bit_for_bit() {
    let settings = Settings::from_env();
    let plan = rig_plan(&settings);
    let steps = settings.steps.max(2 * plan.ladder.len());
    assert!(
        SHORTEST_PROMPT + PROMPT_SPREAD + steps < MAX_MODEL_LEN,
        "{steps} identity steps grow a sequence past the longest the rig holds"
    );
    let mut random = Lcg(0x5EED_2026_0910);
    let (mut harness, capture) = support::build_harness(plan.clone(), &mut random);
    let mut identity = Identity::new(plan.ladder.len());

    for step in 0..steps {
        let mut chosen = support::shuffled(plan.sequences, &mut random);
        chosen.truncate(batch_size(step, &plan.ladder, &mut random));
        chosen.sort_unstable();
        identity_step(&mut harness, &mut identity, &chosen, step);
    }
    let soak = soak(&mut harness);

    print_evidence(&plan, &capture, &identity, soak);
    assert_evidence(&plan, &capture, &identity, soak);
}
