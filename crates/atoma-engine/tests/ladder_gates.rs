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
//! The arena's reuse layout is gated against its reference in a second run, behind the
//! `test-support` feature: one step per bucket over the same sequences and tokens on a rig placed
//! greedily, one placed without reuse, and one placed under the poison layout, whose logits must
//! all be the same bit for bit. Then the same steps over the model's role table with `Normed`'s
//! lifetime declared one op short, which the up projection reads at, show the gates bite: under
//! poison the fill scheduled ahead of that projection turns every row not-a-number, so the
//! logits fail bit-identity against the reference and poison mode has caught the lie; under
//! greedy, where the host proof shows the lie moves no slot, the logits stay the reference's.
//!
//! `LADDER_GATE_MODEL` names the checkpoint, Llama 3.2 1B Instruct unless set;
//! `LADDER_GATE_MAX_BATCH` the maximum batch the bucket ladder is cut at, which must be a bucket
//! of that ladder: 128 unless set, and 512 for the whole Hopper ladder. `LADDER_GATE_STEPS`
//! raises the identity steps, which are never fewer than 32 or twice the bucket count, and never
//! lowers them. Needs a device, the CUDA toolkit and the checkpoint; run through
//! `scripts/ladder-gates.sh`, which enables `test-support`. Under NCCL the decode step stays on
//! candle and there is nothing to gate.

#![cfg(all(feature = "cuda", not(feature = "nccl")))]
// The evidence block is this test's product; it goes to stdout on purpose.
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod support;

use core::fmt;
use std::env;

use atoma_core::dispatch::{BucketLadder, DispatchConfig, Platform};
use atoma_engine::batch::BatchLayout;
use atoma_engine::config::{ModelConfig, ModelId};
use atoma_engine::decode::batch::DecodeBuckets;
use atoma_engine::device::forward::CudaForward;
use atoma_engine::forward::Forward;
use atoma_engine::logits::Logits;
use atoma_engine::readback::Readback;
use atoma_runtime::arena::ArenaLayout;
use atoma_runtime::context::DeviceBytes;

use support::{
    build_harness, decode_commands, free_memory, holding_free_memory, lay_out, live_sequences,
    model_config, requests, shuffled, CaptureEvidence, Harness, Lcg, RigPlan, Sequence,
    PROMPT_SPREAD, SHORTEST_PROMPT,
};

const DEFAULT_MODEL: &str = "unsloth/Llama-3.2-1B-Instruct";
const DEFAULT_MAX_BATCH: usize = 128;
/// The fewest identity steps a run makes. `LADDER_GATE_STEPS` raises it and never lowers it,
/// and a run makes at least twice the bucket count either way, so every bucket is replayed
/// twice.
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
            // Raising the floor, never lowering it: a run that made fewer steps than the gates
            // require would capture the whole ladder and soak it before failing on the count.
            steps: count_from_env("LADDER_GATE_STEPS", DEFAULT_STEPS).max(DEFAULT_STEPS),
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
    let ladder = usable_ladder(settings.max_batch);
    assert_eq!(
        ladder.last().copied(),
        Some(settings.max_batch),
        "LADDER_GATE_MAX_BATCH is {}, which is not a bucket of the Hopper bucket ladder; the \
         soak decodes every sequence in one keyed step, and a batch off the ladder has no graph",
        settings.max_batch
    );
    RigPlan {
        model: settings.model.clone(),
        ladder,
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

/// Reads every baked address again through its owner and holds it to where it was baked,
/// panicking with the first that moved and naming `at`.
///
/// Returns the one check it made, so a running total counts the checks that were performed and
/// never the replays a caller meant to make: a total assembled from a constant would hold
/// however few checks ran.
#[must_use]
fn baked_checked(forward: &CudaForward, at: impl fmt::Display) -> usize {
    forward
        .baked_unmoved()
        .unwrap_or_else(|error| panic!("{at}: {error}"));
    1
}

/// The rows of the logits `keyed` leaves when it runs on the eager decode step: the reference a
/// replay and every reuse layout are held to.
fn forward_rows(
    forward: &mut CudaForward,
    readback: &mut Readback<f32>,
    keyed: &BatchLayout,
) -> Vec<Vec<f32>> {
    let logits = forward
        .forward_logits(keyed, readback)
        .expect("the keyed batch runs on the decode step");
    rows_of(&logits)
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
    let live = live_sequences(sequences, chosen);
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
    identity.baked_checks += baked_checked(forward, format_args!("step {step}, after the replay"));
    let eager_rows: Vec<Vec<f32>> =
        holding_free_memory(context, *pool, step, "the eager decode step", || {
            forward_rows(forward, readback, &keyed)
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
    let mut baked_checks = 0;
    for replay in 0..SOAK_REPLAYS {
        forward
            .forward(&replayed)
            .expect("the keyed batch replays its bucket's graph");
        baked_checks += baked_checked(forward, format_args!("soak replay {replay}"));
    }
    let after = free_memory(context);
    Soak {
        before,
        after,
        baked_checks,
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
    let (mut harness, capture) = build_harness(plan.clone(), &mut random);
    let mut identity = Identity::new(plan.ladder.len());

    for step in 0..steps {
        let mut chosen = shuffled(plan.sequences, &mut random);
        chosen.truncate(batch_size(step, &plan.ladder, &mut random));
        chosen.sort_unstable();
        identity_step(&mut harness, &mut identity, &chosen, step);
    }
    let soak = soak(&mut harness);

    print_evidence(&plan, &capture, &identity, soak);
    assert_evidence(&plan, &capture, &identity, soak);
}

/// The reuse layout against the reference, and the layout gates shown to bite. Behind
/// `test-support`, which is what lets a rig state a role table that is not the model's.
#[cfg(feature = "test-support")]
mod layout {
    use core::fmt;

    use atoma_models::layer::normed_one_op_short;
    use atoma_runtime::arena::{ArenaLayout, RoleTable};

    use crate::support::{
        argmax, build_harness, decode_commands, holding_free_memory, lay_out, live_sequences,
        model_dims, shuffled, Harness, Lcg,
    };
    use crate::{forward_rows, rig_plan, Settings, BLOCK_SIZE};

    /// Every run draws the same sequences and the same rows for each step from this seed.
    const SEED: u64 = 0x5EED_2026_0911;

    /// Which role table a run's arena was placed from.
    #[derive(Clone, Copy)]
    enum Roles {
        /// The model's own, which every serving step places under.
        Model,
        /// The model's with `Normed`'s lifetime declared one op short: the one lie the gates run.
        NormedOneOpShort,
    }

    impl Roles {
        /// The table to state on the plan, or `None` for the model's own, which the decode step
        /// builds for itself.
        fn table(self, settings: &Settings) -> Option<RoleTable> {
            match self {
                Roles::Model => None,
                Roles::NormedOneOpShort => Some(normed_one_op_short(&model_dims(&settings.model))),
            }
        }
    }

    /// How a run's arena was placed: under which layout, and from which role table.
    #[derive(Clone, Copy)]
    struct Placement {
        layout: ArenaLayout,
        roles: Roles,
    }

    impl fmt::Display for Placement {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let table = match self.roles {
                Roles::Model => "",
                Roles::NormedOneOpShort => " over Normed one op short",
            };
            // Padded through `pad` so the evidence block's column width reaches this.
            f.pad(&format!("{}{table}", self.layout))
        }
    }

    /// One rig's run: how its arena was placed, the eager step's logits at every step, and the
    /// tokens the sequences advanced by.
    struct Run {
        placed: Placement,
        logits: Vec<Vec<Vec<f32>>>,
        tokens: Vec<Vec<u32>>,
    }

    /// Runs one step per bucket, ascending, over a fresh rig placed as `placed` says, and
    /// advances each step's sequences by `reference`'s tokens where there is one and by the
    /// step's own argmax otherwise, so every run decodes the same tokens over the same blocks
    /// and its logits compare with the reference's step by step.
    fn run(settings: &Settings, placed: Placement, reference: Option<&Run>) -> Run {
        let mut plan = rig_plan(settings);
        plan.arena_layout = placed.layout;
        plan.roles = placed.roles.table(settings);
        let mut random = Lcg(SEED);
        let (mut harness, _capture) = build_harness(plan.clone(), &mut random);
        let mut run = Run {
            placed,
            logits: Vec::new(),
            tokens: Vec::new(),
        };
        for (step, &rows) in plan.ladder.iter().enumerate() {
            let mut chosen = shuffled(plan.sequences, &mut random);
            chosen.truncate(rows);
            chosen.sort_unstable();
            let logits = eager_step(&mut harness, &chosen, step);
            let tokens: Vec<u32> = reference.map_or_else(
                || logits.iter().map(|row| argmax(row)).collect(),
                |reference| reference.tokens[step].clone(),
            );
            for (&index, &next) in chosen.iter().zip(&tokens) {
                harness.sequences[index].advance(next);
            }
            run.logits.push(logits);
            run.tokens.push(tokens);
        }
        run
    }

    /// Runs the eager decode step over `chosen` and reads its logits back, with free device
    /// memory and the pool held still around it.
    fn eager_step(harness: &mut Harness, chosen: &[usize], step: usize) -> Vec<Vec<f32>> {
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
        let live = live_sequences(sequences, chosen);
        let commands = decode_commands(&live, dispatcher, plan, *vocab, 300 + step as u64);
        let keyed = lay_out(&commands.keyed, BLOCK_SIZE);
        holding_free_memory(context, *pool, step, "the eager decode step", || {
            forward_rows(forward, readback, &keyed)
        })
    }

    /// How a run compares with the reference: rows compared, rows that are not the reference's
    /// bit for bit, and rows holding not-a-number.
    struct Against {
        rows: usize,
        apart: usize,
        nan: usize,
    }

    fn against(run: &Run, reference: &Run) -> Against {
        let mut against = Against {
            rows: 0,
            apart: 0,
            nan: 0,
        };
        for (step, (logits, reference)) in run.logits.iter().zip(&reference.logits).enumerate() {
            assert_eq!(
                logits.len(),
                reference.len(),
                "step {step}: one row per live entry in both runs"
            );
            for (row, (logits, reference)) in logits.iter().zip(reference).enumerate() {
                against.rows += 1;
                if logits.iter().any(|value| value.is_nan()) {
                    against.nan += 1;
                }
                let apart = logits
                    .iter()
                    .zip(reference)
                    .position(|(value, reference)| value.to_bits() != reference.to_bits());
                let Some(index) = apart else {
                    continue;
                };
                against.apart += 1;
                if against.apart == 1 {
                    eprintln!(
                        "step {step} row {row}: {} is not the reference's at index {index}: {} \
                         against {}",
                        run.placed, logits[index], reference[index]
                    );
                }
            }
        }
        against
    }

    #[test]
    #[ignore = "needs a device, the CUDA toolkit and a checkpoint; run scripts/ladder-gates.sh"]
    fn the_reuse_layout_is_the_reference_and_the_layout_gates_bite() {
        let settings = Settings::from_env();
        let placed = |layout, roles| Placement { layout, roles };
        let reference = run(&settings, placed(ArenaLayout::NoReuse, Roles::Model), None);
        let runs = [
            placed(ArenaLayout::Greedy, Roles::Model),
            placed(ArenaLayout::Poison, Roles::Model),
            placed(ArenaLayout::Greedy, Roles::NormedOneOpShort),
            placed(ArenaLayout::Poison, Roles::NormedOneOpShort),
        ]
        .map(|placed| {
            let compared = run(&settings, placed, Some(&reference));
            (placed, against(&compared, &reference))
        });

        println!("=============== arena layout gate evidence ===============");
        println!("model:                {}", settings.model.id);
        println!(
            "steps:                {}, one per bucket, the same rows and tokens in every run",
            reference.logits.len()
        );
        println!("against the no-reuse reference, bit for bit:");
        for (placed, against) in &runs {
            println!(
                "  {placed:<32} rows {:>5}, apart {:>5}, not-a-number {:>5}",
                against.rows, against.apart, against.nan
            );
        }

        let [(_, greedy), (_, poison), (_, greedy_lie), (_, poison_lie)] = runs;
        assert!(greedy.rows > 0, "rows were compared");
        assert_eq!(
            greedy.apart, 0,
            "greedy's logits are the no-reuse reference's bit for bit"
        );
        assert_eq!(
            poison.apart, 0,
            "poison's fills touch no live slot, so its logits are the reference's"
        );
        assert_eq!(
            greedy_lie.apart, 0,
            "greedy places nothing differently over Normed one op short, as the host proof \
             predicts"
        );
        assert_eq!(
            poison_lie.nan, poison_lie.rows,
            "poison fills Normed ahead of the up projection that reads it, so every row is \
             not-a-number: the poison layout caught the lie"
        );
        assert_eq!(
            poison_lie.apart, poison_lie.rows,
            "and no row is the reference's: the bit-identity gate is red under a broken layout"
        );
    }
}
