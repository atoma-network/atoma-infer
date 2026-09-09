//! Capturing the bucket ladder the decode step serves: one graph per bucket, each recorded over
//! a dummy run of that bucket's rows, in the Capture session phase, and what that cost.
//!
//! Before any recording, one eager warmup runs at the largest bucket, so a backend's one-time
//! lazy allocations — cuBLAS workspaces, first-call setup — land before any bucket is recorded.
//! Then each bucket, in bucket order, is warmed up at its own exact shape immediately before it
//! is recorded: the warmup its recording consumes, at the shape the graph bakes. Every warmup
//! stages a dummy run and copies it in through the staging ring's acquire and fence, so a dummy
//! run's copy is fenced as a step's is, and each recording is made over the run its warmup left
//! on the device.
//!
//! A recording holds the bucket's keyed step and nothing else. The copy-in in front of it is
//! eager, run by the warmup, and nothing is read back behind it. The dummy run names no request
//! slot for any row and a live-row count of zero, so the sample the recording holds is launched
//! over every row of the bucket and returns for each: no slot's sampling record or draw counter
//! moves while the bucket ladder is captured.
//!
//! Free device memory is read ahead of and behind the whole capture, behind the one warmup
//! ahead of every recording, and behind each bucket's warmup and recording, so that each bucket
//! starts from the reading the one before it ended on; and each bucket is timed: what one graph
//! cost and what the capture cost at startup, logged here on the executor thread and handed back
//! as a [`CaptureReport`], with graph memory fitted to a fixed term plus a marginal term per
//! graph.
//! A reading can be the first call to check the context after a driver status was deferred
//! onto it, in which case the reading fails under that status's own classification; every
//! reading is propagated, never logged and passed over, so a deferred failure fails the capture
//! rather than being read as a number.
//!
//! A bucket that will not capture is an error, and the session goes with it: there is no
//! recapture and no per-bucket eager fallback, so the graph set is every bucket or nothing.

use std::time::Instant;

use atoma_core::types::BlockId;
use atoma_runtime::context::RuntimeContext;
use atoma_runtime::error::RuntimeError;
use atoma_runtime::graph_memory::GraphMemoryError;
use atoma_runtime::session::{Allocation, BakedBuffers, Capture, GraphIdx, Replay};
use thiserror::Error;
use tracing::info;

use crate::decode::graphs::{dummy_runs, CaptureReport, GraphCost, GraphSet, GraphSetError};
use crate::decode::inputs::InputsError;
use crate::decode::staging::DummyRun;
use crate::device::decode::{DecodeStep, DecodeStepError};
use crate::device::sampler::{DeviceSampler, SamplerError};

/// Why the bucket ladder could not be captured.
#[derive(Debug, Error)]
pub enum CaptureError {
    #[error(transparent)]
    GraphSet(#[from] GraphSetError),
    #[error(transparent)]
    DecodeStep(#[from] DecodeStepError),
    #[error(transparent)]
    Inputs(#[from] InputsError),
    #[error(transparent)]
    Sampler(#[from] SamplerError),
    /// A reading of free device memory refused: under its own status, or one deferred onto the
    /// context that the reading was the first to check.
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
}

/// What capturing the bucket ladder produced: the session in its Replay phase, the graph serving
/// each bucket, and what the capture cost.
pub struct Captured {
    /// The session in its Replay phase, every bucket's graph recorded in it.
    pub session: Replay,
    /// The graph serving each bucket, at the bucket's index in `session`.
    pub graphs: GraphSet,
    /// What the capture cost, as logged.
    pub report: CaptureReport,
}

/// Captures one graph for every bucket `decode_step` serves, over the padding dummies' `blocks`,
/// reading free memory through `context` around each, and hands back the session in its Replay
/// phase with the graph set to serve from and the report of what it cost: the graph at a bucket's
/// index is the recording of that bucket.
///
/// No wait on candle's stream goes in front of a copy-in here: building the step joined that
/// stream, and this runs before any forward is enqueued on it.
///
/// # Errors
///
/// Returns [`CaptureError`] when a bucket has more rows than `blocks` holds or than the sampler
/// holds, a dummy run cannot be staged or copied in, a bucket's keyed step will not warm up or
/// record, or free memory cannot be read. The session is consumed either way.
pub fn capture_bucket_ladder(
    context: &RuntimeContext,
    allocation: Allocation,
    decode_step: &mut DecodeStep,
    sampler: &mut DeviceSampler,
    blocks: &[BlockId],
) -> Result<Captured, CaptureError> {
    let runs = dummy_runs(decode_step.buckets(), blocks)?;
    let started = Instant::now();
    let before = context.free_memory()?;
    let mut capture = allocation.into_capture();
    // Each recording below consumes a warmup at its own bucket's shape; this one lands the
    // backend's one-time lazy allocations first. `DecodeStep::build` refuses a bucket ladder
    // with no usable bucket, so `runs` is never empty through it, and an empty one records
    // nothing.
    if let Some(largest) = runs.iter().max_by_key(|run| run.rows()) {
        warm_up_bucket(&mut capture, decode_step, sampler, largest)?;
    }
    let mut graphs = Vec::with_capacity(runs.len());
    let mut costs = Vec::with_capacity(runs.len());
    let mut free = context.free_memory()?;
    for run in &runs {
        let bucket_started = Instant::now();
        let graph = capture_bucket(&mut capture, decode_step, sampler, run)?;
        let after = context.free_memory()?;
        let cost = GraphCost::measured(run, bucket_started.elapsed(), free, after);
        info!(
            bucket = cost.bucket.0,
            rows = cost.rows,
            elapsed = ?cost.elapsed,
            used_bytes = %cost.used,
            free_bytes = %cost.free,
            "bucket captured"
        );
        free = after;
        graphs.push(graph);
        costs.push(cost);
    }
    let report = CaptureReport::measured(costs, started.elapsed(), before, free);
    log_report(&report);
    Ok(Captured {
        session: capture.into_replay(),
        graphs: GraphSet::new(graphs),
        report,
    })
}

/// Warms `run`'s bucket up and records it: the graph at the index the recording took.
fn capture_bucket(
    capture: &mut Capture,
    decode_step: &mut DecodeStep,
    sampler: &mut DeviceSampler,
    run: &DummyRun,
) -> Result<GraphIdx, CaptureError> {
    warm_up_bucket(capture, decode_step, sampler, run)?;
    let mut step = decode_step.bucket_step(sampler, run)?;
    Ok(capture.record(&mut step, BakedBuffers::default())?)
}

/// Runs `run`'s bucket eagerly at its exact shape: the dummy run staged on the sampler and in a
/// staging entry, its block copied in, then the bucket's keyed step, each waited for. Leaves the
/// session ready to record the same step, and the device block holding the run the recording is
/// made over.
fn warm_up_bucket(
    capture: &mut Capture,
    decode_step: &mut DecodeStep,
    sampler: &mut DeviceSampler,
    run: &DummyRun,
) -> Result<(), CaptureError> {
    sampler.stage_dummy_run(run)?;
    let entry = decode_step.stage_dummy(run)?;
    capture.warm_up(&mut decode_step.copy_in(entry, run.bucket())?)?;
    capture.warm_up(&mut decode_step.bucket_step(sampler, run)?)?;
    Ok(())
}

/// Logs what the capture cost at startup: its time and memory in all, then graph memory as a
/// fixed term plus a marginal term per graph where two or more graphs give the fit its two
/// terms, and that one graph's reading names neither where it does not.
fn log_report(report: &CaptureReport) {
    info!(
        graphs = report.graphs.len(),
        elapsed = ?report.elapsed,
        used_bytes = %report.used,
        free_bytes = %report.free,
        "bucket ladder captured"
    );
    match report.graph_memory() {
        Ok(memory) => info!(
            fixed_bytes = %memory.fixed(),
            marginal_bytes_per_graph = %memory.marginal(),
            "graph memory fitted over what each graph used"
        ),
        Err(GraphMemoryError::FitWithoutTwoGraphs { graphs }) => info!(
            graphs,
            "graph memory not fitted: one graph's used bytes name no fixed and marginal term"
        ),
    }
}
