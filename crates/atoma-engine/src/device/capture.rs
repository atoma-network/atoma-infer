//! Capturing the bucket ladder the decode step serves: one graph per bucket, each recorded over
//! a dummy run of that bucket's rows, in the Capture session phase.
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
//! A bucket that will not capture is an error, and the session goes with it: there is no
//! recapture and no per-bucket eager fallback, so the graph set is every bucket or nothing.

use atoma_core::types::BlockId;
use atoma_runtime::session::{Allocation, BakedBuffers, Capture, Replay};
use thiserror::Error;
use tracing::info;

use crate::decode::graphs::{dummy_runs, GraphSet, GraphSetError};
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
}

/// Captures one graph for every bucket `decode_step` serves, over the padding dummies' `blocks`,
/// and hands back the session in its Replay phase with the graph set to serve from: the graph at
/// a bucket's index is the recording of that bucket.
///
/// No wait on candle's stream goes in front of a copy-in here: building the step joined that
/// stream, and this runs before any forward is enqueued on it.
///
/// # Errors
///
/// Returns [`CaptureError`] when a bucket has more rows than `blocks` holds or than the sampler
/// holds, a dummy run cannot be staged or copied in, or a bucket's keyed step will not warm up
/// or record. The session is consumed either way.
pub fn capture_bucket_ladder(
    allocation: Allocation,
    decode_step: &mut DecodeStep,
    sampler: &mut DeviceSampler,
    blocks: &[BlockId],
) -> Result<(Replay, GraphSet), CaptureError> {
    let runs = dummy_runs(decode_step.buckets(), blocks)?;
    let mut capture = allocation.into_capture();
    // Each recording below consumes a warmup at its own bucket's shape; this one lands the
    // backend's one-time lazy allocations first. `DecodeStep::build` refuses a bucket ladder
    // with no usable bucket, so `runs` is never empty through it, and an empty one records
    // nothing.
    if let Some(largest) = runs.iter().max_by_key(|run| run.rows()) {
        warm_up_bucket(&mut capture, decode_step, sampler, largest)?;
    }
    let mut graphs = Vec::with_capacity(runs.len());
    for run in &runs {
        warm_up_bucket(&mut capture, decode_step, sampler, run)?;
        let mut step = decode_step.bucket_step(sampler, run)?;
        let graph = capture.record(&mut step, BakedBuffers::default())?;
        info!(
            bucket = run.bucket().0,
            rows = run.rows(),
            "bucket captured"
        );
        graphs.push(graph);
    }
    info!(graphs = graphs.len(), "bucket ladder captured");
    Ok((capture.into_replay(), GraphSet::new(graphs)))
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
