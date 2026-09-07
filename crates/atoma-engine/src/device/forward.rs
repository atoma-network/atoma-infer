//! The forward on the device: a keyed decode batch on the step over runtime-owned tensors, every
//! other batch through the Llama forward on candle's stream, and the selected rows sampled on the
//! device either way, so what comes back to the host is one copy of the sampled tokens and never
//! the logits.
//!
//! The logits path stays reachable as [`CudaForward::forward_logits`], which reads them back into
//! a readback the caller owns: what the decode parity harness compares the two forwards through,
//! and the only way logits reach the host.
//!
//! The step over runtime tensors and the sampler bake every address they read, and candle owns
//! the weights and the cache among them. Building the forward reads every baked address by name,
//! and a debug build reads them all again before each keyed step and panics naming the first that
//! moved, before the step's own work is enqueued. Each address is re-read through its owner, and
//! candle's tensors carry cudarc's event guard, so the reading enqueues a wait and a record for
//! each weight and cache first. A release build carries no such list and reads nothing.

use std::sync::Arc;

use atoma_runtime::session::Replay;
use candle_core::{Layout, Storage, Tensor};
use cudarc::driver::{CudaStream, CudaView};
use models::FlashAttentionMetadata;
use thiserror::Error;

use crate::batch::BatchLayout;
use crate::device::sampler::{DeviceSampler, SamplerError};
use crate::device::{KvCache, RankDevice, Weights};
use crate::forward::Forward;
use crate::logits::Logits;
use crate::readback::{Readback, ReadbackError};

#[cfg(not(feature = "nccl"))]
use atoma_core::dispatch::DispatchDecision;
#[cfg(not(feature = "nccl"))]
use tracing::debug;

#[cfg(all(not(feature = "nccl"), debug_assertions))]
use crate::decode::baked::{BakedAddress, BakedAddresses};
#[cfg(not(feature = "nccl"))]
use crate::decode::batch::{Checked, DecodeBatch};
#[cfg(not(feature = "nccl"))]
use crate::device::decode::{DecodeStep, DecodeStepError};

/// Why a step could not be run on the device.
#[derive(Debug, Error)]
pub enum CudaForwardError {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Readback(#[from] ReadbackError),
    #[error(transparent)]
    Sampler(#[from] SamplerError),
    #[cfg(not(feature = "nccl"))]
    #[error(transparent)]
    DecodeStep(Box<DecodeStepError>),
    /// The forward's logits came back on the host, which no device forward should produce.
    #[error("the logits are not on the device")]
    LogitsNotOnDevice,
}

/// The step's error is boxed: it carries the operand report of whichever op refused, and every
/// forward returns this result on its hot path.
#[cfg(not(feature = "nccl"))]
impl From<DecodeStepError> for CudaForwardError {
    fn from(error: DecodeStepError) -> Self {
        Self::DecodeStep(Box::new(error))
    }
}

/// What the Allocation session phase produced for one rank.
pub struct Allocated {
    pub device: RankDevice,
    pub weights: Weights,
    pub kv_cache: KvCache,
    /// Rank zero samples its logits; a follower's are read by nobody, so it holds no sampler and
    /// its forward returns no tokens.
    pub sampler: Option<DeviceSampler>,
    pub vocab: usize,
}

/// The model forward on one rank's device.
///
/// Holds the session's Replay phase for the process lifetime: nothing is captured in this crate,
/// and holding the phase is what keeps the allocation from being reopened. The step over runtime
/// tensors is enqueued through it; under NCCL the decode step stays on candle and there is none.
pub struct CudaForward {
    allocated: Allocated,
    #[cfg(not(feature = "nccl"))]
    decode_step: DecodeStep,
    /// Every address the decode step and the sampler bake, read when the forward was built; a
    /// debug build reads them again before each keyed step, and a release build carries no list.
    #[cfg(all(not(feature = "nccl"), debug_assertions))]
    baked: BakedAddresses,
    /// Held for the process lifetime, which is what keeps the allocation from being reopened;
    /// under NCCL nothing is enqueued through it.
    #[cfg_attr(feature = "nccl", allow(dead_code))]
    session: Replay,
}

impl CudaForward {
    /// Holds what the rank allocated and the step over runtime tensors for the Replay phase
    /// `session`, and, in a debug build, bakes every address the step and the sampler read, by
    /// name, to check before each keyed step.
    ///
    /// # Errors
    ///
    /// Returns [`CudaForwardError`] when an address the step baked cannot be read: a weight or a
    /// cache that is not a bf16 tensor on the device.
    pub fn new(
        allocated: Allocated,
        #[cfg(not(feature = "nccl"))] decode_step: DecodeStep,
        session: Replay,
    ) -> Result<Self, CudaForwardError> {
        #[cfg(all(not(feature = "nccl"), debug_assertions))]
        let baked = BakedAddresses::bake(addresses(&allocated, &decode_step)?);
        Ok(Self {
            allocated,
            #[cfg(not(feature = "nccl"))]
            decode_step,
            #[cfg(all(not(feature = "nccl"), debug_assertions))]
            baked,
            session,
        })
    }

    /// Reads every baked address again and panics naming the first that moved, before the step's
    /// own work is enqueued: what a debug build runs before each keyed step. Each address is read
    /// through its owner, so the reading enqueues cudarc's event guard for each weight and cache
    /// candle holds. A release build trusts the addresses and reads none.
    #[cfg(all(not(feature = "nccl"), debug_assertions))]
    fn assert_unmoved(&self) {
        let current = addresses(&self.allocated, &self.decode_step)
            .expect("every address the forward baked can be read again before a step");
        self.baked.assert_unmoved(current);
    }

    /// Runs `layout` and reads the logits of the rows it selected into `readback`: one row per
    /// selected row, in batch order, a vocabulary wide. Nothing is sampled, and no slot's record
    /// or draw counter moves, so a harness can read a step's logits without disturbing what the
    /// device sampler holds.
    ///
    /// # Errors
    ///
    /// Returns [`CudaForwardError`] when the step could not be run or read back.
    pub fn forward_logits<'a>(
        &mut self,
        layout: &BatchLayout,
        readback: &'a mut Readback<f32>,
    ) -> Result<Logits<'a>, CudaForwardError> {
        #[cfg(not(feature = "nccl"))]
        if let Some(batch) = self.keyed_batch(layout)? {
            #[cfg(debug_assertions)]
            self.assert_unmoved();
            let Self {
                allocated: _,
                decode_step,
                #[cfg(debug_assertions)]
                    baked: _,
                session,
            } = self;
            return Ok(decode_step.run_for_logits(session, layout, batch, readback)?);
        }
        let rows = layout.selected.len();
        let logits = self.candle_logits(layout)?;
        let Allocated {
            device,
            weights: _,
            kv_cache: _,
            sampler: _,
            vocab,
        } = &self.allocated;
        if rows == 0 {
            return Ok(Logits::new(&[], *vocab));
        }
        read_back(readback, device.stream(), &logits, rows, *vocab)
    }

    /// The batch as the decode step serves it, when the layout is keyed and the shape its graphs
    /// bake; a keyed batch it does not serve is logged and runs on candle.
    #[cfg(not(feature = "nccl"))]
    fn keyed_batch(&self, layout: &BatchLayout) -> Result<Option<DecodeBatch>, CudaForwardError> {
        let key = match layout.dispatch {
            DispatchDecision::FullReplay(key) | DispatchDecision::SegmentedReplay(key) => key,
            DispatchDecision::Eager(_) => return Ok(None),
        };
        match self.decode_step.check(layout, key)? {
            Checked::Step(batch) => Ok(Some(batch)),
            Checked::Eager(reason) => {
                debug!(%reason, "keyed batch served on candle");
                Ok(None)
            }
        }
    }

    /// Runs `batch` on the decode step, which stages the sampler for it and samples its live
    /// rows, when this rank samples.
    #[cfg(not(feature = "nccl"))]
    fn run_decode_step(
        &mut self,
        layout: &BatchLayout,
        batch: DecodeBatch,
    ) -> Result<&[u32], CudaForwardError> {
        #[cfg(debug_assertions)]
        self.assert_unmoved();
        let Self {
            allocated,
            decode_step,
            #[cfg(debug_assertions)]
                baked: _,
            session,
        } = self;
        Ok(decode_step.run(session, layout, batch, allocated.sampler.as_mut())?)
    }

    /// Runs `layout` through the Llama forward on candle's stream and samples the selected rows
    /// there, where the forward left its logits, from the sampler's own staging. No row gathers:
    /// candle takes its token ids from the host, and the batch it serves is not the shape the
    /// gather is for.
    fn candle_forward(&mut self, layout: &BatchLayout) -> Result<&[u32], CudaForwardError> {
        if let Some(sampler) = self.allocated.sampler.as_mut() {
            sampler.stage_eager(layout)?;
        }
        let logits = self.candle_logits(layout)?;
        let Allocated {
            device,
            weights: _,
            kv_cache: _,
            sampler,
            vocab: _,
        } = &mut self.allocated;
        let Some(sampler) = sampler else {
            return Ok(&[]);
        };
        if layout.selected.is_empty() {
            return Ok(&[]);
        }
        sample_logits(sampler, device.stream(), &logits)
    }

    /// Runs `layout` through the Llama forward on candle's stream, leaving its logits there.
    fn candle_logits(&mut self, layout: &BatchLayout) -> Result<Tensor, CudaForwardError> {
        let Uploaded {
            tokens,
            positions,
            selected,
            metadata,
        } = self.upload(layout)?;
        let Allocated {
            device: _,
            weights,
            kv_cache,
            sampler: _,
            vocab: _,
        } = &mut self.allocated;
        let kv_caches = kv_cache.layers_mut();
        let logits = weights
            .llama_mut()
            .forward(&tokens, &positions, &selected, &kv_caches, metadata)?;
        #[cfg(not(feature = "nccl"))]
        self.decode_step
            .after_candle(self.allocated.device.stream())?;
        Ok(logits)
    }

    /// The forward's inputs and attention metadata, uploaded from `layout`.
    fn upload(&self, layout: &BatchLayout) -> Result<Uploaded, candle_core::Error> {
        let device = self.allocated.device.candle();
        let tokens = layout.token_count();
        let entries = layout.entry_count();
        // A step in which no entry samples still runs, to write its KV; selecting one row keeps
        // every tensor downstream of the selection nonempty, and nobody reads that row.
        let selected: &[u32] = if layout.selected.is_empty() {
            &[0]
        } else {
            &layout.selected
        };
        let metadata = FlashAttentionMetadata::new(
            Tensor::from_slice(&layout.context_lengths, entries, device)?,
            Tensor::from_slice(&layout.slot_mapping, tokens, device)?,
            Tensor::from_slice(&layout.query_start_locations, entries + 1, device)?,
            layout.prefill_tokens,
            layout.decode_tokens,
            layout.max_query_len,
            layout.max_decode_sequence_len,
            layout.max_prefill_sequence_len,
            layout.prefill_entries,
            Tensor::from_slice(&layout.sequence_start_locations, entries + 1, device)?,
            Tensor::from_slice(&layout.sequence_lengths, entries, device)?,
            Tensor::from_slice(
                &layout.block_tables,
                (entries, layout.block_table_width),
                device,
            )?,
        )?;
        Ok(Uploaded {
            tokens: Tensor::from_slice(&layout.tokens, (1, tokens), device)?,
            positions: Tensor::from_slice(&layout.positions, (1, tokens), device)?,
            selected: Tensor::from_slice(selected, selected.len(), device)?,
            metadata,
        })
    }
}

/// Every address the decode step and the sampler bake, read from the memory that holds each, in
/// one fixed order: the step's, then the sampler's when this rank holds one.
#[cfg(all(not(feature = "nccl"), debug_assertions))]
fn addresses(
    allocated: &Allocated,
    decode_step: &DecodeStep,
) -> Result<Vec<BakedAddress>, CudaForwardError> {
    let stream = allocated.device.stream();
    let mut addresses = decode_step.addresses(&allocated.weights, &allocated.kv_cache, stream)?;
    if let Some(sampler) = &allocated.sampler {
        addresses.extend(sampler.addresses(stream));
    }
    Ok(addresses)
}

/// One step's inputs on the device.
struct Uploaded {
    tokens: Tensor,
    positions: Tensor,
    selected: Tensor,
    metadata: FlashAttentionMetadata,
}

impl Forward for CudaForward {
    type Error = CudaForwardError;

    /// A rank with no sampler samples nothing: it ran the step for its share of the model, and
    /// the leader's tokens are the step's.
    fn forward(&mut self, layout: &BatchLayout) -> Result<&[u32], CudaForwardError> {
        #[cfg(not(feature = "nccl"))]
        if let Some(batch) = self.keyed_batch(layout)? {
            return self.run_decode_step(layout, batch);
        }
        self.candle_forward(layout)
    }
}

/// Samples `sampler`'s staged rows from the `logits` candle left on `stream`.
///
/// Candle hands its storage out behind a read guard, so the view over it lives for as long as
/// that guard does: the guard is held here, and what the call returns borrows the sampler.
fn sample_logits<'a>(
    sampler: &'a mut DeviceSampler,
    stream: &Arc<CudaStream>,
    logits: &Tensor,
) -> Result<&'a [u32], CudaForwardError> {
    let (storage, layout) = logits.storage_and_layout();
    let device_logits = device_logits(&storage, layout)?;
    Ok(sampler.run_on(stream, &device_logits)?)
}

/// Copies the `rows` rows of `logits` back through `readback` on `stream`. The storage guard is
/// held here for the same reason it is in [`sample_logits`].
fn read_back<'a>(
    readback: &'a mut Readback<f32>,
    stream: &Arc<CudaStream>,
    logits: &Tensor,
    rows: usize,
    vocab: usize,
) -> Result<Logits<'a>, CudaForwardError> {
    let (storage, layout) = logits.storage_and_layout();
    let device_logits = device_logits(&storage, layout)?;
    Ok(Logits::new(
        readback.read(stream, &device_logits, rows)?,
        vocab,
    ))
}

/// The f32 logits a tensor's `storage` holds on the device, as the view `layout` places over
/// it; the caller holds the guard `storage` came out of for as long as the view lives.
fn device_logits<'a>(
    storage: &'a Storage,
    layout: &Layout,
) -> Result<CudaView<'a, f32>, CudaForwardError> {
    let Storage::Cuda(storage) = storage else {
        return Err(CudaForwardError::LogitsNotOnDevice);
    };
    let start = layout.start_offset();
    Ok(storage
        .as_cuda_slice::<f32>()?
        .slice(start..start + layout.shape().elem_count()))
}
