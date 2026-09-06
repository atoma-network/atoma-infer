//! The decode step over runtime-owned tensors on one rank's device: built once at Allocation
//! from the addresses candle loaded the weights and cache at, and run for every keyed batch.
//!
//! Candle keeps owning the weights and the cache; this module snapshots their device addresses
//! into tensor views, allocates the arena, the step's fixed buffers and the cuBLAS workspace,
//! resolves every usable bucket's slot tables, and holds the step descriptor over them. A step is
//! then six descriptors on the capture stream: the wait on candle's stream, the sampler's record
//! upload, the upload of the bucket's packed block (the five inputs and the sampler's two
//! per-step arrays in one copy, with the staging entry's fence signaled behind it), the gather
//! that takes each decoding row's token from what the device sampled for its slot, the model
//! step, and the sample, which leaves the tokens on the device and reads them back; then one host
//! wait. Nothing is captured here. Going through the descriptor seam is what lets a later capture
//! record the gather, the model step and the sample unchanged; the two uploads are the host's
//! copies of what changed and stay in front of the graph. A dummy run — a bucket's rows as
//! padding rows over one block each — is staged and uploaded the same way, with no sampler
//! descriptor and no readback: what a capture check or a warmup runs when there is no live batch.
//!
//! The step's outputs reach the sampler and the readback as tensor views narrowed to the live
//! rows: the bucket's logits and the sampler's row tokens are viewed once, at Allocation, over
//! the step's fixed buffers, and a descriptor takes its address, its rows and its width off the
//! view it is handed, so nothing sized by the batch is spelled out at the call.
//!
//! Every address the step bakes can be read again from the memory that holds it, by name, in one
//! fixed order ([`DecodeStep::addresses`]): candle's weights and cache without minting a view, the
//! step's own memory from the allocations it owns. The forward bakes that reading when it is
//! built and a debug build compares a fresh one against it before each keyed step.

use std::sync::Arc;

use atoma_core::dispatch::{DispatchConfig, GraphKey};
use atoma_core::types::TokenCount;
use atoma_models::attention::{block_table_columns, AttentionError, AttentionPlan};
use atoma_models::dims::{DimsError, Llama3RopeScaling, LlamaDims, RopeParams};
use atoma_models::gemm::{GemmError, StepBlas, WORKSPACE_BYTES};
use atoma_models::kernels::RotaryTensors;
use atoma_models::layer::{LayerWeight, LLAMA_LAYER};
use atoma_models::llama::slots::{
    Bucket, BucketSlots, LayerWeights, LlamaCache, LlamaWeights, SlotError, SlotSources,
    StepStatics,
};
use atoma_models::llama::step::{LlamaDecode, LlamaStep, StepError};
use atoma_models::rope::RotaryTables;
use atoma_runtime::arena::{ArenaError, ArenaLayout, BucketIdx, CaptureArena};
use atoma_runtime::error::RuntimeError;
use atoma_runtime::session::{Allocation, Replay};
use atoma_runtime::tensor::{Dtype, Layout, Tensor, TensorError};
use candle_core::cuda::CudaStorageSlice;
use candle_core::{DType, Storage, Tensor as CandleTensor};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr};
use models::llama::{Config, LayerTensors, Llama, Llama3RopeType};
use thiserror::Error;
use tracing::info;

use crate::batch::BatchLayout;
use crate::config::Dtype as ConfiguredDtype;
use crate::decode::baked::{BakedAddress, BakedName};
use crate::decode::batch::{Checked, DecodeBatch, DecodeBatchError, DecodeBuckets};
use crate::decode::inputs::{DecodeInputs, InputsError, Upload, WaitEvent};
use crate::decode::ring::{StagingDepth, StagingEntry};
use crate::decode::staging::{DummyRun, StagingShape};
use crate::device::sampler::{DeviceSampler, SamplerError};
use crate::device::{KvCache, RankDevice, Weights};
use crate::logits::Logits;
use crate::readback::{Readback, ReadbackError};

/// Why the decode step could not be built or run.
#[derive(Debug, Error)]
pub enum DecodeStepError {
    #[error(
        "model.dtype is {dtype:?}; the decode step over runtime tensors runs bf16 only, so set \
         model.dtype = \"bf16\""
    )]
    NotBf16 { dtype: ConfiguredDtype },
    #[error("{what} is {dtype:?} on the device; the step reads bf16")]
    WeightDtype { what: &'static str, dtype: DType },
    #[error("{what} is not on the device")]
    NotOnDevice { what: &'static str },
    #[error("{what} is not contiguous on the device")]
    NotContiguous { what: &'static str },
    #[error("{what} holds {held} elements; the step views {expected}")]
    ElementCount {
        what: &'static str,
        held: usize,
        expected: usize,
    },
    #[error(
        "no entry of engine.dispatch.bucket_ladder is at or below captured_max_requests of \
         {captured_max}; the decode step needs one bucket to serve"
    )]
    NoUsableBucket { captured_max: usize },
    #[error("the device reports {count} multiprocessors, which is not a count")]
    MultiprocessorCount { count: i32 },
    #[error(
        "a layer's cache is rank {rank}; candle allocates [2, blocks, block_size, kv_heads, \
         head_dim]"
    )]
    CacheRank { rank: usize },
    #[error(transparent)]
    Dims(#[from] DimsError),
    #[error(transparent)]
    Arena(#[from] ArenaError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Slot(#[from] SlotError),
    #[error(transparent)]
    Step(#[from] StepError),
    #[error(transparent)]
    Gemm(#[from] GemmError),
    #[error(transparent)]
    Inputs(#[from] InputsError),
    #[error(transparent)]
    Readback(#[from] ReadbackError),
    #[error(transparent)]
    Sampler(#[from] SamplerError),
    #[error(transparent)]
    Batch(#[from] DecodeBatchError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
}

/// What the decode step is sized from: the buckets it serves and the sequence it must hold.
#[derive(Debug, Clone)]
pub struct DecodeStepPlan {
    pub dispatch: DispatchConfig,
    pub max_model_len: TokenCount,
    pub block_size: TokenCount,
    pub dtype: ConfiguredDtype,
    /// How many staging entries the inputs' staging ring holds.
    pub staging_depth: StagingDepth,
}

/// The step's outputs and workspace on the device, owned here for as long as the views over
/// them are read, and read again for the debug check that the views still name them.
struct Statics {
    logits: CudaSlice<u8>,
    softmax_lse: CudaSlice<u8>,
    lse_accum: CudaSlice<u8>,
    o_accum: CudaSlice<u8>,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
}

impl Statics {
    /// Each static's address, read from the memory that holds it, by name in one fixed order.
    fn addresses(&self, stream: &Arc<CudaStream>) -> [BakedAddress; 6] {
        [
            (BakedName::Logits, address(&self.logits, stream)),
            (BakedName::LogSumExp, address(&self.softmax_lse, stream)),
            (BakedName::SplitLogSumExp, address(&self.lse_accum, stream)),
            (BakedName::SplitOutput, address(&self.o_accum, stream)),
            (BakedName::CosineTable, address(&self.cos, stream)),
            (BakedName::SineTable, address(&self.sin, stream)),
        ]
        .map(|(name, address)| BakedAddress { name, address })
    }
}

/// The decode step over runtime-owned tensors, and everything it addresses.
pub struct DecodeStep {
    buckets: DecodeBuckets,
    inputs: DecodeInputs,
    decode: LlamaDecode,
    blas: StepBlas,
    /// Recorded on candle's stream after every candle forward; the step waits on it.
    candle_done: CudaEvent,
    /// The arena's memory, owned here for as long as the slot tables over it are read, and read
    /// again for the debug check that they still name it.
    arena: CudaSlice<u8>,
    statics: Statics,
}

impl DecodeStep {
    /// Builds the step over `weights` and `kv_cache` as candle loaded them, for the buckets
    /// `plan` makes usable, during the Allocation phase.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStepError`] when the model is not loaded in bf16, no bucket is usable, a
    /// weight or cache is not the shape the step reads, or the device refuses an allocation.
    pub fn build(
        allocation: &Allocation,
        device: &RankDevice,
        weights: &Weights,
        kv_cache: &KvCache,
        plan: &DecodeStepPlan,
    ) -> Result<Self, DecodeStepError> {
        if plan.dtype != ConfiguredDtype::Bf16 {
            return Err(DecodeStepError::NotBf16 { dtype: plan.dtype });
        }
        let buckets = DecodeBuckets::usable(&plan.dispatch);
        if buckets.tokens().is_empty() {
            return Err(DecodeStepError::NoUsableBucket {
                captured_max: plan.dispatch.captured_max_requests.get(),
            });
        }
        let llama = weights.llama();
        let dims = llama_dims(llama.get_config())?;
        let stream = device.stream();
        let shape = StagingShape {
            max_tokens: buckets.largest(),
            block_table_width: block_table_columns(
                plan.max_model_len.get(),
                plan.block_size.get(),
                dims.head_dim,
            ),
            max_position: dims.rope.max_position,
            block_size: plan.block_size,
        };
        let sm_count = multiprocessors(stream)?;
        let plans = buckets
            .tokens()
            .iter()
            .map(|&tokens| {
                AttentionPlan::new(
                    &dims,
                    tokens,
                    plan.block_size.get(),
                    shape.block_table_width,
                    sm_count,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sizing = Sizing {
            dims: &dims,
            plans: &plans,
            shape,
            buckets: &buckets,
        };
        let inputs = DecodeInputs::new(allocation, stream, shape, &buckets, plan.staging_depth)?;
        let (statics, step_statics) = allocate_statics(allocation, stream, &sizing)?;
        let (arena_memory, arena_bytes, slots) =
            resolve_slots(allocation, stream, &sizing, &step_statics, &inputs)?;
        let decode = LlamaDecode::new(
            dims,
            snapshot_weights(allocation, llama, stream)?,
            snapshot_cache(allocation, kv_cache, &dims, stream)?,
            step_statics.rotary,
            slots,
        )?;
        let blas = StepBlas::new(allocation, zeroed(stream, WORKSPACE_BYTES)?)?;
        let candle_done = stream
            .context()
            .new_event(None)
            .map_err(RuntimeError::from)?;
        // Every allocation, zero fill and table upload above went to candle's stream, and the
        // first step may be keyed before any candle forward records the event a step waits on;
        // the stream is joined here, in the Allocation phase, where a synchronize is legal.
        stream.synchronize().map_err(RuntimeError::from)?;
        info!(
            buckets = ?buckets.tokens(),
            arena_bytes,
            block_table_width = shape.block_table_width,
            multiprocessors = sm_count,
            "decode step over runtime tensors built"
        );
        Ok(Self {
            buckets,
            inputs,
            decode,
            blas,
            candle_done,
            arena: arena_memory,
            statics,
        })
    }

    /// Every address the step bakes, read from the memory that holds it, in one fixed order:
    /// candle's embedding table, each layer's nine weights, the final norm gain and the head
    /// projection, each layer's cache, then the input block, the arena and the statics. The
    /// forward bakes this reading when it is built and a debug build reads it again before each
    /// keyed step; candle's addresses are the ones that can move, and are read without minting
    /// a view.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStepError`] when a weight or a cache is not a bf16 tensor on the device.
    pub fn addresses(
        &self,
        weights: &Weights,
        kv_cache: &KvCache,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<BakedAddress>, DecodeStepError> {
        let mut addresses = candle_addresses(weights, kv_cache, stream)?;
        addresses.push(BakedAddress {
            name: BakedName::InputBlock,
            address: address(self.inputs.device_block(), stream),
        });
        addresses.push(BakedAddress {
            name: BakedName::Arena,
            address: address(&self.arena, stream),
        });
        addresses.extend(self.statics.addresses(stream));
        Ok(addresses)
    }

    /// Checks a batch keyed by `key` against the shape the step bakes.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStepError::Batch`] when the layout contradicts its key.
    pub fn check(&self, layout: &BatchLayout, key: GraphKey) -> Result<Checked, DecodeStepError> {
        Ok(DecodeBatch::check(
            layout,
            key,
            &self.buckets,
            self.inputs.shape().block_table_width,
        )?)
    }

    /// Runs `batch`'s step through `session` and samples it: a staging entry acquired and the
    /// inputs and the sampler staged into its block, then the wait on candle's stream, the
    /// sampler's record upload, the block's upload, the gather, the model step and the sample
    /// enqueued in that order, then the host wait on the sampled tokens. The batch states that
    /// every entry computes one token, so its rows are the token rows the gather covers. A rank
    /// with no sampler runs the same step without the sampler's descriptors and waits for it
    /// instead, and returns no tokens.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStepError`] when the inputs cannot be staged, a descriptor cannot be
    /// enqueued, or the wait fails.
    pub fn run<'a>(
        &mut self,
        session: &Replay,
        layout: &BatchLayout,
        batch: DecodeBatch,
        sampler: Option<&'a mut DeviceSampler>,
    ) -> Result<&'a [u32], DecodeStepError> {
        let entry = self.inputs.acquire()?;
        let arrays = self.inputs.stage(&entry, layout, &batch)?;
        let Some(sampler) = sampler else {
            session.run(&mut WaitEvent::new(&self.candle_done))?;
            session.run(&mut self.upload(entry, batch.bucket)?)?;
            session.run(&mut self.descriptor(batch.bucket)?)?;
            session.synchronize()?;
            return Ok(&[]);
        };
        sampler.stage(layout, batch.tokens, arrays)?;
        let views = self.inputs.bucket(batch.bucket)?;
        session.run(&mut WaitEvent::new(&self.candle_done))?;
        session.run(&mut sampler.upload_records()?)?;
        session.run(&mut self.upload(entry, batch.bucket)?)?;
        session.run(&mut sampler.gather(&views.inputs.token_ids, &views.gather_slots)?)?;
        session.run(&mut self.descriptor(batch.bucket)?)?;
        let logits = self.live_logits(&batch)?;
        let row_slots = views.row_slots.narrow(0, 0, batch.live)?;
        session.run(&mut sampler.sample(&logits, &row_slots)?)?;
        Ok(sampler.wait()?)
    }

    /// Runs `batch`'s step through `session` and reads the logits of its live rows back into
    /// `readback`: the parity path, which compares the step against the candle forward. Nothing
    /// is sampled, so no slot's record or draw counter moves.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStepError`] when the inputs cannot be staged, a descriptor cannot be
    /// enqueued, or the wait fails.
    pub fn run_for_logits<'a>(
        &mut self,
        session: &Replay,
        layout: &BatchLayout,
        batch: DecodeBatch,
        readback: &'a mut Readback<f32>,
    ) -> Result<Logits<'a>, DecodeStepError> {
        let entry = self.stage(layout, &batch)?;
        session.run(&mut WaitEvent::new(&self.candle_done))?;
        session.run(&mut self.upload(entry, batch.bucket)?)?;
        session.run(&mut self.descriptor(batch.bucket)?)?;
        let logits = self.live_logits(&batch)?;
        session.run(&mut readback.copy(&logits)?)?;
        Ok(Logits::new(readback.wait()?, logits.dim(1)))
    }

    /// The f32 `[live, vocab]` view of the logits `batch`'s step writes for its live rows: the
    /// leading rows of its bucket's logits.
    fn live_logits(&self, batch: &DecodeBatch) -> Result<Tensor, DecodeStepError> {
        let bucket = self.decode.bucket(batch.bucket)?;
        Ok(bucket.statics.logits.narrow(0, 0, batch.live)?)
    }

    /// Acquires a staging entry, waiting until the copy that last read its block has finished,
    /// and writes `batch`'s inputs from `layout` into it; the sampler's two arrays in the block
    /// are left as they are.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStepError::Inputs`] when the fence cannot be waited on or the layout
    /// cannot be staged.
    pub fn stage(
        &mut self,
        layout: &BatchLayout,
        batch: &DecodeBatch,
    ) -> Result<StagingEntry, DecodeStepError> {
        let entry = self.inputs.acquire()?;
        self.inputs.stage(&entry, layout, batch)?;
        Ok(entry)
    }

    /// Acquires a staging entry, waiting until the copy that last read its block has finished,
    /// and writes `run`'s rows into it as padding rows, the sampler's two arrays naming no
    /// request slot: a dummy run's staging, which [`DecodeStep::upload`] carries to the device
    /// as it carries a step's.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStepError::Inputs`] when the fence cannot be waited on, the inputs were
    /// not built for the run's bucket, or the run does not name one block per row of it.
    pub fn stage_dummy(&mut self, run: &DummyRun) -> Result<StagingEntry, DecodeStepError> {
        let entry = self.inputs.acquire()?;
        self.inputs.stage_dummy(&entry, run)?;
        Ok(entry)
    }

    /// The descriptor that copies `bucket`'s packed length from `entry`'s block to the device in
    /// one copy and signals the staging entry's fence behind it.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStepError::Inputs`] when the inputs were not built for `bucket`.
    pub fn upload(
        &self,
        entry: StagingEntry,
        bucket: BucketIdx,
    ) -> Result<Upload<'_>, DecodeStepError> {
        Ok(self.inputs.upload(entry, bucket)?)
    }

    /// The descriptor that enqueues `bucket`'s model step over the uploaded inputs.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStepError::Step`] when no such bucket was resolved.
    pub fn descriptor(&self, bucket: BucketIdx) -> Result<LlamaStep<'_>, DecodeStepError> {
        Ok(self.decode.step(bucket, &self.blas)?)
    }

    /// Records that candle's stream has finished a forward; the next step waits on it before
    /// reading the cache.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStepError::Runtime`] when the event cannot be recorded.
    pub fn after_candle(&self, stream: &Arc<CudaStream>) -> Result<(), DecodeStepError> {
        self.candle_done
            .record(stream)
            .map_err(RuntimeError::from)?;
        Ok(())
    }
}

/// What the arena, the statics and every bucket's tables are sized from.
struct Sizing<'a> {
    dims: &'a LlamaDims,
    plans: &'a [AttentionPlan],
    shape: StagingShape,
    /// The buckets served, one plan each in `plans`, in bucket-ladder order.
    buckets: &'a DecodeBuckets,
}

/// Allocates the arena and resolves every bucket's slot tables over it, each bucket's inputs
/// the views `inputs` minted for it: the arena's memory (owned for as long as the tables are
/// read), its size, and the tables in bucket order.
fn resolve_slots(
    allocation: &Allocation,
    stream: &Arc<CudaStream>,
    sizing: &Sizing<'_>,
    statics: &StepStatics,
    inputs: &DecodeInputs,
) -> Result<(CudaSlice<u8>, usize, Vec<BucketSlots>), DecodeStepError> {
    let dims = sizing.dims;
    let arena = CaptureArena::new(
        dims.layers + 1,
        LLAMA_LAYER.role_table(dims),
        sizing.buckets.tokens(),
        ArenaLayout::Greedy,
    )?;
    let arena_memory = zeroed(stream, arena.total_size())?;
    let memory = Tensor::new(
        allocation,
        address(&arena_memory, stream),
        Layout::contiguous(
            &[arena.total_size() / Dtype::Bf16.size_in_bytes()],
            Dtype::Bf16,
        )?,
    )?;
    let sources = SlotSources {
        memory: &memory,
        arena: &arena,
        statics,
        dims,
    };
    let slots = sizing
        .buckets
        .tokens()
        .iter()
        .zip(sizing.plans)
        .enumerate()
        .map(|(index, (&tokens, attention))| {
            let bucket = Bucket {
                index: BucketIdx(index),
                tokens,
            };
            let views = inputs.bucket(bucket.index)?;
            Ok(BucketSlots::resolve(
                &sources,
                bucket,
                *attention,
                views.inputs,
            )?)
        })
        .collect::<Result<Vec<_>, DecodeStepError>>()?;
    Ok((arena_memory, arena.total_size(), slots))
}

/// The dimensions the step reads off the checkpoint's configuration.
fn llama_dims(config: &Config) -> Result<LlamaDims, DecodeStepError> {
    let scaling = config.rope_scaling.as_ref().and_then(|scaling| {
        matches!(scaling.rope_type, Llama3RopeType::Llama3).then(|| Llama3RopeScaling {
            factor: scaling.factor,
            low_freq_factor: scaling.low_freq_factor,
            high_freq_factor: scaling.high_freq_factor,
            original_max_position_embeddings: scaling.original_max_position_embeddings,
        })
    });
    // The kernel takes a single-precision epsilon; the checkpoint's double is far inside it.
    #[allow(clippy::cast_possible_truncation)]
    let rms_eps = config.rms_norm_eps as f32;
    let dims = LlamaDims {
        layers: config.num_hidden_layers,
        hidden: config.hidden_size,
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.hidden_size / config.num_attention_heads,
        ffn: config.intermediate_size,
        vocab: config.vocab_size,
        rms_eps,
        rope: RopeParams {
            theta: config.rope_theta,
            scaling,
            max_position: config.max_position_embeddings,
        },
    };
    dims.check()?;
    Ok(dims)
}

/// The device's multiprocessor count, which the attention split heuristic sizes by.
fn multiprocessors(stream: &Arc<CudaStream>) -> Result<usize, DecodeStepError> {
    let count = stream
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(RuntimeError::from)?;
    usize::try_from(count).map_err(|_| DecodeStepError::MultiprocessorCount { count })
}

/// `bytes` zeroed bytes on `stream`'s device.
fn zeroed(stream: &Arc<CudaStream>, bytes: usize) -> Result<CudaSlice<u8>, DecodeStepError> {
    Ok(stream
        .alloc_zeros::<u8>(bytes)
        .map_err(RuntimeError::from)?)
}

/// The device address of a buffer. Event tracking is disabled at context creation, so the read
/// guard is a no-op and the address is stable for the buffer's lifetime.
fn address<T>(slice: &CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
    let (address, _reads) = slice.device_ptr(stream);
    address
}

/// The device address of the bf16 tensor candle holds as `tensor`: where its storage is, plus
/// where the tensor starts in it. What [`snapshot`] mints a view at, and what the debug check
/// reads again without minting one.
///
/// Candle opens its own context with event tracking on, so unlike the runtime's own buffers its
/// slices carry read and write events: this read waits `stream` on the slice's write event and
/// records its read event when the guard drops.
fn candle_address(
    what: &'static str,
    tensor: &CandleTensor,
    stream: &Arc<CudaStream>,
) -> Result<u64, DecodeStepError> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        return Err(DecodeStepError::NotOnDevice { what });
    };
    let CudaStorageSlice::BF16(slice) = &storage.slice else {
        return Err(DecodeStepError::WeightDtype {
            what,
            dtype: tensor.dtype(),
        });
    };
    let start = layout.start_offset() * Dtype::Bf16.size_in_bytes();
    Ok(address(slice, stream) + start as u64)
}

/// A view of `dims` over the bf16 device tensor candle holds as `tensor`.
fn snapshot(
    allocation: &Allocation,
    what: &'static str,
    tensor: &CandleTensor,
    dims: &[usize],
    stream: &Arc<CudaStream>,
) -> Result<Tensor, DecodeStepError> {
    let address = candle_address(what, tensor, stream)?;
    if !tensor.layout().is_contiguous() {
        return Err(DecodeStepError::NotContiguous { what });
    }
    let expected: usize = dims.iter().product();
    let held = tensor.elem_count();
    if held != expected {
        return Err(DecodeStepError::ElementCount {
            what,
            held,
            expected,
        });
    }
    let view = Layout::contiguous(dims, Dtype::Bf16)?;
    Ok(Tensor::new(allocation, address, view)?)
}

/// A layer's nine weights, in the order the step bakes them.
const LAYER_WEIGHTS: [LayerWeight; 9] = [
    LayerWeight::InputNorm,
    LayerWeight::Q,
    LayerWeight::K,
    LayerWeight::V,
    LayerWeight::O,
    LayerWeight::PostAttentionNorm,
    LayerWeight::Gate,
    LayerWeight::Up,
    LayerWeight::Down,
];

/// What a refusal calls `weight`.
fn weight_what(weight: LayerWeight) -> &'static str {
    match weight {
        LayerWeight::InputNorm => "an input norm gain",
        LayerWeight::Q => "a query projection",
        LayerWeight::K => "a key projection",
        LayerWeight::V => "a value projection",
        LayerWeight::O => "an output projection",
        LayerWeight::PostAttentionNorm => "a post-attention norm gain",
        LayerWeight::Gate => "a gate projection",
        LayerWeight::Up => "an up projection",
        LayerWeight::Down => "a down projection",
    }
}

/// `weight` of `layer`, as candle holds it.
fn layer_tensor<'a>(layer: &LayerTensors<'a>, weight: LayerWeight) -> &'a CandleTensor {
    match weight {
        LayerWeight::InputNorm => layer.input_norm,
        LayerWeight::Q => layer.q_proj,
        LayerWeight::K => layer.k_proj,
        LayerWeight::V => layer.v_proj,
        LayerWeight::O => layer.o_proj,
        LayerWeight::PostAttentionNorm => layer.post_attention_norm,
        LayerWeight::Gate => layer.gate_proj,
        LayerWeight::Up => layer.up_proj,
        LayerWeight::Down => layer.down_proj,
    }
}

/// Every address candle holds a weight or a cache at, by name, in the order the step bakes
/// them: the embedding table, each layer's nine weights, the final norm gain, the head
/// projection, then each layer's cache. Read from candle's tensors without minting a view.
fn candle_addresses(
    weights: &Weights,
    kv_cache: &KvCache,
    stream: &Arc<CudaStream>,
) -> Result<Vec<BakedAddress>, DecodeStepError> {
    let read = |name: BakedName, what: &'static str, tensor: &CandleTensor| {
        candle_address(what, tensor, stream).map(|address| BakedAddress { name, address })
    };
    let llama = weights.llama();
    let mut addresses = vec![read(
        BakedName::EmbeddingTable,
        "the embedding table",
        llama.embeddings(),
    )?];
    for (layer, tensors) in llama.layer_weights().iter().enumerate() {
        for weight in LAYER_WEIGHTS {
            let name = BakedName::LayerWeight { layer, weight };
            addresses.push(read(
                name,
                weight_what(weight),
                layer_tensor(tensors, weight),
            )?);
        }
    }
    addresses.push(read(
        BakedName::FinalNormGain,
        "the final norm gain",
        llama.final_norm(),
    )?);
    addresses.push(read(
        BakedName::HeadProjection,
        "the head projection",
        llama.lm_head(),
    )?);
    for (layer, cache) in kv_cache.layers().iter().enumerate() {
        addresses.push(read(BakedName::Cache { layer }, "a layer's cache", cache)?);
    }
    Ok(addresses)
}

/// Every weight of `llama`, viewed at the address candle loaded it to.
fn snapshot_weights(
    allocation: &Allocation,
    llama: &Llama,
    stream: &Arc<CudaStream>,
) -> Result<LlamaWeights, DecodeStepError> {
    let view = |what: &'static str, tensor: &CandleTensor| {
        snapshot(allocation, what, tensor, tensor.dims(), stream)
    };
    let layers = llama
        .layer_weights()
        .iter()
        .map(|layer| {
            let weight_view =
                |weight: LayerWeight| view(weight_what(weight), layer_tensor(layer, weight));
            Ok(LayerWeights {
                input_norm: weight_view(LayerWeight::InputNorm)?,
                q: weight_view(LayerWeight::Q)?,
                k: weight_view(LayerWeight::K)?,
                v: weight_view(LayerWeight::V)?,
                o: weight_view(LayerWeight::O)?,
                post_attention_norm: weight_view(LayerWeight::PostAttentionNorm)?,
                gate: weight_view(LayerWeight::Gate)?,
                up: weight_view(LayerWeight::Up)?,
                down: weight_view(LayerWeight::Down)?,
            })
        })
        .collect::<Result<Vec<_>, DecodeStepError>>()?;
    Ok(LlamaWeights {
        embedding: view("the embedding table", llama.embeddings())?,
        layers,
        final_norm: view("the final norm gain", llama.final_norm())?,
        lm_head: view("the head projection", llama.lm_head())?,
    })
}

/// Every layer's cache, viewed as `[2, blocks, block_size, kv_width]` at the address candle
/// allocated it: the key-value heads of a slot flattened, since the view holds four dimensions
/// and the kernel takes the head stride on its own.
fn snapshot_cache(
    allocation: &Allocation,
    kv_cache: &KvCache,
    dims: &LlamaDims,
    stream: &Arc<CudaStream>,
) -> Result<LlamaCache, DecodeStepError> {
    let caches = kv_cache
        .layers()
        .iter()
        .map(|cache| {
            let shape = cache.dims();
            if shape.len() != 5 {
                return Err(DecodeStepError::CacheRank { rank: shape.len() });
            }
            let view = [2, shape[1], shape[2], dims.kv_width()];
            snapshot(allocation, "a layer's cache", cache, &view, stream)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LlamaCache::new(&caches, dims)?)
}

/// Allocates the step's outputs, its attention workspace at the largest any bucket needs, and
/// the rotary tables, and views each.
fn allocate_statics(
    allocation: &Allocation,
    stream: &Arc<CudaStream>,
    sizing: &Sizing<'_>,
) -> Result<(Statics, StepStatics), DecodeStepError> {
    let (dims, plans, shape) = (sizing.dims, sizing.plans, sizing.shape);
    let largest = |bytes: fn(&AttentionPlan) -> usize| plans.iter().map(bytes).max().unwrap_or(0);
    let f32_static = |bytes: usize| -> Result<(CudaSlice<u8>, Tensor), DecodeStepError> {
        let buffer = zeroed(stream, bytes)?;
        let layout = Layout::contiguous(&[bytes / Dtype::F32.size_in_bytes()], Dtype::F32)?;
        let tensor = Tensor::new(allocation, address(&buffer, stream), layout)?;
        Ok((buffer, tensor))
    };
    let logits_bytes = Dtype::F32.width_bytes(shape.max_tokens * dims.vocab);
    let (logits_buffer, logits_flat) = f32_static(logits_bytes)?;
    let logits = logits_flat.reshape(&[shape.max_tokens, dims.vocab])?;
    let (softmax_lse_buffer, softmax_lse) = f32_static(largest(AttentionPlan::softmax_lse_bytes))?;
    let (lse_accum_buffer, lse_accum) = f32_static(largest(AttentionPlan::lse_accum_bytes))?;
    let (o_accum_buffer, o_accum) = f32_static(largest(AttentionPlan::o_accum_bytes))?;

    let tables = RotaryTables::new(dims);
    let upload = |values: &[f32]| -> Result<(CudaSlice<f32>, Tensor), DecodeStepError> {
        let buffer = stream.clone_htod(values).map_err(RuntimeError::from)?;
        let layout = Layout::contiguous(&[tables.max_position(), tables.pairs()], Dtype::F32)?;
        let tensor = Tensor::new(allocation, address(&buffer, stream), layout)?;
        Ok((buffer, tensor))
    };
    let (cos_buffer, cos) = upload(tables.cos())?;
    let (sin_buffer, sin) = upload(tables.sin())?;

    Ok((
        Statics {
            logits: logits_buffer,
            softmax_lse: softmax_lse_buffer,
            lse_accum: lse_accum_buffer,
            o_accum: o_accum_buffer,
            cos: cos_buffer,
            sin: sin_buffer,
        },
        StepStatics {
            logits,
            softmax_lse,
            lse_accum,
            o_accum,
            rotary: RotaryTensors { cos, sin },
        },
    ))
}
