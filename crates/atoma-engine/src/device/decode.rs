//! The decode step over runtime-owned tensors on one rank's device: built once at Allocation
//! from the addresses candle loaded the weights and cache at, and run for every keyed batch.
//!
//! Candle keeps owning the weights and the cache; this module snapshots their device addresses
//! into tensor views, allocates the arena, the step's fixed buffers and the cuBLAS workspace,
//! resolves every usable bucket's slot tables, and holds the step descriptor over them. A step is
//! then six descriptors on the capture stream: the wait on candle's stream, the input upload,
//! the sampler's upload, the gather that takes each decoding row's token from what the device
//! sampled for its slot, the model step, and the sample, which leaves the tokens on the device and
//! reads them back; then one host wait. Nothing is captured here. Going through the descriptor
//! seam is what lets a later capture record the gather, the model step and the sample unchanged;
//! the two uploads are the host's copies of what changed, as many as changed, and stay in front
//! of the graph.

use std::sync::Arc;

use atoma_core::dispatch::{DispatchConfig, GraphKey};
use atoma_core::types::TokenCount;
use atoma_models::attention::{block_table_columns, AttentionError, AttentionPlan};
use atoma_models::dims::{DimsError, Llama3RopeScaling, LlamaDims, RopeParams};
use atoma_models::gemm::{GemmError, StepBlas, WORKSPACE_BYTES};
use atoma_models::kernels::RotaryTensors;
use atoma_models::layer::LLAMA_LAYER;
use atoma_models::llama::slots::{
    Bucket, BucketInputs, BucketSlots, LayerWeights, LlamaCache, LlamaWeights, SlotError,
    SlotSources, StepStatics,
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
use models::llama::{Config, Llama, Llama3RopeType};
use thiserror::Error;
use tracing::info;

use crate::batch::BatchLayout;
use crate::config::Dtype as ConfiguredDtype;
use crate::decode::batch::{Checked, DecodeBatch, DecodeBatchError, DecodeBuckets};
use crate::decode::inputs::{DecodeInputs, InputTensors, InputsError, Upload, WaitEvent};
use crate::decode::staging::StagingShape;
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
}

/// The step's outputs and workspace on the device, owned here for as long as the views over
/// them are read.
struct Statics {
    _logits: CudaSlice<u8>,
    _softmax_lse: CudaSlice<u8>,
    _lse_accum: CudaSlice<u8>,
    _o_accum: CudaSlice<u8>,
    _cos: CudaSlice<f32>,
    _sin: CudaSlice<f32>,
}

/// The decode step over runtime-owned tensors, and everything it addresses.
pub struct DecodeStep {
    buckets: DecodeBuckets,
    inputs: DecodeInputs,
    decode: LlamaDecode,
    blas: StepBlas,
    /// Recorded on candle's stream after every candle forward; the step waits on it.
    candle_done: CudaEvent,
    _arena: CudaSlice<u8>,
    _statics: Statics,
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
        let inputs = DecodeInputs::new(allocation, stream, shape)?;
        let (statics, step_statics) = allocate_statics(allocation, stream, &sizing)?;
        let (arena_memory, arena_bytes, slots) = resolve_slots(
            allocation,
            stream,
            &sizing,
            &step_statics,
            &inputs.tensors(),
        )?;
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
            _arena: arena_memory,
            _statics: statics,
        })
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

    /// Runs `batch`'s step through `session` and samples it: the inputs staged, then the wait on
    /// candle's stream, the input upload, the sampler's upload, the gather, the model step and the
    /// sample enqueued in that order, then the host wait on the sampled tokens. A rank with no
    /// sampler runs the same step without the sampler's descriptors and waits for it instead,
    /// so the next step's staging is fenced either way, and returns no tokens.
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
        self.stage(layout, &batch)?;
        session.run(&mut WaitEvent::new(&self.candle_done))?;
        session.run(&mut self.upload(&batch))?;
        let Some(sampler) = sampler else {
            session.run(&mut self.descriptor(batch.bucket)?)?;
            session.synchronize()?;
            return Ok(&[]);
        };
        session.run(&mut sampler.upload()?)?;
        session.run(&mut sampler.gather(self.token_ids_address())?)?;
        session.run(&mut self.descriptor(batch.bucket)?)?;
        let logits = self.decode.bucket(batch.bucket)?.statics.logits.address();
        session.run(&mut sampler.sample(logits)?)?;
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
        self.stage(layout, &batch)?;
        session.run(&mut WaitEvent::new(&self.candle_done))?;
        session.run(&mut self.upload(&batch))?;
        session.run(&mut self.descriptor(batch.bucket)?)?;
        let vocab = self.decode.dims().vocab;
        let logits = self.decode.bucket(batch.bucket)?.statics.logits.address();
        session.run(&mut readback.copy(logits, batch.live)?)?;
        Ok(Logits::new(readback.wait()?, vocab))
    }

    /// The device address of the uploaded token ids, which the gather overwrites.
    fn token_ids_address(&self) -> u64 {
        self.inputs.tensors().token_ids.address()
    }

    /// Writes `batch`'s inputs from `layout` into the pinned staging.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeStepError::Inputs`] when the layout cannot be staged.
    pub fn stage(
        &mut self,
        layout: &BatchLayout,
        batch: &DecodeBatch,
    ) -> Result<(), DecodeStepError> {
        Ok(self.inputs.stage(layout, batch)?)
    }

    /// The descriptor that copies `batch`'s rows of the staged inputs to the device.
    #[must_use]
    pub fn upload(&self, batch: &DecodeBatch) -> Upload<'_> {
        self.inputs.upload(batch)
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
/// minted from `inputs`: the arena's memory (owned for as long as the tables are read), its
/// size, and the tables in bucket order.
fn resolve_slots(
    allocation: &Allocation,
    stream: &Arc<CudaStream>,
    sizing: &Sizing<'_>,
    statics: &StepStatics,
    inputs: &InputTensors,
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
            let inputs = bucket_inputs(inputs, tokens)?;
            Ok(BucketSlots::resolve(&sources, bucket, *attention, inputs)?)
        })
        .collect::<Result<Vec<_>, DecodeStepError>>()?;
    Ok((arena_memory, arena.total_size(), slots))
}

/// The views a bucket of `tokens` rows reads its inputs through: the leading rows of each
/// device buffer, where the upload puts them.
fn bucket_inputs(inputs: &InputTensors, tokens: usize) -> Result<BucketInputs, TensorError> {
    Ok(BucketInputs {
        token_ids: inputs.token_ids.narrow(0, 0, tokens)?,
        positions: inputs.positions.narrow(0, 0, tokens)?,
        seqlens_k: inputs.seqlens_k.narrow(0, 0, tokens)?,
        slot_mapping: inputs.slot_mapping.narrow(0, 0, tokens)?,
        block_table: inputs.block_table.narrow(0, 0, tokens)?,
    })
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

/// A view of `dims` over the bf16 device tensor candle holds as `tensor`.
fn snapshot(
    allocation: &Allocation,
    what: &'static str,
    tensor: &CandleTensor,
    dims: &[usize],
    stream: &Arc<CudaStream>,
) -> Result<Tensor, DecodeStepError> {
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        return Err(DecodeStepError::NotOnDevice { what });
    };
    if !layout.is_contiguous() {
        return Err(DecodeStepError::NotContiguous { what });
    }
    let CudaStorageSlice::BF16(slice) = &storage.slice else {
        return Err(DecodeStepError::WeightDtype {
            what,
            dtype: tensor.dtype(),
        });
    };
    let expected: usize = dims.iter().product();
    let held = layout.shape().elem_count();
    if held != expected {
        return Err(DecodeStepError::ElementCount {
            what,
            held,
            expected,
        });
    }
    let start = layout.start_offset() * Dtype::Bf16.size_in_bytes();
    let view = Layout::contiguous(dims, Dtype::Bf16)?;
    Ok(Tensor::new(
        allocation,
        address(slice, stream) + start as u64,
        view,
    )?)
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
        .into_iter()
        .map(|layer| {
            Ok(LayerWeights {
                input_norm: view("an input norm gain", layer.input_norm)?,
                q: view("a query projection", layer.q_proj)?,
                k: view("a key projection", layer.k_proj)?,
                v: view("a value projection", layer.v_proj)?,
                o: view("an output projection", layer.o_proj)?,
                post_attention_norm: view("a post-attention norm gain", layer.post_attention_norm)?,
                gate: view("a gate projection", layer.gate_proj)?,
                up: view("an up projection", layer.up_proj)?,
                down: view("a down projection", layer.down_proj)?,
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
            _logits: logits_buffer,
            _softmax_lse: softmax_lse_buffer,
            _lse_accum: lse_accum_buffer,
            _o_accum: o_accum_buffer,
            _cos: cos_buffer,
            _sin: sin_buffer,
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
