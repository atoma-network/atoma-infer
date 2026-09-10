//! The Llama decode step, enqueued from its op table through one launcher seam.
//!
//! The step is a walk: the embedding gather, then for every layer the ops of [`LLAMA_LAYER`] in
//! order, then the final norm and the head projection. Each op is resolved against the bucket's
//! slot tables into the tensors it reads and writes and handed to an [`OpLauncher`]. The CUDA
//! launcher turns each into a kernel call on the capture stream; a recording launcher, in the
//! tests, turns each into the addresses it would have touched, so the walk is held to the table
//! and the arena's lookup without a device.
//!
//! The poison fills the bucket's tables resolved are launches of the walk too, each ahead of the
//! op it is scheduled before in the arena's op timeline: the timeline counts layer `l`'s op `i`
//! at `l * ops_per_layer + i`, so the embedding gather, which writes the first layer's residual,
//! sits at -1, the final norm at `layers * ops_per_layer` and the head projection one after it,
//! and a fill scheduled past that follows the last launch. The greedy and no-reuse layouts
//! schedule none, so a serving step's walk launches none.
//!
//! [`LlamaStep`] is the [`Descriptor`] the session enqueues: one bucket's walk over the CUDA
//! launcher. Nothing in the walk allocates or looks anything up: every view was resolved when the
//! tables were built. Each launch still holds its views to what its kernel assumes, through the
//! operand checks, because that is what the launch then trusts; those checks are host arithmetic
//! over copied layouts and run at capture time or on an eager step, never inside a replay.

use core::ffi::c_void;
use core::fmt;

use atoma_kernels::decode_ops;
use atoma_kernels::error::KernelError;
use atoma_kernels::paged_decode;
use atoma_runtime::arena::{BucketIdx, LayerIdx, POISON_BYTE};
use atoma_runtime::error::RuntimeError;
use atoma_runtime::session::Descriptor;
use atoma_runtime::tensor::Tensor;
use cudarc::driver::{result, sys};
use thiserror::Error;

use crate::attention::{
    decode_call, kv_write_call, AttentionError, AttentionPlan, AttentionTensors, CacheHalves,
};
use crate::dims::LlamaDims;
use crate::gemm::{GemmError, StepBlas};
use crate::kernels::{DecodeKernels, RotaryTensors};
use crate::layer::{LayerOp, Role, RoleRef, LLAMA_LAYER};
use crate::llama::slots::{BucketSlots, LlamaCache, LlamaWeights, PoisonFill, SlotError};
use crate::operand::OperandError;

/// Why the step could not be built or enqueued.
#[derive(Debug, Error)]
pub enum StepError {
    #[error(transparent)]
    Slot(#[from] SlotError),
    #[error(transparent)]
    Operand(#[from] OperandError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
    #[error(transparent)]
    Gemm(#[from] GemmError),
    #[error(transparent)]
    Launch(#[from] KernelError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(
        "bucket {index} holds bucket {held}; the buckets must be resolved in bucket-ladder order"
    )]
    BucketOrder { index: usize, held: usize },
    #[error("no bucket {index} is resolved; {count} are")]
    NoBucket { index: usize, count: usize },
}

/// One op with its operands resolved: the tensors it reads and writes, by reference into the
/// bucket's tables.
#[derive(Debug, Clone, Copy)]
pub enum ResolvedOp<'a> {
    EmbeddingGather {
        table: &'a Tensor,
        token_ids: &'a Tensor,
        out: &'a Tensor,
    },
    RmsNorm {
        input: &'a Tensor,
        gain: &'a Tensor,
        output: &'a Tensor,
    },
    /// `output = input · weightᵀ`; a column-writing projection's output is the column view.
    Projection {
        input: &'a Tensor,
        weight: &'a Tensor,
        output: &'a Tensor,
    },
    Rope {
        qkv: &'a Tensor,
        positions: &'a Tensor,
        tables: &'a RotaryTensors,
    },
    KvWrite {
        qkv: &'a Tensor,
        cache: &'a CacheHalves,
        slot_mapping: &'a Tensor,
    },
    Attention {
        plan: &'a AttentionPlan,
        tensors: AttentionTensors<'a>,
    },
    SiluMul {
        gate: &'a Tensor,
        up: &'a Tensor,
        output: &'a Tensor,
    },
    ResidualAdd {
        residual: &'a Tensor,
        delta: &'a Tensor,
        output: &'a Tensor,
    },
    /// Every byte of `slot` written with [`POISON_BYTE`].
    PoisonFill { slot: &'a Tensor },
}

impl ResolvedOp<'_> {
    /// The addresses this op reads, weights and inputs included.
    #[must_use]
    pub fn reads(&self) -> Vec<u64> {
        let tensors: Vec<&Tensor> = match self {
            ResolvedOp::EmbeddingGather {
                table, token_ids, ..
            } => vec![table, token_ids],
            ResolvedOp::RmsNorm { input, gain, .. } => vec![input, gain],
            ResolvedOp::Projection { input, weight, .. } => vec![input, weight],
            ResolvedOp::Rope {
                qkv,
                positions,
                tables,
            } => vec![qkv, positions, &tables.cos, &tables.sin],
            ResolvedOp::KvWrite {
                qkv, slot_mapping, ..
            } => vec![qkv, slot_mapping],
            ResolvedOp::Attention { tensors, .. } => vec![
                tensors.qkv,
                &tensors.cache.k,
                &tensors.cache.v,
                tensors.seqlens_k,
                tensors.block_table,
            ],
            ResolvedOp::SiluMul { gate, up, .. } => vec![gate, up],
            ResolvedOp::ResidualAdd {
                residual, delta, ..
            } => vec![residual, delta],
            ResolvedOp::PoisonFill { .. } => Vec::new(),
        };
        tensors.into_iter().map(Tensor::address).collect()
    }

    /// The addresses this op writes, the cache and the attention workspace included.
    #[must_use]
    pub fn writes(&self) -> Vec<u64> {
        let tensors: Vec<&Tensor> = match self {
            ResolvedOp::EmbeddingGather { out, .. } => vec![out],
            ResolvedOp::RmsNorm { output, .. }
            | ResolvedOp::Projection { output, .. }
            | ResolvedOp::SiluMul { output, .. }
            | ResolvedOp::ResidualAdd { output, .. } => vec![output],
            ResolvedOp::Rope { qkv, .. } => vec![qkv],
            ResolvedOp::PoisonFill { slot } => vec![slot],
            ResolvedOp::KvWrite { cache, .. } => vec![&cache.k, &cache.v],
            ResolvedOp::Attention { tensors, .. } => vec![
                tensors.out,
                tensors.softmax_lse,
                tensors.lse_accum,
                tensors.o_accum,
            ],
        };
        tensors.into_iter().map(Tensor::address).collect()
    }
}

/// One launch of the step, as logs and recording launchers identify it: an op of one layer's
/// table, or one of the three the step runs outside the layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOp {
    /// The embedding gather into the first row's residual stream.
    EmbeddingGather,
    /// `op` of `layer`'s op table.
    Layer { layer: LayerIdx, op: LayerOp },
    /// The final norm over the last row.
    FinalNorm,
    /// The head projection into the logits.
    LmHead,
    /// A poison fill of one slot, scheduled before the op at `before_op` of the op timeline.
    PoisonFill { before_op: isize },
}

impl StepOp {
    /// What the launch computes, as logs name it; a layer's op is named as its table names it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            StepOp::EmbeddingGather => "embedding_gather",
            StepOp::Layer { op, .. } => op.name(),
            StepOp::FinalNorm => "final_norm",
            StepOp::LmHead => "lm_head",
            StepOp::PoisonFill { .. } => "poison_fill",
        }
    }
}

impl fmt::Display for StepOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepOp::Layer { layer, op } => write!(f, "layer {}'s {}", layer.0, op.name()),
            StepOp::PoisonFill { before_op } => {
                write!(f, "the poison fill before op {before_op}")
            }
            StepOp::EmbeddingGather | StepOp::FinalNorm | StepOp::LmHead => {
                f.write_str(self.name())
            }
        }
    }
}

/// The index of `layer`'s op `op` in the arena's op timeline, which a poison fill's `before_op`
/// indexes: the layers' op orders laid end to end.
fn timeline(layer: usize, op: usize) -> isize {
    isize::try_from(layer * LLAMA_LAYER.ops_per_layer() + op).expect("an op timeline fits isize")
}

/// The poison fills of a step still to be launched, in schedule order: each goes to the launcher
/// ahead of the op it is scheduled before, and the ones scheduled past the last op follow it.
struct PendingFills<'a>(&'a [PoisonFill]);

impl PendingFills<'_> {
    /// Launches every pending fill scheduled at or before the op at `op` of the timeline.
    fn launch_before<L: OpLauncher>(
        &mut self,
        op: isize,
        launcher: &mut L,
    ) -> Result<(), L::Error> {
        while let Some((fill, rest)) = self.0.split_first() {
            if fill.before_op > op {
                break;
            }
            launcher.launch(
                StepOp::PoisonFill {
                    before_op: fill.before_op,
                },
                &ResolvedOp::PoisonFill { slot: &fill.slot },
            )?;
            self.0 = rest;
        }
        Ok(())
    }

    /// Launches every fill still pending: the ones scheduled past the last op, which restore
    /// the pattern for the next replay.
    fn launch_rest<L: OpLauncher>(mut self, launcher: &mut L) -> Result<(), L::Error> {
        self.launch_before(isize::MAX, launcher)
    }
}

/// Where a resolved op goes: onto a stream, or into a recording launcher's list.
pub trait OpLauncher {
    type Error;

    /// Launches `resolved`, which is `op` of the step; the op identifies the launch for logs and
    /// recording launchers.
    ///
    /// # Errors
    ///
    /// Returns the launcher's error when the op cannot be launched; the walk stops there.
    fn launch(&mut self, op: StepOp, resolved: &ResolvedOp<'_>) -> Result<(), Self::Error>;
}

/// The Llama decode step over runtime-owned tensors: the tables every bucket's walk reads.
pub struct LlamaDecode {
    dims: LlamaDims,
    kernels: DecodeKernels,
    weights: LlamaWeights,
    cache: LlamaCache,
    rotary: RotaryTensors,
    buckets: Vec<BucketSlots>,
}

impl LlamaDecode {
    /// The step over `weights` and `cache`, with `buckets` resolved in bucket-ladder order.
    ///
    /// # Errors
    ///
    /// Returns [`StepError`] when the weights are not the model's shape or the buckets are not
    /// in bucket-ladder order.
    pub fn new(
        dims: LlamaDims,
        weights: LlamaWeights,
        cache: LlamaCache,
        rotary: RotaryTensors,
        buckets: Vec<BucketSlots>,
    ) -> Result<Self, StepError> {
        weights.check(&dims)?;
        for (index, slots) in buckets.iter().enumerate() {
            if slots.bucket.index != BucketIdx(index) {
                return Err(StepError::BucketOrder {
                    index,
                    held: slots.bucket.index.0,
                });
            }
        }
        Ok(Self {
            dims,
            kernels: DecodeKernels::new(dims),
            weights,
            cache,
            rotary,
            buckets,
        })
    }

    #[must_use]
    pub fn dims(&self) -> &LlamaDims {
        &self.dims
    }

    /// The buckets resolved, in bucket-ladder order.
    #[must_use]
    pub fn buckets(&self) -> &[BucketSlots] {
        &self.buckets
    }

    /// The tables of `bucket`.
    ///
    /// # Errors
    ///
    /// Returns [`StepError::NoBucket`] when no such bucket was resolved.
    pub fn bucket(&self, bucket: BucketIdx) -> Result<&BucketSlots, StepError> {
        self.buckets.get(bucket.0).ok_or(StepError::NoBucket {
            index: bucket.0,
            count: self.buckets.len(),
        })
    }

    /// The descriptor that enqueues `bucket`'s step through `blas`.
    ///
    /// # Errors
    ///
    /// Returns [`StepError::NoBucket`] when no such bucket was resolved.
    pub fn step<'a>(
        &'a self,
        bucket: BucketIdx,
        blas: &'a StepBlas,
    ) -> Result<LlamaStep<'a>, StepError> {
        let slots = self.bucket(bucket)?;
        Ok(LlamaStep {
            decode: self,
            slots,
            blas,
        })
    }

    /// Walks `slots`' step through `launcher`: the gather, every layer's op table, the final
    /// norm and the head projection, each behind the poison fills scheduled before it in the
    /// op timeline, and the fills scheduled past the head projection last, stopping at the
    /// first launch that fails.
    ///
    /// # Errors
    ///
    /// Returns the launcher's error for the first op it refuses.
    pub fn walk<L: OpLauncher>(
        &self,
        slots: &BucketSlots,
        launcher: &mut L,
    ) -> Result<(), L::Error> {
        let activations = &slots.activations;
        let statics = &slots.statics;
        let mut fills = PendingFills(activations.poison_fills());
        fills.launch_before(timeline(0, 0) - 1, launcher)?;
        launcher.launch(
            StepOp::EmbeddingGather,
            &ResolvedOp::EmbeddingGather {
                table: &self.weights.embedding,
                token_ids: &statics.token_ids,
                out: activations.row(LayerIdx(0)).role(Role::Hidden),
            },
        )?;
        for layer in 0..self.dims.layers {
            for (index, op) in LLAMA_LAYER.ops().iter().enumerate() {
                fills.launch_before(timeline(layer, index), launcher)?;
                let layer = LayerIdx(layer);
                launcher.launch(
                    StepOp::Layer { layer, op: *op },
                    &self.resolve(slots, layer, *op),
                )?;
            }
        }
        let last = activations.final_row();
        fills.launch_before(timeline(self.dims.layers, 0), launcher)?;
        launcher.launch(
            StepOp::FinalNorm,
            &ResolvedOp::RmsNorm {
                input: last.role(Role::Hidden),
                gain: &self.weights.final_norm,
                output: last.role(Role::Normed),
            },
        )?;
        fills.launch_before(timeline(self.dims.layers, 1), launcher)?;
        launcher.launch(
            StepOp::LmHead,
            &ResolvedOp::Projection {
                input: last.role(Role::Normed),
                weight: &self.weights.lm_head,
                output: &statics.logits,
            },
        )?;
        fills.launch_rest(launcher)
    }

    /// `op` of `layer`, with every role it names resolved against `slots`.
    fn resolve<'a>(
        &'a self,
        slots: &'a BucketSlots,
        layer: LayerIdx,
        op: LayerOp,
    ) -> ResolvedOp<'a> {
        let activations = &slots.activations;
        let statics = &slots.statics;
        let weights = &self.weights.layers[layer.0];
        let at = |role: RoleRef| activations.role(layer, role);
        match op {
            LayerOp::RmsNorm {
                input,
                gain,
                output,
            } => ResolvedOp::RmsNorm {
                input: at(input),
                gain: weights.get(gain),
                output: at(output),
            },
            LayerOp::Projection {
                input,
                weight,
                output,
                columns,
            } => ResolvedOp::Projection {
                input: at(input),
                weight: weights.get(weight),
                output: match columns {
                    Some(columns) => activations.frame(layer, output.layer).columns(columns),
                    None => at(output),
                },
            },
            LayerOp::Rope { qkv } => ResolvedOp::Rope {
                qkv: at(qkv),
                positions: &statics.positions,
                tables: &self.rotary,
            },
            LayerOp::KvWrite { qkv } => ResolvedOp::KvWrite {
                qkv: at(qkv),
                cache: self.cache.layer(layer),
                slot_mapping: &statics.slot_mapping,
            },
            LayerOp::Attention { qkv, output } => ResolvedOp::Attention {
                plan: &slots.plan,
                tensors: AttentionTensors {
                    qkv: at(qkv),
                    cache: self.cache.layer(layer),
                    out: at(output),
                    seqlens_k: &statics.seqlens_k,
                    block_table: &statics.block_table,
                    softmax_lse: &statics.softmax_lse,
                    lse_accum: &statics.lse_accum,
                    o_accum: &statics.o_accum,
                },
            },
            LayerOp::SiluMul { gate, up, output } => ResolvedOp::SiluMul {
                gate: at(gate),
                up: at(up),
                output: at(output),
            },
            LayerOp::ResidualAdd {
                residual,
                delta,
                output,
            } => ResolvedOp::ResidualAdd {
                residual: at(residual),
                delta: at(delta),
                output: at(output),
            },
        }
    }
}

/// One bucket's step as the session enqueues it: the walk over the CUDA launcher.
pub struct LlamaStep<'a> {
    decode: &'a LlamaDecode,
    slots: &'a BucketSlots,
    blas: &'a StepBlas,
}

impl Descriptor for LlamaStep<'_> {
    type Error = StepError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), StepError> {
        let mut launcher = CudaLauncher {
            kernels: &self.decode.kernels,
            dims: &self.decode.dims,
            blas: self.blas,
            stream: stream.cast::<c_void>(),
        };
        self.decode.walk(self.slots, &mut launcher)
    }
}

/// The launcher that puts each op on the capture stream: the decode kernels, the attention
/// kernels and the cuBLAS handle, each call assembled from the op's tensors.
pub struct CudaLauncher<'a> {
    kernels: &'a DecodeKernels,
    dims: &'a LlamaDims,
    blas: &'a StepBlas,
    stream: *mut c_void,
}

impl OpLauncher for CudaLauncher<'_> {
    type Error = StepError;

    fn launch(&mut self, _op: StepOp, resolved: &ResolvedOp<'_>) -> Result<(), StepError> {
        let stream = self.stream;
        match *resolved {
            ResolvedOp::EmbeddingGather {
                table,
                token_ids,
                out,
            } => {
                let call = self
                    .kernels
                    .embedding_gather(table, token_ids, out, stream)?;
                // SAFETY: the call was assembled from checked views over live device memory,
                // on the session's own stream.
                unsafe { decode_ops::embedding_gather(&call) }?;
            }
            ResolvedOp::RmsNorm {
                input,
                gain,
                output,
            } => {
                let call = self.kernels.rmsnorm(input, gain, output, stream)?;
                // SAFETY: as for the gather.
                unsafe { decode_ops::rmsnorm(&call) }?;
            }
            ResolvedOp::Projection {
                input,
                weight,
                output,
            } => {
                // SAFETY: three views over live device memory; the handle is bound to the
                // session's stream.
                unsafe { self.blas.enqueue(input, weight, output) }?;
            }
            ResolvedOp::Rope {
                qkv,
                positions,
                tables,
            } => {
                let call = self.kernels.rope(qkv, positions, tables, stream)?;
                // SAFETY: as for the gather.
                unsafe { decode_ops::rope(&call) }?;
            }
            ResolvedOp::KvWrite {
                qkv,
                cache,
                slot_mapping,
            } => {
                let call = kv_write_call(qkv, cache, slot_mapping, self.dims, stream)?;
                // SAFETY: as for the gather; every slot in the mapping was laid out inside the
                // cache by the engine.
                unsafe { paged_decode::write_kv(&call) }?;
            }
            ResolvedOp::Attention { plan, tensors } => {
                let call = decode_call(plan, &tensors, self.dims, stream)?;
                // SAFETY: as for the gather; the accumulators were sized from this plan.
                unsafe { paged_decode::decode_attention(&call) }?;
            }
            ResolvedOp::SiluMul { gate, up, output } => {
                let call = self.kernels.silu_mul(gate, up, output, stream)?;
                // SAFETY: as for the gather.
                unsafe { decode_ops::silu_mul(&call) }?;
            }
            ResolvedOp::ResidualAdd {
                residual,
                delta,
                output,
            } => {
                let call = self.kernels.add(residual, delta, output, stream)?;
                // SAFETY: as for the gather.
                unsafe { decode_ops::add(&call) }?;
            }
            ResolvedOp::PoisonFill { slot } => {
                // SAFETY: the slot is a view over the arena, live device memory the step was
                // resolved over, and the fill is enqueued on the session's own stream.
                unsafe {
                    result::memset_d8_async(
                        slot.address(),
                        POISON_BYTE,
                        slot.extent_bytes(),
                        stream.cast(),
                    )
                }
                .map_err(RuntimeError::from)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use atoma_runtime::arena::{ArenaLayout, CaptureArena};
    use atoma_runtime::tensor::{Dtype, Layout};

    use super::*;
    use crate::dims::test_support::llama_8b;
    use crate::layer::{LayerOffset, QkvColumns, LLAMA_OPS};
    use crate::llama::slots::{Bucket, BucketInputs, LayerWeights, SlotSources, StepStatics};

    const LAYERS: usize = 2;
    const LADDER: [usize; 2] = [1, 8];
    const PAGE_BLOCK: usize = 16;
    const BLOCKS: usize = 64;
    const BLOCK_COLUMNS: usize = 32;
    const SMS: usize = 114;
    const ARENA_BASE: u64 = 0x1000_0000;
    const LOGITS: u64 = 0x7100_0000;

    fn view(address: u64, dims: &[usize], dtype: Dtype) -> Tensor {
        Tensor::for_test(address, Layout::contiguous(dims, dtype).unwrap()).unwrap()
    }

    fn layer_weights(base: u64, dims: &LlamaDims) -> LayerWeights {
        let (hidden, ffn) = (dims.hidden, dims.ffn);
        LayerWeights {
            input_norm: view(base, &[hidden], Dtype::Bf16),
            q: view(base + 0x1_0000, &[dims.q_width(), hidden], Dtype::Bf16),
            k: view(base + 0x2_0000, &[dims.kv_width(), hidden], Dtype::Bf16),
            v: view(base + 0x3_0000, &[dims.kv_width(), hidden], Dtype::Bf16),
            o: view(base + 0x4_0000, &[hidden, dims.q_width()], Dtype::Bf16),
            post_attention_norm: view(base + 0x5_0000, &[hidden], Dtype::Bf16),
            gate: view(base + 0x6_0000, &[ffn, hidden], Dtype::Bf16),
            up: view(base + 0x7_0000, &[ffn, hidden], Dtype::Bf16),
            down: view(base + 0x8_0000, &[hidden, ffn], Dtype::Bf16),
        }
    }

    fn weights(dims: &LlamaDims) -> LlamaWeights {
        LlamaWeights {
            embedding: view(0x2000_0000, &[dims.vocab, dims.hidden], Dtype::Bf16),
            layers: (0..dims.layers)
                .map(|layer| layer_weights(0x3000_0000 + layer as u64 * 0x10_0000, dims))
                .collect(),
            final_norm: view(0x4000_0000, &[dims.hidden], Dtype::Bf16),
            lm_head: view(0x5000_0000, &[dims.vocab, dims.hidden], Dtype::Bf16),
        }
    }

    fn cache(dims: &LlamaDims) -> LlamaCache {
        let caches: Vec<Tensor> = (0..dims.layers)
            .map(|layer| {
                view(
                    0x6000_0000 + layer as u64 * 0x100_0000,
                    &[2, BLOCKS, PAGE_BLOCK, dims.kv_width()],
                    Dtype::Bf16,
                )
            })
            .collect();
        LlamaCache::new(&caches, dims).unwrap()
    }

    fn plan(dims: &LlamaDims, tokens: usize) -> AttentionPlan {
        AttentionPlan::new(dims, tokens, PAGE_BLOCK, BLOCK_COLUMNS, SMS).unwrap()
    }

    /// A bucket's inputs at `tokens` rows, each at its own address.
    fn inputs(tokens: usize) -> BucketInputs {
        BucketInputs {
            token_ids: view(0x8000_0000, &[tokens], Dtype::U32),
            positions: view(0x8001_0000, &[tokens], Dtype::I32),
            seqlens_k: view(0x8002_0000, &[tokens], Dtype::I32),
            slot_mapping: view(0x8003_0000, &[tokens], Dtype::I64),
            block_table: view(0x8004_0000, &[tokens, BLOCK_COLUMNS], Dtype::I32),
        }
    }

    fn statics(dims: &LlamaDims) -> StepStatics {
        let max = *LADDER.iter().max().unwrap();
        let shape = plan(dims, max).shape();
        let splits = plan(dims, 1).num_splits;
        let rotary = [dims.rope.max_position, dims.head_dim / 2];
        StepStatics {
            logits: view(LOGITS, &[max, dims.vocab], Dtype::F32),
            softmax_lse: view(0x7200_0000, &[shape.softmax_lse_len()], Dtype::F32),
            lse_accum: view(0x7300_0000, &[shape.lse_accum_len(splits)], Dtype::F32),
            o_accum: view(0x7400_0000, &[shape.o_accum_len(splits)], Dtype::F32),
            rotary: RotaryTensors {
                cos: view(0x7500_0000, &rotary, Dtype::F32),
                sin: view(0x7600_0000, &rotary, Dtype::F32),
            },
        }
    }

    fn arena(dims: &LlamaDims, layout: ArenaLayout) -> CaptureArena {
        CaptureArena::new(
            dims.layers + 1,
            LLAMA_LAYER.role_table(dims),
            &LADDER,
            layout,
        )
        .unwrap()
    }

    /// The step over a two-layer 8B shape with buckets of one and eight tokens, its slots placed
    /// under the greedy layout.
    fn decode() -> (LlamaDecode, CaptureArena) {
        decode_under(ArenaLayout::Greedy)
    }

    /// The step over the same shape with its slots placed under `layout`.
    fn decode_under(layout: ArenaLayout) -> (LlamaDecode, CaptureArena) {
        let dims = llama_8b(LAYERS);
        let arena = arena(&dims, layout);
        let memory = view(ARENA_BASE, &[arena.total_size() / 2], Dtype::Bf16);
        let statics = statics(&dims);
        let sources = SlotSources {
            memory: &memory,
            arena: &arena,
            statics: &statics,
            dims: &dims,
        };
        let buckets = LADDER
            .iter()
            .enumerate()
            .map(|(index, &tokens)| {
                let bucket = Bucket {
                    index: BucketIdx(index),
                    tokens,
                };
                BucketSlots::resolve(&sources, bucket, plan(&dims, tokens), inputs(tokens)).unwrap()
            })
            .collect();
        let decode =
            LlamaDecode::new(dims, weights(&dims), cache(&dims), statics.rotary, buckets).unwrap();
        (decode, arena)
    }

    /// One launch as the recorder saw it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Launch {
        op: StepOp,
        reads: Vec<u64>,
        writes: Vec<u64>,
    }

    /// A launcher that records what each op would have touched, and can refuse the n-th.
    #[derive(Default)]
    struct Recorder {
        launches: Vec<Launch>,
        refuse_at: Option<usize>,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Refused(usize);

    impl OpLauncher for Recorder {
        type Error = Refused;

        fn launch(&mut self, op: StepOp, resolved: &ResolvedOp<'_>) -> Result<(), Refused> {
            if self.refuse_at == Some(self.launches.len()) {
                return Err(Refused(self.launches.len()));
            }
            self.launches.push(Launch {
                op,
                reads: resolved.reads(),
                writes: resolved.writes(),
            });
            Ok(())
        }
    }

    /// A launcher that cannot fail, for the walks that only count.
    struct Counter(usize);

    impl OpLauncher for Counter {
        type Error = Infallible;

        fn launch(&mut self, _: StepOp, _: &ResolvedOp<'_>) -> Result<(), Infallible> {
            self.0 += 1;
            Ok(())
        }
    }

    fn record(decode: &LlamaDecode, bucket: usize) -> Vec<Launch> {
        let mut recorder = Recorder::default();
        decode
            .walk(decode.bucket(BucketIdx(bucket)).unwrap(), &mut recorder)
            .unwrap();
        recorder.launches
    }

    /// The address the arena places `role` of `layer`'s frame at, under `bucket`.
    fn slot(arena: &CaptureArena, bucket: usize, layer: usize, role: RoleRef) -> u64 {
        let row = match role.layer {
            LayerOffset::Same => layer,
            LayerOffset::Next => layer + 1,
        };
        ARENA_BASE + arena.offset(BucketIdx(bucket), LayerIdx(row), role.role.tensor_role()) as u64
    }

    #[test]
    fn the_walk_is_the_gather_the_op_table_per_layer_the_final_norm_and_the_head() {
        let (decode, _) = decode();
        let ops: Vec<StepOp> = record(&decode, 1).iter().map(|launch| launch.op).collect();

        let mut expected = vec![StepOp::EmbeddingGather];
        for layer in 0..LAYERS {
            let layer = LayerIdx(layer);
            expected.extend(LLAMA_OPS.iter().map(|&op| StepOp::Layer { layer, op }));
        }
        expected.extend([StepOp::FinalNorm, StepOp::LmHead]);
        assert_eq!(ops, expected);
        assert_eq!(ops.len(), 1 + LAYERS * LLAMA_OPS.len() + 2);
    }

    #[test]
    fn a_launch_names_itself_with_its_layer_when_it_has_one() {
        let rope = StepOp::Layer {
            layer: LayerIdx(3),
            op: LLAMA_OPS[4],
        };
        assert_eq!(rope.name(), "rope");
        assert_eq!(rope.to_string(), "layer 3's rope");
        assert_eq!(StepOp::LmHead.to_string(), "lm_head");
        assert_eq!(StepOp::EmbeddingGather.name(), "embedding_gather");
        let fill = StepOp::PoisonFill { before_op: -1 };
        assert_eq!(fill.name(), "poison_fill");
        assert_eq!(fill.to_string(), "the poison fill before op -1");
    }

    fn is_fill(launch: &Launch) -> bool {
        matches!(launch.op, StepOp::PoisonFill { .. })
    }

    #[test]
    fn each_fill_precedes_the_op_it_is_scheduled_before_and_the_trailing_ones_follow_the_head() {
        let (poisoned, arena) = decode_under(ArenaLayout::Poison);
        let (greedy, _) = decode();
        for bucket in 0..LADDER.len() {
            let launches = record(&poisoned, bucket);
            let schedule = arena.poison_fills(BucketIdx(bucket));
            // The ops of the walk laid end to end index the timeline from the gather at -1, so
            // a fill scheduled before op `t` precedes the walk's op `t + 1`, in schedule order;
            // one scheduled past the head projection follows it.
            let mut expected = Vec::new();
            let mut fills = schedule.iter().peekable();
            for (index, launch) in record(&greedy, bucket).iter().enumerate() {
                let at = isize::try_from(index).unwrap() - 1;
                while let Some(fill) = fills.next_if(|fill| fill.before_op <= at) {
                    expected.push(StepOp::PoisonFill {
                        before_op: fill.before_op,
                    });
                }
                expected.push(launch.op);
            }
            expected.extend(fills.map(|fill| StepOp::PoisonFill {
                before_op: fill.before_op,
            }));
            let recorded: Vec<StepOp> = launches.iter().map(|launch| launch.op).collect();
            assert_eq!(recorded, expected, "bucket {bucket}");

            let fill_writes: Vec<Vec<u64>> = launches
                .iter()
                .filter(|launch| is_fill(launch))
                .map(|launch| {
                    assert!(launch.reads.is_empty(), "a fill reads nothing");
                    launch.writes.clone()
                })
                .collect();
            let scheduled: Vec<Vec<u64>> = schedule
                .iter()
                .map(|fill| vec![ARENA_BASE + fill.offset as u64])
                .collect();
            assert_eq!(
                fill_writes, scheduled,
                "bucket {bucket}: each fill writes its slot"
            );
            assert_eq!(fill_writes.len(), 2 * (LAYERS + 1) * Role::ALL.len());
            assert!(
                is_fill(launches.last().unwrap()),
                "the trailing fills close the walk"
            );
        }
    }

    #[test]
    fn a_slot_is_poisoned_before_its_roles_first_use_and_again_after_its_last_use() {
        let (poisoned, arena) = decode_under(ArenaLayout::Poison);
        let launches = record(&poisoned, 0);
        let position = |op: StepOp| launches.iter().position(|launch| launch.op == op).unwrap();
        let layer_op = |layer: usize, op: usize| StepOp::Layer {
            layer: LayerIdx(layer),
            op: LLAMA_OPS[op],
        };
        let fills_over = |address: u64| -> (usize, usize) {
            let fills: Vec<usize> = launches
                .iter()
                .enumerate()
                .filter(|(_, launch)| is_fill(launch) && launch.writes == [address])
                .map(|(index, _)| index)
                .collect();
            assert_eq!(fills.len(), 2, "two fills over the slot at {address:#x}");
            (fills[0], fills[1])
        };

        // Layer 1's `Normed` lives over ops [0, 12): filled after layer 0's last op and before
        // layer 1's first, then after its up projection, the last op that reads it, and before
        // its silu_mul.
        let normed = slot(&arena, 0, 1, RoleRef::same(Role::Normed));
        let (first, last) = fills_over(normed);
        assert!(position(layer_op(0, 14)) < first && first < position(layer_op(1, 0)));
        assert!(position(layer_op(1, 11)) < last && last < position(layer_op(1, 12)));

        // Layer 0's residual enters at -1: its first fill precedes the embedding gather.
        let hidden = slot(&arena, 0, 0, RoleRef::same(Role::Hidden));
        let (first, _) = fills_over(hidden);
        assert!(first < position(StepOp::EmbeddingGather));

        // The final row's `Normed` is written by the final norm and read by the head
        // projection: filled before the one and after the other, past every op of the step.
        let final_normed = slot(&arena, 0, LAYERS, RoleRef::same(Role::Normed));
        let (first, last) = fills_over(final_normed);
        assert!(position(layer_op(LAYERS - 1, 14)) < first);
        assert!(first < position(StepOp::FinalNorm));
        assert!(position(StepOp::LmHead) < last);
    }

    #[test]
    fn the_greedy_and_no_reuse_layouts_add_no_launch_to_the_walk() {
        for layout in [ArenaLayout::Greedy, ArenaLayout::NoReuse] {
            let (decode, _) = decode_under(layout);
            let launches = record(&decode, 1);
            assert!(!launches.iter().any(is_fill), "{layout}");
            assert_eq!(launches.len(), 1 + LAYERS * LLAMA_OPS.len() + 2);
        }
    }

    #[test]
    fn every_launch_reads_and_writes_what_the_table_and_the_arena_lookup_name() {
        let (decode, arena) = decode();
        let dims = *decode.dims();
        for bucket in 0..LADDER.len() {
            let launches = record(&decode, bucket);
            for (layer, ops) in launches[1..=LAYERS * LLAMA_OPS.len()]
                .chunks(LLAMA_OPS.len())
                .enumerate()
            {
                for (op, launch) in LLAMA_OPS.iter().zip(ops) {
                    for read in op.reads() {
                        let address = slot(&arena, bucket, layer, read);
                        assert!(
                            launch.reads.contains(&address),
                            "bucket {bucket} layer {layer} {}: reads {:x?}, not {read:?} at \
                             {address:#x}",
                            op.name(),
                            launch.reads
                        );
                    }
                    let column = match *op {
                        LayerOp::Projection {
                            columns: Some(QkvColumns::K),
                            ..
                        } => dims.q_width(),
                        LayerOp::Projection {
                            columns: Some(QkvColumns::V),
                            ..
                        } => dims.q_width() + dims.kv_width(),
                        _ => 0,
                    };
                    for write in op.writes() {
                        let address = slot(&arena, bucket, layer, write) + (column * 2) as u64;
                        assert!(
                            launch.writes.contains(&address),
                            "bucket {bucket} layer {layer} {}: writes {:x?}, not {write:?} at \
                             {address:#x}",
                            op.name(),
                            launch.writes
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_gather_writes_the_first_rows_residual_and_the_head_reads_the_last_rows_norm() {
        let (decode, arena) = decode();
        let launches = record(&decode, 0);
        let first = &launches[0];
        let hidden = RoleRef::same(Role::Hidden);
        assert_eq!(first.writes, [slot(&arena, 0, 0, hidden)]);
        assert_eq!(first.reads[0], 0x2000_0000, "the embedding table");

        let final_norm = &launches[launches.len() - 2];
        let normed = RoleRef::same(Role::Normed);
        assert_eq!(final_norm.reads[0], slot(&arena, 0, LAYERS, hidden));
        assert_eq!(final_norm.writes, [slot(&arena, 0, LAYERS, normed)]);

        let head = &launches[launches.len() - 1];
        assert_eq!(head.reads, [slot(&arena, 0, LAYERS, normed), 0x5000_0000]);
        assert_eq!(head.writes, [LOGITS]);
    }

    #[test]
    fn each_layers_kv_write_and_attention_touch_that_layers_cache() {
        let (decode, _) = decode();
        let launches = record(&decode, 1);
        for layer in 0..LAYERS {
            let base = 1 + layer * LLAMA_OPS.len();
            let k_cache = 0x6000_0000 + layer as u64 * 0x100_0000;
            let kv_write = &launches[base + 5];
            assert_eq!(kv_write.op.name(), "kv_write");
            assert_eq!(kv_write.writes[0], k_cache);
            let attention = &launches[base + 6];
            assert_eq!(attention.op.name(), "attention");
            assert!(attention.reads.contains(&k_cache));
        }
    }

    #[test]
    fn a_refused_launch_stops_the_walk_there() {
        let (decode, _) = decode();
        let mut recorder = Recorder {
            refuse_at: Some(7),
            ..Recorder::default()
        };
        let refused = decode
            .walk(decode.bucket(BucketIdx(0)).unwrap(), &mut recorder)
            .unwrap_err();
        assert_eq!(refused, Refused(7));
        assert_eq!(recorder.launches.len(), 7);
    }

    #[test]
    fn the_walk_is_the_same_length_for_every_bucket() {
        let (decode, _) = decode();
        for slots in decode.buckets() {
            let mut counter = Counter(0);
            decode.walk(slots, &mut counter).unwrap();
            assert_eq!(counter.0, 1 + LAYERS * LLAMA_OPS.len() + 2);
        }
    }

    #[test]
    fn a_bucket_that_was_not_resolved_is_refused_by_index() {
        let (decode, _) = decode();
        let refused = decode.bucket(BucketIdx(2)).unwrap_err();
        assert!(matches!(
            refused,
            StepError::NoBucket { index: 2, count: 2 }
        ));
        assert!(refused.to_string().contains("no bucket 2"));
    }

    #[test]
    fn buckets_out_of_bucket_ladder_order_are_refused_when_the_step_is_built() {
        let (decode, _) = decode();
        let dims = *decode.dims();
        let mut buckets = decode.buckets().to_vec();
        buckets.swap(0, 1);
        let refused = LlamaDecode::new(
            dims,
            weights(&dims),
            cache(&dims),
            statics(&dims).rotary,
            buckets,
        )
        .err()
        .expect("the buckets are out of order");
        assert!(matches!(
            refused,
            StepError::BucketOrder { index: 0, held: 1 }
        ));
    }
}
