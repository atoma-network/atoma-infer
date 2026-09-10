//! A layer class: the linear op order of one layer of a model, and the activation roles each op
//! reads and writes.
//!
//! The arena places a role's slot from its declared lifetime, the half-open range of the layer's
//! op order in which the slot holds live data. That declaration is only right while it agrees
//! with what the ops do, so the ops are declared too: a host-visible table naming, per op, the
//! roles it reads and the roles it writes. The forward enqueues from the table, and a test holds
//! each role's declared lifetime to the hull of the ops that touch it.
//!
//! A model whose layers differ structurally declares one class per shape. Llama's layers are all
//! alike, so it declares one, [`LLAMA_LAYER`].

use std::fmt;

use atoma_runtime::arena::{Lifetime, RoleDeclaration, RoleTable, TensorRole};
use atoma_runtime::tensor::Dtype;

use crate::dims::LlamaDims;

/// One activation tensor a layer produces, addressed through the arena.
///
/// The discriminant is the arena role index. `Hidden` is the residual stream entering a layer,
/// written by the previous layer's last op; the arena is built with one extra layer row so the
/// last layer's residual add has a slot to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    Hidden = 0,
    /// The `RMSNorm` output feeding the projections: the attention's, then the feed-forward's.
    Normed = 1,
    /// The fused q, k and v projections, `[num_heads + 2 * num_kv_heads, head_dim]` per token.
    Qkv = 2,
    /// Attention output, `[num_heads, head_dim]` per token.
    AttnOut = 3,
    OProj = 4,
    /// The residual stream after the attention residual add.
    Mid = 5,
    Gate = 6,
    Up = 7,
    /// `silu(gate) * up`.
    FfnAct = 8,
    FfnDown = 9,
}

impl Role {
    pub const ALL: [Role; 10] = [
        Role::Hidden,
        Role::Normed,
        Role::Qkv,
        Role::AttnOut,
        Role::OProj,
        Role::Mid,
        Role::Gate,
        Role::Up,
        Role::FfnAct,
        Role::FfnDown,
    ];

    #[must_use]
    pub fn tensor_role(self) -> TensorRole {
        TensorRole(self as usize)
    }

    /// Elements per token the role holds.
    #[must_use]
    pub fn width_elements(self, dims: &LlamaDims) -> usize {
        match self {
            Role::Hidden
            | Role::Normed
            | Role::AttnOut
            | Role::OProj
            | Role::Mid
            | Role::FfnDown => dims.hidden,
            Role::Qkv => dims.qkv_width(),
            Role::Gate | Role::Up | Role::FfnAct => dims.ffn,
        }
    }

    /// The declared lifetime in [`LLAMA_LAYER`]'s op order. `Hidden` enters at -1 because the
    /// previous layer's residual add, its last op, writes it; `Normed` is written twice, before
    /// each projection group, and its declaration spans both.
    #[must_use]
    pub fn lifetime(self) -> Lifetime {
        let (first_use, last_use) = match self {
            Role::Hidden => (-1, 9),
            Role::Normed => (0, 12),
            Role::Qkv => (1, 7),
            Role::AttnOut => (6, 8),
            Role::OProj => (7, 9),
            Role::Mid => (8, 15),
            Role::Gate => (10, 13),
            Role::Up => (11, 13),
            Role::FfnAct => (12, 14),
            Role::FfnDown => (13, 15),
        };
        Lifetime {
            first_use,
            last_use,
        }
    }
}

/// Which layer's slot an op names: its own, or the next layer's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerOffset {
    Same,
    Next,
}

/// A role in a layer's frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoleRef {
    pub role: Role,
    pub layer: LayerOffset,
}

impl RoleRef {
    /// `role` in the op's own layer.
    #[must_use]
    pub const fn same(role: Role) -> Self {
        Self {
            role,
            layer: LayerOffset::Same,
        }
    }

    /// `role` in the layer after the op's.
    #[must_use]
    pub const fn next(role: Role) -> Self {
        Self {
            role,
            layer: LayerOffset::Next,
        }
    }
}

/// One of a layer's weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerWeight {
    InputNorm,
    Q,
    K,
    V,
    O,
    PostAttentionNorm,
    Gate,
    Up,
    Down,
}

impl fmt::Display for LayerWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LayerWeight::InputNorm => "input norm gain",
            LayerWeight::Q => "query projection",
            LayerWeight::K => "key projection",
            LayerWeight::V => "value projection",
            LayerWeight::O => "output projection",
            LayerWeight::PostAttentionNorm => "post-attention norm gain",
            LayerWeight::Gate => "gate projection",
            LayerWeight::Up => "up projection",
            LayerWeight::Down => "down projection",
        })
    }
}

/// The columns of the fused qkv row a projection writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QkvColumns {
    Q,
    K,
    V,
}

/// One op of a layer, with the roles it reads and writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerOp {
    RmsNorm {
        input: RoleRef,
        gain: LayerWeight,
        output: RoleRef,
    },
    /// `output = input · weightᵀ`, into columns of the fused row when `columns` is set.
    Projection {
        input: RoleRef,
        weight: LayerWeight,
        output: RoleRef,
        columns: Option<QkvColumns>,
    },
    /// The rotary embedding over the q and k heads of the fused row, in place.
    Rope { qkv: RoleRef },
    /// The k and v heads of the fused row scattered into the paged cache.
    KvWrite { qkv: RoleRef },
    /// Paged decode attention from the q heads of the fused row over the cache.
    Attention { qkv: RoleRef, output: RoleRef },
    SiluMul {
        gate: RoleRef,
        up: RoleRef,
        output: RoleRef,
    },
    ResidualAdd {
        residual: RoleRef,
        delta: RoleRef,
        output: RoleRef,
    },
}

impl LayerOp {
    /// The roles this op reads.
    #[must_use]
    pub fn reads(&self) -> Vec<RoleRef> {
        match *self {
            LayerOp::RmsNorm { input, .. } | LayerOp::Projection { input, .. } => vec![input],
            LayerOp::Rope { qkv } | LayerOp::KvWrite { qkv } | LayerOp::Attention { qkv, .. } => {
                vec![qkv]
            }
            LayerOp::SiluMul { gate, up, .. } => vec![gate, up],
            LayerOp::ResidualAdd {
                residual, delta, ..
            } => vec![residual, delta],
        }
    }

    /// The roles this op writes.
    #[must_use]
    pub fn writes(&self) -> Vec<RoleRef> {
        match *self {
            LayerOp::RmsNorm { output, .. }
            | LayerOp::Projection { output, .. }
            | LayerOp::Attention { output, .. }
            | LayerOp::SiluMul { output, .. }
            | LayerOp::ResidualAdd { output, .. } => vec![output],
            LayerOp::Rope { qkv } => vec![qkv],
            LayerOp::KvWrite { .. } => Vec::new(),
        }
    }

    /// What the op computes, for logs and launch records.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            LayerOp::RmsNorm { .. } => "rmsnorm",
            LayerOp::Projection { weight, .. } => weight.projection_name(),
            LayerOp::Rope { .. } => "rope",
            LayerOp::KvWrite { .. } => "kv_write",
            LayerOp::Attention { .. } => "attention",
            LayerOp::SiluMul { .. } => "silu_mul",
            LayerOp::ResidualAdd { .. } => "residual_add",
        }
    }
}

impl LayerWeight {
    /// The projection this weight is, as logs name it; a norm gain is no projection.
    #[must_use]
    pub fn projection_name(self) -> &'static str {
        match self {
            LayerWeight::Q => "q_proj",
            LayerWeight::K => "k_proj",
            LayerWeight::V => "v_proj",
            LayerWeight::O => "o_proj",
            LayerWeight::Gate => "gate_proj",
            LayerWeight::Up => "up_proj",
            LayerWeight::Down => "down_proj",
            LayerWeight::InputNorm | LayerWeight::PostAttentionNorm => "projection",
        }
    }
}

/// A class of layers: the linear op order every layer of the class runs, and so the coordinate
/// system its roles' lifetimes index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerClass {
    name: &'static str,
    ops: &'static [LayerOp],
}

impl LayerClass {
    #[must_use]
    pub const fn new(name: &'static str, ops: &'static [LayerOp]) -> Self {
        Self { name, ops }
    }

    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The op order, in enqueue order.
    #[must_use]
    pub fn ops(&self) -> &'static [LayerOp] {
        self.ops
    }

    /// Length of the op order: the frame every lifetime is relative to.
    #[must_use]
    pub fn ops_per_layer(&self) -> usize {
        self.ops.len()
    }

    /// The hull of the ops that touch `role`, in the frame of the layer the role belongs to:
    /// from the first op that reads or writes it to one past the last. An op that names the
    /// role in the next layer touches it at a negative index of that layer's frame. `None` when
    /// no op touches the role.
    ///
    /// # Panics
    ///
    /// Panics when the op order is longer than an `isize`, which no declared class is.
    #[must_use]
    pub fn hull(&self, role: Role) -> Option<Lifetime> {
        let frame = isize::try_from(self.ops.len()).expect("an op order fits in isize");
        let mut touches = self.ops.iter().enumerate().flat_map(|(index, op)| {
            let index = isize::try_from(index).expect("an op index fits in isize");
            op.reads()
                .into_iter()
                .chain(op.writes())
                .filter(move |touched| touched.role == role)
                .map(move |touched| match touched.layer {
                    LayerOffset::Same => index,
                    LayerOffset::Next => index - frame,
                })
        });
        let first = touches.next()?;
        let (first_use, last) = touches.fold((first, first), |(first, last), index| {
            (first.min(index), last.max(index))
        });
        Some(Lifetime {
            first_use,
            last_use: last + 1,
        })
    }

    /// The arena's role table for a model of `dims`: every role's per-token width in bf16 and
    /// its declared lifetime.
    #[must_use]
    pub fn role_table(&self, dims: &LlamaDims) -> RoleTable {
        RoleTable {
            ops_per_layer: self.ops_per_layer(),
            roles: Role::ALL
                .iter()
                .map(|role| RoleDeclaration {
                    width_bytes: Dtype::Bf16.width_bytes(role.width_elements(dims)),
                    lifetime: role.lifetime(),
                })
                .collect(),
        }
    }
}

/// Llama's one layer class: fifteen ops from the input norm to the feed-forward residual add.
pub const LLAMA_LAYER: LayerClass = LayerClass::new("llama", &LLAMA_OPS);

/// The op order of a Llama layer. Reordering it changes every role's lifetime, and the hull test
/// says which declarations no longer hold.
pub const LLAMA_OPS: [LayerOp; 15] = [
    LayerOp::RmsNorm {
        input: RoleRef::same(Role::Hidden),
        gain: LayerWeight::InputNorm,
        output: RoleRef::same(Role::Normed),
    },
    LayerOp::Projection {
        input: RoleRef::same(Role::Normed),
        weight: LayerWeight::Q,
        output: RoleRef::same(Role::Qkv),
        columns: Some(QkvColumns::Q),
    },
    LayerOp::Projection {
        input: RoleRef::same(Role::Normed),
        weight: LayerWeight::K,
        output: RoleRef::same(Role::Qkv),
        columns: Some(QkvColumns::K),
    },
    LayerOp::Projection {
        input: RoleRef::same(Role::Normed),
        weight: LayerWeight::V,
        output: RoleRef::same(Role::Qkv),
        columns: Some(QkvColumns::V),
    },
    LayerOp::Rope {
        qkv: RoleRef::same(Role::Qkv),
    },
    LayerOp::KvWrite {
        qkv: RoleRef::same(Role::Qkv),
    },
    LayerOp::Attention {
        qkv: RoleRef::same(Role::Qkv),
        output: RoleRef::same(Role::AttnOut),
    },
    LayerOp::Projection {
        input: RoleRef::same(Role::AttnOut),
        weight: LayerWeight::O,
        output: RoleRef::same(Role::OProj),
        columns: None,
    },
    LayerOp::ResidualAdd {
        residual: RoleRef::same(Role::Hidden),
        delta: RoleRef::same(Role::OProj),
        output: RoleRef::same(Role::Mid),
    },
    LayerOp::RmsNorm {
        input: RoleRef::same(Role::Mid),
        gain: LayerWeight::PostAttentionNorm,
        output: RoleRef::same(Role::Normed),
    },
    LayerOp::Projection {
        input: RoleRef::same(Role::Normed),
        weight: LayerWeight::Gate,
        output: RoleRef::same(Role::Gate),
        columns: None,
    },
    LayerOp::Projection {
        input: RoleRef::same(Role::Normed),
        weight: LayerWeight::Up,
        output: RoleRef::same(Role::Up),
        columns: None,
    },
    LayerOp::SiluMul {
        gate: RoleRef::same(Role::Gate),
        up: RoleRef::same(Role::Up),
        output: RoleRef::same(Role::FfnAct),
    },
    LayerOp::Projection {
        input: RoleRef::same(Role::FfnAct),
        weight: LayerWeight::Down,
        output: RoleRef::same(Role::FfnDown),
        columns: None,
    },
    LayerOp::ResidualAdd {
        residual: RoleRef::same(Role::Mid),
        delta: RoleRef::same(Role::FfnDown),
        output: RoleRef::next(Role::Hidden),
    },
];

/// The Llama role table with `Normed`'s lifetime declared one op short, `[0, 11)` for `[0, 12)`.
///
/// Op 11 is the up projection, the last op that reads `Normed`, so the declaration says the slot
/// is dead while that projection still reads it. This is the one lie the correctness gates run,
/// and it is stated here once so the host proofs of what it does and the device gate that runs it
/// cannot drift apart.
///
/// Under the poison layout a fill lands over `Normed` ahead of that projection and the step reads
/// not-a-number. Under greedy it moves no slot: `Up`, which the projection writes, is feed-forward
/// wide, and the hole the lie frees is hidden wide with live neighbours.
#[cfg(any(test, feature = "test-support"))]
#[must_use]
pub fn normed_one_op_short(dims: &LlamaDims) -> RoleTable {
    let mut table = LLAMA_LAYER.role_table(dims);
    table.roles[Role::Normed as usize].lifetime.last_use -= 1;
    table
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use atoma_runtime::arena::{ArenaLayout, BucketIdx, CaptureArena, LayerIdx};

    use super::*;
    use crate::dims::test_support::{llama_1b, llama_8b};

    #[test]
    fn every_declared_lifetime_is_the_hull_of_the_ops_that_touch_the_role() {
        for role in Role::ALL {
            assert_eq!(
                LLAMA_LAYER.hull(role),
                Some(role.lifetime()),
                "{role:?}'s declaration disagrees with the op table"
            );
        }
    }

    #[test]
    fn the_residual_stream_enters_a_layer_from_the_previous_layers_last_op() {
        let hull = LLAMA_LAYER.hull(Role::Hidden).unwrap();
        assert_eq!(hull.first_use, -1);
        let last = LLAMA_OPS[LLAMA_OPS.len() - 1];
        assert_eq!(last.writes(), [RoleRef::next(Role::Hidden)]);
    }

    #[test]
    fn a_role_no_op_touches_has_no_hull() {
        const SILENT: LayerClass = LayerClass::new("silent", &[]);
        assert_eq!(SILENT.hull(Role::Qkv), None);
        assert_eq!(SILENT.ops_per_layer(), 0);
    }

    #[test]
    fn the_op_table_names_each_op_and_reads_before_it_writes() {
        let names: Vec<&str> = LLAMA_OPS.iter().map(LayerOp::name).collect();
        assert_eq!(
            names,
            [
                "rmsnorm",
                "q_proj",
                "k_proj",
                "v_proj",
                "rope",
                "kv_write",
                "attention",
                "o_proj",
                "residual_add",
                "rmsnorm",
                "gate_proj",
                "up_proj",
                "silu_mul",
                "down_proj",
                "residual_add",
            ]
        );
        let mut written = vec![RoleRef::same(Role::Hidden)];
        for op in LLAMA_OPS {
            for read in op.reads() {
                assert!(
                    written.contains(&read),
                    "{} reads {read:?} before any op wrote it",
                    op.name()
                );
            }
            written.extend(op.writes());
        }
    }

    #[test]
    fn the_role_table_carries_widths_in_bf16_and_the_declared_lifetimes() {
        let dims = llama_8b(2);
        let table = LLAMA_LAYER.role_table(&dims);
        assert_eq!(table.ops_per_layer, 15);
        assert_eq!(table.roles.len(), Role::ALL.len());
        assert_eq!(table.roles[Role::Hidden as usize].width_bytes, 4096 * 2);
        assert_eq!(table.roles[Role::Qkv as usize].width_bytes, 6144 * 2);
        assert_eq!(table.roles[Role::Gate as usize].width_bytes, 14336 * 2);
        assert_eq!(
            table.roles[Role::Mid as usize].lifetime,
            Role::Mid.lifetime()
        );
    }

    #[test]
    fn the_arena_accepts_the_table_and_addresses_the_final_residual_row() {
        let dims = llama_8b(2);
        let arena = CaptureArena::new(
            dims.layers + 1,
            LLAMA_LAYER.role_table(&dims),
            &[1, 8],
            ArenaLayout::Greedy,
        )
        .expect("the declared lifetimes are valid");
        let final_hidden = arena.offset(BucketIdx(1), LayerIdx(2), Role::Hidden.tensor_role());
        assert_eq!(final_hidden % 256, 0);
        assert!(arena.total_size() > 0);
    }

    /// The bucket ladder the device gates run by default: every bucket of the Hopper ladder at
    /// or below a maximum batch of 128.
    const GATE_LADDER: [usize; 19] = [
        1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
    ];

    /// The byte range of `role`'s slot in `layer`'s frame under `bucket`.
    fn slot_bytes(arena: &CaptureArena, bucket: usize, layer: usize, role: Role) -> Range<usize> {
        let start = arena.offset(BucketIdx(bucket), LayerIdx(layer), role.tensor_role());
        start..start + arena.slot_size(BucketIdx(bucket), role.tensor_role())
    }

    fn arena_under(dims: &LlamaDims, table: RoleTable, layout: ArenaLayout) -> CaptureArena {
        CaptureArena::new(dims.layers + 1, table, &GATE_LADDER, layout)
            .expect("the table declares valid lifetimes")
    }

    #[test]
    fn the_up_projection_is_the_last_op_to_read_normed() {
        assert!(matches!(
            LLAMA_OPS[11],
            LayerOp::Projection {
                input: RoleRef {
                    role: Role::Normed,
                    layer: LayerOffset::Same
                },
                weight: LayerWeight::Up,
                ..
            }
        ));
        assert_eq!(Role::Normed.lifetime().last_use, 12);
        let dims = llama_8b(2);
        let table = normed_one_op_short(&dims);
        assert_eq!(
            table.roles[Role::Normed as usize].lifetime,
            Lifetime {
                first_use: 0,
                last_use: 11
            }
        );
    }

    #[test]
    fn normed_declared_dead_one_op_early_is_poisoned_ahead_of_the_projection_that_reads_it() {
        for dims in [llama_8b(4), llama_1b(16)] {
            let arena = arena_under(&dims, normed_one_op_short(&dims), ArenaLayout::Poison);
            let per_layer = isize::try_from(LLAMA_LAYER.ops_per_layer()).unwrap();
            for (bucket, &tokens) in GATE_LADDER.iter().enumerate() {
                let fills = arena.poison_fills(BucketIdx(bucket));
                for layer in 0..dims.layers {
                    let normed = slot_bytes(&arena, bucket, layer, Role::Normed);
                    let up_projection = isize::try_from(layer).unwrap() * per_layer + 11;
                    assert!(
                        fills.iter().any(|fill| fill.before_op == up_projection
                            && fill.offset == normed.start
                            && fill.len == normed.len()),
                        "hidden {}, bucket {tokens}, layer {layer}: no fill over Normed's slot \
                         before its up projection",
                        dims.hidden
                    );
                }
            }
        }
    }

    #[test]
    fn normed_declared_dead_one_op_early_moves_no_slot_under_greedy() {
        // `Up`, which the up projection writes, is feed-forward wide; the hole the lie frees is
        // hidden wide with live neighbours, so greedy has nothing to place in it. The greedy
        // step over the lie is the declared step, and a gate over it under greedy reads the
        // reference's logits back: only the poison layout can turn the lie into a failure.
        for dims in [llama_8b(4), llama_1b(16)] {
            let declared = arena_under(&dims, LLAMA_LAYER.role_table(&dims), ArenaLayout::Greedy);
            let lied = arena_under(&dims, normed_one_op_short(&dims), ArenaLayout::Greedy);
            for (bucket, &tokens) in GATE_LADDER.iter().enumerate() {
                for layer in 0..=dims.layers {
                    for role in Role::ALL {
                        assert_eq!(
                            slot_bytes(&lied, bucket, layer, role),
                            slot_bytes(&declared, bucket, layer, role),
                            "hidden {}, bucket {tokens}, layer {layer}, {role:?}: the lie moved \
                             the slot, so a greedy step over it is no longer the declared step",
                            dims.hidden
                        );
                    }
                }
                assert_eq!(
                    lied.bucket_footprint(BucketIdx(bucket)),
                    declared.bucket_footprint(BucketIdx(bucket))
                );
            }
        }
    }
}
