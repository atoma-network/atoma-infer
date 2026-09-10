//! The dimensions a Llama decode step is shaped by.
//!
//! Read from a checkpoint's configuration by whoever loads it, then checked once here: every width
//! the step derives — the fused qkv row, the per-head scale, the rotary pairs — follows from these
//! numbers, so a set that does not hang together is refused before any buffer is sized from it.

use std::fmt;

use thiserror::Error;

/// Llama 3's rotary frequency scaling: frequencies below the low-frequency wavelength are
/// divided by `factor`, those above the high-frequency wavelength are left alone, and the band
/// between is blended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Llama3RopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
}

/// The rotary embedding's parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeParams {
    /// The base of the frequency ladder.
    pub theta: f32,
    /// Llama 3's scaling, or none for the unscaled ladder.
    pub scaling: Option<Llama3RopeScaling>,
    /// Positions the tables cover: the longest sequence the model serves.
    pub max_position: usize,
}

/// A set of dimensions that does not describe a Llama.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DimsError {
    #[error("hidden size {hidden} is not {num_heads} heads of {head_dim}")]
    HiddenNotHeads {
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    },
    #[error("{num_heads} query heads do not group over {num_kv_heads} key-value heads")]
    HeadsNotGrouped {
        num_heads: usize,
        num_kv_heads: usize,
    },
    #[error("head dimension {head_dim} has no rotary pairs; it must be even and nonzero")]
    HeadDimNotPaired { head_dim: usize },
    #[error("{dimension} is zero")]
    Zero { dimension: LlamaDimension },
}

/// A dimension the step needs nonzero, as a refusal names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaDimension {
    Layers,
    Ffn,
    Vocab,
    NumKvHeads,
    MaxPosition,
}

impl fmt::Display for LlamaDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LlamaDimension::Layers => "the layer count",
            LlamaDimension::Ffn => "the feed-forward width",
            LlamaDimension::Vocab => "the vocabulary",
            LlamaDimension::NumKvHeads => "the key-value head count",
            LlamaDimension::MaxPosition => "the rotary position range",
        })
    }
}

/// The dimensions of one Llama, as the decode step reads them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlamaDims {
    pub layers: usize,
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// The feed-forward intermediate width.
    pub ffn: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope: RopeParams,
}

impl LlamaDims {
    /// Refuses a set of dimensions the step could not be shaped by.
    ///
    /// # Errors
    ///
    /// Returns [`DimsError`] when a dimension is zero, the head dimension has no rotary pairs,
    /// the hidden size is not the query heads, or the heads do not group over the key-value
    /// heads.
    pub fn check(&self) -> Result<(), DimsError> {
        for (dimension, value) in [
            (LlamaDimension::Layers, self.layers),
            (LlamaDimension::Ffn, self.ffn),
            (LlamaDimension::Vocab, self.vocab),
            (LlamaDimension::NumKvHeads, self.num_kv_heads),
            (LlamaDimension::MaxPosition, self.rope.max_position),
        ] {
            if value == 0 {
                return Err(DimsError::Zero { dimension });
            }
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(DimsError::HeadDimNotPaired {
                head_dim: self.head_dim,
            });
        }
        if self.hidden != self.num_heads * self.head_dim {
            return Err(DimsError::HiddenNotHeads {
                hidden: self.hidden,
                num_heads: self.num_heads,
                head_dim: self.head_dim,
            });
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(DimsError::HeadsNotGrouped {
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
            });
        }
        Ok(())
    }

    /// Elements of the query projection per token.
    #[must_use]
    pub fn q_width(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Elements of the key (or value) projection per token.
    #[must_use]
    pub fn kv_width(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Elements of the fused qkv row per token: the query heads, then the key heads, then the
    /// value heads.
    #[must_use]
    pub fn qkv_width(&self) -> usize {
        self.q_width() + 2 * self.kv_width()
    }

    /// The heads the rotary embedding turns: the query heads and the key heads, which lead the
    /// fused row.
    #[must_use]
    pub fn rotary_heads(&self) -> usize {
        self.num_heads + self.num_kv_heads
    }

    /// The attention scale, one over the root of the head dimension.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn softmax_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{LlamaDims, RopeParams};

    /// Llama 3.1 8B's shape with `layers` layers: hidden 4096, 32 query and 8 key-value heads
    /// of 128, feed-forward 14336, vocabulary 128256.
    pub(crate) fn llama_8b(layers: usize) -> LlamaDims {
        LlamaDims {
            layers,
            hidden: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            ffn: 14336,
            vocab: 128_256,
            rms_eps: 1e-5,
            rope: RopeParams {
                theta: 500_000.0,
                scaling: None,
                max_position: 8192,
            },
        }
    }

    /// Llama 3.2 1B's shape with `layers` layers: hidden 2048, 32 query and 8 key-value heads
    /// of 64, feed-forward 8192, vocabulary 128256.
    pub(crate) fn llama_1b(layers: usize) -> LlamaDims {
        LlamaDims {
            layers,
            hidden: 2048,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            ffn: 8192,
            vocab: 128_256,
            rms_eps: 1e-5,
            rope: RopeParams {
                theta: 500_000.0,
                scaling: None,
                max_position: 8192,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::llama_8b;
    use super::*;

    #[test]
    fn the_8b_shape_checks_and_derives_its_widths() {
        let dims = llama_8b(32);
        assert_eq!(dims.check(), Ok(()));
        assert_eq!(dims.q_width(), 4096);
        assert_eq!(dims.kv_width(), 1024);
        assert_eq!(dims.qkv_width(), 6144);
        assert_eq!(dims.rotary_heads(), 40);
        assert!((dims.softmax_scale() - 1.0 / 128f32.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn a_hidden_size_that_is_not_the_heads_is_refused() {
        let dims = LlamaDims {
            hidden: 4000,
            ..llama_8b(1)
        };
        assert_eq!(
            dims.check(),
            Err(DimsError::HiddenNotHeads {
                hidden: 4000,
                num_heads: 32,
                head_dim: 128
            })
        );
    }

    #[test]
    fn query_heads_must_group_over_the_key_value_heads() {
        let dims = LlamaDims {
            num_kv_heads: 6,
            ..llama_8b(1)
        };
        assert_eq!(
            dims.check(),
            Err(DimsError::HeadsNotGrouped {
                num_heads: 32,
                num_kv_heads: 6
            })
        );
    }

    #[test]
    fn an_odd_head_dimension_has_no_rotary_pairs() {
        let dims = LlamaDims {
            hidden: 32 * 127,
            head_dim: 127,
            ..llama_8b(1)
        };
        assert_eq!(
            dims.check(),
            Err(DimsError::HeadDimNotPaired { head_dim: 127 })
        );
    }

    #[test]
    fn a_zero_dimension_is_refused_by_name() {
        let dims = LlamaDims {
            layers: 0,
            ..llama_8b(1)
        };
        let refused = dims.check().unwrap_err();
        assert_eq!(
            refused,
            DimsError::Zero {
                dimension: LlamaDimension::Layers
            }
        );
        assert!(refused.to_string().contains("layer count"));
    }
}
