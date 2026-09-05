//! The decode step over runtime-owned tensors: the host side that needs no device.
//!
//! A keyed step command is checked against the bucket its key names, laid out as the fixed
//! inputs the step reads at the widths the captured graphs bake, and carried to the device
//! through the descriptor seam. Nothing here needs candle or a compiled kernel, so it builds and
//! tests without a device; the model step itself sits behind the `cuda` feature.
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`batch`] | A keyed batch held to its bucket, and the buckets the decode step serves |
//! | [`baked`] | Every address the step bakes, by name, and the check that none of them moved |
//! | [`staging`] | One step's inputs written into staging at full width, ready to upload |
//! | [`inputs`] | The staging ring's blocks, the device block, each bucket's views; the upload and wait descriptors |
//! | [`ring`] | The staging ring: a fence per staging entry and a cursor; `acquire` waits, `try_acquire` asks |

pub mod baked;
pub mod batch;
pub mod inputs;
pub mod ring;
pub mod staging;

use atoma_core::attention::{BackendDeclaration, SupportLevel};

/// What the decode step over runtime tensors declares to the capture contract: it serves every
/// uniform single-token decode batch, so the engine keys those and pads them to their bucket.
/// Under NCCL the decode step stays on candle and nothing is keyed.
#[must_use]
pub fn declaration() -> BackendDeclaration {
    BackendDeclaration::new("flash-attention", support_level())
}

#[cfg(not(feature = "nccl"))]
fn support_level() -> SupportLevel {
    SupportLevel::UniformSingleTokenDecode
}

#[cfg(feature = "nccl")]
fn support_level() -> SupportLevel {
    SupportLevel::Never
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declaration_keys_uniform_single_token_decodes_unless_ranks_are_coupled() {
        let declaration = declaration();
        assert_eq!(declaration.name(), "flash-attention");
        #[cfg(not(feature = "nccl"))]
        assert_eq!(
            declaration.support_level(),
            SupportLevel::UniformSingleTokenDecode
        );
        #[cfg(feature = "nccl")]
        assert_eq!(declaration.support_level(), SupportLevel::Never);
        assert!(declaration.break_points().is_empty());
    }
}
