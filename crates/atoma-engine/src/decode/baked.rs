//! Every address the decode step bakes, by name, and the check that none of them has moved.
//!
//! A step over runtime-owned tensors reads and writes fixed device addresses: the views it was
//! built over name them, and a replay trusts them. Candle owns the weights and the cache, so those
//! are the addresses that can move; the device block, the arena, the step's fixed buffers and the
//! sampler's arrays are owned for as long as the views over them are read, and are listed with
//! them so the check covers every address a step bakes. Each address is read from the memory that
//! owns it when the forward is built and, in a debug build, again before each keyed step; the
//! first that differs ends the step with a panic naming it, before the step's own work is
//! enqueued. Nothing here needs a device: the forward reads the addresses and this module
//! compares them.

use std::fmt;

use atoma_models::layer::LayerWeight;
use thiserror::Error;

/// What one baked address is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakedName {
    /// Candle's embedding table.
    EmbeddingTable,
    /// One of a layer's nine weights, candle's.
    LayerWeight { layer: usize, weight: LayerWeight },
    /// Candle's final norm gain.
    FinalNormGain,
    /// Candle's head projection.
    HeadProjection,
    /// One layer's paged cache, candle's.
    Cache { layer: usize },
    /// The device block every bucket's input views are minted over.
    DeviceBlock,
    /// The arena every activation is addressed in.
    Arena,
    /// The step's logits.
    Logits,
    /// The attention's log-sum-exp output.
    LogSumExp,
    /// The attention's split log-sum-exp accumulator.
    SplitLogSumExp,
    /// The attention's split output accumulator.
    SplitOutput,
    /// The rotary cosine table.
    CosineTable,
    /// The rotary sine table.
    SineTable,
    /// The sampler's per-slot sampling records.
    SamplingRecords,
    /// The sampler's per-slot last sampled tokens.
    SampledTokens,
    /// The sampler's own row slots, which an eager step samples under. The per-step row slots a
    /// keyed step samples under are staged into the packed block and are not baked.
    SamplerRowSlots,
    /// The sampler's row tokens, which the readback copies through.
    RowTokens,
}

impl fmt::Display for BakedName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BakedName::EmbeddingTable => "the embedding table",
            BakedName::LayerWeight { layer, weight } => {
                return write!(f, "layer {layer}'s {weight}");
            }
            BakedName::FinalNormGain => "the final norm gain",
            BakedName::HeadProjection => "the head projection",
            BakedName::Cache { layer } => return write!(f, "layer {layer}'s cache"),
            BakedName::DeviceBlock => "the device block",
            BakedName::Arena => "the arena",
            BakedName::Logits => "the logits",
            BakedName::LogSumExp => "the log-sum-exp output",
            BakedName::SplitLogSumExp => "the split log-sum-exp accumulator",
            BakedName::SplitOutput => "the split output accumulator",
            BakedName::CosineTable => "the cosine table",
            BakedName::SineTable => "the sine table",
            BakedName::SamplingRecords => "the sampling records",
            BakedName::SampledTokens => "the sampled tokens",
            BakedName::SamplerRowSlots => "the sampler's row slots",
            BakedName::RowTokens => "the row tokens",
        })
    }
}

/// One baked address: which memory it names, and the address that memory was at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BakedAddress {
    pub name: BakedName,
    pub address: u64,
}

/// Why the addresses read again before a step are not the baked ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BakedError {
    /// A baked address is not where the memory that owns it now is.
    #[error("{name} was baked at {baked:#x} and is at {current:#x} before this step")]
    Moved {
        name: BakedName,
        baked: u64,
        current: u64,
    },
    /// Other memory was read at a baked position: the addresses are read again in the order they
    /// were baked, and this reading does not follow it.
    #[error("position {position} was baked as {baked} and read again as {read}")]
    OtherName {
        position: usize,
        baked: BakedName,
        read: BakedName,
    },
    /// Fewer or more addresses were read again than were baked.
    #[error("{baked} addresses were baked and {read} were read again")]
    Count { baked: usize, read: usize },
}

/// Every address the step bakes, by name, in the order they were read.
#[derive(Debug)]
pub struct BakedAddresses {
    addresses: Vec<BakedAddress>,
}

impl BakedAddresses {
    /// Bakes `addresses`, in the order read: what [`BakedAddresses::check`] compares a later
    /// reading against.
    #[must_use]
    pub fn bake(addresses: impl IntoIterator<Item = BakedAddress>) -> Self {
        Self {
            addresses: addresses.into_iter().collect(),
        }
    }

    /// Compares `reading`, the baked addresses read again in the order they were baked, against
    /// the baked ones.
    ///
    /// # Errors
    ///
    /// Returns [`BakedError`] for the first address that differs: moved, other memory at its
    /// position, or a reading that ends early or runs on.
    pub fn check(&self, reading: impl IntoIterator<Item = BakedAddress>) -> Result<(), BakedError> {
        let mut reading = reading.into_iter();
        for (position, baked) in self.addresses.iter().enumerate() {
            let Some(read) = reading.next() else {
                return Err(BakedError::Count {
                    baked: self.addresses.len(),
                    read: position,
                });
            };
            if read.name != baked.name {
                return Err(BakedError::OtherName {
                    position,
                    baked: baked.name,
                    read: read.name,
                });
            }
            if read.address != baked.address {
                return Err(BakedError::Moved {
                    name: baked.name,
                    baked: baked.address,
                    current: read.address,
                });
            }
        }
        let past_the_baked = reading.count();
        if past_the_baked > 0 {
            return Err(BakedError::Count {
                baked: self.addresses.len(),
                read: self.addresses.len() + past_the_baked,
            });
        }
        Ok(())
    }

    /// Checks `reading` as [`BakedAddresses::check`] does and panics with the first address that
    /// differs, naming it: what a debug build runs before each keyed step.
    ///
    /// # Panics
    ///
    /// Panics with the [`BakedError`] the check returns.
    pub fn assert_unmoved(&self, reading: impl IntoIterator<Item = BakedAddress>) {
        if let Err(error) = self.check(reading) {
            panic!("{error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;

    /// Where a fake device holds its memory, aligned as a device allocation is.
    const BASE: u64 = 0x7f00_0000_0000;

    /// Memory the test moves: what a fake candle and device hold, read by name in one order.
    struct Memory(Vec<BakedAddress>);

    impl Memory {
        /// Five addresses a page apart, candle's first as the forward reads them.
        fn held() -> Self {
            let names = [
                BakedName::EmbeddingTable,
                BakedName::LayerWeight {
                    layer: 3,
                    weight: LayerWeight::K,
                },
                BakedName::Cache { layer: 0 },
                BakedName::Arena,
                BakedName::RowTokens,
            ];
            Self(
                names
                    .into_iter()
                    .enumerate()
                    .map(|(page, name)| BakedAddress {
                        name,
                        address: BASE + page as u64 * 0x1000,
                    })
                    .collect(),
            )
        }

        fn addresses(&self) -> impl Iterator<Item = BakedAddress> + '_ {
            self.0.iter().copied()
        }

        fn relocate(&mut self, name: BakedName, to: u64) {
            let held = self
                .0
                .iter_mut()
                .find(|held| held.name == name)
                .expect("a name the memory holds");
            held.address = to;
        }
    }

    #[test]
    fn every_address_read_again_where_it_was_baked_passes() {
        let memory = Memory::held();
        let baked = BakedAddresses::bake(memory.addresses());

        assert_eq!(baked.check(memory.addresses()), Ok(()));
        assert_eq!(baked.check(memory.addresses()), Ok(()), "read again");
    }

    #[test]
    fn an_empty_set_passes() {
        let baked = BakedAddresses::bake(iter::empty());
        assert_eq!(baked.check(iter::empty()), Ok(()));
    }

    #[test]
    fn the_first_address_that_moved_is_named_with_where_it_was_and_where_it_is() {
        let mut memory = Memory::held();
        let baked = BakedAddresses::bake(memory.addresses());
        // Layer 3's key projection was baked at page 1 and moves to page 9; the arena, baked at
        // page 3, moves too, but is after it in the order read.
        let key = BakedName::LayerWeight {
            layer: 3,
            weight: LayerWeight::K,
        };
        memory.relocate(key, BASE + 0x9000);
        memory.relocate(BakedName::Arena, BASE + 0xa000);

        let refused = baked.check(memory.addresses()).unwrap_err();

        assert_eq!(
            refused,
            BakedError::Moved {
                name: key,
                baked: BASE + 0x1000,
                current: BASE + 0x9000,
            }
        );
        assert_eq!(
            refused.to_string(),
            "layer 3's key projection was baked at 0x7f0000001000 and is at 0x7f0000009000 before \
             this step"
        );
    }

    #[test]
    fn other_memory_at_a_baked_position_is_refused_by_both_names() {
        let memory = Memory::held();
        let baked = BakedAddresses::bake(memory.addresses());
        // The reading swaps positions 1 and 2: layer 3's key projection and layer 0's cache.
        let mut swapped: Vec<BakedAddress> = memory.addresses().collect();
        swapped.swap(1, 2);

        assert_eq!(
            baked.check(swapped),
            Err(BakedError::OtherName {
                position: 1,
                baked: BakedName::LayerWeight {
                    layer: 3,
                    weight: LayerWeight::K,
                },
                read: BakedName::Cache { layer: 0 },
            })
        );
    }

    #[test]
    fn fewer_or_more_addresses_read_again_than_were_baked_are_refused() {
        let memory = Memory::held();
        let baked = BakedAddresses::bake(memory.addresses());

        // Three of the five, every one where it was baked.
        assert_eq!(
            baked.check(memory.addresses().take(3)),
            Err(BakedError::Count { baked: 5, read: 3 })
        );
        // The five, then one more.
        let extra = BakedAddress {
            name: BakedName::SineTable,
            address: BASE + 0xf000,
        };
        assert_eq!(
            baked.check(memory.addresses().chain(iter::once(extra))),
            Err(BakedError::Count { baked: 5, read: 6 })
        );
        assert_eq!(
            BakedError::Count { baked: 5, read: 3 }.to_string(),
            "5 addresses were baked and 3 were read again"
        );
    }

    #[test]
    #[should_panic(
        expected = "layer 3's key projection was baked at 0x7f0000001000 and is at 0x7f0000009000 \
                    before this step"
    )]
    fn the_assert_panics_naming_the_first_address_that_moved() {
        let mut memory = Memory::held();
        let baked = BakedAddresses::bake(memory.addresses());
        memory.relocate(
            BakedName::LayerWeight {
                layer: 3,
                weight: LayerWeight::K,
            },
            BASE + 0x9000,
        );
        memory.relocate(BakedName::RowTokens, BASE + 0xb000);

        baked.assert_unmoved(memory.addresses());
    }

    #[test]
    fn only_the_first_baked_address_moving_is_refused() {
        let mut memory = Memory::held();
        let baked = BakedAddresses::bake(memory.addresses());
        // The embedding table is first in the order read, and the only thing that moves.
        memory.relocate(BakedName::EmbeddingTable, BASE + 0x9000);

        assert_eq!(
            baked.check(memory.addresses()),
            Err(BakedError::Moved {
                name: BakedName::EmbeddingTable,
                baked: BASE,
                current: BASE + 0x9000,
            })
        );
    }

    #[test]
    fn only_the_last_baked_address_moving_is_refused() {
        let mut memory = Memory::held();
        let baked = BakedAddresses::bake(memory.addresses());
        // The row tokens are last in the order read, and the only thing that moves.
        memory.relocate(BakedName::RowTokens, BASE + 0x9000);

        assert_eq!(
            baked.check(memory.addresses()),
            Err(BakedError::Moved {
                name: BakedName::RowTokens,
                baked: BASE + 0x4000,
                current: BASE + 0x9000,
            })
        );
    }

    #[test]
    fn one_baked_address_passes_where_it_was_baked_and_is_refused_where_it_moved_to() {
        let arena = BakedAddress {
            name: BakedName::Arena,
            address: BASE,
        };
        let baked = BakedAddresses::bake(iter::once(arena));

        assert_eq!(baked.check(iter::once(arena)), Ok(()));

        let moved = BakedAddress {
            address: BASE + 0x1000,
            ..arena
        };
        assert_eq!(
            baked.check(iter::once(moved)),
            Err(BakedError::Moved {
                name: BakedName::Arena,
                baked: BASE,
                current: BASE + 0x1000,
            })
        );
    }

    #[test]
    fn other_memory_at_the_first_or_the_last_baked_position_is_refused() {
        let memory = Memory::held();
        let baked = BakedAddresses::bake(memory.addresses());

        // The sine table is read where the embedding table was baked, at the first position.
        let mut first: Vec<BakedAddress> = memory.addresses().collect();
        first[0].name = BakedName::SineTable;
        assert_eq!(
            baked.check(first),
            Err(BakedError::OtherName {
                position: 0,
                baked: BakedName::EmbeddingTable,
                read: BakedName::SineTable,
            })
        );

        // The sampling records are read where the row tokens were baked, at the last position.
        let mut last: Vec<BakedAddress> = memory.addresses().collect();
        let end = last.len() - 1;
        last[end].name = BakedName::SamplingRecords;
        assert_eq!(
            baked.check(last),
            Err(BakedError::OtherName {
                position: end,
                baked: BakedName::RowTokens,
                read: BakedName::SamplingRecords,
            })
        );
    }

    /// Every name, with the words it reads as: one of every kind, all nine of a layer's weights
    /// and three layers, so a weight or a cache reading as another layer's is caught too.
    fn every_name_and_what_it_reads_as() -> Vec<(BakedName, &'static str)> {
        let weight = |layer, weight| BakedName::LayerWeight { layer, weight };
        vec![
            (BakedName::EmbeddingTable, "the embedding table"),
            (
                weight(0, LayerWeight::InputNorm),
                "layer 0's input norm gain",
            ),
            (weight(0, LayerWeight::Q), "layer 0's query projection"),
            (weight(0, LayerWeight::K), "layer 0's key projection"),
            (weight(0, LayerWeight::V), "layer 0's value projection"),
            (weight(0, LayerWeight::O), "layer 0's output projection"),
            (
                weight(0, LayerWeight::PostAttentionNorm),
                "layer 0's post-attention norm gain",
            ),
            (weight(0, LayerWeight::Gate), "layer 0's gate projection"),
            (weight(0, LayerWeight::Up), "layer 0's up projection"),
            (weight(0, LayerWeight::Down), "layer 0's down projection"),
            (weight(7, LayerWeight::Q), "layer 7's query projection"),
            (weight(31, LayerWeight::Down), "layer 31's down projection"),
            (BakedName::FinalNormGain, "the final norm gain"),
            (BakedName::HeadProjection, "the head projection"),
            (BakedName::Cache { layer: 0 }, "layer 0's cache"),
            (BakedName::Cache { layer: 7 }, "layer 7's cache"),
            (BakedName::Cache { layer: 31 }, "layer 31's cache"),
            (BakedName::DeviceBlock, "the device block"),
            (BakedName::Arena, "the arena"),
            (BakedName::Logits, "the logits"),
            (BakedName::LogSumExp, "the log-sum-exp output"),
            (
                BakedName::SplitLogSumExp,
                "the split log-sum-exp accumulator",
            ),
            (BakedName::SplitOutput, "the split output accumulator"),
            (BakedName::CosineTable, "the cosine table"),
            (BakedName::SineTable, "the sine table"),
            (BakedName::SamplingRecords, "the sampling records"),
            (BakedName::SampledTokens, "the sampled tokens"),
            (BakedName::SamplerRowSlots, "the sampler's row slots"),
            (BakedName::RowTokens, "the row tokens"),
        ]
    }

    #[test]
    fn every_name_reads_as_the_thing_it_names() {
        for (name, reads_as) in every_name_and_what_it_reads_as() {
            assert_eq!(name.to_string(), reads_as, "{name:?}");
        }
    }

    #[test]
    fn no_two_names_read_as_the_same_thing() {
        let names = every_name_and_what_it_reads_as();
        for (position, (name, _)) in names.iter().enumerate() {
            for (other, _) in &names[position + 1..] {
                assert_ne!(
                    name.to_string(),
                    other.to_string(),
                    "{name:?} and {other:?} read as the same thing"
                );
            }
        }
    }
}
