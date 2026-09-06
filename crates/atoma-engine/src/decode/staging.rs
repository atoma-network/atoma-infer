//! A bucket's staged arrays packed into one block, and one step's inputs written into them at
//! full width.
//!
//! A bucket stages seven arrays for a step: the five the model step reads (token ids, positions,
//! key lengths, slot mapping, block table) and the two the sampler reads (row slots, gather
//! slots). [`StagingLayout::packed`] lays them consecutively at the bucket's rows, each at a
//! 256-byte boundary, so the bucket's staging is one block, [`StagingLayout::bytes`] long and
//! proportional to the batch rather than to the largest bucket, that one copy can carry.
//! [`StagingLayout::carve`] carves the seven arrays out of such a block; [`stage`] writes the
//! model's five from the batch layout, and [`stage_sampler`] the sampler's two from what the
//! sampler decided for the step. [`stage_dummy`] writes all seven for a [`DummyRun`]: every row
//! a padding row over one block, and the sampler's two naming no request slot, which is what a
//! capture check or a warmup runs when there is no live batch.
//!
//! The block table is staged at the full width a sequence can reach, never at the layout's
//! batch-local width: the width is baked into the attention launch, so it cannot follow the
//! batch.
//!
//! The arrays are borrowed rather than owned, so the same fill writes pinned host memory in
//! serving and a block of plain words in tests.

use std::fmt;
use std::mem;
use std::slice;

use atoma_core::request::PADDING_TOKEN;
use atoma_core::types::{BlockId, RequestSlot, TokenCount};
use atoma_runtime::arena::BucketIdx;
use thiserror::Error;

use crate::batch::BatchLayout;
use crate::decode::batch::DecodeBatch;
use crate::sampling::inputs::SamplerInputs;

/// One of the seven arrays a bucket stages, as the staging names it: the five inputs the model
/// step reads, then the two the sampler reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagedInput {
    TokenIds,
    Positions,
    KeyLengths,
    SlotMapping,
    BlockTable,
    /// The slot each selected row samples under.
    RowSlots,
    /// The slot each token row takes its token from, or negative to keep the host's.
    GatherSlots,
}

impl fmt::Display for StagedInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            StagedInput::TokenIds => "token ids",
            StagedInput::Positions => "positions",
            StagedInput::KeyLengths => "key lengths",
            StagedInput::SlotMapping => "slot mapping",
            StagedInput::BlockTable => "block table",
            StagedInput::RowSlots => "row slots",
            StagedInput::GatherSlots => "gather slots",
        })
    }
}

/// Why the inputs could not be staged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum StagingError {
    #[error("the {input} array holds {len} values; a bucket of {tokens} tokens stages {needed}")]
    ArrayTooShort {
        input: StagedInput,
        len: usize,
        tokens: usize,
        needed: usize,
    },
    #[error("the {input} array holds {len} values; {rows} rows stage this step")]
    SamplerArrayTooShort {
        input: StagedInput,
        len: usize,
        rows: usize,
    },
    #[error("{value} in the {input} does not fit the kernel's 32-bit input")]
    Overflow { input: StagedInput, value: i64 },
    #[error("position {position} is past the rotary tables, which cover {max_position} positions")]
    PositionPastTables {
        position: usize,
        max_position: usize,
    },
    #[error(
        "the packed block's base {address:#x} is not {}-byte aligned, as the slot mapping's i64 \
         values need",
        BASE_ALIGNMENT
    )]
    BlockMisaligned { address: usize },
    #[error("the packed block holds {len} bytes; a bucket of {rows} rows packs {needed}")]
    BlockTooShort {
        len: usize,
        rows: usize,
        needed: usize,
    },
    #[error(
        "a bucket of {rows} rows at block table width {width} packs more bytes than the host can \
         address"
    )]
    BlockUnaddressable { rows: usize, width: usize },
}

/// How wide the staged arrays are, and the KV geometry a row is written against: the shape
/// every bucket's inputs are carved from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagingShape {
    /// Rows the arrays hold: the largest bucket.
    pub max_tokens: usize,
    /// Columns of the block table: the blocks a sequence of the model's maximum length holds,
    /// covering whole key tiles of the attention kernel.
    pub block_table_width: usize,
    /// Positions the rotary tables cover.
    pub max_position: usize,
    /// Tokens per KV block: a block id times it is the block's first KV slot, where a padding
    /// row's token is written.
    pub block_size: TokenCount,
}

/// A bucket's rows run as padding rows, over one KV block each: what a capture check or a warmup
/// runs when there is no live batch to run. Each row is what a padding dummy's row is in a live
/// step — the padding token at position 0, a key length of one, the block's first KV slot, and a
/// block table of that one block — so the step computes what it computes for a dummy, and the
/// only cache it writes is each block's first KV slot. No row samples, so no sampler descriptor
/// runs over a dummy run and nothing is read back; its sampler arrays are staged all the same,
/// naming no request slot, so its copy-in carries nothing stale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DummyRun {
    bucket: BucketIdx,
    /// One block per row of the bucket.
    blocks: Vec<BlockId>,
}

impl DummyRun {
    /// A dummy run of `bucket` over `blocks`, one per row. That the count is the bucket's rows
    /// is checked where the bucket's layout is known, when the run is staged.
    #[must_use]
    pub fn new(bucket: BucketIdx, blocks: Vec<BlockId>) -> Self {
        Self { bucket, blocks }
    }

    #[must_use]
    pub fn bucket(&self) -> BucketIdx {
        self.bucket
    }

    /// The rows the run fills: one per block.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.blocks.len()
    }
}

/// Each array in a packed block begins at a multiple of this many bytes: the alignment a device
/// allocation of its own would have, so a kernel handed a view into the block sees what it would
/// see over a buffer of its own.
const ALIGNMENT: usize = 256;

/// What a block's base must be aligned to for the carve: the widest element staged, the slot
/// mapping's `i64`. Every offset is a multiple of [`ALIGNMENT`], so a base aligned to this aligns
/// every array.
const BASE_ALIGNMENT: usize = align_of::<i64>();

/// Where each of one bucket's seven arrays sits in its packed block, and how long the block is.
///
/// The arrays are laid consecutively at the bucket's rows, in [`StagedInput`]'s order, each
/// beginning at a multiple of [`ALIGNMENT`]; the block's length is the last array's end, padded
/// the same way. The layout is the bucket's alone: the largest bucket sizes nothing in it, so a
/// smaller bucket's block is proportionally smaller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagingLayout {
    rows: usize,
    block_table_width: usize,
    token_ids: usize,
    positions: usize,
    seqlens_k: usize,
    slot_mapping: usize,
    block_table: usize,
    row_slots: usize,
    gather_slots: usize,
    bytes: usize,
}

impl StagingLayout {
    /// The layout of a bucket of `rows` rows at `shape`'s block table width.
    ///
    /// # Errors
    ///
    /// Returns [`StagingError::BlockUnaddressable`] when the block would be longer than the host
    /// can address: an array's bytes, an array's end or its padding does not fit a `usize`.
    pub fn packed(shape: StagingShape, rows: usize) -> Result<Self, StagingError> {
        let width = shape.block_table_width;
        let unaddressable = StagingError::BlockUnaddressable { rows, width };
        let mut end = 0;
        let mut place = |row_bytes: usize| -> Result<usize, StagingError> {
            let offset = end;
            end = rows
                .checked_mul(row_bytes)
                .and_then(|bytes| bytes.checked_add(offset))
                .and_then(|array_end| array_end.checked_next_multiple_of(ALIGNMENT))
                .ok_or(unaddressable)?;
            Ok(offset)
        };
        let token_ids = place(size_of::<u32>())?;
        let positions = place(size_of::<i32>())?;
        let seqlens_k = place(size_of::<i32>())?;
        let slot_mapping = place(size_of::<i64>())?;
        let table_row_bytes = width.checked_mul(size_of::<i32>()).ok_or(unaddressable)?;
        let block_table = place(table_row_bytes)?;
        let row_slots = place(size_of::<i32>())?;
        let gather_slots = place(size_of::<i32>())?;
        Ok(Self {
            rows,
            block_table_width: width,
            token_ids,
            positions,
            seqlens_k,
            slot_mapping,
            block_table,
            row_slots,
            gather_slots,
            bytes: end,
        })
    }

    /// Rows every array holds: the bucket.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The byte offset of `input`'s array from the block's base.
    #[must_use]
    pub fn offset(&self, input: StagedInput) -> usize {
        match input {
            StagedInput::TokenIds => self.token_ids,
            StagedInput::Positions => self.positions,
            StagedInput::KeyLengths => self.seqlens_k,
            StagedInput::SlotMapping => self.slot_mapping,
            StagedInput::BlockTable => self.block_table,
            StagedInput::RowSlots => self.row_slots,
            StagedInput::GatherSlots => self.gather_slots,
        }
    }

    /// The block's length: what one copy of the bucket's staging carries.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Carves the seven arrays out of `block`, each at its offset and the bucket's rows long.
    ///
    /// # Errors
    ///
    /// Returns [`StagingError`] when the block's base is not aligned for the widest element, or
    /// the block is shorter than the layout packs.
    pub fn carve<'a>(&self, block: &'a mut [u8]) -> Result<BucketArrays<'a>, StagingError> {
        let address = block.as_ptr().addr();
        if !address.is_multiple_of(BASE_ALIGNMENT) {
            return Err(StagingError::BlockMisaligned { address });
        }
        if block.len() < self.bytes {
            return Err(StagingError::BlockTooShort {
                len: block.len(),
                rows: self.rows,
                needed: self.bytes,
            });
        }
        let rows = self.rows;
        // `packed` fit `rows * width` four-byte block table entries in a `usize`.
        let table_len = rows * self.block_table_width;
        let mut carver = Carver {
            rest: block,
            end: 0,
        };
        Ok(BucketArrays {
            inputs: StagingArrays {
                token_ids: carver.take(self.token_ids, rows),
                positions: carver.take(self.positions, rows),
                seqlens_k: carver.take(self.seqlens_k, rows),
                slot_mapping: carver.take(self.slot_mapping, rows),
                block_table: carver.take(self.block_table, table_len),
            },
            sampler: SamplerArrays {
                row_slots: carver.take(self.row_slots, rows),
                gather_slots: carver.take(self.gather_slots, rows),
            },
        })
    }
}

/// An element every byte pattern is a value of, so a carve can mint one over raw bytes.
trait Plain: Copy {}
impl Plain for u32 {}
impl Plain for i32 {}
impl Plain for i64 {}

/// Walks a block from front to back, handing out each array as a typed slice over its bytes.
struct Carver<'a> {
    /// The block past everything handed out so far.
    rest: &'a mut [u8],
    /// The block offset `rest` begins at: the end of everything handed out so far.
    end: usize,
}

impl<'a> Carver<'a> {
    /// `len` values of `T` at `offset` from the block's base. The layout put `offset` at or past
    /// the previous array's end and at a multiple of [`ALIGNMENT`], and the carve checked the
    /// base and the block's length, so the splits cannot fail.
    fn take<T: Plain>(&mut self, offset: usize, len: usize) -> &'a mut [T] {
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("the layout fit this array's bytes, and the whole block, in a usize");
        let rest = mem::take(&mut self.rest);
        let (_gap, from_offset) = rest.split_at_mut(offset - self.end);
        let (array, rest) = from_offset.split_at_mut(bytes);
        self.rest = rest;
        self.end = offset + bytes;
        debug_assert!(
            array.as_ptr().addr().is_multiple_of(align_of::<T>()),
            "an aligned base and an aligned offset align every array"
        );
        // SAFETY: `array` is exactly `len` values of `T` in bytes, at an address aligned for `T`,
        // exclusively borrowed for `'a`; every byte pattern is a `T`, as `Plain` says.
        unsafe { slice::from_raw_parts_mut(array.as_mut_ptr().cast::<T>(), len) }
    }
}

/// The model step's five arrays, each holding at least the bucket's rows.
#[derive(Debug)]
pub struct StagingArrays<'a> {
    pub token_ids: &'a mut [u32],
    /// Each token's position: its context length.
    pub positions: &'a mut [i32],
    /// Each sequence's key length after this step's token.
    pub seqlens_k: &'a mut [i32],
    pub slot_mapping: &'a mut [i64],
    /// Row-major, [`StagingShape::block_table_width`] columns per row.
    pub block_table: &'a mut [i32],
}

/// The sampler's two per-step arrays, one value per row of the bucket, as the kernels index
/// them.
#[derive(Debug)]
pub struct SamplerArrays<'a> {
    /// The slot each selected row samples under.
    pub row_slots: &'a mut [i32],
    /// The slot each token row takes its token from, or negative to keep the host's.
    pub gather_slots: &'a mut [i32],
}

/// One bucket's seven arrays over its block: the model step's and the sampler's.
#[derive(Debug)]
pub struct BucketArrays<'a> {
    pub inputs: StagingArrays<'a>,
    pub sampler: SamplerArrays<'a>,
}

/// Writes `batch`'s inputs from `layout` into the leading rows of `arrays`, the block table at
/// full width with each row zero-filled past the sequence's blocks.
///
/// # Errors
///
/// Returns [`StagingError`] when an array is shorter than the bucket, a value does not fit the
/// kernel's input, or a position is past the rotary tables.
pub fn stage(
    layout: &BatchLayout,
    batch: &DecodeBatch,
    shape: StagingShape,
    arrays: StagingArrays<'_>,
) -> Result<(), StagingError> {
    let tokens = batch.tokens;
    let width = shape.block_table_width;
    let StagingArrays {
        token_ids,
        positions,
        seqlens_k,
        slot_mapping,
        block_table,
    } = arrays;
    fits(StagedInput::TokenIds, token_ids.len(), tokens, tokens)?;
    fits(StagedInput::Positions, positions.len(), tokens, tokens)?;
    fits(StagedInput::KeyLengths, seqlens_k.len(), tokens, tokens)?;
    fits(StagedInput::SlotMapping, slot_mapping.len(), tokens, tokens)?;
    fits(
        StagedInput::BlockTable,
        block_table.len(),
        tokens,
        tokens * width,
    )?;

    token_ids[..tokens].copy_from_slice(&layout.tokens[..tokens]);
    slot_mapping[..tokens].copy_from_slice(&layout.slot_mapping[..tokens]);
    for (entry, &position) in layout.positions[..tokens].iter().enumerate() {
        let overflow = || StagingError::Overflow {
            input: StagedInput::Positions,
            value: position,
        };
        let index = usize::try_from(position).map_err(|_| overflow())?;
        if index >= shape.max_position {
            return Err(StagingError::PositionPastTables {
                position: index,
                max_position: shape.max_position,
            });
        }
        positions[entry] = i32::try_from(position).map_err(|_| overflow())?;
    }
    for (entry, sequence_len) in layout.sequence_lengths[..tokens].iter().enumerate() {
        seqlens_k[entry] = i32::try_from(*sequence_len).map_err(|_| StagingError::Overflow {
            input: StagedInput::KeyLengths,
            value: i64::from(*sequence_len),
        })?;
    }
    let laid_out = layout.block_table_width;
    for entry in 0..tokens {
        let row = &mut block_table[entry * width..(entry + 1) * width];
        let blocks = &layout.block_tables[entry * laid_out..(entry + 1) * laid_out];
        for (slot, block) in row.iter_mut().zip(blocks) {
            *slot = i32::try_from(*block).map_err(|_| StagingError::Overflow {
                input: StagedInput::BlockTable,
                value: i64::from(*block),
            })?;
        }
        row[laid_out..].fill(0);
    }
    Ok(())
}

/// Writes `inputs`' two arrays into the leading rows of `arrays`, as the kernels index them: the
/// slot of every selected row, and for every covered token row the slot it takes its token from,
/// or a negative value where the host's token stands.
///
/// # Errors
///
/// Returns [`StagingError`] when an array is shorter than the rows it stages, or a slot does not
/// fit the kernel's index.
pub fn stage_sampler(
    inputs: &SamplerInputs,
    arrays: SamplerArrays<'_>,
) -> Result<(), StagingError> {
    let SamplerArrays {
        row_slots,
        gather_slots,
    } = arrays;
    holds(
        StagedInput::RowSlots,
        row_slots.len(),
        inputs.row_slots.len(),
    )?;
    holds(
        StagedInput::GatherSlots,
        gather_slots.len(),
        inputs.gather.len(),
    )?;
    for (staged, slot) in row_slots.iter_mut().zip(&inputs.row_slots) {
        *staged = slot_index(StagedInput::RowSlots, *slot)?;
    }
    for (staged, slot) in gather_slots.iter_mut().zip(&inputs.gather) {
        *staged = match slot {
            Some(slot) => slot_index(StagedInput::GatherSlots, *slot)?,
            None => KEEP_HOST_TOKEN,
        };
    }
    Ok(())
}

/// Writes `run`'s rows into the leading rows of `arrays` as padding rows: the padding token at
/// position 0 with a key length of one, the row's block's first KV slot as its slot, and the
/// block alone in its block table row, zero past it. The sampler's two arrays name no request
/// slot for any row: the gather keeps the host's padding token, as it does for a live step's
/// padding rows, and nothing samples.
///
/// # Errors
///
/// Returns [`StagingError`] when an array is shorter than the run's rows, or a block id does not
/// fit the kernel's input.
pub fn stage_dummy(
    run: &DummyRun,
    shape: StagingShape,
    arrays: BucketArrays<'_>,
) -> Result<(), StagingError> {
    let rows = run.rows();
    let width = shape.block_table_width;
    let BucketArrays {
        inputs:
            StagingArrays {
                token_ids,
                positions,
                seqlens_k,
                slot_mapping,
                block_table,
            },
        sampler: SamplerArrays {
            row_slots,
            gather_slots,
        },
    } = arrays;
    fits(StagedInput::TokenIds, token_ids.len(), rows, rows)?;
    fits(StagedInput::Positions, positions.len(), rows, rows)?;
    fits(StagedInput::KeyLengths, seqlens_k.len(), rows, rows)?;
    fits(StagedInput::SlotMapping, slot_mapping.len(), rows, rows)?;
    fits(
        StagedInput::BlockTable,
        block_table.len(),
        rows,
        rows * width,
    )?;
    holds(StagedInput::RowSlots, row_slots.len(), rows)?;
    holds(StagedInput::GatherSlots, gather_slots.len(), rows)?;

    token_ids[..rows].fill(PADDING_TOKEN);
    positions[..rows].fill(0);
    seqlens_k[..rows].fill(1);
    for (row, &block) in run.blocks.iter().enumerate() {
        slot_mapping[row] = first_slot(block, shape.block_size);
        // The width covers a sequence of the model's maximum length, so a row has a first cell.
        let table_row = &mut block_table[row * width..(row + 1) * width];
        table_row[0] = i32::try_from(block.get()).map_err(|_| StagingError::Overflow {
            input: StagedInput::BlockTable,
            value: i64::from(block.get()),
        })?;
        table_row[1..].fill(0);
    }
    row_slots[..rows].fill(NO_REQUEST_SLOT);
    gather_slots[..rows].fill(KEEP_HOST_TOKEN);
    Ok(())
}

/// The first KV slot of `block`: where a padding row's token is written, derived as the batch
/// layout derives every slot.
fn first_slot(block: BlockId, block_size: TokenCount) -> i64 {
    // A block id is 32 bits and a block size is a token count, so the product fits.
    i64::try_from(block.index() * block_size.get()).expect("a block's first KV slot fits i64")
}

/// The gather slot of a row whose token the host's upload serves: negative, as the kernel reads
/// it.
const KEEP_HOST_TOKEN: i32 = -1;

/// The row slot of a row that does not sample: negative, the value no request slot has.
const NO_REQUEST_SLOT: i32 = -1;

/// `slot` as the kernels index a per-slot array.
fn slot_index(input: StagedInput, slot: RequestSlot) -> Result<i32, StagingError> {
    i32::try_from(slot.get()).map_err(|_| StagingError::Overflow {
        input,
        value: i64::from(slot.get()),
    })
}

/// Holds a sampler array of `len` values to the `rows` the step stages in it.
fn holds(input: StagedInput, len: usize, rows: usize) -> Result<(), StagingError> {
    if len < rows {
        return Err(StagingError::SamplerArrayTooShort { input, len, rows });
    }
    Ok(())
}

/// Holds an array of `len` values to the `needed` a bucket of `tokens` stages.
fn fits(input: StagedInput, len: usize, tokens: usize, needed: usize) -> Result<(), StagingError> {
    if len < needed {
        return Err(StagingError::ArrayTooShort {
            input,
            len,
            tokens,
            needed,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::slice;

    use atoma_core::dispatch::DispatchDecision;
    use atoma_core::step::CommandEntry;
    use atoma_core::types::RequestSlot;

    use super::*;
    use crate::decode::batch::{Checked, DecodeBuckets};
    use crate::test_support::{command, dummy, engine_config, entry, keyed_command, BLOCK_SIZE};

    const MAX_TOKENS: usize = 4;
    const WIDTH: usize = 8;

    fn shape() -> StagingShape {
        StagingShape {
            max_tokens: MAX_TOKENS,
            block_table_width: WIDTH,
            max_position: 32,
            block_size: BLOCK_SIZE,
        }
    }

    /// Storage for a packed block: `u64` words, so the base is aligned as the carve requires,
    /// with every byte set to `0x5A`, which no input stages, so an untouched value reads as
    /// `0x5A5A_5A5A` and never as the `-1` the sampler's arrays carry.
    struct Block(Vec<u64>);

    impl Block {
        fn sized(bytes: usize) -> Self {
            let words = bytes.div_ceil(size_of::<u64>());
            Self(vec![u64::from_ne_bytes([0x5A; 8]); words])
        }

        fn bytes(&mut self) -> &mut [u8] {
            let len = self.0.len() * size_of::<u64>();
            // SAFETY: the words are allocated and exclusively borrowed for as long as the bytes
            // are, and every byte of a `u64` is a `u8`.
            unsafe { slice::from_raw_parts_mut(self.0.as_mut_ptr().cast::<u8>(), len) }
        }
    }

    fn routed(live: Vec<CommandEntry>) -> (BatchLayout, DecodeBatch) {
        let layout = BatchLayout::lay_out(&keyed_command(live), BLOCK_SIZE).unwrap();
        let DispatchDecision::FullReplay(key) = layout.dispatch else {
            panic!("keyed: {:?}", layout.dispatch);
        };
        let buckets = DecodeBuckets::usable(&engine_config().dispatch);
        let Checked::Step(batch) = DecodeBatch::check(&layout, key, &buckets, WIDTH).unwrap()
        else {
            panic!("served by the decode step");
        };
        (layout, batch)
    }

    #[test]
    fn a_buckets_seven_arrays_are_packed_in_order_each_at_a_256_byte_boundary() {
        let wide = StagingShape {
            max_tokens: 128,
            block_table_width: 64,
            ..shape()
        };
        let staging = StagingLayout::packed(wide, 100).unwrap();

        // 100 rows of u32 or i32 are 400 bytes, padded to 512; 100 i64s are 800, padded to 1024;
        // the block table's 100 * 64 * 4 = 25600 bytes are already a multiple of 256.
        assert_eq!(staging.rows(), 100);
        assert_eq!(staging.offset(StagedInput::TokenIds), 0);
        assert_eq!(staging.offset(StagedInput::Positions), 512);
        assert_eq!(staging.offset(StagedInput::KeyLengths), 1024);
        assert_eq!(staging.offset(StagedInput::SlotMapping), 1536);
        assert_eq!(staging.offset(StagedInput::BlockTable), 2560);
        assert_eq!(staging.offset(StagedInput::RowSlots), 28160);
        assert_eq!(staging.offset(StagedInput::GatherSlots), 28672);
        assert_eq!(staging.bytes(), 29184);

        // One row of anything is under 256 bytes, so every array takes one alignment.
        let one = StagingLayout::packed(shape(), 1).unwrap();
        assert_eq!(one.offset(StagedInput::BlockTable), 1024);
        assert_eq!(one.offset(StagedInput::GatherSlots), 1536);
        assert_eq!(one.bytes(), 1792);
    }

    #[test]
    fn the_packed_size_follows_the_bucket_and_not_the_largest() {
        let wide = StagingShape {
            max_tokens: 32,
            block_table_width: 64,
            ..shape()
        };
        let widest = StagingShape {
            max_tokens: 4096,
            ..wide
        };

        // Bucket 8: 32-byte rows pad to 256 each, the 64-wide table is 8 * 256 = 2048 bytes.
        assert_eq!(StagingLayout::packed(wide, 8).unwrap().bytes(), 3584);
        // Bucket 32: 128-byte rows pad to 256, the slot mapping's 256 fit exactly, the table is
        // 32 * 256 = 8192.
        assert_eq!(StagingLayout::packed(wide, 32).unwrap().bytes(), 9728);
        assert_eq!(
            StagingLayout::packed(widest, 8).unwrap(),
            StagingLayout::packed(wide, 8).unwrap(),
            "the largest bucket sizes nothing in a bucket's layout"
        );
    }

    #[test]
    fn the_carved_arrays_are_the_buckets_rows_at_the_layouts_offsets_and_nothing_else() {
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes());
        let BucketArrays { inputs, sampler } = staging.carve(block.bytes()).unwrap();

        assert_eq!(inputs.token_ids.len(), 4);
        assert_eq!(inputs.positions.len(), 4);
        assert_eq!(inputs.seqlens_k.len(), 4);
        assert_eq!(inputs.slot_mapping.len(), 4);
        assert_eq!(inputs.block_table.len(), 4 * WIDTH);
        assert_eq!(sampler.row_slots.len(), 4);
        assert_eq!(sampler.gather_slots.len(), 4);

        inputs.token_ids[3] = 0x1111_1111;
        inputs.positions[3] = 0x2222_2222;
        inputs.seqlens_k[3] = 0x3333_3333;
        inputs.slot_mapping[3] = 0x4444_4444_4444_4444;
        inputs.block_table[4 * WIDTH - 1] = 0x5555_5555;
        sampler.row_slots[3] = 0x6666_6666;
        sampler.gather_slots[3] = 0x7777_7777;

        // Each array's last value lands at its offset plus three rows (31 for the table), at its
        // element's width; every other byte of the block is as the storage set it.
        let mut expected = vec![0x5A; 1792];
        expected[12..16].fill(0x11);
        expected[256 + 12..256 + 16].fill(0x22);
        expected[512 + 12..512 + 16].fill(0x33);
        expected[768 + 24..768 + 32].fill(0x44);
        expected[1024 + 124..1024 + 128].fill(0x55);
        expected[1280 + 12..1280 + 16].fill(0x66);
        expected[1536 + 12..1536 + 16].fill(0x77);
        assert_eq!(block.bytes(), expected);
    }

    #[test]
    fn a_block_whose_base_is_not_aligned_for_the_slot_mapping_is_refused() {
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes() + 8);
        let bytes = block.bytes();
        let address = bytes.as_ptr().addr() + 4;

        assert_eq!(
            staging.carve(&mut bytes[4..]).unwrap_err(),
            StagingError::BlockMisaligned { address }
        );
    }

    #[test]
    fn a_block_shorter_than_the_layout_packs_is_refused_with_the_bytes_it_needs() {
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes());

        assert_eq!(
            staging.carve(&mut block.bytes()[..1791]).unwrap_err(),
            StagingError::BlockTooShort {
                len: 1791,
                rows: 4,
                needed: 1792
            }
        );
        assert!(
            staging.carve(&mut block.bytes()[..1792]).is_ok(),
            "a block of exactly the layout's length carves"
        );
    }

    #[test]
    fn a_bucket_whose_packed_block_the_host_cannot_address_is_refused() {
        let cases = [
            // 2^62 rows of four-byte token ids are 2^64 bytes: one past a usize.
            (shape(), 1 << 62),
            // 2^59 - 1 rows: every array fits a usize on its own, but the four before the block
            // table end at 5 * 2^61 after padding and the table is 2^64 - 32 bytes long.
            (shape(), (1 << 59) - 1),
            // At three columns the table ends at 2^64 - 12 instead; padding it to 256 goes past.
            (
                StagingShape {
                    block_table_width: 3,
                    ..shape()
                },
                (1 << 59) - 1,
            ),
            // One row of a 2^62-column block table is 2^64 bytes.
            (
                StagingShape {
                    block_table_width: 1 << 62,
                    ..shape()
                },
                1,
            ),
        ];
        for (shape, rows) in cases {
            let width = shape.block_table_width;
            assert_eq!(
                StagingLayout::packed(shape, rows).unwrap_err(),
                StagingError::BlockUnaddressable { rows, width },
                "{rows} rows at width {width}"
            );
        }
    }

    #[test]
    fn a_padded_batch_stages_its_leading_rows_and_the_block_table_at_full_width() {
        let (layout, batch) = routed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 8, vec![7], &[20, 21, 22], true),
            entry(3, 1, vec![5], &[30], true),
        ]);
        assert_eq!(batch.tokens, 4, "three live entries and one dummy");
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes());

        let BucketArrays { inputs, .. } = staging.carve(block.bytes()).unwrap();
        stage(&layout, &batch, shape(), inputs).unwrap();

        let BucketArrays { inputs, sampler } = staging.carve(block.bytes()).unwrap();
        assert_eq!(inputs.token_ids[..3], [9, 7, 5]);
        assert_eq!(inputs.positions, [3, 8, 1, 0]);
        assert_eq!(inputs.seqlens_k, [4, 9, 2, 1]);
        assert_eq!(
            inputs.slot_mapping,
            [43, 88, 121, layout.slot_mapping[3]],
            "block times block size plus offset; the dummy's is its own block's first slot"
        );
        let rows: Vec<&[i32]> = inputs.block_table.chunks(WIDTH).collect();
        assert_eq!(rows[0], [10, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(rows[1], [20, 21, 22, 0, 0, 0, 0, 0]);
        assert_eq!(rows[2], [30, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            rows[3][1..],
            [0; WIDTH - 1],
            "the dummy's row is its block, then zero"
        );
        assert_eq!(
            sampler.row_slots, [0x5A5A_5A5A; 4],
            "the sampler's arrays are the sampler's to write"
        );
        assert_eq!(sampler.gather_slots, [0x5A5A_5A5A; 4]);
    }

    #[test]
    fn a_smaller_bucket_leaves_the_rows_past_it_untouched() {
        let (layout, batch) = routed(vec![entry(1, 3, vec![9], &[10], true)]);
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes());

        let BucketArrays { inputs, .. } = staging.carve(block.bytes()).unwrap();
        stage(&layout, &batch, shape(), inputs).unwrap();

        let BucketArrays { inputs, .. } = staging.carve(block.bytes()).unwrap();
        assert_eq!(inputs.token_ids, [9, 0x5A5A_5A5A, 0x5A5A_5A5A, 0x5A5A_5A5A]);
        assert_eq!(inputs.positions, [3, 0x5A5A_5A5A, 0x5A5A_5A5A, 0x5A5A_5A5A]);
        assert_eq!(inputs.block_table[WIDTH..], vec![0x5A5A_5A5A; 3 * WIDTH]);
    }

    #[test]
    fn an_array_shorter_than_the_bucket_is_refused_by_name() {
        let (layout, batch) = routed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 3, vec![9], &[20], true),
        ]);
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes());
        let BucketArrays { inputs, .. } = staging.carve(block.bytes()).unwrap();
        let mut short = vec![0; 1];
        let arrays = StagingArrays {
            seqlens_k: &mut short,
            ..inputs
        };

        assert_eq!(
            stage(&layout, &batch, shape(), arrays).unwrap_err(),
            StagingError::ArrayTooShort {
                input: StagedInput::KeyLengths,
                len: 1,
                tokens: 2,
                needed: 2
            }
        );
    }

    #[test]
    fn a_position_the_rotary_tables_do_not_cover_is_refused() {
        let (layout, batch) = routed(vec![entry(1, 3, vec![9], &[10], true)]);
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes());
        let BucketArrays { inputs, .. } = staging.carve(block.bytes()).unwrap();
        let short_tables = StagingShape {
            max_position: 3,
            ..shape()
        };

        assert_eq!(
            stage(&layout, &batch, short_tables, inputs).unwrap_err(),
            StagingError::PositionPastTables {
                position: 3,
                max_position: 3
            }
        );
    }

    #[test]
    fn a_block_id_past_the_kernels_input_width_is_refused() {
        let (mut layout, batch) = routed(vec![entry(1, 3, vec![9], &[10], true)]);
        layout.block_tables[0] = u32::MAX;
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes());
        let BucketArrays { inputs, .. } = staging.carve(block.bytes()).unwrap();

        assert_eq!(
            stage(&layout, &batch, shape(), inputs).unwrap_err(),
            StagingError::Overflow {
                input: StagedInput::BlockTable,
                value: i64::from(u32::MAX)
            }
        );
    }

    /// A dummy run of the bucket of four over `blocks`.
    fn dummy_run(blocks: [u32; 4]) -> DummyRun {
        DummyRun::new(BucketIdx(2), blocks.map(BlockId::new).to_vec())
    }

    #[test]
    fn a_dummy_run_stages_every_row_as_a_padding_row_over_its_own_block() {
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes());

        let arrays = staging.carve(block.bytes()).unwrap();
        stage_dummy(&dummy_run([10, 20, 21, 22]), shape(), arrays).unwrap();

        let BucketArrays { inputs, sampler } = staging.carve(block.bytes()).unwrap();
        assert_eq!(inputs.token_ids, [PADDING_TOKEN; 4]);
        assert_eq!(inputs.positions, [0; 4]);
        assert_eq!(
            inputs.seqlens_k, [1; 4],
            "the one token, at position 0, is the key"
        );
        assert_eq!(
            inputs.slot_mapping,
            [40, 80, 84, 88],
            "four-token blocks: a block's first slot is four times its id"
        );
        let rows: Vec<&[i32]> = inputs.block_table.chunks(WIDTH).collect();
        assert_eq!(rows[0], [10, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(rows[1], [20, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(rows[2], [21, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(rows[3], [22, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(sampler.row_slots, [-1; 4], "no row samples");
        assert_eq!(
            sampler.gather_slots, [-1; 4],
            "every row keeps the host's padding token"
        );
    }

    #[test]
    fn a_dummy_runs_rows_are_what_a_live_step_stages_for_its_padding_rows() {
        // One live decode and three padding dummies over blocks 101, 102 and 103, laid out as
        // the engine lays a padded step out; the batch is what the check yields for it.
        let padded = command(
            vec![
                entry(1, 3, vec![9], &[10], true),
                dummy(2, 101),
                dummy(3, 102),
                dummy(4, 103),
            ],
            3,
        );
        let layout = BatchLayout::lay_out(&padded, BLOCK_SIZE).unwrap();
        let batch = DecodeBatch {
            bucket: BucketIdx(2),
            tokens: 4,
            live: 1,
        };
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut live = Block::sized(staging.bytes());
        let BucketArrays { inputs, sampler } = staging.carve(live.bytes()).unwrap();
        stage(&layout, &batch, shape(), inputs).unwrap();
        stage_sampler(&sampler_inputs(&[1], &[None; 4]), sampler).unwrap();
        let mut dummies = Block::sized(staging.bytes());
        let arrays = staging.carve(dummies.bytes()).unwrap();
        stage_dummy(&dummy_run([10, 101, 102, 103]), shape(), arrays).unwrap();

        // Rows 1..4 are padding rows on both sides; row 0 is the live decode on one side only.
        let live = staging.carve(live.bytes()).unwrap();
        let dummies = staging.carve(dummies.bytes()).unwrap();
        assert_eq!(live.inputs.token_ids[1..], dummies.inputs.token_ids[1..]);
        assert_eq!(live.inputs.positions[1..], dummies.inputs.positions[1..]);
        assert_eq!(live.inputs.seqlens_k[1..], dummies.inputs.seqlens_k[1..]);
        assert_eq!(
            live.inputs.slot_mapping[1..],
            dummies.inputs.slot_mapping[1..]
        );
        assert_eq!(
            live.inputs.block_table[WIDTH..],
            dummies.inputs.block_table[WIDTH..]
        );
        assert_eq!(
            live.sampler.gather_slots[1..],
            dummies.sampler.gather_slots[1..],
            "a padding row keeps the host's token on both sides"
        );
        assert_ne!(
            live.inputs.token_ids[0], dummies.inputs.token_ids[0],
            "the live row is a decode, the dummy run's row 0 a padding row"
        );
        assert_eq!(
            dummies.sampler.row_slots, [-1; 4],
            "the live step leaves a padding row's row slot; a dummy run names no request slot"
        );
    }

    #[test]
    fn a_dummy_run_over_an_array_shorter_than_its_rows_is_refused_by_name() {
        let run = DummyRun::new(BucketIdx(1), vec![BlockId::new(10), BlockId::new(20)]);
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes());
        let mut short = [0; 1];

        let BucketArrays { inputs, sampler } = staging.carve(block.bytes()).unwrap();
        let arrays = BucketArrays {
            inputs: StagingArrays {
                positions: &mut short,
                ..inputs
            },
            sampler,
        };
        assert_eq!(
            stage_dummy(&run, shape(), arrays).unwrap_err(),
            StagingError::ArrayTooShort {
                input: StagedInput::Positions,
                len: 1,
                tokens: 2,
                needed: 2
            }
        );

        let BucketArrays { inputs, sampler } = staging.carve(block.bytes()).unwrap();
        let arrays = BucketArrays {
            inputs,
            sampler: SamplerArrays {
                gather_slots: &mut short,
                ..sampler
            },
        };
        assert_eq!(
            stage_dummy(&run, shape(), arrays).unwrap_err(),
            StagingError::SamplerArrayTooShort {
                input: StagedInput::GatherSlots,
                len: 1,
                rows: 2
            }
        );
    }

    #[test]
    fn a_dummy_runs_block_past_the_kernels_input_width_is_refused() {
        let run = DummyRun::new(BucketIdx(0), vec![BlockId::new(u32::MAX)]);
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes());
        let arrays = staging.carve(block.bytes()).unwrap();

        assert_eq!(
            stage_dummy(&run, shape(), arrays).unwrap_err(),
            StagingError::Overflow {
                input: StagedInput::BlockTable,
                value: i64::from(u32::MAX)
            }
        );
    }

    fn sampler_inputs(row_slots: &[u32], gather: &[Option<u32>]) -> SamplerInputs {
        SamplerInputs {
            records: Vec::new(),
            row_slots: row_slots.iter().copied().map(RequestSlot::new).collect(),
            gather: gather
                .iter()
                .map(|slot| slot.map(RequestSlot::new))
                .collect(),
        }
    }

    #[test]
    fn the_samplers_two_arrays_are_written_as_the_kernels_index_them_and_rows_past_the_step_kept() {
        let staging = StagingLayout::packed(shape(), MAX_TOKENS).unwrap();
        let mut block = Block::sized(staging.bytes());
        let BucketArrays { sampler, .. } = staging.carve(block.bytes()).unwrap();
        sampler.row_slots.fill(7);
        sampler.gather_slots.fill(7);

        let BucketArrays { sampler, .. } = staging.carve(block.bytes()).unwrap();
        stage_sampler(&sampler_inputs(&[3, 5], &[Some(3), None, Some(5)]), sampler).unwrap();

        let BucketArrays { sampler, .. } = staging.carve(block.bytes()).unwrap();
        assert_eq!(sampler.row_slots, [3, 5, 7, 7], "one slot per selected row");
        assert_eq!(
            sampler.gather_slots,
            [3, -1, 5, 7],
            "one slot per covered token row, negative where the host's token stands"
        );
    }

    #[test]
    fn an_eager_step_that_gathers_nothing_needs_no_gather_array() {
        let mut row_slots = [7; 2];
        let arrays = SamplerArrays {
            row_slots: &mut row_slots,
            gather_slots: &mut [],
        };

        stage_sampler(&sampler_inputs(&[4, 2], &[]), arrays).unwrap();

        assert_eq!(row_slots, [4, 2]);
    }

    #[test]
    fn a_sampler_array_shorter_than_the_step_is_refused_by_name() {
        let inputs = sampler_inputs(&[1, 2, 3], &[Some(1), Some(2), Some(3)]);
        let mut short = [0; 2];
        let mut long = [0; 4];

        assert_eq!(
            stage_sampler(
                &inputs,
                SamplerArrays {
                    row_slots: &mut short,
                    gather_slots: &mut long,
                }
            )
            .unwrap_err(),
            StagingError::SamplerArrayTooShort {
                input: StagedInput::RowSlots,
                len: 2,
                rows: 3
            }
        );
        assert_eq!(
            stage_sampler(
                &inputs,
                SamplerArrays {
                    row_slots: &mut long,
                    gather_slots: &mut short,
                }
            )
            .unwrap_err(),
            StagingError::SamplerArrayTooShort {
                input: StagedInput::GatherSlots,
                len: 2,
                rows: 3
            }
        );
        let mut exact = [0; 3];
        assert!(
            stage_sampler(
                &inputs,
                SamplerArrays {
                    row_slots: &mut exact,
                    gather_slots: &mut [0; 3],
                }
            )
            .is_ok(),
            "arrays of exactly the step's rows stage"
        );
    }

    #[test]
    fn a_slot_past_the_kernels_index_is_refused_by_name() {
        let mut row_slots = [0; 2];
        let mut gather_slots = [0; 2];
        let overflow = i64::from(u32::MAX);

        assert_eq!(
            stage_sampler(
                &sampler_inputs(&[1, u32::MAX], &[]),
                SamplerArrays {
                    row_slots: &mut row_slots,
                    gather_slots: &mut gather_slots,
                }
            )
            .unwrap_err(),
            StagingError::Overflow {
                input: StagedInput::RowSlots,
                value: overflow
            }
        );
        assert_eq!(
            stage_sampler(
                &sampler_inputs(&[1], &[None, Some(u32::MAX)]),
                SamplerArrays {
                    row_slots: &mut row_slots,
                    gather_slots: &mut gather_slots,
                }
            )
            .unwrap_err(),
            StagingError::Overflow {
                input: StagedInput::GatherSlots,
                value: overflow
            }
        );
    }
}
