//! One step's inputs on their way to the device: the staging ring's pinned blocks, the one
//! device block every bucket's views are minted over, and the descriptors that carry a step from
//! one to the other.
//!
//! A bucket's seven arrays are packed into one block, as [`StagingLayout`] lays them out: the
//! five the model step reads and the two the sampler reads. The staging ring holds `depth`
//! pinned blocks, one fence each, and the device holds one block, all of the largest bucket's
//! packed length. Before a step the host acquires a staging entry and writes the bucket's arrays
//! into the staging entry's pinned block at the bucket's offsets; the copy-in descriptor then
//! copies the bucket's packed length to the device block in one asynchronous copy and signals
//! the staging entry's fence behind it. Every bucket reads the device block through views minted
//! once, in the Allocation session phase, at the bucket's own offsets, so no address a step
//! reads follows the batch. A dummy run is staged the same way, every row a padding row, and
//! copied in through the same staging entry and fence.
//!
//! Reuse of a pinned block is what the fence guards: the staging ring hands a staging entry out
//! again only once the copy that last read its block has finished, and the blocks are freed only
//! once every fence is passed.
//!
//! [`WaitEvent`] orders the decode step after candle's stream: a prefill runs there and writes
//! the cache, and the step must not read it before those writes have landed.

use std::iter;
use std::sync::Arc;

use atoma_models::llama::slots::BucketInputs;
use atoma_runtime::arena::BucketIdx;
use atoma_runtime::error::RuntimeError;
use atoma_runtime::fence::{FenceSignal, StagingFence};
use atoma_runtime::session::{Allocation, Descriptor};
use atoma_runtime::tensor::{Dtype, Layout, Tensor, TensorError};
use cudarc::driver::result::{memcpy_htod_async, stream};
use cudarc::driver::sys::{self, CUevent_wait_flags};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr};
use thiserror::Error;
use tracing::warn;

use crate::batch::BatchLayout;
use crate::decode::batch::{DecodeBatch, DecodeBuckets};
use crate::decode::ring::{StagingDepth, StagingEntry, StagingRing};
use crate::decode::staging::{
    stage, stage_dummy, BucketArrays, DummyRun, SamplerArrays, StagedInput, StagingError,
    StagingLayout, StagingShape,
};
use crate::pinned::Pinned;

/// Why the inputs could not be allocated, staged or copied in.
#[derive(Debug, Error)]
pub enum InputsError {
    /// A batch of a bucket the inputs were not built for: the engine and the executor disagree
    /// on the bucket ladder.
    #[error("bucket {} is past the {buckets} buckets the inputs stage for", bucket.0)]
    UnknownBucket { bucket: BucketIdx, buckets: usize },
    /// No bucket to stage for, so nothing sizes the blocks.
    #[error("no bucket is usable; the inputs stage for at least one")]
    NoBucket,
    /// A dummy run with other than one block per row of its bucket.
    #[error(
        "a dummy run of bucket {} names {blocks} blocks; the bucket has {rows} rows, one block \
         each",
        bucket.0
    )]
    DummyRunNotBucket {
        bucket: BucketIdx,
        rows: usize,
        blocks: usize,
    },
    #[error(transparent)]
    Driver(#[from] RuntimeError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Staging(#[from] StagingError),
}

/// One bucket's views over the device block, each at the bucket's packed offset: the model
/// step's five inputs, and the sampler's two per-step arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketViews {
    pub inputs: BucketInputs,
    /// i32 `[tokens]`: the slot each selected row samples under.
    pub row_slots: Tensor,
    /// i32 `[tokens]`: the slot each token row takes its token from, or negative to keep the
    /// host's.
    pub gather_slots: Tensor,
}

impl BucketViews {
    /// The views of the bucket `packed` lays out, over the block at `base`: each array's view
    /// is minted by `mint` at the array's offset, with the array's dtype at the bucket's rows and
    /// the block table `width` wide.
    fn minted(
        base: u64,
        packed: &StagingLayout,
        width: usize,
        mint: impl Fn(u64, Layout) -> Result<Tensor, TensorError>,
    ) -> Result<Self, TensorError> {
        let rows = packed.rows();
        let view = |input: StagedInput, dims: &[usize], dtype: Dtype| {
            // The block was allocated at `base` for every offset the layout places, so the sum is
            // an address inside it.
            let address = base + packed.offset(input) as u64;
            mint(address, Layout::contiguous(dims, dtype)?)
        };
        Ok(Self {
            inputs: BucketInputs {
                token_ids: view(StagedInput::TokenIds, &[rows], Dtype::U32)?,
                positions: view(StagedInput::Positions, &[rows], Dtype::I32)?,
                seqlens_k: view(StagedInput::KeyLengths, &[rows], Dtype::I32)?,
                slot_mapping: view(StagedInput::SlotMapping, &[rows], Dtype::I64)?,
                block_table: view(StagedInput::BlockTable, &[rows, width], Dtype::I32)?,
            },
            row_slots: view(StagedInput::RowSlots, &[rows], Dtype::I32)?,
            gather_slots: view(StagedInput::GatherSlots, &[rows], Dtype::I32)?,
        })
    }
}

/// Every bucket's packed layout at one shape, and the length of a block that holds any of them:
/// the largest bucket's. The host side of the inputs, which needs no device.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PackedBuckets {
    shape: StagingShape,
    /// One layout per bucket, in bucket order.
    layouts: Vec<StagingLayout>,
    block_bytes: usize,
}

impl PackedBuckets {
    fn new(shape: StagingShape, buckets: &DecodeBuckets) -> Result<Self, InputsError> {
        let layouts = buckets
            .tokens()
            .iter()
            .map(|&rows| StagingLayout::packed(shape, rows))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(block_bytes) = layouts.iter().map(StagingLayout::bytes).max() else {
            return Err(InputsError::NoBucket);
        };
        Ok(Self {
            shape,
            layouts,
            block_bytes,
        })
    }

    fn layout(&self, bucket: BucketIdx) -> Result<&StagingLayout, InputsError> {
        self.layouts
            .get(bucket.0)
            .ok_or(InputsError::UnknownBucket {
                bucket,
                buckets: self.layouts.len(),
            })
    }

    /// Writes `batch`'s inputs from `layout` into `block` at its bucket's offsets, and hands
    /// back the sampler's two arrays, carved from the same block, for the sampler to write.
    fn stage<'a>(
        &self,
        block: &'a mut [u8],
        layout: &BatchLayout,
        batch: &DecodeBatch,
    ) -> Result<SamplerArrays<'a>, InputsError> {
        let BucketArrays { inputs, sampler } = self.layout(batch.bucket)?.carve(block)?;
        stage(layout, batch, self.shape, inputs)?;
        Ok(sampler)
    }

    /// Writes `run`'s rows into `block` at its bucket's offsets as padding rows, the sampler's
    /// two arrays included, once the run names one block per row of the bucket.
    fn stage_dummy(&self, block: &mut [u8], run: &DummyRun) -> Result<(), InputsError> {
        let packed = self.layout(run.bucket())?;
        if run.rows() != packed.rows() {
            return Err(InputsError::DummyRunNotBucket {
                bucket: run.bucket(),
                rows: packed.rows(),
                blocks: run.rows(),
            });
        }
        Ok(stage_dummy(run, self.shape, packed.carve(block)?)?)
    }

    /// The leading bytes of `block` one copy of `bucket` carries: the bucket's packed length.
    fn staged<'a>(&self, block: &'a [u8], bucket: BucketIdx) -> Result<&'a [u8], InputsError> {
        let packed = self.layout(bucket)?;
        block
            .get(..packed.bytes())
            .ok_or(InputsError::Staging(StagingError::BlockTooShort {
                len: block.len(),
                rows: packed.rows(),
                needed: packed.bytes(),
            }))
    }
}

/// One step's inputs: the staging ring's pinned blocks, the device block, and every bucket's
/// views over it, all of the largest bucket's packed length.
pub struct DecodeInputs {
    packed: PackedBuckets,
    /// One bucket's views each, in bucket order.
    views: Vec<BucketViews>,
    ring: StagingRing<StagingFence>,
    /// One pinned block per staging entry, indexed by the staging entry.
    blocks: Vec<Pinned<u8>>,
    /// Owned here so the address every view names stays allocated for as long as the views do,
    /// and read again for the debug check that the views still name it.
    device: CudaSlice<u8>,
    device_address: u64,
}

impl DecodeInputs {
    /// Allocates a staging ring of `depth` pinned blocks and the device block for `buckets` at
    /// `shape`, and mints every bucket's views, during the Allocation session phase, on
    /// `stream`'s device.
    ///
    /// # Errors
    ///
    /// Returns [`InputsError`] when there is no bucket, a bucket's block is longer than the host
    /// can address, the driver cannot pin or allocate a block or create a fence, or a view cannot
    /// be minted.
    pub fn new(
        allocation: &Allocation,
        stream: &Arc<CudaStream>,
        shape: StagingShape,
        buckets: &DecodeBuckets,
        depth: StagingDepth,
    ) -> Result<Self, InputsError> {
        let packed = PackedBuckets::new(shape, buckets)?;
        let context = stream.context();
        context.bind_to_thread().map_err(RuntimeError::from)?;
        let ring = StagingRing::new(depth, || StagingFence::new(context))?;
        let blocks = iter::repeat_with(|| Pinned::zeroed(packed.block_bytes))
            .take(depth.get())
            .collect::<Result<Vec<_>, _>>()?;
        let device = stream
            .alloc_zeros::<u8>(packed.block_bytes)
            .map_err(RuntimeError::from)?;
        // The address is read before the block moves into this value, and the read guard is
        // dropped with it; device allocations do not move.
        let device_address = {
            let (address, _reads) = device.device_ptr(stream);
            address
        };
        let views = packed
            .layouts
            .iter()
            .map(|layout| {
                BucketViews::minted(
                    device_address,
                    layout,
                    shape.block_table_width,
                    |at, view| Tensor::new(allocation, at, view),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            packed,
            views,
            ring,
            blocks,
            device,
            device_address,
        })
    }

    /// The staging shape the inputs were built at: what every bucket's arrays are staged at and
    /// their views minted against.
    #[must_use]
    pub fn shape(&self) -> StagingShape {
        self.packed.shape
    }

    /// The device block every bucket's views are minted over. The block's current address is
    /// read from here rather than from the copy the views were minted at, which is what the
    /// debug check compares; its bytes are what a readback of a copy-in copies out.
    #[must_use]
    pub fn device_block(&self) -> &CudaSlice<u8> {
        &self.device
    }

    /// The views `bucket`'s step reads the device block through.
    ///
    /// # Errors
    ///
    /// Returns [`InputsError::UnknownBucket`] when the inputs were not built for `bucket`.
    pub fn bucket(&self, bucket: BucketIdx) -> Result<&BucketViews, InputsError> {
        self.views.get(bucket.0).ok_or(InputsError::UnknownBucket {
            bucket,
            buckets: self.views.len(),
        })
    }

    /// A staging entry whose pinned block the host may write: waits, blocking, until the copy
    /// that last read the block has finished.
    ///
    /// # Errors
    ///
    /// Returns [`InputsError::Driver`] when the fence cannot be waited on.
    pub fn acquire(&mut self) -> Result<StagingEntry, InputsError> {
        Ok(self.ring.acquire()?)
    }

    /// A staging entry whose pinned block the host may write, if the copy that last read the
    /// block has finished: `None` without waiting while it is still in flight, leaving the same
    /// staging entry to be asked about again.
    ///
    /// # Errors
    ///
    /// Returns [`InputsError::Driver`] when the fence cannot be queried.
    pub fn try_acquire(&mut self) -> Result<Option<StagingEntry>, InputsError> {
        Ok(self.ring.try_acquire()?)
    }

    /// Writes `batch`'s inputs from `layout` into `entry`'s pinned block at the bucket's offsets,
    /// and hands back the sampler's two arrays in the same block for the sampler to write.
    ///
    /// # Errors
    ///
    /// Returns [`InputsError`] when the inputs were not built for the batch's bucket or the
    /// layout cannot be staged.
    pub fn stage(
        &mut self,
        entry: &StagingEntry,
        layout: &BatchLayout,
        batch: &DecodeBatch,
    ) -> Result<SamplerArrays<'_>, InputsError> {
        // The staging ring minted the staging entry below its depth, which is the block count.
        let block = self.blocks[entry.index()].as_mut_slice();
        self.packed.stage(block, layout, batch)
    }

    /// Writes `run`'s rows into `entry`'s pinned block at the bucket's offsets as padding rows,
    /// the sampler's two arrays naming no request slot: a dummy run's staging, copied in as a
    /// step's is.
    ///
    /// # Errors
    ///
    /// Returns [`InputsError`] when the inputs were not built for the run's bucket, the run does
    /// not name one block per row of it, or a block id cannot be staged.
    pub fn stage_dummy(&mut self, entry: &StagingEntry, run: &DummyRun) -> Result<(), InputsError> {
        // As in `stage`: the staging entry indexes a block.
        let block = self.blocks[entry.index()].as_mut_slice();
        self.packed.stage_dummy(block, run)
    }

    /// The descriptor that copies `bucket`'s packed length from `entry`'s pinned block to the
    /// device block in one copy and signals the staging entry's fence behind it. Taking the
    /// staging entry is the only way to copy in, so every copy is fenced, a dummy run's as a
    /// step's.
    ///
    /// # Errors
    ///
    /// Returns [`InputsError::UnknownBucket`] when the inputs were not built for `bucket`.
    // The staging entry is taken by value on purpose: one acquire hands out one, and the copy-in
    // is its one use.
    #[allow(clippy::needless_pass_by_value)]
    pub fn copy_in(
        &self,
        entry: StagingEntry,
        bucket: BucketIdx,
    ) -> Result<CopyIn<'_>, InputsError> {
        // As in `stage`: the staging entry indexes a block.
        let source = self
            .packed
            .staged(self.blocks[entry.index()].as_slice(), bucket)?;
        Ok(CopyIn {
            source,
            destination: self.device_address,
            signal: self.ring.fence(&entry).signal(),
        })
    }
}

impl Drop for DecodeInputs {
    fn drop(&mut self) {
        // A copy-in from any staging entry may still be reading its block; every fence is waited
        // on before the blocks go. A failure here cannot be acted on beyond saying so.
        if let Err(error) = self.ring.wait_all() {
            warn!(%error, "a copy-in could not be waited on before its staging goes");
        }
    }
}

/// The copy-in of one bucket's packed block: one copy to the device block, then the signal of the
/// staging entry's fence.
pub struct CopyIn<'a> {
    source: &'a [u8],
    destination: u64,
    signal: FenceSignal<'a>,
}

impl Descriptor for CopyIn<'_> {
    type Error = InputsError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), InputsError> {
        // SAFETY: the session hands a live stream in the blocks' context; the destination is the
        // device block, which holds every bucket's packed length; the source is a pinned block
        // that outlives the copy, since the fence signaled behind the copy is waited on before
        // the block is written again or freed.
        unsafe {
            memcpy_htod_async(self.destination, self.source, stream).map_err(RuntimeError::from)?;
            self.signal.enqueue(stream)?;
        }
        Ok(())
    }
}

/// A wait on `event` from the capture stream: the step runs after everything enqueued before
/// the event was recorded.
pub struct WaitEvent<'a> {
    event: &'a CudaEvent,
}

impl<'a> WaitEvent<'a> {
    /// A wait on `event`, borrowed for as long as the wait lives.
    #[must_use]
    pub fn new(event: &'a CudaEvent) -> Self {
        Self { event }
    }
}

impl Descriptor for WaitEvent<'_> {
    type Error = RuntimeError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), RuntimeError> {
        // SAFETY: the session hands a live stream, and the event is live for as long as this
        // descriptor borrows it. An event never recorded is complete, so the wait is a no-op.
        unsafe {
            stream::wait_event(
                stream,
                self.event.cu_event(),
                CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use atoma_core::dispatch::{BucketLadder, DispatchConfig, DispatchDecision};
    use atoma_core::step::CommandEntry;
    use atoma_core::types::RequestCount;

    use atoma_core::request::PADDING_TOKEN;
    use atoma_core::types::BlockId;

    use super::*;
    use crate::decode::batch::Checked;
    use crate::decode::staging::DummyRun;
    use crate::test_support::{engine_config, entry, keyed_command, BLOCK_SIZE};

    const WIDTH: usize = 64;
    /// Where the device block sits, aligned as a device allocation is.
    const BASE: u64 = 0x7f00_0000_0000;

    fn shape() -> StagingShape {
        StagingShape {
            max_tokens: 4,
            block_table_width: WIDTH,
            max_position: 32,
            block_size: BLOCK_SIZE,
        }
    }

    /// The buckets [`engine_config`] serves: one, two and four rows.
    fn buckets() -> DecodeBuckets {
        DecodeBuckets::usable(&engine_config().dispatch)
    }

    fn packed() -> PackedBuckets {
        PackedBuckets::new(shape(), &buckets()).unwrap()
    }

    fn dispatched(live: Vec<CommandEntry>) -> (BatchLayout, DecodeBatch) {
        let layout = BatchLayout::lay_out(&keyed_command(live), BLOCK_SIZE).unwrap();
        let DispatchDecision::FullReplay(key) = layout.dispatch else {
            panic!("keyed: {:?}", layout.dispatch);
        };
        let Checked::Step(batch) = DecodeBatch::check(&layout, key, &buckets(), WIDTH).unwrap()
        else {
            panic!("served by the decode step");
        };
        (layout, batch)
    }

    /// Storage for a packed block with every byte set to `0x5A`, which no input stages, so an
    /// untouched value reads as neither a `0` nor the `-1` the sampler's arrays carry; handed out
    /// from the first base aligned as the carve requires.
    struct Block(Vec<u8>);

    impl Block {
        fn sized(bytes: usize) -> Self {
            Self(vec![0x5A; bytes + align_of::<i64>()])
        }

        fn bytes(&mut self) -> &mut [u8] {
            let address = self.0.as_ptr().addr();
            let start = address.next_multiple_of(align_of::<i64>()) - address;
            &mut self.0[start..]
        }
    }

    #[test]
    fn every_buckets_views_sit_at_its_own_packed_offsets_over_the_device_block() {
        let packed = packed();
        let views: Vec<BucketViews> = packed
            .layouts
            .iter()
            .map(|layout| BucketViews::minted(BASE, layout, WIDTH, Tensor::for_test).unwrap())
            .collect();
        let at = |view: Tensor| (view.address() - BASE, view.dims().to_vec(), view.dtype());

        // Bucket 2 at width 64: each 8-byte array pads to 256, the 16-byte slot mapping too, the
        // block table's 2 * 64 * 4 = 512 bytes end at 1536, and the sampler's two follow.
        let two = &views[1];
        assert_eq!(at(two.inputs.token_ids), (0, vec![2], Dtype::U32));
        assert_eq!(at(two.inputs.positions), (256, vec![2], Dtype::I32));
        assert_eq!(at(two.inputs.seqlens_k), (512, vec![2], Dtype::I32));
        assert_eq!(at(two.inputs.slot_mapping), (768, vec![2], Dtype::I64));
        assert_eq!(at(two.inputs.block_table), (1024, vec![2, 64], Dtype::I32));
        assert_eq!(at(two.row_slots), (1536, vec![2], Dtype::I32));
        assert_eq!(at(two.gather_slots), (1792, vec![2], Dtype::I32));

        // Bucket 4's block table is 1024 bytes, so its sampler arrays sit 512 further on; bucket
        // 1's is 256, so they sit 256 nearer.
        assert_eq!(
            at(views[2].inputs.block_table),
            (1024, vec![4, 64], Dtype::I32)
        );
        assert_eq!(at(views[2].row_slots), (2048, vec![4], Dtype::I32));
        assert_eq!(at(views[2].gather_slots), (2304, vec![4], Dtype::I32));
        assert_eq!(
            at(views[0].inputs.block_table),
            (1024, vec![1, 64], Dtype::I32)
        );
        assert_eq!(at(views[0].row_slots), (1280, vec![1], Dtype::I32));
        assert_eq!(at(views[0].gather_slots), (1536, vec![1], Dtype::I32));
    }

    #[test]
    fn the_device_block_is_the_largest_buckets_packed_length() {
        // Bucket 4 at width 64: four 256-byte arrays, a 1024-byte block table, then two more
        // 256-byte arrays; buckets 1 and 2 pack 1792 and 2048, so the block is neither a sum nor
        // the first bucket's.
        assert_eq!(packed().block_bytes, 2560);
    }

    #[test]
    fn a_batch_is_staged_at_its_buckets_offsets_and_the_samplers_arrays_are_the_buckets() {
        let (layout, batch) = dispatched(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 8, vec![7], &[20, 21, 22], true),
        ]);
        assert_eq!(batch.bucket, BucketIdx(1), "two live entries fill bucket 2");
        let packed = packed();
        let mut block = Block::sized(packed.block_bytes);

        let sampler = packed.stage(block.bytes(), &layout, &batch).unwrap();
        assert_eq!(sampler.row_slots.len(), 2, "one row slot per bucket row");
        assert_eq!(
            sampler.gather_slots.len(),
            2,
            "one gather slot per bucket row"
        );
        sampler.row_slots[1] = 0x6666_6666;
        sampler.gather_slots[0] = 0x7777_7777;

        // Bucket 2's offsets: token ids at 0, positions at 256, key lengths at 512, slot
        // mapping at 768, block table rows at 1024 and 1280, row slots at 1536, gather slots at
        // 1792; a key length is the context plus this token, a slot is the block times the
        // block size plus the offset.
        let bytes = block.bytes();
        assert_eq!(
            bytes[..8],
            [9u32.to_ne_bytes(), 7u32.to_ne_bytes()].concat()
        );
        assert_eq!(
            bytes[256..264],
            [3i32.to_ne_bytes(), 8i32.to_ne_bytes()].concat()
        );
        assert_eq!(
            bytes[512..520],
            [4i32.to_ne_bytes(), 9i32.to_ne_bytes()].concat()
        );
        assert_eq!(
            bytes[768..784],
            [43i64.to_ne_bytes(), 88i64.to_ne_bytes()].concat()
        );
        assert_eq!(
            bytes[1024..1032],
            [10i32.to_ne_bytes(), 0i32.to_ne_bytes()].concat()
        );
        assert_eq!(
            bytes[1280..1296],
            [20i32, 21, 22, 0]
                .iter()
                .flat_map(|block| block.to_ne_bytes())
                .collect::<Vec<u8>>()
        );
        assert_eq!(bytes[1540..1544], [0x66; 4]);
        assert_eq!(bytes[1792..1796], [0x77; 4]);
        assert!(
            bytes[2048..2560].iter().all(|&byte| byte == 0x5A),
            "past bucket 2's packed length the block is as the storage set it"
        );
    }

    #[test]
    fn the_copy_in_carries_the_buckets_packed_length_and_no_more() {
        let packed = packed();
        let mut block = Block::sized(packed.block_bytes);
        let (_, one) = dispatched(vec![entry(1, 3, vec![9], &[10], true)]);
        let (_, two) = dispatched(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 3, vec![9], &[20], true),
        ]);

        // Bucket 1 packs 1792 bytes and bucket 2 packs 2048: the 2560-byte block is never copied
        // whole. A dummy run copies in by its bucket through the same call.
        assert_eq!(
            packed.staged(block.bytes(), one.bucket).unwrap().len(),
            1792
        );
        assert_eq!(
            packed.staged(block.bytes(), two.bucket).unwrap().len(),
            2048
        );
        assert!(matches!(
            packed.staged(&block.bytes()[..2047], two.bucket),
            Err(InputsError::Staging(StagingError::BlockTooShort {
                len: 2047,
                rows: 2,
                needed: 2048
            }))
        ));
    }

    #[test]
    fn a_batch_of_a_bucket_the_inputs_do_not_stage_for_is_refused() {
        let (layout, batch) = dispatched(vec![entry(1, 3, vec![9], &[10], true)]);
        let past = DecodeBatch {
            bucket: BucketIdx(3),
            ..batch
        };
        let packed = packed();
        let mut block = Block::sized(packed.block_bytes);
        let unknown = |error: &InputsError| {
            matches!(
                error,
                InputsError::UnknownBucket {
                    bucket: BucketIdx(3),
                    buckets: 3
                }
            )
        };

        assert!(unknown(
            &packed.stage(block.bytes(), &layout, &past).unwrap_err()
        ));
        assert!(unknown(
            &packed.staged(block.bytes(), past.bucket).unwrap_err()
        ));
        assert!(unknown(&packed.layout(BucketIdx(3)).unwrap_err()));
        let run = DummyRun::new(BucketIdx(3), vec![BlockId::new(7)]);
        assert!(unknown(
            &packed.stage_dummy(block.bytes(), &run).unwrap_err()
        ));
    }

    #[test]
    fn a_dummy_run_is_staged_at_its_buckets_offsets_with_every_row_a_padding_row() {
        let packed = packed();
        let mut block = Block::sized(packed.block_bytes);
        let run = DummyRun::new(BucketIdx(1), vec![BlockId::new(15), BlockId::new(3)]);

        packed.stage_dummy(block.bytes(), &run).unwrap();

        // Bucket 2's offsets, as above; four-token blocks, so block 15's first slot is 60 and
        // block 3's is 12; each table row is its block then zero to the width; the sampler's
        // two arrays name no request slot.
        let bytes = block.bytes();
        let twice = |value: [u8; 4]| [value, value].concat();
        assert_eq!(bytes[..8], twice(PADDING_TOKEN.to_ne_bytes()));
        assert_eq!(bytes[256..264], twice(0i32.to_ne_bytes()));
        assert_eq!(bytes[512..520], twice(1i32.to_ne_bytes()));
        assert_eq!(
            bytes[768..784],
            [60i64.to_ne_bytes(), 12i64.to_ne_bytes()].concat()
        );
        assert_eq!(bytes[1024..1028], 15i32.to_ne_bytes());
        assert!(bytes[1028..1280].iter().all(|&byte| byte == 0));
        assert_eq!(bytes[1280..1284], 3i32.to_ne_bytes());
        assert!(bytes[1284..1536].iter().all(|&byte| byte == 0));
        assert_eq!(bytes[1536..1544], twice((-1i32).to_ne_bytes()));
        assert_eq!(bytes[1792..1800], twice((-1i32).to_ne_bytes()));
        assert!(
            bytes[2048..2560].iter().all(|&byte| byte == 0x5A),
            "past bucket 2's packed length the block is as the storage set it"
        );
    }

    #[test]
    fn a_dummy_run_with_other_than_one_block_per_row_of_its_bucket_is_refused() {
        let packed = packed();
        let mut block = Block::sized(packed.block_bytes);

        for blocks in [1, 3] {
            let run = DummyRun::new(BucketIdx(1), vec![BlockId::new(7); blocks]);
            let refused = packed.stage_dummy(block.bytes(), &run).unwrap_err();
            assert!(
                matches!(
                    refused,
                    InputsError::DummyRunNotBucket {
                        bucket: BucketIdx(1),
                        rows: 2,
                        blocks: named
                    } if named == blocks
                ),
                "{blocks} blocks for the bucket of two: {refused}"
            );
        }
        assert!(
            block.bytes()[..2048].iter().all(|&byte| byte == 0x5A),
            "a refused dummy run writes nothing"
        );
    }

    #[test]
    fn inputs_for_no_bucket_are_refused() {
        let none = DecodeBuckets::usable(&DispatchConfig {
            bucket_ladder: BucketLadder::new(vec![2, 4]).unwrap(),
            captured_max_requests: RequestCount::new(1).unwrap(),
        });
        assert!(
            none.tokens().is_empty(),
            "no bucket is at or below one request"
        );

        assert!(matches!(
            PackedBuckets::new(shape(), &none),
            Err(InputsError::NoBucket)
        ));
    }
}
