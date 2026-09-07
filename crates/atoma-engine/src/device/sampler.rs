//! The sampler on one rank's device: every request slot's record and last sampled token, the
//! rows' inputs and outputs, all at fixed addresses from the Allocation session phase on, and the
//! descriptors that run a step's sampling through the seam.
//!
//! What crosses to the device each step is small and goes up, not down. The sampling records of
//! the slots that changed hands are staged in the sampler's own pinned memory and uploaded as
//! sparse copies, one per record, in front of everything else the step copies. The slot each
//! selected row samples under, and which token rows take their token from the device, are the
//! sampler's two per-step arrays, and [`DeviceSampler::stage`] writes them where the caller says:
//! the decode step stages them beside the model's inputs in its packed block, uploads the block
//! in one copy, and hands [`DeviceSampler::gather`] and [`DeviceSampler::sample`] tensor views
//! over where the two arrays landed and over the logits. Each view is held to the staged step's
//! rows and the dtype the kernel reads, so a view handed to the wrong argument is refused by
//! name. What comes back is one asynchronous copy of the rows' tokens, through the leading rows
//! of the row tokens view, waited on once the step is enqueued through the readback's own event
//! and nothing else: the host learns what was sampled for detokenisation and finish detection,
//! and the device never waits for it. The sampled tokens stay in the per-slot array the next
//! step's gather reads.
//!
//! The candle forward samples through the same state on candle's stream. An eager step gathers
//! nothing, so [`DeviceSampler::stage_eager`] writes its row slots into the sampler's own pinned
//! pair and [`DeviceSampler::run_on`] uploads them there with the records; the host wait that
//! ends every step, on either stream, is what orders the two streams' use of the sampler's
//! device state.
//!
//! Every descriptor names the sampler's four device arrays by the addresses read once at
//! Allocation. [`DeviceSampler::addresses`] reads them again from the arrays themselves, by name
//! in one fixed order: what the forward bakes when it is built and a debug build compares a
//! fresh reading against before each keyed step.

use std::ffi::c_void;
use std::fmt;
use std::mem::size_of;
use std::ptr;
use std::sync::Arc;

use atoma_core::types::RequestCount;
use atoma_kernels::error::KernelError;
use atoma_kernels::sampler::{gather, sample, GatherCall, SampleCall};
use atoma_runtime::error::RuntimeError;
use atoma_runtime::session::{Allocation, Descriptor};
use atoma_runtime::tensor::{Dtype, Layout, Tensor, TensorError};
use cudarc::driver::result::{event, memcpy_htod_async};
use cudarc::driver::sys::{self, CUevent_flags};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr};
use thiserror::Error;
use tracing::{info, warn};

use crate::batch::BatchLayout;
use crate::decode::baked::{BakedAddress, BakedName};
use crate::decode::staging::{stage_sampler, SamplerArrays, StagingError};
use crate::pinned::Pinned;
use crate::readback::{Readback, ReadbackCopy, ReadbackError};
use crate::sampling::inputs::{SamplerInputs, SamplerInputsError};
use crate::sampling::owners::SlotOwners;
use crate::sampling::record::{SlotRecord, RECORD_BYTES};

/// Why the sampler could not be built or run.
#[derive(Debug, Error)]
pub enum SamplerError {
    /// More selected rows than the sampler was sized for.
    #[error("{rows} rows sample this step but the sampler holds {max_rows} at most")]
    TooManyRows { rows: usize, max_rows: usize },
    /// More token rows to gather for than the sampler was sized for.
    #[error("{rows} token rows this step but the sampler gathers for {max_rows} at most")]
    TooManyGatherRows { rows: usize, max_rows: usize },
    /// The slot count does not fit the kernel's slot index.
    #[error("{slots} request slots do not fit the sampler's 32-bit slot index")]
    TooManySlots { slots: usize },
    /// A descriptor was asked for with no step staged, or a wait with none enqueued.
    #[error("no step is staged; stage the layout before running its sampling")]
    NoStepStaged,
    /// An upload from the other staging: an eager step's arrays are in the sampler's own memory,
    /// a decode step's in the caller's.
    #[error("the staged step's arrays are in {0}; upload them from there")]
    ArraysElsewhere(ArraysIn),
    /// A view handed to the sampler is of another dtype than the kernel reads `what` as.
    #[error("the {what} view is {held:?}; the sampler reads {expected:?}")]
    ViewDtype {
        what: &'static str,
        held: Dtype,
        expected: Dtype,
    },
    /// A view handed to the sampler is gapped; the kernel reads `what` as one contiguous array.
    #[error("the {what} view has strides {strides:?}; the sampler reads it contiguous")]
    ViewNotContiguous {
        what: &'static str,
        strides: Vec<usize>,
    },
    /// A view handed to the sampler is not the rows the staged step covers.
    #[error("the {what} view is {held:?}; the sampler reads {expected:?} this step")]
    ViewShape {
        what: &'static str,
        held: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error(transparent)]
    Inputs(#[from] SamplerInputsError),
    #[error(transparent)]
    Staging(#[from] StagingError),
    #[error(transparent)]
    Readback(#[from] ReadbackError),
    #[error(transparent)]
    Launch(#[from] KernelError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Driver(#[from] RuntimeError),
}

/// Whose memory a staged step's two per-step arrays are in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArraysIn {
    /// The sampler's own pinned pair: an eager step's, uploaded by [`DeviceSampler::run_on`].
    Sampler,
    /// The caller's staging: a decode step's, uploaded with the step's inputs.
    Caller,
}

impl fmt::Display for ArraysIn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ArraysIn::Sampler => "the sampler's own memory",
            ArraysIn::Caller => "the caller's staging",
        })
    }
}

/// A device array: the memory, owned here so the address stays allocated for as long as it is
/// named and read again for the debug check, and the address itself, read once at Allocation.
struct DeviceArray {
    memory: CudaSlice<u8>,
    address: u64,
}

impl DeviceArray {
    /// `bytes` zeroed bytes on `stream`'s device.
    fn zeroed(stream: &Arc<CudaStream>, bytes: usize) -> Result<Self, SamplerError> {
        let memory = stream
            .alloc_zeros::<u8>(bytes)
            .map_err(RuntimeError::from)?;
        let address = address(&memory, stream);
        Ok(Self { memory, address })
    }
}

/// A device array and the view minted over it at Allocation: the array is owned here, so the
/// address the view names stays allocated for as long as the view is read. Named for the array
/// rather than the view, since a tensor view never owns what it names.
struct ViewedArray {
    view: Tensor,
    array: DeviceArray,
}

impl ViewedArray {
    /// `layout`'s elements, zeroed, on `stream`'s device, viewed as `layout`.
    fn zeroed(
        allocation: &Allocation,
        stream: &Arc<CudaStream>,
        layout: Layout,
    ) -> Result<Self, SamplerError> {
        let array = DeviceArray::zeroed(stream, layout.extent_bytes())?;
        let view = Tensor::new(allocation, array.address, layout)?;
        Ok(Self { view, array })
    }
}

/// A device array of `len` values, and the pinned host memory a step writes before copying it
/// there.
struct StagedArray<T> {
    host: Pinned<T>,
    device: DeviceArray,
}

impl<T> StagedArray<T> {
    fn new(stream: &Arc<CudaStream>, len: usize) -> Result<Self, SamplerError> {
        Ok(Self {
            host: Pinned::new(len)?,
            device: DeviceArray::zeroed(stream, len * size_of::<T>())?,
        })
    }

    fn address(&self) -> u64 {
        self.device.address
    }
}

/// The step staged and not yet waited on.
#[derive(Debug, Clone, Copy)]
struct StagedStep {
    /// Selected rows sampling this step.
    rows: usize,
    /// Token rows the gather covers: the batch's tokens for a uniform decode, none otherwise.
    gather_rows: usize,
    /// Whose memory the row slots and gather slots were written to.
    arrays: ArraysIn,
}

impl StagedStep {
    /// The step `inputs` decided, its two arrays written to `arrays`.
    fn of(inputs: &SamplerInputs, arrays: ArraysIn) -> Self {
        Self {
            rows: inputs.row_slots.len(),
            gather_rows: inputs.gather.len(),
            arrays,
        }
    }
}

/// The sampler's device state and staging for one rank.
pub struct DeviceSampler {
    max_rows: usize,
    vocab: usize,
    /// One record per request slot, as the kernel reads and advances it; the host copy holds
    /// what was last written for the slot.
    records: StagedArray<SlotRecord>,
    /// The slots whose record changed since the last upload.
    pending_records: Vec<usize>,
    /// u32 per slot: the token last sampled there.
    sampled: DeviceArray,
    /// i32 per row: the slot each selected row samples under, as the kernel indexes it, for an
    /// eager step: staged in the pinned half and read from the device half. A decode step's row
    /// slots are in its own upload.
    row_slots: StagedArray<i32>,
    /// u32 `[max_rows]`: the token sampled for each selected row this step, viewed for the
    /// readback to copy the leading rows through.
    row_tokens: ViewedArray,
    readback: Readback<u32>,
    /// Recorded behind every upload from the sampler's own staging; waited on before the staging
    /// is freed.
    uploaded: CudaEvent,
    owners: SlotOwners,
    staged: Option<StagedStep>,
}

impl DeviceSampler {
    /// Allocates the sampler for `slots` request slots and up to `max_rows` sampling rows of
    /// `vocab` logits, during the Allocation session phase, on `stream`'s device.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError`] when the slot count does not fit the kernel's index or the
    /// driver cannot pin or allocate a buffer.
    pub fn new(
        allocation: &Allocation,
        stream: &Arc<CudaStream>,
        slots: usize,
        max_rows: RequestCount,
        vocab: usize,
    ) -> Result<Self, SamplerError> {
        if i32::try_from(slots).is_err() {
            return Err(SamplerError::TooManySlots { slots });
        }
        let max_rows = max_rows.get();
        let context = stream.context();
        context.bind_to_thread().map_err(RuntimeError::from)?;
        let uploaded = context
            .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .map_err(RuntimeError::from)?;
        let row_tokens = Layout::contiguous(&[max_rows], Dtype::U32)?;
        let sampler = Self {
            max_rows,
            vocab,
            records: StagedArray::new(stream, slots)?,
            pending_records: Vec::new(),
            sampled: DeviceArray::zeroed(stream, slots * size_of::<u32>())?,
            row_slots: StagedArray::new(stream, max_rows)?,
            row_tokens: ViewedArray::zeroed(allocation, stream, row_tokens)?,
            readback: Readback::new(allocation, context, max_rows, 1)?,
            uploaded,
            owners: SlotOwners::new(slots),
            staged: None,
        };
        info!(slots, max_rows, vocab, "device sampler allocated");
        Ok(sampler)
    }

    /// Every address the sampler's descriptors bake, read from the array that holds it, by name
    /// in one fixed order: the sampling records, the sampled tokens, the sampler's own row slots
    /// and the row tokens. The forward bakes this reading when it is built and a debug build
    /// reads it again before each keyed step.
    #[must_use]
    pub fn addresses(&self, stream: &Arc<CudaStream>) -> [BakedAddress; 4] {
        [
            (BakedName::SamplingRecords, &self.records.device),
            (BakedName::SampledTokens, &self.sampled),
            (BakedName::SamplerRowSlots, &self.row_slots.device),
            (BakedName::RowTokens, &self.row_tokens.array),
        ]
        .map(|(name, array)| BakedAddress {
            name,
            address: address(&array.memory, stream),
        })
    }

    /// Decides `layout`'s step and stages its inputs: the records of the slots that changed
    /// hands into the sampler's own staging, and the slot of every selected row and which token
    /// rows gather into `arrays`, as the kernels index them.
    ///
    /// `gather_rows` is how many leading token rows the gather covers, which only a caller
    /// holding a batch of one token per entry may state.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError`] when the layout has more rows than the sampler or `arrays` hold,
    /// or its inputs cannot be decided.
    pub fn stage(
        &mut self,
        layout: &BatchLayout,
        gather_rows: usize,
        arrays: SamplerArrays<'_>,
    ) -> Result<(), SamplerError> {
        self.staged = None;
        let inputs = self.stage_records(layout, Some(gather_rows))?;
        stage_sampler(&inputs, arrays)?;
        self.staged = Some(StagedStep::of(&inputs, ArraysIn::Caller));
        Ok(())
    }

    /// Decides `layout`'s eager step, which gathers nothing, and stages its inputs in the
    /// sampler's own memory: the records of the slots that changed hands, and the slot of every
    /// selected row.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError`] when the layout has more rows than the sampler holds, or its
    /// inputs cannot be decided.
    pub fn stage_eager(&mut self, layout: &BatchLayout) -> Result<(), SamplerError> {
        self.staged = None;
        let inputs = self.stage_records(layout, None)?;
        stage_sampler(
            &inputs,
            SamplerArrays {
                row_slots: self.row_slots.host.as_mut_slice(),
                gather_slots: &mut [],
            },
        )?;
        self.staged = Some(StagedStep::of(&inputs, ArraysIn::Sampler));
        Ok(())
    }

    /// Decides `layout`'s step against the slot mirror, holding its rows to what the sampler's
    /// device arrays hold, and writes the records of the slots that changed hands into their
    /// staging.
    fn stage_records(
        &mut self,
        layout: &BatchLayout,
        gather_rows: Option<usize>,
    ) -> Result<SamplerInputs, SamplerError> {
        let rows = layout.sampling.len();
        if rows > self.max_rows {
            return Err(SamplerError::TooManyRows {
                rows,
                max_rows: self.max_rows,
            });
        }
        let inputs = SamplerInputs::for_step(layout, &mut self.owners, gather_rows)?;
        if inputs.gather.len() > self.max_rows {
            return Err(SamplerError::TooManyGatherRows {
                rows: inputs.gather.len(),
                max_rows: self.max_rows,
            });
        }
        self.pending_records.clear();
        let records_host = self.records.host.as_mut_slice();
        for (slot, record) in &inputs.records {
            let index = slot.index();
            records_host[index] = *record;
            self.pending_records.push(index);
        }
        Ok(inputs)
    }

    /// The descriptor that copies the staged step's changed records to the device, one sparse
    /// copy per record.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError::NoStepStaged`] when no step is staged.
    pub fn upload_records(&self) -> Result<RecordUpload<'_>, SamplerError> {
        if self.staged.is_none() {
            return Err(SamplerError::NoStepStaged);
        }
        Ok(RecordUpload { sampler: self })
    }

    /// The descriptor that copies an eager step's inputs from the sampler's own staging: the
    /// changed records, then the rows' slots.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError`] when no step is staged, or the staged step's arrays are the
    /// caller's.
    fn upload_eager(&self) -> Result<EagerUpload<'_>, SamplerError> {
        let staged = self.staged.ok_or(SamplerError::NoStepStaged)?;
        if staged.arrays != ArraysIn::Sampler {
            return Err(SamplerError::ArraysElsewhere(staged.arrays));
        }
        Ok(EagerUpload {
            sampler: self,
            rows: staged.rows,
        })
    }

    /// The descriptor that overwrites the gathering token rows of the u32 `token_ids` with the
    /// token last sampled for their slot, as the i32 `gather_slots` name it for every covered
    /// token row: the array the staged step wrote, viewed where the caller's upload put it on
    /// the device. Both views are the covered token rows exactly.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError::NoStepStaged`] when no step is staged, or a `View` variant when
    /// a view is not the covered rows of the dtype the kernel reads.
    pub fn gather(
        &self,
        token_ids: &Tensor,
        gather_slots: &Tensor,
    ) -> Result<Gather, SamplerError> {
        let staged = self.staged.ok_or(SamplerError::NoStepStaged)?;
        let rows = &[staged.gather_rows];
        check_view("token ids", token_ids, Dtype::U32, rows)?;
        check_view("gather slots", gather_slots, Dtype::I32, rows)?;
        Ok(Gather {
            call: GatherCall {
                token_ids: token_ids.address(),
                gather_slots: gather_slots.address(),
                sampled: self.sampled.address,
                n_rows: staged.gather_rows,
                stream: ptr::null_mut(),
            },
        })
    }

    /// The descriptor that samples every selected row from the f32 `logits`, one row per
    /// selected row a vocabulary wide, under the i32 `row_slots`, one per selected row viewed
    /// where the staged step's upload put them, and copies the tokens back for
    /// [`DeviceSampler::wait`]. Both views are the selected rows exactly: the caller narrows
    /// them to the live rows.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError::NoStepStaged`] when no step is staged, or a `View` variant when
    /// a view is not the selected rows of the dtype the kernel reads.
    pub fn sample(
        &mut self,
        logits: &Tensor,
        row_slots: &Tensor,
    ) -> Result<Sample<'_>, SamplerError> {
        let staged = self.staged.ok_or(SamplerError::NoStepStaged)?;
        check_view("logits", logits, Dtype::F32, &[staged.rows, self.vocab])?;
        check_view("row slots", row_slots, Dtype::I32, &[staged.rows])?;
        self.sample_at(logits.address(), row_slots.address())
    }

    /// The sample of the staged step's rows from the f32 logits at `logits` under the i32 row
    /// slots at `row_slots`, and the readback of its tokens through the leading rows of the row
    /// tokens view. The addresses are the caller's to have checked: a view for a decode step,
    /// candle's logits and the sampler's own row slots for an eager one.
    fn sample_at(&mut self, logits: u64, row_slots: u64) -> Result<Sample<'_>, SamplerError> {
        let staged = self.staged.ok_or(SamplerError::NoStepStaged)?;
        let tokens = self.row_tokens.view.narrow(0, 0, staged.rows)?;
        let call = SampleCall {
            logits,
            row_slots,
            records: self.records.address(),
            sampled: self.sampled.address,
            out: tokens.address(),
            vocab: self.vocab,
            n_rows: staged.rows,
            stream: ptr::null_mut(),
        };
        let copy = self.readback.copy(&tokens)?;
        Ok(Sample { call, copy })
    }

    /// Waits for the staged step's tokens, and that copy alone, and returns them: one per
    /// selected row, in batch order.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError`] when no step is staged, no copy was enqueued, or the wait
    /// fails.
    pub fn wait(&mut self) -> Result<&[u32], SamplerError> {
        if self.staged.take().is_none() {
            return Err(SamplerError::NoStepStaged);
        }
        Ok(self.readback.wait()?)
    }

    /// Runs the eager step staged by [`DeviceSampler::stage_eager`] on `stream` — the upload
    /// from the sampler's own staging, the sample and the readback — over the f32 logits `logits`
    /// holds, and waits for its tokens: the candle path, which has no descriptor seam.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError`] when no eager step is staged or the driver refuses a copy or a
    /// launch.
    pub fn run_on<S: DevicePtr<f32>>(
        &mut self,
        stream: &Arc<CudaStream>,
        logits: &S,
    ) -> Result<&[u32], SamplerError> {
        stream
            .context()
            .bind_to_thread()
            .map_err(RuntimeError::from)?;
        let (address, _reads) = logits.device_ptr(stream);
        let row_slots = self.row_slots.address();
        // SAFETY: candle's stream is live in the sampler's context, and every address the
        // descriptors name is this sampler's, or the logits the stream's earlier work wrote.
        unsafe {
            self.upload_eager()?.enqueue(stream.cu_stream())?;
            self.sample_at(address, row_slots)?
                .enqueue(stream.cu_stream())?;
        }
        self.wait()
    }
}

impl Drop for DeviceSampler {
    fn drop(&mut self) {
        // The last upload may still be reading the staging; the event waits for it before the
        // arrays go. A failure here cannot be acted on beyond saying so.
        if let Err(error) = self.uploaded.synchronize() {
            warn!(
                %error,
                "the sampler's last upload could not be waited on before its staging goes"
            );
        }
    }
}

/// The upload of one step's changed records: one sparse copy per record, and the sampler's
/// event recorded behind them.
pub struct RecordUpload<'a> {
    sampler: &'a DeviceSampler,
}

impl Descriptor for RecordUpload<'_> {
    type Error = SamplerError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), SamplerError> {
        // SAFETY: the session hands a live stream in the buffers' context.
        unsafe {
            copy_records(self.sampler, stream)?;
            event::record(self.sampler.uploaded.cu_event(), stream).map_err(RuntimeError::from)?;
        }
        Ok(())
    }
}

/// The upload of an eager step's inputs from the sampler's own staging: the changed records,
/// then the rows' slots, with the sampler's event recorded behind them.
struct EagerUpload<'a> {
    sampler: &'a DeviceSampler,
    rows: usize,
}

impl Descriptor for EagerUpload<'_> {
    type Error = SamplerError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), SamplerError> {
        let sampler = self.sampler;
        // SAFETY: candle's stream is live in the buffers' context; the destination is this
        // sampler's device array and the source its pinned staging, which outlives the copy
        // through the event recorded behind it.
        unsafe {
            copy_records(sampler, stream)?;
            memcpy_htod_async(
                sampler.row_slots.address(),
                &sampler.row_slots.host.as_slice()[..self.rows],
                stream,
            )
            .map_err(RuntimeError::from)?;
            event::record(sampler.uploaded.cu_event(), stream).map_err(RuntimeError::from)?;
        }
        Ok(())
    }
}

/// Copies every pending record to its slot's place in the device records, one copy each.
///
/// # Safety
///
/// `stream` must be a live stream in the sampler's context.
unsafe fn copy_records(sampler: &DeviceSampler, stream: sys::CUstream) -> Result<(), RuntimeError> {
    let records = sampler.records.host.as_slice();
    for &slot in &sampler.pending_records {
        let destination = sampler.records.address() + (slot * RECORD_BYTES) as u64;
        // SAFETY: the destination is this sampler's device records and the source its pinned
        // staging, which outlives the copy through the event the caller records behind it.
        unsafe { memcpy_htod_async(destination, &records[slot..=slot], stream) }?;
    }
    Ok(())
}

/// The gather of the token rows whose token the device sampled last.
pub struct Gather {
    call: GatherCall,
}

impl Descriptor for Gather {
    type Error = SamplerError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), SamplerError> {
        let call = GatherCall {
            stream: stream.cast::<c_void>(),
            ..self.call
        };
        // SAFETY: the session hands a live stream; the token ids are the step's uploaded inputs
        // and the slots were staged against this sampler's arrays.
        unsafe { gather(&call) }?;
        Ok(())
    }
}

/// The sample of every selected row and the readback of its tokens.
pub struct Sample<'a> {
    call: SampleCall,
    copy: ReadbackCopy<'a, u32>,
}

impl Descriptor for Sample<'_> {
    type Error = SamplerError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), SamplerError> {
        let call = SampleCall {
            stream: stream.cast::<c_void>(),
            ..self.call
        };
        // SAFETY: the session hands a live stream; the logits are what the stream's earlier
        // work wrote, and every other address is this sampler's, staged for these rows.
        unsafe {
            sample(&call)?;
            self.copy.enqueue(stream)?;
        }
        Ok(())
    }
}

/// Holds `view` to contiguous `dims` of `dtype`, which is what the kernel reads `what` as,
/// refusing by the first of dtype, contiguity and shape that fails.
fn check_view(
    what: &'static str,
    view: &Tensor,
    dtype: Dtype,
    dims: &[usize],
) -> Result<(), SamplerError> {
    if view.dtype() != dtype {
        return Err(SamplerError::ViewDtype {
            what,
            held: view.dtype(),
            expected: dtype,
        });
    }
    if !view.is_contiguous() {
        return Err(SamplerError::ViewNotContiguous {
            what,
            strides: view.strides().to_vec(),
        });
    }
    if view.dims() != dims {
        return Err(SamplerError::ViewShape {
            what,
            held: view.dims().to_vec(),
            expected: dims.to_vec(),
        });
    }
    Ok(())
}

/// The device address of a buffer; event tracking is disabled at context creation, so the read
/// guard is a no-op and the address is stable for the buffer's lifetime.
fn address<T>(slice: &CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
    let (address, _reads) = slice.device_ptr(stream);
    address
}

#[cfg(test)]
mod tests {
    use atoma_runtime::tensor::{Dtype, Layout, Tensor};

    use super::{check_view, SamplerError};

    /// Where a device buffer sits, aligned as a device allocation is.
    const BASE: u64 = 0x7f00_0000_0000;
    /// The width the sampler reads a logits row as.
    const VOCAB: usize = 4096;

    fn view(layout: Layout) -> Tensor {
        Tensor::for_test(BASE, layout).unwrap()
    }

    #[test]
    fn logits_narrowed_to_the_live_rows_are_taken_and_logits_a_row_apart_are_refused() {
        // Two live rows of a bucket of eight: narrowed from the contiguous `[8, VOCAB]` logits
        // the rows stay VOCAB apart, so the view is the `[2, VOCAB]` array the kernel reads.
        let bucket = view(Layout::contiguous(&[8, VOCAB], Dtype::F32).unwrap());
        let live = bucket.narrow(0, 0, 2).unwrap();
        check_view("logits", &live, Dtype::F32, &[2, VOCAB]).unwrap();

        // The same two rows 2 * VOCAB apart: the kernel would read the row between as logits.
        let gapped = Layout::strided(&[2, VOCAB], &[2 * VOCAB, 1], Dtype::F32).unwrap();
        let refused = check_view("logits", &view(gapped), Dtype::F32, &[2, VOCAB]).unwrap_err();
        assert!(
            matches!(
                refused,
                SamplerError::ViewNotContiguous {
                    what: "logits",
                    ref strides
                } if *strides == [2 * VOCAB, 1]
            ),
            "logits a row apart are refused as gapped: {refused}"
        );
    }

    #[test]
    fn gather_slots_two_apart_are_refused_since_the_kernel_reads_them_as_one_array() {
        // Three covered rows whose slots sit every other element: not the `[3]` array.
        let strided = Layout::strided(&[3], &[2], Dtype::I32).unwrap();
        let refused = check_view("gather slots", &view(strided), Dtype::I32, &[3]).unwrap_err();
        assert!(
            matches!(
                refused,
                SamplerError::ViewNotContiguous {
                    what: "gather slots",
                    ref strides
                } if *strides == [2]
            ),
            "gather slots two apart are refused as gapped: {refused}"
        );
    }

    #[test]
    fn token_ids_of_other_rows_than_the_gather_covers_are_refused_by_name() {
        // The step covers three token rows; four rows are the batch's, not the covered rows.
        let four = view(Layout::contiguous(&[4], Dtype::U32).unwrap());
        let refused = check_view("token ids", &four, Dtype::U32, &[3]).unwrap_err();
        assert!(
            matches!(
                refused,
                SamplerError::ViewShape {
                    what: "token ids",
                    ref held,
                    ref expected
                } if *held == [4] && *expected == [3]
            ),
            "token ids of other rows than the gather covers are refused: {refused}"
        );
    }
}
