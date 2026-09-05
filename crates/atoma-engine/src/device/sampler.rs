//! The sampler on one rank's device: every request slot's record and last sampled token, the
//! rows' inputs and outputs, all at fixed addresses from the Allocation session phase on, and the
//! descriptors that run a step's sampling through the seam.
//!
//! What crosses to the device each step is small and goes up, not down: the records of the slots
//! that changed hands, the slot each selected row samples under, and, for a uniform decode, which
//! token rows take their token from the device. What comes back is one asynchronous copy of the
//! rows' tokens, fenced by the readback's event and waited on once the step is enqueued: the host
//! learns what was sampled for detokenisation and finish detection, and the device never waits
//! for it. The sampled tokens stay in the per-slot array the next step's gather reads.
//!
//! The candle forward samples through the same state on candle's stream, enqueuing the same
//! descriptors there; the host wait that ends every step, on either stream, is what orders the
//! two streams' use of the sampler's device state.

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;
use std::sync::Arc;

use atoma_core::types::{RequestCount, RequestSlot};
use atoma_kernels::error::KernelError;
use atoma_kernels::sampler::{gather, sample, GatherCall, SampleCall};
use atoma_runtime::error::RuntimeError;
use atoma_runtime::session::{Allocation, Descriptor};
use cudarc::driver::result::{event, memcpy_htod_async};
use cudarc::driver::sys::{self, CUevent_flags};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr};
use thiserror::Error;
use tracing::{info, warn};

use crate::batch::BatchLayout;
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
    #[error(transparent)]
    Inputs(#[from] SamplerInputsError),
    #[error(transparent)]
    Readback(#[from] ReadbackError),
    #[error(transparent)]
    Launch(#[from] KernelError),
    #[error(transparent)]
    Driver(#[from] RuntimeError),
}

/// A device array: the memory, owned here so the address stays allocated for as long as it is
/// named, and the address itself.
struct DeviceArray {
    _memory: CudaSlice<u8>,
    address: u64,
}

impl DeviceArray {
    /// `bytes` zeroed bytes on `stream`'s device.
    fn zeroed(stream: &Arc<CudaStream>, bytes: usize) -> Result<Self, SamplerError> {
        let memory = stream
            .alloc_zeros::<u8>(bytes)
            .map_err(RuntimeError::from)?;
        let address = address(&memory, stream);
        Ok(Self {
            _memory: memory,
            address,
        })
    }
}

/// A device array of `len` values with the pinned host staging a step writes it from.
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
    /// i32 per row: the slot each selected row samples under, as the kernel indexes it.
    row_slots: StagedArray<i32>,
    /// i32 per token row: the slot the row takes its token from, or negative to keep the host's.
    gather_slots: StagedArray<i32>,
    /// u32 per row: the token sampled for each selected row this step.
    out: DeviceArray,
    readback: Readback<u32>,
    /// Recorded behind every upload; waited on before the staging is freed.
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
        let sampler = Self {
            max_rows,
            vocab,
            records: StagedArray::new(stream, slots)?,
            pending_records: Vec::new(),
            sampled: DeviceArray::zeroed(stream, slots * size_of::<u32>())?,
            row_slots: StagedArray::new(stream, max_rows)?,
            gather_slots: StagedArray::new(stream, max_rows)?,
            out: DeviceArray::zeroed(stream, max_rows * size_of::<u32>())?,
            readback: Readback::new(allocation, context, max_rows, 1)?,
            uploaded,
            owners: SlotOwners::new(slots),
            staged: None,
        };
        info!(slots, max_rows, vocab, "device sampler allocated");
        Ok(sampler)
    }

    /// Decides `layout`'s step and writes its inputs into the staging: the records of the slots
    /// that changed hands, the slot of every selected row, and which token rows gather.
    ///
    /// `gather_rows` is how many leading token rows the gather covers, which only a caller
    /// holding a batch of one token per entry may state; every other caller states none and the
    /// host's uploaded token ids stand.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError`] when the layout has more rows than the sampler holds, or its
    /// inputs cannot be decided.
    pub fn stage(
        &mut self,
        layout: &BatchLayout,
        gather_rows: Option<usize>,
    ) -> Result<(), SamplerError> {
        self.staged = None;
        let rows = layout.sampling.len();
        if rows > self.max_rows {
            return Err(SamplerError::TooManyRows {
                rows,
                max_rows: self.max_rows,
            });
        }
        let SamplerInputs {
            records,
            row_slots,
            gather,
        } = SamplerInputs::for_step(layout, &mut self.owners, gather_rows)?;
        if gather.len() > self.max_rows {
            return Err(SamplerError::TooManyGatherRows {
                rows: gather.len(),
                max_rows: self.max_rows,
            });
        }
        self.pending_records.clear();
        let records_host = self.records.host.as_mut_slice();
        for (slot, record) in records {
            let index = slot.index();
            records_host[index] = record;
            self.pending_records.push(index);
        }
        let row_slots_host = self.row_slots.host.as_mut_slice();
        for (row, slot) in row_slots.iter().enumerate() {
            row_slots_host[row] = slot_i32(*slot);
        }
        let gather_host = self.gather_slots.host.as_mut_slice();
        for (row, slot) in gather.iter().enumerate() {
            gather_host[row] = slot.map_or(-1, slot_i32);
        }
        self.staged = Some(StagedStep {
            rows,
            gather_rows: gather.len(),
        });
        Ok(())
    }

    /// The descriptor that copies the staged step's inputs to the device.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError::NoStepStaged`] when no step is staged.
    pub fn upload(&self) -> Result<Upload<'_>, SamplerError> {
        let staged = self.staged.ok_or(SamplerError::NoStepStaged)?;
        Ok(Upload {
            sampler: self,
            staged,
        })
    }

    /// The descriptor that overwrites the gathering token rows of the u32 token ids at
    /// `token_ids` with the token last sampled for their slot.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError::NoStepStaged`] when no step is staged.
    pub fn gather(&self, token_ids: u64) -> Result<Gather, SamplerError> {
        let staged = self.staged.ok_or(SamplerError::NoStepStaged)?;
        Ok(Gather {
            call: GatherCall {
                token_ids,
                gather_slots: self.gather_slots.address(),
                sampled: self.sampled.address,
                n_rows: staged.gather_rows,
                stream: ptr::null_mut(),
            },
        })
    }

    /// The descriptor that samples every selected row from the f32 logits at `logits`, one row
    /// per selected row a vocabulary wide, and copies the tokens back for [`DeviceSampler::wait`].
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError::NoStepStaged`] when no step is staged.
    pub fn sample(&mut self, logits: u64) -> Result<Sample<'_>, SamplerError> {
        let staged = self.staged.ok_or(SamplerError::NoStepStaged)?;
        let call = SampleCall {
            logits,
            row_slots: self.row_slots.address(),
            records: self.records.address(),
            sampled: self.sampled.address,
            out: self.out.address,
            vocab: self.vocab,
            n_rows: staged.rows,
            stream: ptr::null_mut(),
        };
        let copy = self.readback.copy(self.out.address, staged.rows)?;
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

    /// Runs the staged step's sampling on `stream` — the upload, the sample and the readback —
    /// over the f32 logits `logits` holds, and waits for its tokens: the candle path, which has
    /// no descriptor seam.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError`] when no step is staged or the driver refuses a copy or a launch.
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
        // SAFETY: candle's stream is live in the sampler's context, and every address the
        // descriptors name is this sampler's, or the logits the stream's earlier work wrote.
        unsafe {
            self.upload()?.enqueue(stream.cu_stream())?;
            self.sample(address)?.enqueue(stream.cu_stream())?;
        }
        self.wait()
    }
}

impl Drop for DeviceSampler {
    fn drop(&mut self) {
        // The last upload may still be reading the staging; the event waits for it before the
        // arrays go. A failure here cannot be acted on beyond saying so.
        if let Err(error) = self.uploaded.synchronize() {
            warn!(%error, "the sampler's last upload could not be waited on before its staging goes");
        }
    }
}

/// The upload of one step's sampler inputs: the changed records, the rows' slots and the gather
/// slots.
pub struct Upload<'a> {
    sampler: &'a DeviceSampler,
    staged: StagedStep,
}

impl Descriptor for Upload<'_> {
    type Error = SamplerError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), SamplerError> {
        let sampler = self.sampler;
        let records = sampler.records.host.as_slice();
        // SAFETY: the session hands a live stream in the buffers' context; every destination is
        // this sampler's device array and every source its pinned staging, which outlives the
        // copies through the event recorded behind them.
        unsafe {
            for &slot in &sampler.pending_records {
                let destination = sampler.records.address() + (slot * RECORD_BYTES) as u64;
                memcpy_htod_async(destination, &records[slot..=slot], stream)
                    .map_err(RuntimeError::from)?;
            }
            memcpy_htod_async(
                sampler.row_slots.address(),
                &sampler.row_slots.host.as_slice()[..self.staged.rows],
                stream,
            )
            .map_err(RuntimeError::from)?;
            memcpy_htod_async(
                sampler.gather_slots.address(),
                &sampler.gather_slots.host.as_slice()[..self.staged.gather_rows],
                stream,
            )
            .map_err(RuntimeError::from)?;
            event::record(sampler.uploaded.cu_event(), stream).map_err(RuntimeError::from)?;
        }
        Ok(())
    }
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

/// A staged slot as the kernel indexes it; the mirror refused any past the count that fits.
fn slot_i32(slot: RequestSlot) -> i32 {
    i32::try_from(slot.get()).expect("the mirror holds only slots that fit the kernel's index")
}

/// The device address of a buffer; event tracking is disabled at context creation, so the read
/// guard is a no-op and the address is stable for the buffer's lifetime.
fn address<T>(slice: &CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
    let (address, _reads) = slice.device_ptr(stream);
    address
}
