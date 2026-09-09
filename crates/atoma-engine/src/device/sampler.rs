//! The sampler on one rank's device: every request slot's record and last sampled token, the
//! rows' inputs and outputs, all at fixed addresses from the Allocation session phase on, and the
//! descriptors that run a step's sampling through the seam.
//!
//! What crosses to the device each step is small and goes up, not down. The sampling records of
//! the slots that changed hands are staged in the sampler's own pinned memory and uploaded as
//! sparse copies, one per record, in front of everything else the step copies. The slot each
//! selected row samples under, and which token rows take their token from the device, are the
//! sampler's two per-step arrays, and [`DeviceSampler::stage`] writes them, with the count of
//! rows that sample, where the caller says: the decode step stages them beside the model's
//! inputs in its packed block, which one copy-in carries, and hands [`DeviceSampler::gather`]
//! and [`DeviceSampler::sample`] tensor views over where the two arrays and the count landed
//! and over the logits. Each view is held to the staged step's rows and the dtype the kernel
//! reads, so a view handed to the wrong argument is refused by name, and each descriptor holds
//! the sampler borrowed for its life, so it cannot outlive the staged step whose rows it names.
//! A dummy run is staged by [`DeviceSampler::stage_dummy_run`] over the caller's staging of it,
//! with every row of its bucket covered and none live, which is what a recording of the bucket
//! is made over. The sample describes the launch alone, bounded by the count that came up with the
//! arrays — the kernel returns for a row at or past it, reading neither its logits nor its slot —
//! and leaves each live row's token on the device; [`DeviceSampler::read_tokens`] describes the one
//! asynchronous copy that brings them back, through the leading rows of the row tokens view —
//! the staged step's rows and no others — enqueued behind the sample and waited on once the
//! step is enqueued through the readback's own event and nothing else: the host learns what was
//! sampled for detokenisation and finish detection, and the device never waits for it. The
//! readback is described behind a [`Sampled`] witness, minted where a sample is enqueued and
//! nowhere else, so a copy of tokens no sample wrote cannot be asked for. The sampled tokens
//! stay in the per-slot array the next step's gather reads.
//!
//! The candle forward samples through the same state on candle's stream. An eager step gathers
//! nothing, so [`DeviceSampler::stage_eager`] writes its row slots and its live-row count into
//! the sampler's own pinned memory and [`DeviceSampler::run_on`] uploads them there with the
//! records and enqueues the sample and the readback in turn; the host wait that ends every step,
//! on either stream, is what orders the two streams' use of the sampler's device state.
//!
//! Every descriptor names the sampler's five device arrays by the addresses read once at
//! Allocation. [`DeviceSampler::addresses`] reads them again from the arrays themselves, by name
//! in one fixed order: what the forward bakes when it is built and a debug build compares a
//! fresh reading against before each keyed step.

use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;
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
use crate::decode::staging::{stage_sampler, DummyRun, SamplerArrays, StagingError};
use crate::pinned::Pinned;
use crate::readback::{Readback, ReadbackCopy, ReadbackError};
use crate::sampling::inputs::{SamplerInputs, SamplerInputsError};
use crate::sampling::owners::SlotOwners;
use crate::sampling::record::{SlotRecord, RECORD_BYTES};

/// Why the sampler could not be built or run.
#[derive(Debug, Error)]
pub enum SamplerError {
    /// More rows staged than the sampler was sized for.
    #[error("{rows} rows are staged this step but the sampler holds {max_rows} at most")]
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
    /// A sample's witness was asked for before the sample was enqueued.
    #[error(
        "the sample was not enqueued; enqueue it before describing the readback of its tokens"
    )]
    SampleNotEnqueued,
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

/// Whose memory a staged step's two per-step arrays and its live-row count are in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArraysIn {
    /// The sampler's own pinned memory: an eager step's, uploaded by [`DeviceSampler::run_on`].
    Sampler,
    /// The caller's staging: a decode step's, copied in with the step's inputs.
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
/// address the view names stays allocated for as long as the view is read.
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
    /// Whose memory the row slots, the gather slots and the live-row count were written to.
    arrays: ArraysIn,
}

impl StagedStep {
    /// The step `inputs` decided, its row slots, gather slots and live-row count written to
    /// `arrays`.
    fn of(inputs: &SamplerInputs, arrays: ArraysIn) -> Self {
        Self {
            rows: inputs.row_slots.len(),
            gather_rows: inputs.gather.len(),
            arrays,
        }
    }
}

/// Where a sample reads its three per-step inputs on the device: the logits, the slot of every
/// selected row, and the count of the rows that are live. Each is taken from the thing that
/// holds it, under the name the kernel reads it by, since neither the launch nor the kernel can
/// tell one device address from another.
#[derive(Debug, Clone, Copy)]
struct SampleAddresses {
    logits: u64,
    row_slots: u64,
    live_rows: u64,
}

impl SampleAddresses {
    /// Where an eager step reads: `logits` on `stream`, and the sampler's own row slots and
    /// live-row count, which its upload wrote. Each array is taken as the element type the
    /// kernel reads it as, so no two can be passed over each other.
    fn eager<S: DevicePtr<f32>>(
        logits: &S,
        stream: &Arc<CudaStream>,
        row_slots: &StagedArray<i32>,
        live_rows: &StagedArray<u32>,
    ) -> Self {
        let (logits, _reads) = logits.device_ptr(stream);
        Self {
            logits,
            row_slots: row_slots.address(),
            live_rows: live_rows.address(),
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
    /// One u32: how many rows an eager step samples, which is what its sample launch is bounded
    /// by. A decode step's count comes up in its own copy-in, beside its row slots.
    live_rows: StagedArray<u32>,
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
            live_rows: StagedArray::new(stream, 1)?,
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
    /// and live-row count, and the row tokens. The forward bakes this reading when it is built
    /// and a debug build reads it again before each keyed step.
    #[must_use]
    pub fn addresses(&self, stream: &Arc<CudaStream>) -> [BakedAddress; 5] {
        [
            (BakedName::SamplingRecords, &self.records.device),
            (BakedName::SampledTokens, &self.sampled),
            (BakedName::SamplerRowSlots, &self.row_slots.device),
            (BakedName::SamplerLiveRows, &self.live_rows.device),
            (BakedName::RowTokens, &self.row_tokens.array),
        ]
        .map(|(name, array)| BakedAddress {
            name,
            address: address(&array.memory, stream),
        })
    }

    /// Decides `layout`'s step and stages its inputs: the records of the slots that changed
    /// hands into the sampler's own staging, and the slot of every selected row, which token
    /// rows gather and how many rows sample into `arrays`, as the kernels read them.
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
    /// sampler's own memory: the records of the slots that changed hands, the slot of every
    /// selected row, and the count of the rows that sample, which the step's launch is bounded
    /// by.
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
                live_rows: &mut self.live_rows.host.as_mut_slice()[0],
            },
        )?;
        self.staged = Some(StagedStep::of(&inputs, ArraysIn::Sampler));
        Ok(())
    }

    /// Stages `run`: what a recording of its bucket is made over. No record changes hands and
    /// nothing is written here — a dummy run's row slots, gather slots and live-row count are the
    /// caller's to stage, as [`stage_dummy`](crate::decode::staging::stage_dummy) does, naming no
    /// request slot and a count of zero — so what this settles is the rows the run's descriptors
    /// cover: every row of the bucket, for the gather and the sample alike, since a graph bakes
    /// the bucket's rows and the sample takes its live count from the device. No row samples
    /// under the step it leaves — the sample launched over it returns for every row — and no
    /// readback follows it, so there is nothing to wait for; the next [`DeviceSampler::stage`]
    /// or [`DeviceSampler::stage_eager`] replaces it.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError::TooManyRows`] when the run has more rows than the sampler holds.
    pub fn stage_dummy_run(&mut self, run: &DummyRun) -> Result<(), SamplerError> {
        self.staged = None;
        let rows = run.rows();
        if rows > self.max_rows {
            return Err(SamplerError::TooManyRows {
                rows,
                max_rows: self.max_rows,
            });
        }
        self.staged = Some(StagedStep {
            rows,
            gather_rows: rows,
            arrays: ArraysIn::Caller,
        });
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
    /// changed records, then the rows' slots and the live-row count.
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
    /// token row: the array the staged step wrote, viewed where the caller's copy-in put it on
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
    ) -> Result<Gather<'_>, SamplerError> {
        let staged = self.staged.ok_or(SamplerError::NoStepStaged)?;
        let rows = &[staged.gather_rows];
        let token_ids = check_view("token ids", token_ids, Dtype::U32, rows)?;
        let gather_slots = check_view("gather slots", gather_slots, Dtype::I32, rows)?;
        Ok(Gather {
            call: GatherCall {
                token_ids,
                gather_slots,
                sampled: self.sampled.address,
                n_rows: staged.gather_rows,
                stream: ptr::null_mut(),
            },
            _while_staged: PhantomData,
        })
    }

    /// The descriptor that samples from the f32 `logits`, one row per selected row a vocabulary
    /// wide, under the i32 `row_slots`, one per selected row viewed where the staged step's
    /// copy-in put them, and leaves each row's token in the row tokens. Those two views are the
    /// selected rows exactly: the caller narrows them to the live rows. The launch is bounded by
    /// the u32 `live_rows`, the one value the same copy-in carried: the kernel returns for a row
    /// at or past it, so a row the batch does not fill samples nothing. Nothing is read back
    /// here; [`DeviceSampler::read_tokens`] describes the readback.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError::NoStepStaged`] when no step is staged, or a `View` variant when
    /// a view is not the rows and the dtype the kernel reads it as.
    pub fn sample(
        &self,
        logits: &Tensor,
        row_slots: &Tensor,
        live_rows: &Tensor,
    ) -> Result<Sample<'_>, SamplerError> {
        let staged = self.staged.ok_or(SamplerError::NoStepStaged)?;
        self.sample_at(SampleAddresses {
            logits: check_view("logits", logits, Dtype::F32, &[staged.rows, self.vocab])?,
            row_slots: check_view("row slots", row_slots, Dtype::I32, &[staged.rows])?,
            live_rows: check_view("live rows", live_rows, Dtype::U32, &[1])?,
        })
    }

    /// The sample of the staged step's rows from `at`, over the sampler's own records, sampled
    /// tokens and row tokens.
    fn sample_at(&self, at: SampleAddresses) -> Result<Sample<'_>, SamplerError> {
        let staged = self.staged.ok_or(SamplerError::NoStepStaged)?;
        let tokens = self.staged_tokens(staged.rows)?;
        let SampleAddresses {
            logits,
            row_slots,
            live_rows,
        } = at;
        let call = SampleCall {
            logits,
            row_slots,
            records: self.records.address(),
            sampled: self.sampled.address,
            out: tokens.address(),
            live_rows,
            vocab: self.vocab,
            n_rows: staged.rows,
            stream: ptr::null_mut(),
        };
        Ok(Sample {
            call,
            enqueued: false,
            _while_staged: PhantomData,
        })
    }

    /// The descriptor that copies the staged step's tokens to the host, through the leading rows
    /// of the row tokens view: the rows the sample writes and no others, brought back in one
    /// asynchronous copy for [`DeviceSampler::wait`]. It is described behind `sampled`, the
    /// witness that a sample of the staged step reached the stream, and is the caller's to
    /// enqueue behind that sample, on the stream the sample is enqueued on; describing it is
    /// already what [`DeviceSampler::wait`] waits for, so one dropped unenqueued leaves that wait
    /// on an earlier copy's event and returns stale rows. Nothing here checks either clause.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError::NoStepStaged`] when no step is staged.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the witness is consumed on purpose: one sample, one readback described behind it"
    )]
    pub fn read_tokens(&mut self, sampled: Sampled) -> Result<ReadbackCopy<'_, u32>, SamplerError> {
        let Sampled { _witnessed: () } = sampled;
        let staged = self.staged.ok_or(SamplerError::NoStepStaged)?;
        let tokens = self.staged_tokens(staged.rows)?;
        Ok(self.readback.copy(&tokens)?)
    }

    /// The u32 view of the row tokens `rows` rows are sampled into: the leading rows of the row
    /// tokens view.
    fn staged_tokens(&self, rows: usize) -> Result<Tensor, SamplerError> {
        Ok(self.row_tokens.view.narrow(0, 0, rows)?)
    }

    /// Waits for the staged step's tokens, and that copy alone, and returns them: one per
    /// selected row, in batch order.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError`] when no step is staged, [`DeviceSampler::read_tokens`] described
    /// no copy, or the wait fails.
    pub fn wait(&mut self) -> Result<&[u32], SamplerError> {
        if self.staged.take().is_none() {
            return Err(SamplerError::NoStepStaged);
        }
        Ok(self.readback.wait()?)
    }

    /// Runs the eager step staged by [`DeviceSampler::stage_eager`] on `stream` — the upload
    /// from the sampler's own staging, the sample and the readback of its tokens — over the f32
    /// logits `logits` holds, and waits for those tokens: the candle path, which has no
    /// descriptor seam.
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
        let at = SampleAddresses::eager(logits, stream, &self.row_slots, &self.live_rows);
        // SAFETY: candle's stream is live in the sampler's context, and every address the
        // descriptors name is this sampler's, or the logits the stream's earlier work wrote.
        let sampled = unsafe {
            self.upload_eager()?.enqueue(stream.cu_stream())?;
            let mut sample = self.sample_at(at)?;
            sample.enqueue(stream.cu_stream())?;
            sample.sampled()?
        };
        // SAFETY: as above; the readback copies what the sample just enqueued wrote.
        unsafe {
            self.read_tokens(sampled)?.enqueue(stream.cu_stream())?;
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
/// then the rows' slots and the live-row count, with the sampler's event recorded behind them.
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
            memcpy_htod_async(
                sampler.live_rows.address(),
                sampler.live_rows.host.as_slice(),
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

/// Borrow marker: a descriptor is minted for the step staged at the time and names its rows, so it
/// holds the sampler borrowed for as long as it lives. Every `&mut self` method the sampler exposes
/// is blocked while one is live — [`DeviceSampler::stage`], [`DeviceSampler::stage_eager`],
/// [`DeviceSampler::read_tokens`], [`DeviceSampler::wait`] and [`DeviceSampler::run_on`] — so
/// restaging, waiting the staged step out, or describing its readback while a descriptor is still
/// to be enqueued is a compile error rather than a launch whose row count is the old step's over
/// the arrays the new one wrote. `read_tokens` among them is what makes the readback's place —
/// behind the launches, never among them — a type error rather than a convention.
type WhileStaged<'a> = PhantomData<&'a DeviceSampler>;

/// The gather of the token rows whose token the device sampled last.
///
/// It holds the sampler borrowed while it lives, so staging another step over the arrays it
/// names, while it is still to be enqueued, does not compile:
///
/// ```compile_fail
/// use atoma_engine::batch::BatchLayout;
/// use atoma_engine::device::sampler::{DeviceSampler, SamplerError};
/// use atoma_runtime::tensor::Tensor;
///
/// fn stage_behind_the_gather(
///     sampler: &mut DeviceSampler,
///     token_ids: &Tensor,
///     gather_slots: &Tensor,
///     layout: &BatchLayout,
/// ) -> Result<(), SamplerError> {
///     let gather = sampler.gather(token_ids, gather_slots)?;
///     sampler.stage_eager(layout)?;
///     drop(gather);
///     Ok(())
/// }
/// ```
///
/// Staging once the gather is done with does:
///
/// ```
/// use atoma_engine::batch::BatchLayout;
/// use atoma_engine::device::sampler::{DeviceSampler, SamplerError};
/// use atoma_runtime::tensor::Tensor;
///
/// fn stage_after_the_gather(
///     sampler: &mut DeviceSampler,
///     token_ids: &Tensor,
///     gather_slots: &Tensor,
///     layout: &BatchLayout,
/// ) -> Result<(), SamplerError> {
///     let gather = sampler.gather(token_ids, gather_slots)?;
///     drop(gather);
///     sampler.stage_eager(layout)?;
///     Ok(())
/// }
/// ```
pub struct Gather<'a> {
    call: GatherCall,
    _while_staged: WhileStaged<'a>,
}

impl Descriptor for Gather<'_> {
    type Error = SamplerError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), SamplerError> {
        let call = GatherCall {
            stream: stream.cast::<c_void>(),
            ..self.call
        };
        // SAFETY: the session hands a live stream; the token ids are the step's copied-in inputs
        // and the slots were staged against this sampler's arrays.
        unsafe { gather(&call) }?;
        Ok(())
    }
}

/// The sample of the staged step's rows, bounded by the live-row count on the device, which
/// leaves each live row's token in the row tokens.
///
/// It holds the sampler borrowed while it lives, so waiting the staged step out, while the
/// sample is still to be enqueued, does not compile:
///
/// ```compile_fail
/// use atoma_engine::device::sampler::{DeviceSampler, SamplerError};
/// use atoma_runtime::tensor::Tensor;
///
/// fn wait_out_the_sample(
///     sampler: &mut DeviceSampler,
///     logits: &Tensor,
///     row_slots: &Tensor,
///     live_rows: &Tensor,
/// ) -> Result<(), SamplerError> {
///     let sample = sampler.sample(logits, row_slots, live_rows)?;
///     sampler.wait()?;
///     drop(sample);
///     Ok(())
/// }
/// ```
///
/// Waiting once the sample is done with does:
///
/// ```
/// use atoma_engine::device::sampler::{DeviceSampler, SamplerError};
/// use atoma_runtime::tensor::Tensor;
///
/// fn wait_after_the_sample(
///     sampler: &mut DeviceSampler,
///     logits: &Tensor,
///     row_slots: &Tensor,
///     live_rows: &Tensor,
/// ) -> Result<(), SamplerError> {
///     let sample = sampler.sample(logits, row_slots, live_rows)?;
///     drop(sample);
///     sampler.wait()?;
///     Ok(())
/// }
/// ```
pub struct Sample<'a> {
    call: SampleCall,
    enqueued: bool,
    _while_staged: WhileStaged<'a>,
}

impl Sample<'_> {
    /// The witness that this sample reached the stream, once it has been enqueued: what the
    /// readback of its tokens is described behind.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerError::SampleNotEnqueued`] when it has not been.
    pub fn sampled(self) -> Result<Sampled, SamplerError> {
        if !self.enqueued {
            return Err(SamplerError::SampleNotEnqueued);
        }
        Ok(Sampled::witnessed())
    }
}

impl Descriptor for Sample<'_> {
    type Error = SamplerError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), SamplerError> {
        let call = SampleCall {
            stream: stream.cast::<c_void>(),
            ..self.call
        };
        // SAFETY: the session hands a live stream; the logits are what the stream's earlier
        // work wrote, the row slots and the live-row count came up in front of it, and every
        // other address is this sampler's, staged for these rows.
        unsafe { sample(&call) }?;
        self.enqueued = true;
        Ok(())
    }
}

/// That a sample of the staged step reached the stream: what a readback of its tokens is
/// described behind. Minted by a [`Sample`] once it has been enqueued, and by the decode step
/// once it has replayed a graph that holds one, and nowhere else, so
/// [`DeviceSampler::read_tokens`] cannot be asked for a copy of tokens no sample wrote.
pub struct Sampled {
    _witnessed: (),
}

impl Sampled {
    /// The witness, for an enqueued sample and for the replay of a graph that holds one.
    pub(crate) fn witnessed() -> Self {
        Self { _witnessed: () }
    }
}

/// Holds `view` to contiguous `dims` of `dtype`, which is what the kernel reads `what` as,
/// refusing by the first of dtype, contiguity and shape that fails, and takes the address the
/// launch reads: what a kernel is handed under a name is what was checked under it.
fn check_view(
    what: &'static str,
    view: &Tensor,
    dtype: Dtype,
    dims: &[usize],
) -> Result<u64, SamplerError> {
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
    Ok(view.address())
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
        let taken = check_view("logits", &live, Dtype::F32, &[2, VOCAB]).unwrap();
        assert_eq!(taken, BASE, "the address taken is the view's own");

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
