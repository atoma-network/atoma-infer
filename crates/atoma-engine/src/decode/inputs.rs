//! One step's inputs on their way to the device: pinned host staging, the fixed device buffers
//! the graphs bake, and the descriptors that carry a step from one to the other.
//!
//! Each input has one pinned host array and one device buffer, both sized at the largest bucket
//! and allocated once, in the Allocation phase. Before a step the host arrays are written from
//! the batch layout; the upload descriptor then copies the bucket's rows of each to the device
//! on the capture stream, asynchronously, which is what pinned memory is for. The device buffers
//! never move, so the views the step reads them through are minted once.
//!
//! Reuse is fenced by the step itself: every step ends in a readback wait, and the upload and the
//! step precede the readback on the same stream, so by the time the host writes the next step's
//! inputs the previous step has finished reading both copies.
//!
//! [`WaitEvent`] orders the decode step after candle's stream: a prefill runs there, and the
//! step must not read the cache it wrote until it is written.

use std::sync::Arc;

use atoma_runtime::error::RuntimeError;
use atoma_runtime::session::{Allocation, Descriptor};
use atoma_runtime::tensor::{Dtype, Layout, Tensor, TensorError};
use cudarc::driver::result::{event, memcpy_htod_async, stream};
use cudarc::driver::sys::{self, CUevent_flags, CUevent_wait_flags};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr};
use thiserror::Error;
use tracing::warn;

use crate::batch::BatchLayout;
use crate::decode::batch::DecodeBatch;
use crate::decode::staging::{stage, StagingArrays, StagingError, StagingShape};
use crate::pinned::Pinned;

/// Why the inputs could not be allocated, staged or uploaded.
#[derive(Debug, Error)]
pub enum InputsError {
    #[error(transparent)]
    Driver(#[from] RuntimeError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Staging(#[from] StagingError),
}

/// The layouts of the five inputs at `shape`: what each device buffer holds and each view reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputLayouts {
    /// u32 `[max_tokens]`.
    pub token_ids: Layout,
    /// i32 `[max_tokens]`.
    pub positions: Layout,
    /// i32 `[max_tokens]`.
    pub seqlens_k: Layout,
    /// i64 `[max_tokens]`.
    pub slot_mapping: Layout,
    /// i32 `[max_tokens, block_table_width]`.
    pub block_table: Layout,
}

impl InputLayouts {
    /// The layouts every input takes at `shape`.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError`] only for a shape a layout cannot hold, which no staging shape is.
    pub fn new(shape: StagingShape) -> Result<Self, TensorError> {
        let rows = [shape.max_tokens];
        Ok(Self {
            token_ids: Layout::contiguous(&rows, Dtype::U32)?,
            positions: Layout::contiguous(&rows, Dtype::I32)?,
            seqlens_k: Layout::contiguous(&rows, Dtype::I32)?,
            slot_mapping: Layout::contiguous(&rows, Dtype::I64)?,
            block_table: Layout::contiguous(
                &[shape.max_tokens, shape.block_table_width],
                Dtype::I32,
            )?,
        })
    }
}

/// The views the step reads each input through, each over its whole device buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputTensors {
    pub token_ids: Tensor,
    pub positions: Tensor,
    pub seqlens_k: Tensor,
    pub slot_mapping: Tensor,
    pub block_table: Tensor,
}

/// One input: its pinned staging, its device buffer, and the view over the buffer.
struct Staged<T> {
    host: Pinned<T>,
    /// Owned here so the address the view names stays allocated for as long as the view does.
    _device: CudaSlice<u8>,
    tensor: Tensor,
}

impl<T> Staged<T> {
    fn new(
        allocation: &Allocation,
        stream: &Arc<CudaStream>,
        layout: Layout,
    ) -> Result<Self, InputsError> {
        let host = Pinned::new(layout.element_count())?;
        let device = stream
            .alloc_zeros::<u8>(layout.extent_bytes())
            .map_err(RuntimeError::from)?;
        // The address is read before the buffer moves into this value, and the read guard is
        // dropped with the block; device allocations do not move.
        let address = {
            let (address, _reads) = device.device_ptr(stream);
            address
        };
        Ok(Self {
            host,
            tensor: Tensor::new(allocation, address, layout)?,
            _device: device,
        })
    }

    /// Copies the first `elements` staged values to the device on `stream`.
    ///
    /// # Safety
    ///
    /// `stream` must be a live stream in the buffers' context.
    unsafe fn upload(&self, elements: usize, stream: sys::CUstream) -> Result<(), RuntimeError> {
        let source = &self.host.as_slice()[..elements];
        // SAFETY: the destination is this input's device buffer, which holds at least as many
        // elements as the staging it mirrors; the source is pinned and outlives the copy, which
        // the event recorded after every upload fences before the staging is freed.
        unsafe { memcpy_htod_async(self.tensor.address(), source, stream) }?;
        Ok(())
    }
}

/// One step's inputs: pinned staging and fixed device buffers for each, sized at the largest
/// bucket.
pub struct DecodeInputs {
    shape: StagingShape,
    token_ids: Staged<u32>,
    positions: Staged<i32>,
    seqlens_k: Staged<i32>,
    slot_mapping: Staged<i64>,
    block_table: Staged<i32>,
    /// Recorded behind every upload; waited on before the staging is freed.
    uploaded: CudaEvent,
}

impl DecodeInputs {
    /// Allocates the staging and the device buffers of `shape`, during the Allocation session
    /// phase, on `stream`'s device.
    ///
    /// # Errors
    ///
    /// Returns [`InputsError`] when the driver cannot pin or allocate a buffer, or a view over
    /// one cannot be minted.
    pub fn new(
        allocation: &Allocation,
        stream: &Arc<CudaStream>,
        shape: StagingShape,
    ) -> Result<Self, InputsError> {
        let layouts = InputLayouts::new(shape)?;
        let context = stream.context();
        context.bind_to_thread().map_err(RuntimeError::from)?;
        let uploaded = context
            .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .map_err(RuntimeError::from)?;
        Ok(Self {
            shape,
            token_ids: Staged::new(allocation, stream, layouts.token_ids)?,
            positions: Staged::new(allocation, stream, layouts.positions)?,
            seqlens_k: Staged::new(allocation, stream, layouts.seqlens_k)?,
            slot_mapping: Staged::new(allocation, stream, layouts.slot_mapping)?,
            block_table: Staged::new(allocation, stream, layouts.block_table)?,
            uploaded,
        })
    }

    #[must_use]
    pub fn shape(&self) -> StagingShape {
        self.shape
    }

    /// The views the step reads the inputs through.
    #[must_use]
    pub fn tensors(&self) -> InputTensors {
        InputTensors {
            token_ids: self.token_ids.tensor,
            positions: self.positions.tensor,
            seqlens_k: self.seqlens_k.tensor,
            slot_mapping: self.slot_mapping.tensor,
            block_table: self.block_table.tensor,
        }
    }

    /// Writes `batch`'s inputs from `layout` into the staging.
    ///
    /// # Errors
    ///
    /// Returns [`InputsError::Staging`] when the layout cannot be staged at this shape.
    pub fn stage(&mut self, layout: &BatchLayout, batch: &DecodeBatch) -> Result<(), InputsError> {
        stage(
            layout,
            batch,
            self.shape,
            StagingArrays {
                token_ids: self.token_ids.host.as_mut_slice(),
                positions: self.positions.host.as_mut_slice(),
                seqlens_k: self.seqlens_k.host.as_mut_slice(),
                slot_mapping: self.slot_mapping.host.as_mut_slice(),
                block_table: self.block_table.host.as_mut_slice(),
            },
        )?;
        Ok(())
    }

    /// The descriptor that copies `batch`'s rows of every input to the device.
    #[must_use]
    pub fn upload(&self, batch: &DecodeBatch) -> Upload<'_> {
        Upload {
            inputs: self,
            tokens: batch.tokens,
        }
    }
}

impl Drop for DecodeInputs {
    fn drop(&mut self) {
        // The last upload may still be reading the staging; the event waits for it before the
        // arrays go. A failure here cannot be acted on beyond saying so.
        if let Err(error) = self.uploaded.synchronize() {
            warn!(%error, "the last input upload could not be waited on before its staging goes");
        }
    }
}

/// The upload of one bucket's rows of every input, enqueued on the capture stream.
pub struct Upload<'a> {
    inputs: &'a DecodeInputs,
    tokens: usize,
}

impl Descriptor for Upload<'_> {
    type Error = InputsError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), InputsError> {
        let inputs = self.inputs;
        let tokens = self.tokens;
        // SAFETY: the session hands a live stream in the buffers' context, and every input
        // holds at least the largest bucket's rows.
        unsafe {
            inputs.token_ids.upload(tokens, stream)?;
            inputs.positions.upload(tokens, stream)?;
            inputs.seqlens_k.upload(tokens, stream)?;
            inputs.slot_mapping.upload(tokens, stream)?;
            inputs
                .block_table
                .upload(tokens * inputs.shape.block_table_width, stream)?;
            event::record(inputs.uploaded.cu_event(), stream).map_err(RuntimeError::from)?;
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
    use super::*;

    #[test]
    fn the_input_layouts_are_the_kernels_dtypes_at_the_largest_bucket() {
        let layouts = InputLayouts::new(StagingShape {
            max_tokens: 32,
            block_table_width: 256,
            max_position: 8192,
        })
        .unwrap();

        assert_eq!(layouts.token_ids.dims(), [32]);
        assert_eq!(layouts.token_ids.dtype(), Dtype::U32);
        assert_eq!(layouts.positions.dtype(), Dtype::I32);
        assert_eq!(layouts.seqlens_k.dtype(), Dtype::I32);
        assert_eq!(layouts.slot_mapping.dtype(), Dtype::I64);
        assert_eq!(layouts.slot_mapping.extent_bytes(), 32 * 8);
        assert_eq!(layouts.block_table.dims(), [32, 256]);
        assert_eq!(layouts.block_table.dtype(), Dtype::I32);
        assert_eq!(layouts.block_table.extent_bytes(), 32 * 256 * 4);
    }
}
