//! Reading what a step produced for the host back in one event-fenced copy: rows of one plain
//! value type, a fixed width each.
//!
//! The pinned host buffer is allocated once, sized for the most rows a step can select, and every
//! step copies its rows into it with one asynchronous device-to-host copy on the forward's
//! stream, records the buffer's own event behind the copy, and waits on that event and nothing
//! else: no stream synchronize and no device-wide wait, so whatever else the stream holds behind
//! the copy is not waited for.
//!
//! Two paths reach the buffer. The candle forward copies from a device tensor on candle's stream
//! and waits in one call. The decode step over runtime tensors enqueues the copy through the seam,
//! as the last descriptor of a step on the capture stream, and waits on it separately once the step
//! is enqueued; that copy takes a tensor view of the rows, narrowed to the live ones, and reads
//! its address, its row count and its width off the view, refusing one that is not contiguous
//! rows of the readback's value type and width.

use std::slice;
use std::sync::Arc;

use atoma_runtime::error::RuntimeError;
use atoma_runtime::session::{Allocation, Descriptor};
use atoma_runtime::tensor::{Dtype, Element, Layout, Tensor};
use cudarc::driver::result::{event, memcpy_dtoh_async};
use cudarc::driver::sys::{self, CUevent_flags};
use cudarc::driver::{CudaContext, CudaEvent, CudaStream, DevicePtr};
use thiserror::Error;
use tracing::warn;

use crate::pinned::Pinned;

/// Why what a step produced could not be read back.
#[derive(Debug, Error)]
pub enum ReadbackError {
    /// The caller selected more rows than the readback was sized for.
    #[error("{rows} rows were selected but the readback holds {max_rows} at most")]
    TooManyRows { rows: usize, max_rows: usize },
    /// The source is not `rows` rows of the readback's width.
    #[error("the device holds {len} values, not {rows} rows of {width}")]
    Shape {
        len: usize,
        rows: usize,
        width: usize,
    },
    /// The view is of another value type than the readback copies.
    #[error("the view holds {held:?} values; the readback copies {expected:?}")]
    Dtype { held: Dtype, expected: Dtype },
    /// A gapped view: one copy carries the gaps as values.
    #[error(
        "the view has strides {:?}; one copy brings back contiguous rows",
        layout.strides()
    )]
    NotContiguous { layout: Layout },
    /// A scalar view: nothing to count rows of.
    #[error("a scalar view has no rows; the readback copies rows of {width}")]
    Scalar { width: usize },
    /// A wait with no copy described before it.
    #[error("no readback copy is pending; describe one with `copy` before waiting on it")]
    NoCopyPending,
    #[error(transparent)]
    Driver(#[from] RuntimeError),
}

/// The pinned host buffer a step's rows of `T` are copied into.
pub struct Readback<T> {
    /// `max_rows * width` values, written by the copy and read after its wait.
    buffer: Pinned<T>,
    /// Recorded behind every copy; waited on before the host reads, and before the buffer is
    /// freed.
    event: CudaEvent,
    /// Values per row.
    width: usize,
    max_rows: usize,
    /// Values the copy described by [`Readback::copy`] brings back, until waited on.
    pending: Option<usize>,
}

impl<T: Element> Readback<T> {
    /// A readback for up to `max_rows` rows of `width` values, pinned in `context`'s host
    /// memory during the Allocation session phase, which is taken as a witness.
    ///
    /// # Errors
    ///
    /// Returns [`ReadbackError::Driver`] when the driver cannot pin the buffer or create its
    /// event.
    pub fn new(
        _allocation: &Allocation,
        context: &Arc<CudaContext>,
        max_rows: usize,
        width: usize,
    ) -> Result<Self, ReadbackError> {
        let event = context
            .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .map_err(RuntimeError::from)?;
        context.bind_to_thread().map_err(RuntimeError::from)?;
        let buffer = Pinned::new(max_rows * width)?;
        Ok(Self {
            buffer,
            event,
            width,
            max_rows,
            pending: None,
        })
    }

    /// The descriptor that copies the rows `view` holds back on the stream it is enqueued on,
    /// and records the event behind the copy for [`Readback::wait`]. The view's first dimension
    /// is its rows and the rest are the row's values; a rank-one view is rows of one value.
    ///
    /// # Errors
    ///
    /// Returns [`ReadbackError`] when the view is not of `T`, not contiguous, a scalar, not rows
    /// of the readback's width, or more rows than the readback holds.
    pub fn copy(&mut self, view: &Tensor) -> Result<ReadbackCopy<'_, T>, ReadbackError> {
        let len = view_len::<T>(view, self.width, self.max_rows)?;
        self.pending = Some(len);
        Ok(ReadbackCopy {
            host: self.buffer.as_mut_ptr(),
            len,
            device: view.address(),
            event: &self.event,
        })
    }

    /// Waits for the copy the last [`Readback::copy`] described, and that copy alone, and
    /// returns its rows, flat.
    ///
    /// # Errors
    ///
    /// Returns [`ReadbackError::NoCopyPending`] when no copy was described since the last wait,
    /// or [`ReadbackError::Driver`] when the wait fails.
    pub fn wait(&mut self) -> Result<&[T], ReadbackError> {
        let Some(len) = self.pending.take() else {
            return Err(ReadbackError::NoCopyPending);
        };
        self.event.synchronize().map_err(RuntimeError::from)?;
        Ok(&self.buffer.as_slice()[..len])
    }

    /// Copies `rows` rows of `source` back on `stream` and waits for that copy alone.
    ///
    /// # Errors
    ///
    /// Returns [`ReadbackError`] when `rows` is more than the readback holds, `source` is not
    /// that many rows of the width, or the driver fails the copy or the wait.
    pub fn read<S: DevicePtr<T>>(
        &mut self,
        stream: &Arc<CudaStream>,
        source: &S,
        rows: usize,
    ) -> Result<&[T], ReadbackError> {
        let len = selected_len(rows, self.width, self.max_rows, source.len())?;
        self.pending = None;
        stream
            .context()
            .bind_to_thread()
            .map_err(RuntimeError::from)?;
        let host = &mut self.buffer.as_mut_slice()[..len];
        let (device, _reads) = source.device_ptr(stream);
        // SAFETY: `device` addresses the `len` values the stream's earlier work wrote, `host` is
        // `len` pinned values, and the event recorded next fences the copy before the host reads.
        unsafe { memcpy_dtoh_async(host, device, stream.cu_stream()) }
            .map_err(RuntimeError::from)?;
        self.event.record(stream).map_err(RuntimeError::from)?;
        self.event.synchronize().map_err(RuntimeError::from)?;
        Ok(host)
    }
}

impl<T> Drop for Readback<T> {
    fn drop(&mut self) {
        // The last copy may still be in flight; the event waits for it before the buffer, which
        // frees itself once this body returns, goes. A failure here cannot be acted on beyond
        // saying so.
        if let Err(error) = self.event.synchronize() {
            warn!(%error, "the readback's last copy could not be waited on before its buffer goes");
        }
    }
}

/// One step's copy of its rows, enqueued on the capture stream as the step's last descriptor.
pub struct ReadbackCopy<'a, T> {
    host: *mut T,
    len: usize,
    device: u64,
    event: &'a CudaEvent,
}

impl<T> Descriptor for ReadbackCopy<'_, T> {
    type Error = ReadbackError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), ReadbackError> {
        // SAFETY: `len` values lie within the pinned buffer and nothing else touches them until
        // the wait; `device` addresses the values the stream's earlier work wrote; the session
        // hands a live stream; and the event recorded behind the copy is what the wait fences.
        unsafe {
            let host = slice::from_raw_parts_mut(self.host, self.len);
            memcpy_dtoh_async(host, self.device, stream).map_err(RuntimeError::from)?;
            event::record(self.event.cu_event(), stream).map_err(RuntimeError::from)?;
        }
        Ok(())
    }
}

/// How many values `view` brings back, once it is contiguous rows of `T` of the readback's
/// width that fit in it.
fn view_len<T: Element>(
    view: &Tensor,
    width: usize,
    max_rows: usize,
) -> Result<usize, ReadbackError> {
    if view.dtype() != T::DTYPE {
        return Err(ReadbackError::Dtype {
            held: view.dtype(),
            expected: T::DTYPE,
        });
    }
    if !view.is_contiguous() {
        return Err(ReadbackError::NotContiguous {
            layout: *view.layout(),
        });
    }
    let [rows, ..] = view.dims() else {
        return Err(ReadbackError::Scalar { width });
    };
    selected_len(*rows, width, max_rows, view.element_count())
}

/// How many values `rows` rows of `width` are, once they fit the readback and match what the
/// device holds.
fn selected_len(
    rows: usize,
    width: usize,
    max_rows: usize,
    device_len: usize,
) -> Result<usize, ReadbackError> {
    if rows > max_rows {
        return Err(ReadbackError::TooManyRows { rows, max_rows });
    }
    let len = rows * width;
    if device_len != len {
        return Err(ReadbackError::Shape {
            len: device_len,
            rows,
            width,
        });
    }
    Ok(len)
}

#[cfg(test)]
mod tests {
    use atoma_runtime::tensor::{Dtype, Layout, Tensor};

    use super::{selected_len, view_len, ReadbackError};

    /// Where a device buffer sits, aligned as a device allocation is.
    const BASE: u64 = 0x7f00_0000_0000;

    fn view(dims: &[usize], dtype: Dtype) -> Tensor {
        Tensor::for_test(BASE, Layout::contiguous(dims, dtype).unwrap()).unwrap()
    }

    #[test]
    fn the_copy_is_the_views_rows_of_the_width_and_nothing_else() {
        // A readback of four rows of eight f32: three rows are 24 values, four fill it, and a
        // step selecting nothing brings back nothing.
        assert_eq!(
            view_len::<f32>(&view(&[3, 8], Dtype::F32), 8, 4).unwrap(),
            24
        );
        assert_eq!(
            view_len::<f32>(&view(&[4, 8], Dtype::F32), 8, 4).unwrap(),
            32
        );
        assert_eq!(
            view_len::<f32>(&view(&[0, 8], Dtype::F32), 8, 4).unwrap(),
            0
        );
        // The sampler's tokens: rows of one u32, so a rank-one view of the rows.
        assert_eq!(view_len::<u32>(&view(&[3], Dtype::U32), 1, 8).unwrap(), 3);
        assert!(matches!(
            view_len::<f32>(&view(&[5, 8], Dtype::F32), 8, 4).unwrap_err(),
            ReadbackError::TooManyRows {
                rows: 5,
                max_rows: 4
            }
        ));
        // Three rows of seven are 21 values, not three rows of eight.
        assert!(matches!(
            view_len::<f32>(&view(&[3, 7], Dtype::F32), 8, 4).unwrap_err(),
            ReadbackError::Shape {
                len: 21,
                rows: 3,
                width: 8
            }
        ));
    }

    #[test]
    fn a_view_of_another_value_type_is_refused_by_both_types() {
        assert!(matches!(
            view_len::<f32>(&view(&[3, 8], Dtype::U32), 8, 4).unwrap_err(),
            ReadbackError::Dtype {
                held: Dtype::U32,
                expected: Dtype::F32
            }
        ));
        assert!(matches!(
            view_len::<u32>(&view(&[3], Dtype::I32), 1, 8).unwrap_err(),
            ReadbackError::Dtype {
                held: Dtype::I32,
                expected: Dtype::U32
            }
        ));
    }

    #[test]
    fn a_gapped_view_is_refused_since_one_copy_carries_contiguous_rows() {
        // Three rows of eight, sixteen apart: the copy would carry the gaps as values.
        let gapped = Layout::strided(&[3, 8], &[16, 1], Dtype::F32).unwrap();
        let refused = view_len::<f32>(&Tensor::for_test(BASE, gapped).unwrap(), 8, 4).unwrap_err();
        assert!(matches!(refused, ReadbackError::NotContiguous { .. }));
        assert!(refused.to_string().contains("[16, 1]"), "{refused}");
    }

    #[test]
    fn a_scalar_view_has_no_rows_to_copy() {
        assert!(matches!(
            view_len::<f32>(&view(&[], Dtype::F32), 8, 4).unwrap_err(),
            ReadbackError::Scalar { width: 8 }
        ));
    }

    #[test]
    fn the_copy_is_the_selected_rows_of_the_width_and_nothing_else() {
        assert_eq!(selected_len(3, 8, 4, 24).unwrap(), 24);
        assert_eq!(
            selected_len(0, 8, 4, 0).unwrap(),
            0,
            "a step selecting nothing"
        );
        assert!(matches!(
            selected_len(5, 8, 4, 40).unwrap_err(),
            ReadbackError::TooManyRows {
                rows: 5,
                max_rows: 4
            }
        ));
        assert!(matches!(
            selected_len(3, 8, 4, 25).unwrap_err(),
            ReadbackError::Shape {
                len: 25,
                rows: 3,
                width: 8
            }
        ));
    }
}
