//! Driver error taxonomy: named errors carrying actionable remediation text.
//!
//! A capture failure on the rig has to be classifiable — kernel-illegal operation, cudarc
//! behaviour, hidden FFI synchronization, or NCCL — without re-deriving the taxonomy each time,
//! so every capture-relevant driver status maps to a named error whose message says what likely
//! went wrong and what to do about it.
//!
//! Variants carry the raw [`CUresult`] status, never [`DriverError`]: both `Display` and `Debug`
//! on `DriverError` ask the driver for the error string via `cuGetErrorString`, which panics
//! when no `libcuda` is loadable, and these errors must be printable in GPU-free tests and in
//! the loud startup failure of a misconfigured deployment.

use cudarc::driver::sys::CUresult;
use cudarc::driver::DriverError;
use thiserror::Error;

use crate::capture::CaptureState;

/// Errors from the capture substrate: driver statuses the taxonomy names, this crate's own
/// refusals, and a read named for its call where the taxonomy names nothing.
#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(
        "capture-illegal operation ({0:?}): the captured region performed an operation the \
         driver forbids under capture, such as a device allocation, a synchronize, or legacy \
         default-stream work; move it before capture or drop it from the step"
    )]
    CaptureUnsupported(CUresult),

    #[error(
        "capture invalidated ({0:?}): a previous operation broke the recording; discard it with \
         end_capture_discard, then look for a hidden FFI synchronize or allocation inside the \
         captured region"
    )]
    CaptureInvalidated(CUresult),

    #[error(
        "capture merge rejected ({0:?}): two independent captures met on one stream; keep each \
         capture on its own dedicated side stream"
    )]
    CaptureMerge(CUresult),

    #[error(
        "capture unmatched ({0:?}): the capture was ended on a stream other than the one that \
         began it; end it on the same CaptureStream"
    )]
    CaptureUnmatched(CUresult),

    #[error(
        "capture unjoined ({0:?}): work forked from the capture stream was never joined back; \
         rejoin every forked stream before ending the capture"
    )]
    CaptureUnjoined(CUresult),

    #[error(
        "capture isolation violated ({0:?}): a dependency crossed the capture boundary from a \
         non-captured stream; keep cross-stream waits inside the captured set"
    )]
    CaptureIsolation(CUresult),

    #[error(
        "implicit legacy-stream capture ({0:?}): the operation would have captured the legacy \
         default stream; never launch on the default stream while a capture is active"
    )]
    CaptureImplicit(CUresult),

    #[error(
        "wait on a pre-capture event ({0:?}): an event recorded before the capture was waited on \
         inside it, which invalidates the capture; RuntimeContext disables cudarc's event \
         tracking at init — check for buffers allocated before that, or explicit event waits"
    )]
    CapturedEventWait(CUresult),

    #[error(
        "capture touched from the wrong thread ({0:?}): graphs are not internally synchronized; \
         warmup, capture, and replay must all run on the executor thread that owns the stream"
    )]
    CaptureWrongThread(CUresult),

    #[error(
        "no usable CUDA driver ({0:?}): the driver library is missing, not initialized, or no \
         device is visible; this deployment cannot serve — install or expose the driver and \
         restart"
    )]
    NoDriver(CUresult),

    #[error(
        "device out of memory ({0:?}): the arena, graph pool, or KV budget exceeds the device; \
         shrink the bucket ladder or the KV allocation"
    )]
    OutOfDeviceMemory(CUresult),

    #[error(
        "driver call failed ({0:?}): not a capture-classified status; consult the CUDA driver docs"
    )]
    Driver(CUresult),

    #[error(
        "free device memory could not be read ({0:?}): a status this taxonomy does not name, \
         which may come from the cuMemGetInfo call, from the context bind cudarc runs ahead of \
         it, or from an earlier operation whose status cudarc deferred onto this one; consult \
         the CUDA driver docs, and restart the process if it repeats"
    )]
    FreeMemoryUnreadable(CUresult),

    #[error(
        "begin_capture while a capture is already active on this stream; end or discard the \
         current capture first"
    )]
    BeginWhileActive,

    #[error(
        "begin_capture on a stream whose previous capture was invalidated and still holds \
         recorded state; call end_capture_discard first"
    )]
    BeginAfterInvalidation,

    #[error("end_capture on a stream with no capture in progress; call begin_capture first")]
    EndWithoutCapture,

    #[error(
        "end_capture on an invalidated capture, which cannot be instantiated; call \
         end_capture_discard to drain it, then inspect what invalidated the recording"
    )]
    EndAfterInvalidation,

    #[error("end_capture_discard on a stream with no capture in progress; nothing to discard")]
    DiscardWithoutCapture,

    #[error(
        "record with no warmup pass since the session's last recording; run the exact step at \
         the graph's exact shape eagerly (Capture::warm_up) immediately before each record, so \
         every lazy backend allocation lands outside the recording"
    )]
    RecordWithoutWarmup,

    #[error(
        "synchronize while the stream capture state is {0:?}: a synchronize would invalidate an \
         active recording; end or discard the capture first"
    )]
    SyncWhileCapturing(CaptureState),

    #[error(
        "dot-print path contains an interior NUL byte and cannot be passed to the driver; \
         choose a different output path"
    )]
    DotPrintPathHasNul,

    #[error(
        "NCCL call failed ({0:?}): the communicator could not be created or the collective was \
         rejected; check rank and world_size, and that libnccl is loadable"
    )]
    #[cfg(feature = "nccl")]
    Nccl(cudarc::nccl::sys::ncclResult_t),
}

#[cfg(feature = "nccl")]
impl From<cudarc::nccl::result::NcclError> for RuntimeError {
    fn from(err: cudarc::nccl::result::NcclError) -> Self {
        Self::Nccl(err.0)
    }
}

impl From<DriverError> for RuntimeError {
    fn from(err: DriverError) -> Self {
        match err.0 {
            CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED => Self::CaptureUnsupported(err.0),
            CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED => Self::CaptureInvalidated(err.0),
            CUresult::CUDA_ERROR_STREAM_CAPTURE_MERGE => Self::CaptureMerge(err.0),
            CUresult::CUDA_ERROR_STREAM_CAPTURE_UNMATCHED => Self::CaptureUnmatched(err.0),
            CUresult::CUDA_ERROR_STREAM_CAPTURE_UNJOINED => Self::CaptureUnjoined(err.0),
            CUresult::CUDA_ERROR_STREAM_CAPTURE_ISOLATION => Self::CaptureIsolation(err.0),
            CUresult::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT => Self::CaptureImplicit(err.0),
            CUresult::CUDA_ERROR_CAPTURED_EVENT => Self::CapturedEventWait(err.0),
            CUresult::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD => Self::CaptureWrongThread(err.0),
            CUresult::CUDA_ERROR_NO_DEVICE
            | CUresult::CUDA_ERROR_NOT_INITIALIZED
            | CUresult::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => Self::NoDriver(err.0),
            CUresult::CUDA_ERROR_OUT_OF_MEMORY => Self::OutOfDeviceMemory(err.0),
            // CUresult is a foreign enum with ~90 non-capture statuses; Driver is the deliberate
            // catch-all for everything the taxonomy does not name.
            _ => Self::Driver(err.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classified(status: CUresult) -> RuntimeError {
        RuntimeError::from(DriverError(status))
    }

    #[test]
    fn capture_statuses_map_to_named_errors() {
        assert!(matches!(
            classified(CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED),
            RuntimeError::CaptureUnsupported(_)
        ));
        assert!(matches!(
            classified(CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED),
            RuntimeError::CaptureInvalidated(_)
        ));
        assert!(matches!(
            classified(CUresult::CUDA_ERROR_CAPTURED_EVENT),
            RuntimeError::CapturedEventWait(_)
        ));
        assert!(matches!(
            classified(CUresult::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD),
            RuntimeError::CaptureWrongThread(_)
        ));
    }

    #[test]
    fn missing_driver_statuses_map_to_no_driver() {
        for status in [
            CUresult::CUDA_ERROR_NO_DEVICE,
            CUresult::CUDA_ERROR_NOT_INITIALIZED,
            CUresult::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
        ] {
            assert!(matches!(classified(status), RuntimeError::NoDriver(_)));
        }
    }

    #[test]
    fn remediation_text_names_the_likely_cause_and_the_fix() {
        let invalidated = classified(CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
        assert!(invalidated.to_string().contains("end_capture_discard"));

        let event_wait = classified(CUresult::CUDA_ERROR_CAPTURED_EVENT);
        assert!(event_wait.to_string().contains("event tracking"));

        let wrong_thread = classified(CUresult::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD);
        assert!(wrong_thread.to_string().contains("executor thread"));
    }

    #[test]
    fn session_errors_direct_the_operator() {
        let unwarmed = RuntimeError::RecordWithoutWarmup;
        assert!(unwarmed.to_string().contains("warm_up"));
        assert!(unwarmed.to_string().contains("exact shape"));

        let mid_capture = RuntimeError::SyncWhileCapturing(CaptureState::Active);
        assert!(mid_capture.to_string().contains("Active"));
        assert!(mid_capture.to_string().contains("discard"));
    }

    #[test]
    fn a_failed_memory_reading_names_the_call_and_the_fix() {
        let unreadable = RuntimeError::FreeMemoryUnreadable(CUresult::CUDA_ERROR_DEINITIALIZED);
        assert!(unreadable.to_string().contains("cuMemGetInfo"));
        assert!(unreadable.to_string().contains("CUDA_ERROR_DEINITIALIZED"));
        assert!(unreadable.to_string().contains("restart the process"));
        // The call is one of three sources the message may not close over: the bind cudarc runs
        // ahead of it, and an earlier operation whose status it deferred, reach an operator here
        // too.
        assert!(unreadable.to_string().contains("deferred"));
    }

    #[test]
    fn messages_format_without_a_driver_present() {
        // DriverError's own Display and Debug ask libcuda for the error string; ours must not.
        let unclassified = classified(CUresult::CUDA_ERROR_UNKNOWN);
        assert!(unclassified.to_string().contains("CUDA_ERROR_UNKNOWN"));
    }

    #[test]
    #[cfg(feature = "nccl")]
    fn nccl_errors_carry_the_raw_status_and_remediation() {
        use cudarc::nccl::result::NcclError;
        use cudarc::nccl::sys::ncclResult_t;

        let err = RuntimeError::from(NcclError(ncclResult_t::ncclInvalidArgument));
        assert!(err.to_string().contains("ncclInvalidArgument"));
        assert!(err.to_string().contains("world_size"));
    }
}
