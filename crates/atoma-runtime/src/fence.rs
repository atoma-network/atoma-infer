//! The staging fence: one event, recorded on the capture stream behind the copy that reads a
//! staging entry, and waited on by the host before it writes that staging entry again.
//!
//! A copy-in copies a pinned staging entry to the device asynchronously, so the host must not
//! write the staging entry again until that copy has read it. The fence is what says when: its
//! [`FenceSignal`] is enqueued through the [`Descriptor`] seam behind the copy, and the host
//! then asks the fence and nothing else — [`StagingFence::wait`] blocks until the signal has
//! passed, [`StagingFence::try_wait`] says whether it has without blocking. Neither reaches the
//! stream, so the capture stream's surface keeps its no-synchronize rule.
//!
//! A fence nobody has signaled is passed: a fresh staging entry is written without a wait.
//! Dropping a fence does not wait; whoever owns the memory the copy reads waits on the fence
//! before letting the memory go.
//!
//! The protocol for one staging entry; the example compiles on a machine with no GPU and runs
//! on none:
//!
//! ```no_run
//! use atoma_runtime::context::RuntimeContext;
//! use atoma_runtime::error::RuntimeError;
//! use atoma_runtime::fence::StagingFence;
//! use atoma_runtime::session::Allocation;
//!
//! fn one_staging_entry() -> Result<(), RuntimeError> {
//!     let ctx = RuntimeContext::new(0)?;
//!     let allocation = Allocation::new(&ctx)?;
//!     let fence = StagingFence::new(ctx.cuda())?;
//!     let replay = allocation.into_capture().into_replay();
//!     // The copy that reads the staging entry goes here; the signal follows it on the stream.
//!     replay.run(&mut fence.signal())?;
//!     if !fence.try_wait()? {
//!         fence.wait()?;
//!     }
//!     // The staging entry may be written again.
//!     Ok(())
//! }
//! ```

use std::sync::Arc;

use cudarc::driver::result::event;
use cudarc::driver::sys::{self, CUevent_flags, CUresult};
use cudarc::driver::{CudaContext, CudaEvent, DriverError};

use crate::error::RuntimeError;
use crate::session::Descriptor;

/// The fence guarding one staging entry's reuse: passed once the copy that read the staging entry
/// has finished, which is when the host may write it again.
pub struct StagingFence {
    event: CudaEvent,
}

impl StagingFence {
    /// A fence in `context` that nothing has signaled, so it is passed.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the driver cannot create the event.
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, RuntimeError> {
        // Timing off and no `CU_EVENT_BLOCKING_SYNC`: a wait that misses lasts microseconds, and
        // the executor thread spins through it on its own core instead of being parked.
        let event = context.new_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))?;
        Ok(Self { event })
    }

    /// The descriptor that signals the fence from the stream it is enqueued on: the fence is
    /// passed once everything enqueued ahead of it there has finished.
    #[must_use]
    pub fn signal(&self) -> FenceSignal<'_> {
        FenceSignal { event: &self.event }
    }

    /// Waits until the fence is passed, blocking the calling thread.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the driver fails the wait.
    pub fn wait(&self) -> Result<(), RuntimeError> {
        self.event.synchronize()?;
        Ok(())
    }

    /// Whether the fence is passed, without waiting: `false` while the copy ahead of the last
    /// signal is still in flight.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the driver fails the query for any reason but not ready.
    pub fn try_wait(&self) -> Result<bool, RuntimeError> {
        // SAFETY: the event is live for as long as `self` owns it.
        passed(unsafe { event::query(self.event.cu_event()) })
    }
}

/// What one query of the fence's event means: passed, still in flight, or a driver failure.
/// Not ready is the one status that is an answer rather than an error.
fn passed(queried: Result<(), DriverError>) -> Result<bool, RuntimeError> {
    match queried {
        Ok(()) => Ok(true),
        Err(DriverError(CUresult::CUDA_ERROR_NOT_READY)) => Ok(false),
        Err(err) => Err(RuntimeError::from(err)),
    }
}

/// The signal of one fence, enqueued on the capture stream behind the copy it guards.
pub struct FenceSignal<'a> {
    event: &'a CudaEvent,
}

impl Descriptor for FenceSignal<'_> {
    type Error = RuntimeError;

    unsafe fn enqueue(&mut self, stream: sys::CUstream) -> Result<(), RuntimeError> {
        // SAFETY: the session hands a live stream, and the event is live for as long as this
        // descriptor borrows it.
        unsafe { event::record(self.event.cu_event(), stream) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use cudarc::driver::sys::CUresult;
    use cudarc::driver::DriverError;

    use super::passed;
    use crate::error::RuntimeError;

    #[test]
    fn a_fence_whose_event_has_completed_is_passed() {
        assert!(passed(Ok(())).unwrap());
    }

    #[test]
    fn a_fence_whose_copy_is_still_in_flight_is_not_passed_and_not_an_error() {
        let not_ready = Err(DriverError(CUresult::CUDA_ERROR_NOT_READY));
        assert!(!passed(not_ready).unwrap());
    }

    #[test]
    fn every_other_query_failure_surfaces_classified() {
        assert!(matches!(
            passed(Err(DriverError(CUresult::CUDA_ERROR_INVALID_HANDLE))),
            Err(RuntimeError::Driver(CUresult::CUDA_ERROR_INVALID_HANDLE))
        ));
        assert!(matches!(
            passed(Err(DriverError(CUresult::CUDA_ERROR_NOT_INITIALIZED))),
            Err(RuntimeError::NoDriver(CUresult::CUDA_ERROR_NOT_INITIALIZED))
        ));
    }
}
