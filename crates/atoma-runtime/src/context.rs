//! Device context ownership: construction, the global event-tracking disable, the device's free
//! memory, and loud failure when no driver is present.

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{CudaContext, DriverError};

use crate::error::RuntimeError;

/// A quantity of device memory, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceBytes(usize);

impl DeviceBytes {
    /// The quantity `bytes` bytes.
    #[must_use]
    pub const fn new(bytes: usize) -> Self {
        Self(bytes)
    }

    /// The quantity as a plain number of bytes.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }

    /// The drop from this reading to `later`, and zero where free memory rose instead.
    ///
    /// A reading is the device's free memory, not this process's, so anything else on the device
    /// freeing between two readings makes the later one the larger. A capture that cost nothing
    /// measurable is the honest floor for that; a wrap or a debug panic in the caller is not.
    #[must_use]
    pub const fn saturating_sub(self, later: Self) -> Self {
        Self(self.0.saturating_sub(later.0))
    }
}

impl fmt::Display for DeviceBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// The engine's handle to one CUDA device, and the only way this crate opens one.
///
/// Construction disables cudarc's event tracking globally, before anything is allocated: cudarc
/// attaches a tracking event to every buffer at allocation time, and a wait on an event recorded
/// before a capture began invalidates the capture. Disabling per capture would be too late — the
/// events attach at allocation, not at use.
pub struct RuntimeContext {
    ctx: Arc<CudaContext>,
}

impl RuntimeContext {
    /// Opens device `ordinal` and disables event tracking for everything allocated afterward.
    ///
    /// Fails loudly and early when no usable driver or device exists, so a misconfigured
    /// deployment dies at startup rather than mid-serve: with no driver library at all, cudarc's
    /// loader panics; with a driver but no usable device, this returns
    /// [`RuntimeError::NoDriver`] carrying the remediation text.
    pub fn new(ordinal: usize) -> Result<Self, RuntimeError> {
        let ctx = CudaContext::new(ordinal)?;
        // SAFETY: cross-stream synchronization is this crate's responsibility from here on. The
        // capture substrate orders work explicitly — buffers are allocated before capture, the
        // capture stream never frees them (GraphEntry owns every buffer for the graph's
        // lifetime), and replay is serialized on the executor thread — so no CudaSlice relies on
        // cudarc's event-based synchronization.
        unsafe { ctx.disable_event_tracking() };
        Ok(Self { ctx })
    }

    /// The underlying cudarc context, for allocating buffers and creating streams.
    pub fn cuda(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// How much memory this context's device has free right now.
    ///
    /// The read goes through the context, which binds it to the calling thread first, so the
    /// count is this context's device even where another context is the current one. It is what
    /// the device has free, not what this process holds: anything else allocating on the same
    /// device moves it.
    ///
    /// # Errors
    ///
    /// Returns the classification of whatever status the reading fails on, which cudarc may have
    /// carried over from an earlier operation on this context, and
    /// [`RuntimeError::FreeMemoryUnreadable`] where the taxonomy does not name that status.
    pub fn free_memory(&self) -> Result<DeviceBytes, RuntimeError> {
        free_bytes(self.ctx.mem_get_info())
    }
}

/// What one `cuMemGetInfo` reading means: the free half of the driver's `(free, total)` pair, or
/// what the rejecting status classifies as, with the read named as what failed only where the
/// taxonomy names nothing.
///
/// Naming the call is worth more than the status only where the status says nothing. cudarc
/// surfaces a status recorded by an earlier drop on the next call that checks the context — the
/// bind ahead of this read is such a call — so a capture failure can reach this reading before
/// the driver is asked at all, and it has to stay a capture failure.
fn free_bytes(reading: Result<(usize, usize), DriverError>) -> Result<DeviceBytes, RuntimeError> {
    match reading {
        Ok((free, _total)) => Ok(DeviceBytes::new(free)),
        Err(rejected) => Err(match RuntimeError::from(rejected) {
            // Driver is the taxonomy's catch-all for a status it does not name; this is where
            // naming the read adds what the status left out. Every other classification stands.
            RuntimeError::Driver(status) => RuntimeError::FreeMemoryUnreadable(status),
            classified => classified,
        }),
    }
}

#[cfg(test)]
mod tests {
    use cudarc::driver::sys::CUresult;
    use cudarc::driver::DriverError;

    use super::{free_bytes, DeviceBytes};
    use crate::error::RuntimeError;

    #[test]
    fn a_drop_between_two_readings_is_what_the_later_one_lost() {
        let before = DeviceBytes::new(64 * 1024);
        let after = DeviceBytes::new(24 * 1024);
        assert_eq!(before.saturating_sub(after), DeviceBytes::new(40 * 1024));
    }

    #[test]
    fn a_reading_that_rose_is_no_drop_at_all() {
        // Free memory is the device's, not this process's, so the later reading can be the larger.
        let before = DeviceBytes::new(24 * 1024);
        let after = DeviceBytes::new(64 * 1024);
        assert_eq!(before.saturating_sub(after), DeviceBytes::new(0));
    }

    #[test]
    fn a_quantity_prints_as_its_bytes() {
        assert_eq!(DeviceBytes::new(40 * 1024).to_string(), "40960");
    }

    #[test]
    fn a_reading_is_the_free_half_of_the_drivers_pair() {
        let reading = free_bytes(Ok((3 * 1024, 8 * 1024))).unwrap();
        assert_eq!(reading, DeviceBytes::new(3 * 1024));
    }

    #[test]
    fn a_rejected_read_keeps_the_classification_its_status_earns() {
        // cudarc surfaces a status recorded by an earlier drop on the next call that checks the
        // context, and the bind ahead of this read is such a call, so a capture failure can reach
        // this reading before the driver is asked at all. What the status says outranks which
        // call returned it.
        assert!(matches!(
            free_bytes(Err(DriverError(
                CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
            ))),
            Err(RuntimeError::CaptureUnsupported(_))
        ));
        assert!(matches!(
            free_bytes(Err(DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY))),
            Err(RuntimeError::OutOfDeviceMemory(_))
        ));
        assert!(matches!(
            free_bytes(Err(DriverError(CUresult::CUDA_ERROR_NOT_INITIALIZED))),
            Err(RuntimeError::NoDriver(_))
        ));
    }

    #[test]
    fn a_read_rejected_by_an_unnamed_status_is_named_for_the_read() {
        // The statuses the taxonomy leaves to its Driver catch-all are the ones where naming the
        // call that failed is all there is to say.
        let status = CUresult::CUDA_ERROR_DEINITIALIZED;
        assert!(matches!(
            free_bytes(Err(DriverError(status))),
            Err(RuntimeError::FreeMemoryUnreadable(carried)) if carried == status
        ));
    }
}
