//! Pinned host memory with one owner: `len` values allocated from the driver once and freed on
//! drop.
//!
//! Every host array a step copies to or from the device is one of these: the input staging, the
//! sampler's records and row arrays, and the readback's buffer. Pinned memory is what makes an
//! asynchronous copy asynchronous: the driver copies straight from and to it, with no staging
//! copy of its own. The memory is cacheable, never write-combined, because the readback reads
//! every value it brings back and reads from write-combined memory are uncached.
//!
//! The values are unwritten until the owner writes them or a copy lands in them, and the owner
//! reads only what was written; a block from [`Pinned::zeroed`] is written whole at allocation,
//! so a slice over any of it reads initialised memory. Waiting is the owner's too: a `Pinned`
//! frees its memory on drop without waiting for a copy that may still be reading or writing it,
//! so the owner waits on the event recorded behind its last copy before letting it go.

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;
use std::slice;

use atoma_runtime::error::RuntimeError;
use cudarc::driver::result::{free_host, malloc_host};
use tracing::warn;

/// `cuMemHostAlloc` flags: none of them, which is pinned and cacheable memory this context
/// alone reaches. Neither `CU_MEMHOSTALLOC_WRITECOMBINED`, since the owner reads what comes
/// back, nor `CU_MEMHOSTALLOC_DEVICEMAP`, since no kernel reaches the memory directly.
const CACHEABLE_PINNED: u32 = 0;

/// `len` values of pinned host memory, allocated once and freed on drop.
pub struct Pinned<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Pinned<T> {
    /// `len` values of pinned, cacheable host memory in the current context.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the driver cannot pin the memory.
    pub fn new(len: usize) -> Result<Self, RuntimeError> {
        // SAFETY: a driver allocation of the size asked for, freed once, in `Drop`.
        let ptr = unsafe { malloc_host(len * size_of::<T>(), CACHEABLE_PINNED) }?.cast::<T>();
        Ok(Self { ptr, len })
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `len` values were allocated at `ptr` and nothing writes them while this borrow
        // is live: the writer takes `&mut self`.
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as above, exclusively through `&mut self`.
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// The address a copy into the memory writes at. Values a copy writes through it are read,
    /// through [`Pinned::as_slice`], only after that copy has been waited on.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl Pinned<u8> {
    /// `len` zeroed bytes of pinned, cacheable host memory in the current context: every byte is
    /// written here, so a slice over the block reads initialised memory from the start.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the driver cannot pin the memory.
    pub fn zeroed(len: usize) -> Result<Self, RuntimeError> {
        let block = Self::new(len)?;
        // SAFETY: `len` bytes were allocated at the pointer and nothing else holds them yet.
        unsafe { ptr::write_bytes(block.ptr, 0, len) };
        Ok(block)
    }
}

impl<T> Drop for Pinned<T> {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `malloc_host` and is freed here alone.
        if let Err(error) = unsafe { free_host(self.ptr.cast::<c_void>()) } {
            warn!(%error, "pinned host memory could not be freed");
        }
    }
}
