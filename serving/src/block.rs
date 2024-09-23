use std::{
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
    time::Instant,
};

use thiserror::Error;
use tracing::{error, info_span, instrument, Span};

use crate::types::{ReadLock, WriteLock};

/// A mapping between logical and physical KV (Key-Value) blocks for each request.
///
/// Each entry in the `BlockTable` represents:
/// - The corresponding physical block for a logical block
/// - The number of filled positions in that block
pub type BlockTable = Vec<SyncPhysicalTokenBlock>;

/// Represents the device on which a block is allocated.
///
/// `Cpu`: The block is allocated on the CPU
///
/// `Gpu`: The block is allocated on the GPU
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockDevice {
    Cpu,
    Gpu,
}

/// Represents a contiguous chunk of tokens in the logical space.
///
/// `LogicalTokenBlock` is used to track the state of corresponding physical blocks
/// in the KV cache (typically allocated on the GPU). It stores tokens sequentially
/// from left to right.
#[derive(Clone, Debug, PartialEq)]
pub struct LogicalTokenBlock {
    /// Unique identifier for this block
    block_number: usize,
    /// Maximum number of tokens this block can hold
    block_size: usize,
    /// Sequence of token IDs, with a maximum length of `block_size`
    token_ids: Vec<u32>,
    /// Current number of tokens stored in this block
    num_tokens: usize,
    /// Tracing span for observability
    span: Span,
}

impl LogicalTokenBlock {
    /// Constructor
    pub fn new(block_number: usize, block_size: usize) -> Self {
        Self {
            block_number,
            block_size,
            token_ids: Vec::with_capacity(block_size),
            num_tokens: 0,
            span: info_span!("block"),
        }
    }

    /// Getter for `block_number`
    pub fn block_number(&self) -> usize {
        self.block_number
    }

    /// Getter for `block_size`
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Checks if `token_ids` is empty
    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    /// Check if `token_ids` is full
    pub fn is_full(&self) -> bool {
        self.num_tokens == self.block_size
    }

    /// Get the number of additional token ids that can be added to the current `LogicalTokenBlock`
    pub fn get_num_empty_slots(&self) -> usize {
        self.block_size - self.num_tokens
    }

    /// Appends a new set of token ids to the current `LogicalTokenBlock`.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - A slice of u32 values representing the token IDs to be appended.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the tokens were successfully appended.
    /// * `Err(BlockError::AllocationError)` if there isn't enough space in the block.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut block = LogicalTokenBlock::new(0, 10);
    /// assert!(block.append_tokens(&[1, 2, 3]).is_ok());
    /// assert!(block.append_tokens(&[4, 5, 6, 7, 8, 9, 10]).is_err());
    /// ```    #[instrument(skip_all)]
    pub fn append_tokens(&mut self, token_ids: &[u32]) -> Result<(), BlockError> {
        if token_ids.len() <= self.get_num_empty_slots() {
            self.token_ids.extend(token_ids);
            self.num_tokens += token_ids.len();
            return Ok(());
        }
        error!("Not enough space for allocation");
        Err(BlockError::AllocationError(
            "Not enough space for allocation".into(),
        ))
    }

    /// Getter for `token_ids`
    pub fn get_token_ids(&self) -> Vec<u32> {
        self.token_ids.clone()
    }

    /// Getter for last element in `token_ids`
    pub fn get_last_token_id(&self) -> Option<u32> {
        self.token_ids.last().cloned()
    }
}

impl Eq for LogicalTokenBlock {}

/// Represents a contiguous memory block in the KV cache, typically allocated on a GPU device.
///
/// This structure is used to manage physical memory allocation and track the state
/// of each block in the KV cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalTokenBlock {
    /// Unique identifier for this block
    block_number: u32,
    /// Maximum number of KV vectors this block can hold
    block_size: usize,
    /// Indicates whether the block's content has been computed
    computed: bool,
    /// The device (CPU or GPU) on which this block is allocated
    device: BlockDevice,
    /// Timestamp of the most recent access to this block
    last_accessed: Option<Instant>,
    /// Number of tokens that have been hashed and stored in this block
    num_hashed_tokens: usize,
    /// Reference count for Copy-on-Write operations in advanced decoding techniques
    ref_count: usize,
}

impl PhysicalTokenBlock {
    /// Constructor
    pub fn new(block_number: u32, block_size: usize, device: BlockDevice) -> Self {
        Self {
            block_number,
            block_size,
            computed: false,
            device,
            last_accessed: None,
            num_hashed_tokens: 0,
            ref_count: 0,
        }
    }

    /// Getter for `block_number`
    pub fn block_number(&self) -> u32 {
        self.block_number
    }

    /// Getter for `block_size`
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Getter for `computed`
    pub fn computed(&self) -> bool {
        self.computed
    }

    /// Set `computed`
    pub fn set_computed(&mut self, value: bool) {
        self.computed = value
    }

    /// Getter for `device`
    pub fn device(&self) -> BlockDevice {
        self.device.clone()
    }

    /// Getter for `num_hashed_tokens`
    pub fn num_hashed_tokens(&self) -> usize {
        self.num_hashed_tokens
    }

    /// Getter for `last_accessed`
    pub fn last_accessed(&self) -> Option<Instant> {
        self.last_accessed
    }

    /// Sets `last_accessed`
    pub fn set_last_accessed(&mut self, instant: Instant) {
        self.last_accessed = Some(instant)
    }

    /// Getter for `ref_count`
    pub fn ref_count(&self) -> usize {
        self.ref_count
    }

    /// Set `num_hashed_tokens`
    pub fn update_num_hashed_tokens(&mut self, num_hashed_tokens: usize) {
        self.num_hashed_tokens = num_hashed_tokens
    }

    /// Set `computed` to false
    pub fn not_computed(&mut self) {
        self.computed = false;
    }

    /// Increments the `ref_count` variable by +1
    pub fn increment_ref_count(&mut self) {
        self.ref_count += 1;
    }

    /// Sets the `ref_count` by `value`
    pub fn set_ref_count_by(&mut self, value: usize) {
        self.ref_count = value;
    }

    /// Decreases the reference count by 1.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the reference count was successfully decreased.
    /// - `Err(BlockError::ReferenceCountError)` if the reference count is already zero.
    ///
    /// # Errors
    ///
    /// This method will return an error if the reference count is already zero,
    /// as it's not possible to decrease it further.
    pub fn decrease_ref_count(&mut self) -> Result<(), BlockError> {
        if self.ref_count > 0 {
            self.ref_count -= 1;
            Ok(())
        } else {
            error!(
                "Reference counter is already zero, trying to dereference once more which should not be possible.."
            );
            Err(BlockError::ReferenceCountError)
        }
    }
}

/// A thread-safe, shared-ownership wrapper for `PhysicalTokenBlock`.
///
/// This type provides synchronized read and write access to a `PhysicalTokenBlock`
/// across multiple threads. It combines `Arc` for shared ownership and `RwLock`
/// for interior mutability with multiple reader / single writer access.
pub type SyncPhysicalTokenBlock = Arc<RwLock<PhysicalTokenBlock>>;

impl ReadLock for SyncPhysicalTokenBlock {
    type Error = BlockError;
    type Inner = PhysicalTokenBlock;

    fn read_lock(&self) -> Result<RwLockReadGuard<Self::Inner>, Self::Error> {
        self.read()
            .map_err(|e| Self::Error::PoisonError(e.to_string()))
    }
}

impl WriteLock for SyncPhysicalTokenBlock {
    type Error = BlockError;
    type Inner = PhysicalTokenBlock;

    fn write_lock(&self) -> Result<RwLockWriteGuard<Self::Inner>, Self::Error> {
        self.write()
            .map_err(|e| Self::Error::PoisonError(e.to_string()))
    }
}

#[derive(Debug, Error)]
pub enum BlockError {
    #[error("Poison error: `{0}`")]
    PoisonError(String),
    #[error("Allocation error: `{0}`Not enough space for allocation")]
    AllocationError(String),
    #[error("Reference counter error, it cannot be negative")]
    ReferenceCountError,
}
