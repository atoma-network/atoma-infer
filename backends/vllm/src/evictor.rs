use indexmap::IndexMap;
use thiserror::Error;

use crate::block::PhysicalTokenBlock;

pub trait Evictor {
    /// Checks if the evictor contains a block with the given block number.
    ///
    /// # Arguments
    ///
    /// * `block_number` - The block number to check for.
    ///
    /// # Returns
    ///
    /// `true` if the block is present, `false` otherwise.
    fn contains(&self, block_number: u32) -> bool;

    /// Evicts a block based on the eviction policy.
    ///
    /// # Returns
    ///
    /// `Ok(PhysicalTokenBlock)` if a block was successfully evicted, or
    /// `Err(EvictorError)` if the free table is empty.
    fn evict(&mut self) -> Result<PhysicalTokenBlock, EvictorError>;

    /// Adds a new block to the evictor.
    ///
    /// # Arguments
    ///
    /// * `block` - The block to add.
    fn add(&mut self, block: PhysicalTokenBlock);

    /// Removes a block with the given block number, if it exists.
    ///
    /// # Arguments
    ///
    /// * `block_number` - The block number of the block to remove.
    ///
    /// # Returns
    ///
    /// `Some(PhysicalTokenBlock)` if the block was found and removed, or
    /// `None` if the block was not found.
    fn remove(&mut self, block_number: u32) -> Option<PhysicalTokenBlock>;

    /// Gets the number of blocks currently in the evictor.
    ///
    /// # Returns
    ///
    /// The number of blocks.
    fn num_blocks(&self) -> usize;
}

/// The `LRUEvictor` struct implements an eviction policy based on the Least Recently Used (LRU) algorithm.
/// It maintains a cache of blocks, where each block has a `last_accessed` timestamp indicating the last time it was accessed.
///
/// When the cache needs to evict a block, the block with the oldest `last_accessed` timestamp is chosen.
/// If there are multiple blocks with the same `last_accessed` timestamp, the block with the highest `num_hashed_tokens` is evicted.
/// If multiple blocks have the same `last_accessed` timestamp and the highest `num_hashed_tokens` value, one of them is chosen arbitrarily.
#[derive(Debug)]
pub struct LRUEvictor {
    /// An `IndexMap` that stores the cached blocks, where the key is the block number and the value is the `PhysicalTokenBlock`.
    pub free_table: IndexMap<u32, PhysicalTokenBlock>,
}

impl LRUEvictor {
    /// Constructor
    pub fn new() -> Self {
        Self {
            free_table: IndexMap::new(),
        }
    }
}

impl Default for LRUEvictor {
    fn default() -> Self {
        Self::new()
    }
}

impl Evictor for LRUEvictor {
    fn contains(&self, block_number: u32) -> bool {
        self.free_table.contains_key(&block_number)
    }

    fn evict(&mut self) -> Result<PhysicalTokenBlock, EvictorError> {
        if self.free_table.is_empty() {
            return Err(EvictorError::EmptyFreeTable);
        }

        // Step 1: Find the block to evict
        let mut evicted_block_key = None;
        let mut evicted_block: Option<PhysicalTokenBlock> = None;

        // The blocks with the lowest `last_accessed` should be placed consecutively
        // at the start of `free_table`. Loop through all these blocks to
        // find the one with maximum number of hashed tokens.
        for (key, block) in &self.free_table {
            if let Some(current_evicted_block) = &evicted_block {
                if current_evicted_block.last_accessed() < block.last_accessed() {
                    break;
                }
                if current_evicted_block.num_hashed_tokens() < block.num_hashed_tokens() {
                    evicted_block = Some(block.clone());
                    evicted_block_key = Some(*key);
                }
            } else {
                evicted_block = Some(block.clone());
                evicted_block_key = Some(*key);
            }
        }

        // Step 2: Remove the block from the free table
        if let Some(key) = evicted_block_key {
            let mut evicted_block = self.free_table.shift_remove(&key).unwrap(); // DON'T PANIC: we already checked that `free_table` is not empty
            evicted_block.not_computed();
            return Ok(evicted_block);
        }

        Err(EvictorError::EmptyFreeTable)
    }

    fn add(&mut self, block: PhysicalTokenBlock) {
        self.free_table.insert(block.block_number(), block);
    }

    fn remove(&mut self, block_number: u32) -> Option<PhysicalTokenBlock> {
        self.free_table.shift_remove(&block_number)
    }

    fn num_blocks(&self) -> usize {
        self.free_table.len()
    }
}

#[derive(Debug, Error)]
pub enum EvictorError {
    #[error("Free table is empty")]
    EmptyFreeTable,
}
