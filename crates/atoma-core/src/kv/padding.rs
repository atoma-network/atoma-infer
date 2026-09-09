//! Padding dummies: the permanent block leases behind graph padding.
//!
//! A live batch is padded up to its bucket with dummy requests, and a dummy's attention must
//! read valid KV, so every dummy owns its own block for the process lifetime. A dummy run has
//! no live entry at all and fills every row of its bucket from the same dummies, which is what
//! the reservation is sized for. The engine reserves them from the pool once at startup — a held
//! lease already makes a block un-evictable, so permanence needs no second mechanism — and hands
//! the block ids to the executor, whose capture of the bucket ladder fills every dummy run from
//! them. What a configuration's padding costs is answerable before any pool exists.

use thiserror::Error;

use crate::kv::{BlockLease, BlockPool, KvCacheSpec};
use crate::types::{BlockId, RequestCount};

/// The padding dummies' blocks, held as leases for the process lifetime.
///
/// A dummy run fills every row of its bucket, so the reservation holds one block per dummy and
/// as many dummies as the configured maximum batch holds entries.
#[derive(Debug)]
pub struct PaddingReservation {
    leases: Vec<BlockLease>,
}

impl PaddingReservation {
    /// Dummies a maximum batch of `max_batch` entries can ever need: as many as the batch
    /// itself. Padding a live batch needs one fewer, since a batch that is padded at all holds at
    /// least one live entry, but a dummy run has no live entry and fills every row of its bucket.
    /// The decode step serves no bucket above the maximum batch, so no dummy run has more rows
    /// than this, and [`PaddingCannotCoverBucket`] refuses a configuration whose maximum batch
    /// pads to a bucket above itself, so that cap drops no bucket a live batch reaches.
    ///
    /// [`PaddingCannotCoverBucket`]: crate::engine::EngineError::PaddingCannotCoverBucket
    #[must_use]
    pub fn dummies_for(max_batch: RequestCount) -> usize {
        max_batch.get()
    }

    /// Reserves one block per dummy from `pool`, permanently.
    ///
    /// # Errors
    ///
    /// Returns [`PaddingError::NotEnoughFreeBlocks`] when the pool cannot cover the reservation;
    /// nothing is reserved in that case.
    pub fn reserve(pool: &mut BlockPool, max_batch: RequestCount) -> Result<Self, PaddingError> {
        let dummy_count = Self::dummies_for(max_batch);
        let mut leases = Vec::with_capacity(dummy_count);
        for _ in 0..dummy_count {
            let Some(lease) = pool.lease() else {
                let free = pool.free_count() + leases.len();
                for lease in leases {
                    pool.release(lease);
                }
                return Err(PaddingError::NotEnoughFreeBlocks {
                    needed: dummy_count,
                    free,
                });
            };
            leases.push(lease);
        }
        Ok(Self { leases })
    }

    /// Bytes the reservation costs under `spec` — answerable at configuration time, before any
    /// pool exists.
    #[must_use]
    pub fn cost_bytes(spec: &KvCacheSpec, max_batch: RequestCount) -> usize {
        Self::dummies_for(max_batch) * spec.bytes_per_block()
    }

    /// Dummies reserved: as many as the configured maximum batch holds entries.
    #[must_use]
    pub fn dummy_count(&self) -> usize {
        self.leases.len()
    }

    /// Each dummy's block, in reservation order — what the executor receives to capture over.
    #[must_use]
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.leases.iter().map(BlockLease::block).collect()
    }

    /// Surrenders the reservation. The engine never calls this — the dummies live as long as
    /// the process — but shutdown and tests return the blocks cleanly.
    pub fn release(self, pool: &mut BlockPool) {
        for lease in self.leases {
            pool.release(lease);
        }
    }
}

/// A padding reservation the pool cannot cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PaddingError {
    /// The configured maximum batch needs more dummy blocks than the pool has free.
    #[error("padding needs {needed} free blocks for its dummies but the pool has {free}")]
    NotEnoughFreeBlocks { needed: usize, free: usize },
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{PaddingError, PaddingReservation};
    use crate::kv::test_support::full_attention_group;
    use crate::kv::{BlockPool, KvCacheSpec};
    use crate::test_support::requests as max_batch;

    #[test]
    fn the_reservation_holds_one_distinct_block_per_dummy() {
        let mut pool = BlockPool::new(8);
        let reservation = PaddingReservation::reserve(&mut pool, max_batch(4)).unwrap();

        assert_eq!(
            reservation.dummy_count(),
            4,
            "as many as the maximum batch, so a dummy run can fill every row"
        );
        let ids = reservation.block_ids();
        let distinct: HashSet<_> = ids.iter().map(|id| id.get()).collect();
        assert_eq!(distinct.len(), 4, "every dummy owns its own block");
        assert_eq!(pool.free_count(), 4);

        reservation.release(&mut pool);
        assert_eq!(pool.free_count(), 8, "released only for shutdown and tests");
    }

    #[test]
    fn reserved_blocks_stay_out_of_reach_for_the_pool_lifetime() {
        let mut pool = BlockPool::new(4);
        let reservation = PaddingReservation::reserve(&mut pool, max_batch(3)).unwrap();

        assert_eq!(
            pool.available(),
            1,
            "the dummies' blocks are not obtainable"
        );
        let only = pool.lease().expect("one block is left");
        assert!(pool.lease().is_none());

        pool.release(only);
        reservation.release(&mut pool);
    }

    #[test]
    fn a_maximum_batch_of_one_reserves_the_row_a_dummy_run_fills() {
        let mut pool = BlockPool::new(2);
        let reservation = PaddingReservation::reserve(&mut pool, max_batch(1)).unwrap();
        assert_eq!(
            reservation.dummy_count(),
            1,
            "a live batch of one pads with nothing, but a dummy run still fills the row"
        );
        assert_eq!(reservation.block_ids().len(), 1);
        assert_eq!(pool.free_count(), 1);
        reservation.release(&mut pool);
        assert_eq!(pool.free_count(), 2);
    }

    #[test]
    fn a_pool_too_small_for_the_dummies_is_reported_with_both_numbers() {
        let mut pool = BlockPool::new(2);
        let error = PaddingReservation::reserve(&mut pool, max_batch(4)).unwrap_err();
        assert_eq!(
            error,
            PaddingError::NotEnoughFreeBlocks { needed: 4, free: 2 }
        );
        assert_eq!(pool.free_count(), 2, "a refused reservation takes nothing");
    }

    #[test]
    fn padding_cost_is_reported_at_configuration_time() {
        let spec = KvCacheSpec::new(vec![full_attention_group(0)]).unwrap();
        assert_eq!(
            PaddingReservation::cost_bytes(&spec, max_batch(4)),
            4 * 2 * 1024 * 1024
        );
        assert_eq!(
            PaddingReservation::cost_bytes(&spec, max_batch(1)),
            2 * 1024 * 1024
        );
    }
}
