//! What one capture of the bucket ladder produces, and what each of its recordings is made over:
//! the graph serving each bucket, and the dummy run that fills each bucket's rows from the
//! padding dummies' blocks.
//!
//! Neither needs a device. Which block a bucket's row is filled from and which graph serves a
//! bucket are settled here; driving the session over them is `device::capture`'s.

use atoma_core::types::BlockId;
use atoma_runtime::arena::BucketIdx;
use atoma_runtime::session::GraphIdx;
use thiserror::Error;

use crate::decode::batch::DecodeBuckets;
use crate::decode::staging::DummyRun;

/// Why the bucket ladder's dummy runs could not be made over the blocks handed over, or a bucket
/// has no graph in the set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum GraphSetError {
    /// A bucket with more rows than the padding dummies' blocks handed over, one of which fills
    /// each row.
    #[error(
        "a bucket of {rows} rows fills each row from a padding dummy's block, but {blocks} were \
         handed over; hand the capture one block per dummy of the reservation"
    )]
    NotEnoughDummyBlocks { rows: usize, blocks: usize },
    /// A bucket past the buckets the set holds a graph for: the batch's bucket and the captured
    /// bucket ladder disagree.
    #[error("bucket {} is past the {graphs} buckets the graph set was captured for", bucket.0)]
    NoGraphForBucket { bucket: BucketIdx, graphs: usize },
}

/// The dummy run each of `buckets` is captured over, in bucket order: a bucket's rows filled
/// from the leading blocks of `blocks`, one per row.
///
/// The reservation holds one dummy per entry of the maximum batch and the decode step serves no
/// bucket above it, so a rank's own blocks always cover its bucket ladder. That is guarded here
/// rather than trusted, because the two are sized in different crates and a harness hands over
/// blocks of its own.
///
/// # Errors
///
/// Returns [`GraphSetError::NotEnoughDummyBlocks`] for the first bucket, in bucket order, with
/// more rows than `blocks` holds.
pub fn dummy_runs(
    buckets: &DecodeBuckets,
    blocks: &[BlockId],
) -> Result<Vec<DummyRun>, GraphSetError> {
    buckets
        .iter()
        .map(|(bucket, rows)| {
            let Some(taken) = blocks.get(..rows) else {
                return Err(GraphSetError::NotEnoughDummyBlocks {
                    rows,
                    blocks: blocks.len(),
                });
            };
            Ok(DummyRun::new(bucket, taken.to_vec()))
        })
        .collect()
}

/// Which graph serves each bucket: the recordings one capture of the bucket ladder made, at the
/// index of the bucket each was made over.
///
/// The set holds what it is given. That the graphs are one per bucket in bucket order is the
/// capture's to uphold, and a [`GraphIdx`] is minted nowhere else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphSet {
    graphs: Vec<GraphIdx>,
}

impl GraphSet {
    /// The set of `graphs`, one per bucket in bucket order.
    #[must_use]
    pub fn new(graphs: Vec<GraphIdx>) -> Self {
        Self { graphs }
    }

    /// The graph recorded for `bucket`.
    ///
    /// # Errors
    ///
    /// Returns [`GraphSetError::NoGraphForBucket`] when the set holds no bucket at that index.
    pub fn graph(&self, bucket: BucketIdx) -> Result<GraphIdx, GraphSetError> {
        self.graphs
            .get(bucket.0)
            .copied()
            .ok_or(GraphSetError::NoGraphForBucket {
                bucket,
                graphs: self.graphs.len(),
            })
    }
}

#[cfg(test)]
mod tests {
    use atoma_core::dispatch::BucketLadder;
    use atoma_core::types::RequestCount;

    use super::*;
    use crate::test_support::engine_config;

    fn blocks(ids: impl IntoIterator<Item = u32>) -> Vec<BlockId> {
        ids.into_iter().map(BlockId::new).collect()
    }

    /// The buckets of a bucket ladder of `sizes`, at a maximum batch of the largest of them.
    fn buckets(sizes: Vec<usize>) -> DecodeBuckets {
        let largest =
            RequestCount::new(sizes.iter().copied().max().expect("nonempty")).expect("nonzero");
        let mut config = engine_config().dispatch;
        config.bucket_ladder = BucketLadder::new(sizes).expect("nonempty");
        config.captured_max_requests = largest;
        DecodeBuckets::usable(&config, largest)
    }

    #[test]
    fn each_bucket_fills_its_rows_from_the_leading_blocks_in_bucket_order() {
        let runs = dummy_runs(&buckets(vec![4, 1, 2]), &blocks([70, 71, 72, 73])).unwrap();

        let described: Vec<(BucketIdx, usize)> =
            runs.iter().map(|run| (run.bucket(), run.rows())).collect();
        assert_eq!(
            described,
            [(BucketIdx(0), 4), (BucketIdx(1), 1), (BucketIdx(2), 2)],
            "one run per bucket, in bucket order, with the bucket's rows"
        );
        assert_eq!(
            runs.iter().map(DummyRun::blocks).collect::<Vec<_>>(),
            [
                &blocks([70, 71, 72, 73])[..],
                &blocks([70])[..],
                &blocks([70, 71])[..],
            ],
            "every bucket's rows are filled from the same leading blocks"
        );
    }

    #[test]
    fn blocks_past_the_largest_bucket_fill_nothing() {
        let runs = dummy_runs(&buckets(vec![1, 2]), &blocks([5, 6, 7, 8, 9])).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[1].blocks(), &blocks([5, 6])[..]);
    }

    #[test]
    fn a_bucket_with_more_rows_than_blocks_is_refused_by_its_rows_and_the_blocks_held() {
        let refused = dummy_runs(&buckets(vec![1, 2, 4]), &blocks([5, 6, 7])).unwrap_err();
        assert_eq!(
            refused,
            GraphSetError::NotEnoughDummyBlocks { rows: 4, blocks: 3 }
        );

        assert_eq!(
            dummy_runs(&buckets(vec![2]), &[]).unwrap_err(),
            GraphSetError::NotEnoughDummyBlocks { rows: 2, blocks: 0 },
            "no blocks at all cover no bucket"
        );
    }

    #[test]
    fn the_refusal_names_the_first_short_bucket_in_bucket_order_not_the_largest() {
        // Bucket 0 has four rows and bucket 1 has eight; both are short of two blocks, and the
        // one named is the first the capture would reach.
        let refused = dummy_runs(&buckets(vec![4, 8]), &blocks([1, 2])).unwrap_err();
        assert_eq!(
            refused,
            GraphSetError::NotEnoughDummyBlocks { rows: 4, blocks: 2 }
        );
    }

    #[test]
    fn each_bucket_is_served_by_the_graph_at_its_index_and_a_bucket_past_the_set_by_none() {
        // Indices that are not the buckets' own, so a lookup by the wrong index or by position
        // arithmetic cannot land on the right graph by coincidence.
        let set = GraphSet::new(vec![
            GraphIdx::for_test(7),
            GraphIdx::for_test(2),
            GraphIdx::for_test(5),
        ]);

        assert_eq!(set.graph(BucketIdx(0)), Ok(GraphIdx::for_test(7)));
        assert_eq!(set.graph(BucketIdx(1)), Ok(GraphIdx::for_test(2)));
        assert_eq!(set.graph(BucketIdx(2)), Ok(GraphIdx::for_test(5)));
        assert_eq!(
            set.graph(BucketIdx(3)),
            Err(GraphSetError::NoGraphForBucket {
                bucket: BucketIdx(3),
                graphs: 3,
            }),
            "no fourth bucket was captured"
        );
    }

    #[test]
    fn an_empty_graph_set_serves_no_bucket() {
        assert_eq!(
            GraphSet::new(Vec::new()).graph(BucketIdx(0)),
            Err(GraphSetError::NoGraphForBucket {
                bucket: BucketIdx(0),
                graphs: 0,
            })
        );
    }
}
