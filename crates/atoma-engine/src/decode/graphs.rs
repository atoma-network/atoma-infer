//! What one capture of the bucket ladder produces, and what each of its recordings is made over:
//! the graph serving each bucket, what capturing it cost, and the dummy run that fills each
//! bucket's rows from the padding dummies' blocks.
//!
//! None of it needs a device. Which block a bucket's row is filled from, which graph serves a
//! bucket and what two readings of free memory say a capture used are settled here; driving the
//! session over them and taking the readings is `device::capture`'s.

use std::time::Duration;

use atoma_core::types::BlockId;
use atoma_runtime::arena::BucketIdx;
use atoma_runtime::context::DeviceBytes;
use atoma_runtime::graph_memory::{GraphMemory, GraphMemoryError};
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

/// What recording one bucket's graph cost: how long its warmup and recording took, what device
/// memory they used, and what the device had free once the graph was recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphCost {
    pub bucket: BucketIdx,
    pub rows: usize,
    pub elapsed: Duration,
    /// The drop in free device memory across the bucket's warmup and recording, and none where
    /// it rose: a reading is the device's free memory, not this process's.
    pub used: DeviceBytes,
    /// Free device memory once the graph was recorded.
    pub free: DeviceBytes,
}

impl GraphCost {
    /// The cost of `run`'s bucket, whose warmup and recording took `elapsed` and left `after`
    /// free where `before` was free ahead of them.
    #[must_use]
    pub fn measured(
        run: &DummyRun,
        elapsed: Duration,
        before: DeviceBytes,
        after: DeviceBytes,
    ) -> Self {
        Self {
            bucket: run.bucket(),
            rows: run.rows(),
            elapsed,
            used: before.saturating_sub(after),
            free: after,
        }
    }
}

/// What capturing the bucket ladder cost at startup: each graph's cost, and the whole capture's
/// time and memory, the one warmup ahead of every recording included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureReport {
    graphs: Vec<GraphCost>,
    elapsed: Duration,
    used: DeviceBytes,
    free: DeviceBytes,
}

impl CaptureReport {
    /// The report of a capture over `graphs` that took `elapsed` in all and left `after` free
    /// where `before` was free ahead of it.
    #[must_use]
    pub fn measured(
        graphs: Vec<GraphCost>,
        elapsed: Duration,
        before: DeviceBytes,
        after: DeviceBytes,
    ) -> Self {
        Self {
            graphs,
            elapsed,
            used: before.saturating_sub(after),
            free: after,
        }
    }

    /// Each graph's cost, in bucket order.
    #[must_use]
    pub fn graphs(&self) -> &[GraphCost] {
        &self.graphs
    }

    /// How long the whole capture took.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// The drop in free device memory across the whole capture, and none where it rose.
    #[must_use]
    pub fn used(&self) -> DeviceBytes {
        self.used
    }

    /// Free device memory once the capture was over.
    #[must_use]
    pub fn free(&self) -> DeviceBytes {
        self.free
    }

    /// Graph memory as a fixed term plus a marginal term per graph, fitted over what each graph
    /// used: a report of what these captures cost, and no ceiling for one that has not run.
    ///
    /// # Errors
    ///
    /// Returns [`GraphMemoryError::FitWithoutTwoGraphs`] when fewer than two graphs were
    /// captured; [`CaptureReport::used`] is the whole of what there is to report then.
    pub fn graph_memory(&self) -> Result<GraphMemory, GraphMemoryError> {
        let used: Vec<DeviceBytes> = self.graphs.iter().map(|graph| graph.used).collect();
        GraphMemory::fit(&used)
    }
}

#[cfg(test)]
mod tests {
    use atoma_core::dispatch::BucketLadder;
    use atoma_core::types::RequestCount;

    use super::*;
    use crate::test_support::engine_config;

    const MIB: usize = 1024 * 1024;

    fn bytes(value: usize) -> DeviceBytes {
        DeviceBytes::new(value)
    }

    fn run(bucket: usize, rows: usize) -> DummyRun {
        DummyRun::new(BucketIdx(bucket), vec![BlockId::new(9); rows])
    }

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

    #[test]
    fn a_graphs_cost_is_the_drop_in_free_memory_across_it_and_none_where_it_rose() {
        let cost = GraphCost::measured(
            &run(2, 4),
            Duration::from_millis(12),
            bytes(40 * MIB),
            bytes(38 * MIB),
        );
        assert_eq!(
            cost,
            GraphCost {
                bucket: BucketIdx(2),
                rows: 4,
                elapsed: Duration::from_millis(12),
                used: bytes(2 * MIB),
                free: bytes(38 * MIB),
            }
        );

        // Something else on the device freed memory between the two readings.
        let rose = GraphCost::measured(&run(0, 1), Duration::ZERO, bytes(MIB), bytes(3 * MIB));
        assert_eq!((rose.used, rose.free), (bytes(0), bytes(3 * MIB)));
    }

    #[test]
    fn the_reports_totals_are_the_whole_captures_not_the_graphs_summed() {
        // The warmup ahead of every recording is in the whole capture's readings and in no
        // graph's, so the total can exceed what the graphs used between them.
        let graphs = vec![
            GraphCost::measured(
                &run(0, 1),
                Duration::from_millis(5),
                bytes(70 * MIB),
                bytes(68 * MIB),
            ),
            GraphCost::measured(
                &run(1, 2),
                Duration::from_millis(7),
                bytes(68 * MIB),
                bytes(68 * MIB),
            ),
        ];
        let report = CaptureReport::measured(
            graphs.clone(),
            Duration::from_millis(30),
            bytes(100 * MIB),
            bytes(68 * MIB),
        );

        assert_eq!(report.graphs(), &graphs[..]);
        assert_eq!(report.elapsed(), Duration::from_millis(30));
        assert_eq!(report.used(), bytes(32 * MIB));
        assert_eq!(report.free(), bytes(68 * MIB));
    }

    #[test]
    fn graph_memory_is_fitted_over_what_each_graph_used() {
        // Three graphs whose readings follow an affine rule — 8 MiB fixed and 2 MiB a graph, the
        // first paying the fixed term as well as its own — fit back to its two terms.
        let readings = [(0, 10 * MIB), (1, 2 * MIB), (2, 2 * MIB)];
        let graphs: Vec<GraphCost> = readings
            .iter()
            .map(|&(bucket, used)| {
                GraphCost::measured(&run(bucket, 1), Duration::ZERO, bytes(used), bytes(0))
            })
            .collect();
        let report = CaptureReport::measured(graphs, Duration::ZERO, bytes(0), bytes(0));

        let memory = report.graph_memory().unwrap();

        assert_eq!(memory.fixed(), bytes(8 * MIB));
        assert_eq!(memory.marginal(), bytes(2 * MIB));
    }

    #[test]
    fn one_graphs_reading_names_no_fixed_and_marginal_term() {
        let one = vec![GraphCost::measured(
            &run(0, 1),
            Duration::ZERO,
            bytes(MIB),
            bytes(0),
        )];
        let report = CaptureReport::measured(one, Duration::ZERO, bytes(MIB), bytes(0));
        assert_eq!(
            report.graph_memory().unwrap_err(),
            GraphMemoryError::FitWithoutTwoGraphs { graphs: 1 }
        );
        assert_eq!(
            report.used(),
            bytes(MIB),
            "the total is what there is to report"
        );
    }
}
