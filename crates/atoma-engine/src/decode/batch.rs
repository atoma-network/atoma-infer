//! A keyed batch held to its bucket, and the buckets the decode step serves.
//!
//! The engine pads a uniform-decode batch to its bucket and keys it; the executor acts on the key
//! without re-deriving it. What the decode step still has to establish is that the batch it was
//! handed is the shape its graphs bake: one token per entry, as many entries as the bucket, the
//! live entries leading and every one of them sampling, so the logits it reads back are exactly
//! the leading rows. A batch that is keyed but not that shape is not an error: a one-token
//! prefill chunk that does not sample yet, or a bucket the decode step did not resolve, is named
//! and served eagerly. A batch that contradicts its own key is the engine breaking the step
//! protocol, and is refused.

use atoma_core::dispatch::{DispatchConfig, GraphKey};
use atoma_core::types::RequestCount;
use atoma_runtime::arena::BucketIdx;
use thiserror::Error;

use crate::batch::BatchLayout;

/// A keyed batch that contradicts its key or the layout protocol: the engine's bug, not a
/// runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DecodeBatchError {
    #[error("the key is not uniform decode, but only uniform-decode batches are keyed")]
    KeyNotUniformDecode,
    #[error("{entries} entries compute {tokens} tokens; a keyed batch computes one per entry")]
    EntryNotOneToken { entries: usize, tokens: usize },
    #[error("{entries} entries were laid out for a bucket of {bucket}")]
    EntriesNotBucket { entries: usize, bucket: usize },
    #[error("{entries} entries are all {padding} padding dummies")]
    NoLiveEntries { entries: usize, padding: usize },
    #[error("a padding dummy at entry {entry} samples")]
    DummySamples { entry: usize },
    #[error(
        "the block table is {width} blocks wide; the decode step stages {max_width}, the most \
         a sequence can hold"
    )]
    BlockTableTooWide { width: usize, max_width: usize },
}

/// Why a keyed batch runs eagerly after all: a shape the decode step does not serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum Unservable {
    /// A live entry that does not sample: a prefill chunk whose last token is not the prompt's.
    /// The logits read back are the leading rows, one per live entry, so every live entry must
    /// want one.
    #[error(
        "live entry {entry} does not sample; the decode step reads back one row per live entry"
    )]
    LiveEntryNotSampling { entry: usize },
    /// The key's bucket is above every bucket the decode step resolved.
    #[error("no resolved bucket holds {tokens} tokens; the largest is {largest}")]
    NoBucket { tokens: usize, largest: usize },
}

/// What checking a keyed batch found: the decode step serves it, or candle does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Checked {
    /// Served by the decode step, at this bucket.
    Step(DecodeBatch),
    /// Served eagerly, on candle, for this reason.
    Eager(Unservable),
}

/// The buckets the decode step serves: the configured bucket ladder's distinct entries at or
/// below the maximum batch, in configured order. The arena and every slot table are built over
/// exactly these, so an entry's position here is its bucket index everywhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeBuckets {
    tokens: Vec<usize>,
}

impl DecodeBuckets {
    /// The entries of `config`'s bucket ladder a step of at most `max_batch` entries can fill: a
    /// uniform decode gives every entry one token, so a bucket above the maximum batch never
    /// fills. A size the ladder repeats is served by its first entry, so the repeats are dropped
    /// rather than given tables and a graph nothing routes to.
    #[must_use]
    pub fn usable(config: &DispatchConfig, max_batch: RequestCount) -> Self {
        let mut tokens: Vec<usize> = Vec::new();
        for &bucket in config.bucket_ladder.buckets() {
            if bucket <= max_batch.get() && !tokens.contains(&bucket) {
                tokens.push(bucket);
            }
        }
        Self { tokens }
    }

    /// The buckets, in index order.
    #[must_use]
    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    /// The largest bucket, or zero when nothing is usable.
    #[must_use]
    pub fn largest(&self) -> usize {
        self.tokens.iter().copied().max().unwrap_or(0)
    }

    /// The bucket serving exactly `tokens`: the first entry of that size.
    #[must_use]
    pub fn index_of(&self, tokens: usize) -> Option<BucketIdx> {
        self.tokens
            .iter()
            .position(|&bucket| bucket == tokens)
            .map(BucketIdx)
    }
}

/// A keyed batch the decode step serves: its bucket, and how many of the bucket's rows are live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeBatch {
    pub bucket: BucketIdx,
    /// Entries in the batch, dummies included: the bucket's size.
    pub tokens: usize,
    /// The leading entries that are live; each samples, so these are the logits rows read back.
    pub live: usize,
}

impl DecodeBatch {
    /// Holds `layout`, keyed by `key`, to the shape the decode step bakes, and finds its bucket
    /// among `buckets`; the block table must fit `max_block_table_width` columns.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeBatchError`] when the layout contradicts its key: an entry computing more
    /// than one token, an entry count that is not the bucket, no live entry, a dummy that
    /// samples, or a block table wider than a sequence can be.
    pub fn check(
        layout: &BatchLayout,
        key: GraphKey,
        buckets: &DecodeBuckets,
        max_block_table_width: usize,
    ) -> Result<Checked, DecodeBatchError> {
        if !key.uniform_decode() {
            return Err(DecodeBatchError::KeyNotUniformDecode);
        }
        let entries = layout.entry_count();
        let tokens = layout.token_count();
        if layout.prefill_entries != 0 || tokens != entries {
            return Err(DecodeBatchError::EntryNotOneToken { entries, tokens });
        }
        let bucket = key.padded_token_count().get();
        if entries != bucket {
            return Err(DecodeBatchError::EntriesNotBucket { entries, bucket });
        }
        let padding = layout.padding_count;
        let Some(live) = entries.checked_sub(padding).filter(|&live| live > 0) else {
            return Err(DecodeBatchError::NoLiveEntries { entries, padding });
        };
        if layout.block_table_width > max_block_table_width {
            return Err(DecodeBatchError::BlockTableTooWide {
                width: layout.block_table_width,
                max_width: max_block_table_width,
            });
        }
        // Every entry computes one token, so a selected row is its entry's index, and the live
        // entries lead: the sampling entries are the leading rows exactly when each live entry
        // samples and no dummy does.
        let selected: Vec<usize> = layout.selected.iter().map(|&row| row as usize).collect();
        if let Some(&entry) = selected.iter().find(|&&entry| entry >= live) {
            return Err(DecodeBatchError::DummySamples { entry });
        }
        if let Some(entry) = (0..live).find(|entry| !selected.contains(entry)) {
            return Ok(Checked::Eager(Unservable::LiveEntryNotSampling { entry }));
        }
        let Some(index) = buckets.index_of(bucket) else {
            return Ok(Checked::Eager(Unservable::NoBucket {
                tokens: bucket,
                largest: buckets.largest(),
            }));
        };
        Ok(Checked::Step(Self {
            bucket: index,
            tokens: bucket,
            live,
        }))
    }
}

#[cfg(test)]
mod tests {
    use atoma_core::dispatch::{BucketLadder, DispatchDecision};
    use atoma_core::step::CommandEntry;
    use atoma_core::types::RequestCount;

    use super::*;
    use crate::test_support::{dummy, engine_config, entry, keyed_command, BLOCK_SIZE};

    const MAX_WIDTH: usize = 8;

    fn buckets() -> DecodeBuckets {
        let config = engine_config();
        DecodeBuckets::usable(&config.dispatch, config.scheduler.max_batch)
    }

    /// Lays a keyed command out and takes its key.
    fn keyed(live: Vec<CommandEntry>) -> (BatchLayout, GraphKey) {
        let command = keyed_command(live);
        let layout = BatchLayout::lay_out(&command, BLOCK_SIZE).unwrap();
        let DispatchDecision::FullReplay(key) = layout.dispatch else {
            panic!("a uniform-decode batch is keyed: {:?}", layout.dispatch);
        };
        (layout, key)
    }

    #[test]
    fn the_usable_buckets_are_the_ladders_distinct_entries_up_to_the_maximum_batch_in_order() {
        let mut config = engine_config().dispatch;
        config.bucket_ladder = BucketLadder::new(vec![8, 1, 4, 2, 16, 4]).unwrap();

        let buckets = DecodeBuckets::usable(&config, RequestCount::new(4).unwrap());

        assert_eq!(
            buckets.tokens(),
            [1, 4, 2],
            "the repeated four is served by its first entry and gets no second table"
        );
        assert_eq!(buckets.largest(), 4);
        assert_eq!(buckets.index_of(4), Some(BucketIdx(1)));
        assert_eq!(buckets.index_of(2), Some(BucketIdx(2)));
        assert_eq!(buckets.index_of(8), None);
        assert_eq!(buckets.index_of(3), None);
    }

    #[test]
    fn a_bucket_a_uniform_decode_could_never_fill_is_not_usable() {
        let mut config = engine_config().dispatch;
        config.bucket_ladder = BucketLadder::new(vec![1, 2, 4, 8]).unwrap();
        config.captured_max_requests = RequestCount::new(8).unwrap();

        let buckets = DecodeBuckets::usable(&config, RequestCount::new(4).unwrap());

        assert_eq!(
            buckets.tokens(),
            [1, 2, 4],
            "a batch of at most four entries decodes at most four tokens, whatever the captured \
             maximum"
        );
        assert_eq!(buckets.largest(), 4);
        assert_eq!(buckets.index_of(8), None);
    }

    #[test]
    fn three_decodes_pad_to_the_bucket_of_four_with_three_live_rows() {
        let (layout, key) = keyed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 8, vec![9], &[20, 21, 22], true),
            entry(3, 1, vec![9], &[30], true),
        ]);
        assert_eq!(layout.padding_count, 1);

        let checked = DecodeBatch::check(&layout, key, &buckets(), MAX_WIDTH).unwrap();

        assert_eq!(
            checked,
            Checked::Step(DecodeBatch {
                bucket: BucketIdx(2),
                tokens: 4,
                live: 3,
            })
        );
    }

    #[test]
    fn one_decode_fills_the_bucket_of_one_with_no_padding() {
        let (layout, key) = keyed(vec![entry(1, 3, vec![9], &[10], true)]);
        let checked = DecodeBatch::check(&layout, key, &buckets(), MAX_WIDTH).unwrap();
        let Checked::Step(batch) = checked else {
            panic!("{checked:?}");
        };
        assert_eq!(
            (batch.bucket, batch.tokens, batch.live),
            (BucketIdx(0), 1, 1)
        );
    }

    #[test]
    fn a_one_token_chunk_that_does_not_sample_yet_goes_eager_by_name() {
        let (layout, key) = keyed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 2, vec![9], &[20, 21], false),
        ]);
        let checked = DecodeBatch::check(&layout, key, &buckets(), MAX_WIDTH).unwrap();
        assert_eq!(
            checked,
            Checked::Eager(Unservable::LiveEntryNotSampling { entry: 1 })
        );
    }

    #[test]
    fn a_bucket_the_decode_step_did_not_resolve_goes_eager_with_the_largest_it_has() {
        let (layout, key) = keyed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 3, vec![9], &[20], true),
            entry(3, 3, vec![9], &[30], true),
        ]);
        let buckets =
            DecodeBuckets::usable(&engine_config().dispatch, RequestCount::new(2).unwrap());

        let checked = DecodeBatch::check(&layout, key, &buckets, MAX_WIDTH).unwrap();

        assert_eq!(
            checked,
            Checked::Eager(Unservable::NoBucket {
                tokens: 4,
                largest: 2
            })
        );
    }

    #[test]
    fn a_layout_that_contradicts_its_key_is_refused_by_name() {
        let (mut layout, key) = keyed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 3, vec![9], &[20], true),
            entry(3, 3, vec![9], &[30], true),
        ]);
        assert_eq!((layout.entry_count(), layout.padding_count), (4, 1));
        let buckets = buckets();

        let mut wide = layout.clone();
        wide.block_table_width = MAX_WIDTH + 1;
        assert_eq!(
            DecodeBatch::check(&wide, key, &buckets, MAX_WIDTH).unwrap_err(),
            DecodeBatchError::BlockTableTooWide {
                width: MAX_WIDTH + 1,
                max_width: MAX_WIDTH
            }
        );

        let mut sampling_dummy = layout.clone();
        sampling_dummy.selected.push(3);
        assert_eq!(
            DecodeBatch::check(&sampling_dummy, key, &buckets, MAX_WIDTH).unwrap_err(),
            DecodeBatchError::DummySamples { entry: 3 }
        );

        let mut all_padding = layout.clone();
        all_padding.padding_count = 4;
        assert_eq!(
            DecodeBatch::check(&all_padding, key, &buckets, MAX_WIDTH).unwrap_err(),
            DecodeBatchError::NoLiveEntries {
                entries: 4,
                padding: 4
            }
        );

        layout.tokens.push(9);
        assert_eq!(
            DecodeBatch::check(&layout, key, &buckets, MAX_WIDTH).unwrap_err(),
            DecodeBatchError::EntryNotOneToken {
                entries: 4,
                tokens: 5
            }
        );
    }

    #[test]
    fn a_batch_laid_out_for_another_bucket_than_its_key_is_refused() {
        let (layout, key) = keyed(vec![
            entry(1, 3, vec![9], &[10], true),
            entry(2, 3, vec![9], &[20], true),
        ]);
        let (_, key_of_one) = keyed(vec![entry(1, 3, vec![9], &[10], true)]);
        assert_ne!(key, key_of_one);
        assert_eq!(
            DecodeBatch::check(&layout, key_of_one, &buckets(), MAX_WIDTH).unwrap_err(),
            DecodeBatchError::EntriesNotBucket {
                entries: 2,
                bucket: 1
            }
        );
    }

    #[test]
    fn a_prefill_in_a_keyed_batch_is_refused_as_not_one_token() {
        let command = keyed_command(vec![entry(1, 3, vec![9], &[10], true)]);
        let mut with_prefill = command.clone();
        with_prefill
            .entries
            .push(entry(2, 0, vec![1, 2, 3], &[20], true));
        with_prefill.entries.push(dummy(3, 30));
        with_prefill.padding_count = 1;
        let layout = BatchLayout::lay_out(&with_prefill, BLOCK_SIZE).unwrap();
        let DispatchDecision::FullReplay(key) = command.dispatch else {
            panic!("keyed");
        };
        assert_eq!(
            DecodeBatch::check(&layout, key, &buckets(), MAX_WIDTH).unwrap_err(),
            DecodeBatchError::EntryNotOneToken {
                entries: 3,
                tokens: 5
            }
        );
    }
}
