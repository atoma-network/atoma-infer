//! Laying a step command out as the batch arrays the model forward takes.
//!
//! The attention kernels take a batch prefill entries first, then decode entries: the tokens
//! flattened in that order, one block-table row per entry in that order, and cumulative starts
//! carrying one value more than the batch has entries. A step command holds its entries in the
//! engine's order — live entries, then the padding dummies — so the layout reorders them and keeps
//! the way back: which logits row each sampling entry reads. Pure host arithmetic, with no device
//! and no tensor: what is laid out here is copied in as it stands.
//!
//! The command's dispatch decision and padding count travel with the layout, so a forward that
//! routes on the decision reads it from the one value it is handed.

use atoma_core::dispatch::DispatchDecision;
use atoma_core::request::SamplingParams;
use atoma_core::step::{CommandEntry, StepCommand};
use atoma_core::types::{RequestId, RequestSlot, TokenCount};
use thiserror::Error;

/// A step command that cannot be laid out. Each is the engine breaking the step protocol, not a
/// runtime state: the engine issues no empty command, schedules no entry that computes nothing,
/// and grows every block table before it schedules an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LayoutError {
    #[error("the step command has no entries")]
    NoEntries,
    #[error("the entry for request {} computes no tokens", request.get())]
    EmptyQuery { request: RequestId },
    #[error(
        "the entry for request {} holds {blocks} blocks, which do not cover its sequence length \
         of {sequence_len}",
        request.get()
    )]
    BlockTableShort {
        request: RequestId,
        blocks: usize,
        sequence_len: usize,
    },
}

/// What one selected row samples under: whose token it draws, the request slot that token is
/// kept in, and the parameters it is drawn with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowSampling {
    pub request: RequestId,
    pub slot: RequestSlot,
    pub params: SamplingParams,
}

/// A step command laid out prefill-first, as the forward takes it.
///
/// An entry computing more than one token is a prefill and leads the batch; one computing a
/// single token is a decode and follows, the padding dummies among them. Per-token arrays hold
/// every entry's tokens flattened in that order, per-entry arrays hold one value per entry in
/// that order, and the two cumulative starts hold one value more. The block tables are one row
/// per entry, right-padded with zero to the widest.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchLayout {
    pub tokens: Vec<u32>,
    /// Each token's position in its sequence: `context_len..sequence_len` per entry.
    pub positions: Vec<i64>,
    /// Each token's KV slot: its block id times the block size, plus its offset in the block.
    pub slot_mapping: Vec<i64>,
    pub context_lengths: Vec<u32>,
    pub sequence_lengths: Vec<u32>,
    pub query_start_locations: Vec<u32>,
    pub sequence_start_locations: Vec<u32>,
    /// Row-major, `entry_count` rows of `block_table_width`.
    pub block_tables: Vec<u32>,
    pub block_table_width: usize,
    /// How many entries lead the batch as prefills.
    pub prefill_entries: usize,
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub max_query_len: usize,
    pub max_prefill_sequence_len: usize,
    pub max_decode_sequence_len: usize,
    /// The flattened index of each sampling entry's last token, in batch order: the rows the
    /// forward selects logits for. Entries that do not sample select nothing.
    pub selected: Vec<u32>,
    /// What each selected row samples under, in the same order as `selected`.
    pub sampling: Vec<RowSampling>,
    /// Which captured graph serves the batch, or why it runs eagerly, as the engine decided.
    pub dispatch: DispatchDecision,
    /// How many trailing entries of the command are padding dummies.
    pub padding_count: usize,
    /// For each sampling entry in command order, its row among `selected`.
    rows: Vec<usize>,
}

impl BatchLayout {
    /// Lays `command` out over `block_size`-token blocks.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] when the command has no entries, an entry computes no tokens, or
    /// an entry's block table does not cover its sequence length.
    pub fn lay_out(command: &StepCommand, block_size: TokenCount) -> Result<Self, LayoutError> {
        let block_size = block_size.get();
        if command.entries.is_empty() {
            return Err(LayoutError::NoEntries);
        }
        for entry in &command.entries {
            if entry.query_len() == 0 {
                return Err(LayoutError::EmptyQuery {
                    request: entry.request,
                });
            }
            if entry.block_table.len() * block_size < entry.sequence_len() {
                return Err(LayoutError::BlockTableShort {
                    request: entry.request,
                    blocks: entry.block_table.len(),
                    sequence_len: entry.sequence_len(),
                });
            }
        }
        Ok(Self::build(command, block_size))
    }

    /// The layout of a command every entry of which computes at least one token over a block
    /// table that covers it.
    fn build(command: &StepCommand, block_size: usize) -> Self {
        let entries = &command.entries;
        let (prefills, decodes): (Vec<usize>, Vec<usize>) =
            (0..entries.len()).partition(|&index| is_prefill(&entries[index]));
        let token_count = entries.iter().map(CommandEntry::query_len).sum();
        let block_table_width = entries
            .iter()
            .map(|entry| entry.block_table.len())
            .max()
            .unwrap_or(0);
        let mut layout = Self {
            tokens: Vec::with_capacity(token_count),
            positions: Vec::with_capacity(token_count),
            slot_mapping: Vec::with_capacity(token_count),
            context_lengths: Vec::with_capacity(entries.len()),
            sequence_lengths: Vec::with_capacity(entries.len()),
            query_start_locations: vec![0],
            sequence_start_locations: vec![0],
            block_tables: Vec::with_capacity(entries.len() * block_table_width),
            block_table_width,
            prefill_entries: prefills.len(),
            prefill_tokens: 0,
            decode_tokens: 0,
            max_query_len: 0,
            max_prefill_sequence_len: 0,
            max_decode_sequence_len: 0,
            selected: Vec::with_capacity(command.sampling_count()),
            sampling: Vec::with_capacity(command.sampling_count()),
            dispatch: command.dispatch,
            padding_count: command.padding_count,
            rows: Vec::with_capacity(command.sampling_count()),
        };
        let mut row_by_index = vec![None; entries.len()];
        for index in prefills.into_iter().chain(decodes) {
            let entry = &entries[index];
            if let Some(params) = entry.sampling {
                row_by_index[index] = Some(layout.selected.len());
                layout
                    .selected
                    .push(fits_u32(layout.tokens.len() + entry.query_len() - 1));
                layout.sampling.push(RowSampling {
                    request: entry.request,
                    slot: entry.slot,
                    params,
                });
            }
            layout.push_entry(entry, block_size);
        }
        layout.rows.extend(row_by_index.into_iter().flatten());
        layout
    }

    /// Appends one entry's tokens, positions, slots, lengths, starts and padded block-table row.
    fn push_entry(&mut self, entry: &CommandEntry, block_size: usize) {
        let query_len = entry.query_len();
        let sequence_len = entry.sequence_len();
        self.tokens.extend_from_slice(&entry.input_tokens);
        for position in entry.context_len..sequence_len {
            let block = entry.block_table[position / block_size];
            let slot = block.index() * block_size + position % block_size;
            self.positions.push(fits_i64(position));
            self.slot_mapping.push(fits_i64(slot));
        }
        self.context_lengths.push(fits_u32(entry.context_len));
        self.sequence_lengths.push(fits_u32(sequence_len));
        self.query_start_locations.push(fits_u32(self.tokens.len()));
        let sequence_start = self.sequence_start_locations[self.sequence_start_locations.len() - 1];
        self.sequence_start_locations
            .push(sequence_start + fits_u32(sequence_len));
        self.block_tables
            .extend(entry.block_table.iter().map(|block| block.get()));
        self.block_tables.resize(
            self.block_tables.len() + self.block_table_width - entry.block_table.len(),
            0,
        );
        self.max_query_len = self.max_query_len.max(query_len);
        if is_prefill(entry) {
            self.prefill_tokens += query_len;
            self.max_prefill_sequence_len = self.max_prefill_sequence_len.max(sequence_len);
        } else {
            self.decode_tokens += query_len;
            self.max_decode_sequence_len = self.max_decode_sequence_len.max(sequence_len);
        }
    }

    /// Entries in the batch, dummies included.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.context_lengths.len()
    }

    /// Tokens the batch computes, summed over entries.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// For each sampling entry in command order, the row of the selected logits it reads.
    #[must_use]
    pub fn sampling_rows(&self) -> &[usize] {
        &self.rows
    }
}

/// An entry computing more than one token this step goes through the prefill kernel; one
/// computing a single token — a decode, a dummy, or the last token of a chunked prefill — goes
/// through the single-query decode kernel, which is the same attention over the block table.
fn is_prefill(entry: &CommandEntry) -> bool {
    entry.query_len() > 1
}

fn fits_u32(value: usize) -> u32 {
    u32::try_from(value).expect("a token count or index fits u32")
}

fn fits_i64(value: usize) -> i64 {
    i64::try_from(value).expect("a position or slot fits i64")
}

#[cfg(test)]
mod tests {
    use atoma_core::request::{SamplingParams, PADDING_TOKEN};
    use atoma_core::step::StepCommand;
    use atoma_core::types::{RequestId, RequestSlot, TokenCount};

    use super::{BatchLayout, LayoutError, RowSampling};
    use crate::test_support::{command, dummy, entry, sampling_entry};

    const BLOCK_SIZE: TokenCount = TokenCount::new(4).expect("nonzero");

    /// A decode, a prefill, a dummy and another decode, in the engine's order.
    fn mixed() -> StepCommand {
        command(
            vec![
                entry(1, 7, vec![70], &[10, 11], true),
                entry(2, 0, vec![1, 2, 3, 4, 5], &[20, 21], true),
                dummy(3, 30),
                entry(4, 3, vec![33], &[40], true),
            ],
            1,
        )
    }

    #[test]
    fn prefills_lead_the_batch_and_decodes_follow_in_command_order() {
        let layout = BatchLayout::lay_out(&mixed(), BLOCK_SIZE).unwrap();
        assert_eq!(layout.entry_count(), 4);
        assert_eq!(layout.token_count(), 8);
        assert_eq!(layout.prefill_entries, 1);
        assert_eq!(layout.prefill_tokens, 5);
        assert_eq!(layout.decode_tokens, 3);
        assert_eq!(layout.tokens, [1, 2, 3, 4, 5, 70, PADDING_TOKEN, 33]);
        assert_eq!(layout.positions, [0, 1, 2, 3, 4, 7, 0, 3]);
        assert_eq!(layout.context_lengths, [0, 7, 0, 3]);
        assert_eq!(layout.sequence_lengths, [5, 8, 1, 4]);
        assert_eq!(layout.query_start_locations, [0, 5, 6, 7, 8]);
        assert_eq!(layout.sequence_start_locations, [0, 5, 13, 14, 18]);
        assert_eq!(layout.max_query_len, 5);
        assert_eq!(layout.max_prefill_sequence_len, 5);
        assert_eq!(layout.max_decode_sequence_len, 8);
    }

    #[test]
    fn slots_are_block_times_block_size_plus_offset_for_every_computed_token() {
        let layout = BatchLayout::lay_out(&mixed(), BLOCK_SIZE).unwrap();
        // The prefill fills block 20 and the first slot of block 21; the first decode's token
        // sits at position 7, the last slot of its second block; the dummy's token is the first
        // slot of its own block; the second decode's token is the last slot of block 40.
        assert_eq!(
            layout.slot_mapping,
            [80, 81, 82, 83, 84, 47, 120, 163],
            "{:?}",
            layout.slot_mapping
        );
    }

    #[test]
    fn block_tables_are_padded_to_the_widest_row() {
        let layout = BatchLayout::lay_out(&mixed(), BLOCK_SIZE).unwrap();
        assert_eq!(layout.block_table_width, 2);
        assert_eq!(layout.block_tables, [20, 21, 10, 11, 30, 0, 40, 0]);
    }

    #[test]
    fn selected_rows_are_each_sampling_entrys_last_token_mapped_back_to_command_order() {
        let layout = BatchLayout::lay_out(&mixed(), BLOCK_SIZE).unwrap();
        assert_eq!(
            layout.selected,
            [4, 5, 7],
            "the prefill's last token, then each decode's one token; the dummy selects nothing"
        );
        assert_eq!(
            layout.sampling_rows(),
            [1, 0, 2],
            "command order is decode 1, prefill 2, decode 4: rows 1, 0, 2"
        );
    }

    #[test]
    fn each_selected_row_carries_the_request_slot_and_parameters_it_samples_under() {
        let drawn = SamplingParams {
            do_sample: true,
            temperature: 0.5,
            seed: 11,
            ..SamplingParams::default()
        };
        let command = command(
            vec![
                sampling_entry(1, 6, drawn),
                entry(2, 0, vec![1, 2, 3], &[20], true),
                dummy(3, 30),
            ],
            1,
        );
        let layout = BatchLayout::lay_out(&command, BLOCK_SIZE).unwrap();
        assert_eq!(
            layout.selected,
            [2, 3],
            "the prefill leads, then the decode"
        );
        assert_eq!(
            layout.sampling,
            [
                RowSampling {
                    request: RequestId::new(2),
                    slot: RequestSlot::new(2),
                    params: SamplingParams::default(),
                },
                RowSampling {
                    request: RequestId::new(1),
                    slot: RequestSlot::new(6),
                    params: drawn,
                },
            ],
            "one per selected row, in batch order; the dummy has none"
        );
        assert_eq!(layout.sampling.len(), layout.selected.len());
    }

    #[test]
    fn a_chunked_prefill_continues_from_its_context_and_samples_only_when_final() {
        let command = command(
            vec![
                entry(1, 4, vec![5, 6, 7], &[10, 11], false),
                entry(2, 6, vec![8], &[20, 21], true),
            ],
            0,
        );
        let layout = BatchLayout::lay_out(&command, BLOCK_SIZE).unwrap();
        assert_eq!(layout.prefill_entries, 1, "three tokens is a prefill chunk");
        assert_eq!(layout.positions, [4, 5, 6, 6]);
        assert_eq!(layout.slot_mapping, [44, 45, 46, 86]);
        assert_eq!(layout.context_lengths, [4, 6]);
        assert_eq!(layout.sequence_lengths, [7, 7]);
        assert_eq!(
            layout.selected,
            [3],
            "the non-final chunk selects nothing; the final one-token chunk does"
        );
        assert_eq!(layout.sampling_rows(), [0]);
        assert_eq!(layout.max_prefill_sequence_len, 7);
        assert_eq!(layout.max_decode_sequence_len, 7);
    }

    #[test]
    fn an_all_decode_batch_has_no_prefill_side() {
        let command = command(
            vec![
                entry(1, 3, vec![9], &[10], true),
                entry(2, 8, vec![9], &[20, 21, 22], true),
                dummy(3, 30),
                dummy(4, 31),
            ],
            2,
        );
        let layout = BatchLayout::lay_out(&command, BLOCK_SIZE).unwrap();
        assert_eq!(layout.prefill_entries, 0);
        assert_eq!(layout.prefill_tokens, 0);
        assert_eq!(layout.decode_tokens, 4);
        assert_eq!(layout.max_query_len, 1);
        assert_eq!(layout.max_prefill_sequence_len, 0);
        assert_eq!(layout.max_decode_sequence_len, 9);
        assert_eq!(layout.query_start_locations, [0, 1, 2, 3, 4]);
        assert_eq!(layout.block_table_width, 3);
        assert_eq!(layout.selected, [0, 1]);
        assert_eq!(layout.sampling_rows(), [0, 1]);
    }

    #[test]
    fn the_layout_carries_the_commands_dispatch_decision_and_padding_count() {
        let command = mixed();
        let layout = BatchLayout::lay_out(&command, BLOCK_SIZE).unwrap();
        assert_eq!(layout.dispatch, command.dispatch);
        assert_eq!(layout.padding_count, 1);
    }

    #[test]
    fn a_command_that_breaks_the_protocol_is_refused_by_name() {
        assert_eq!(
            BatchLayout::lay_out(&command(Vec::new(), 0), BLOCK_SIZE).unwrap_err(),
            LayoutError::NoEntries
        );
        let empty_query = command(vec![entry(7, 3, Vec::new(), &[10], true)], 0);
        assert_eq!(
            BatchLayout::lay_out(&empty_query, BLOCK_SIZE).unwrap_err(),
            LayoutError::EmptyQuery {
                request: RequestId::new(7)
            }
        );
        let short_table = command(vec![entry(8, 4, vec![1], &[10], true)], 0);
        assert_eq!(
            BatchLayout::lay_out(&short_table, BLOCK_SIZE).unwrap_err(),
            LayoutError::BlockTableShort {
                request: RequestId::new(8),
                blocks: 1,
                sequence_len: 5,
            }
        );
    }
}
