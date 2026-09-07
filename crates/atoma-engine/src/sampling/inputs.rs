//! What the sampler puts on the device before a step, decided on the host from the batch layout
//! and the mirror of what each request slot holds: the records to write, because their slots
//! changed hands; the slot each selected row samples under; and the rows that take their token
//! from the device rather than from the host's copy-in.
//!
//! Which rows those are is the caller's to state, never this module's to derive: only a batch
//! every entry of which computes one token has a token row per entry, and that is what the
//! dispatch decision already settled on the engine thread.

use atoma_core::types::{RequestId, RequestSlot};
use thiserror::Error;

use crate::batch::BatchLayout;
use crate::sampling::owners::{Claim, OwnersError, SlotOwners};
use crate::sampling::record::SlotRecord;

/// A step whose sampler inputs cannot be decided. Each is the engine breaking the step protocol,
/// not a runtime state: two rows in one request slot would race on that slot's record, and a
/// gather past the batch would read token rows the step does not hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SamplerInputsError {
    #[error(
        "requests {} and {} both sample in slot {} this step",
        request.get(),
        other.get(),
        slot.get()
    )]
    SlotTwice {
        slot: RequestSlot,
        request: RequestId,
        other: RequestId,
    },
    #[error("the gather covers {rows} token rows but the batch holds {tokens}")]
    GatherPastBatch { rows: usize, tokens: usize },
    #[error(transparent)]
    Owners(#[from] OwnersError),
}

/// One step's sampler inputs, in the order the layout holds its rows.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerInputs {
    /// The slots whose record this step writes, each with the record: the slots that changed
    /// hands since the last step.
    pub records: Vec<(RequestSlot, SlotRecord)>,
    /// The slot each selected row samples under, in batch order.
    pub row_slots: Vec<RequestSlot>,
    /// One entry per token row the caller said the gather covers: the slot whose last sampled
    /// token is the row's input, so the row takes it from the device; none for a row the host's
    /// copy-in serves, which is one whose request no step has sampled for yet. Empty when the
    /// caller covers none.
    pub gather: Vec<Option<RequestSlot>>,
}

impl SamplerInputs {
    /// Decides `layout`'s step against `owners`, claiming every selected row's slot for its
    /// request and marking each as sampled for. `gather_rows` is how many leading token rows the
    /// gather covers, which only a caller holding a batch of one token per entry may state; a
    /// caller that states none copies in every token.
    ///
    /// # Errors
    ///
    /// Returns [`SamplerInputsError`] when a row names a slot past the mirror, two rows name one
    /// slot, or more rows are covered than the batch holds.
    pub fn for_step(
        layout: &BatchLayout,
        owners: &mut SlotOwners,
        gather_rows: Option<usize>,
    ) -> Result<Self, SamplerInputsError> {
        let gather_rows = gather_rows.unwrap_or(0);
        if gather_rows > layout.token_count() {
            return Err(SamplerInputsError::GatherPastBatch {
                rows: gather_rows,
                tokens: layout.token_count(),
            });
        }
        let mut records = Vec::new();
        let mut row_slots: Vec<RequestSlot> = Vec::with_capacity(layout.sampling.len());
        let mut gather = vec![None; gather_rows];
        for (sampling, &token_row) in layout.sampling.iter().zip(&layout.selected) {
            if let Some(row) = row_slots.iter().position(|held| *held == sampling.slot) {
                return Err(SamplerInputsError::SlotTwice {
                    slot: sampling.slot,
                    request: layout.sampling[row].request,
                    other: sampling.request,
                });
            }
            if owners.claim(sampling.slot, sampling.request)? == Claim::Taken {
                records.push((sampling.slot, SlotRecord::new(&sampling.params)));
            }
            // The caller states that every entry computes one token, so a selected row's token
            // row is its entry's row; it gathers on what the steps before this one sampled.
            let token_row = token_row as usize;
            if token_row < gather_rows && owners.gathers(sampling.slot, sampling.request) {
                gather[token_row] = Some(sampling.slot);
            }
            owners.samples(sampling.slot)?;
            row_slots.push(sampling.slot);
        }
        Ok(Self {
            records,
            row_slots,
            gather,
        })
    }
}

#[cfg(test)]
mod tests {
    use atoma_core::request::SamplingParams;
    use atoma_core::types::{RequestId, RequestSlot};

    use super::{SamplerInputs, SamplerInputsError};
    use crate::batch::BatchLayout;
    use crate::sampling::owners::{OwnersError, SlotOwners};
    use crate::sampling::record::SlotRecord;
    use crate::test_support::{command, dummy, entry, sampling_entry, BLOCK_SIZE};

    fn slot(slot: u32) -> RequestSlot {
        RequestSlot::new(slot)
    }

    fn drawn(seed: u64) -> SamplingParams {
        SamplingParams {
            do_sample: true,
            temperature: 0.7,
            seed,
            ..SamplingParams::default()
        }
    }

    /// Two decodes and a dummy: request 1 in slot 1 decoding token 5, request 2 in slot 2
    /// decoding token 8.
    fn decodes() -> BatchLayout {
        let command = command(
            vec![
                sampling_entry(1, 1, drawn(3)),
                entry(2, 4, vec![8], &[20, 21], true),
                dummy(3, 30),
            ],
            1,
        );
        let mut command = command;
        command.entries[0].input_tokens = vec![5];
        BatchLayout::lay_out(&command, BLOCK_SIZE).unwrap()
    }

    #[test]
    fn a_first_step_writes_every_rows_record_and_gathers_nothing() {
        let mut owners = SlotOwners::new(8);
        let inputs = SamplerInputs::for_step(&decodes(), &mut owners, Some(3)).unwrap();
        assert_eq!(
            inputs.records,
            [
                (slot(1), SlotRecord::new(&drawn(3))),
                (slot(2), SlotRecord::new(&SamplingParams::default())),
            ]
        );
        assert_eq!(inputs.row_slots, [slot(1), slot(2)]);
        assert_eq!(
            inputs.gather,
            [None, None, None],
            "one entry per token row, none of them sampled for yet"
        );
    }

    #[test]
    fn a_row_gathers_once_a_step_has_sampled_for_its_slot_whatever_the_host_copies_in() {
        let mut owners = SlotOwners::new(8);
        SamplerInputs::for_step(&decodes(), &mut owners, Some(3)).unwrap();
        let mut layout = decodes();
        layout.tokens[0] = 77;
        let inputs = SamplerInputs::for_step(&layout, &mut owners, Some(3)).unwrap();
        assert!(
            inputs.records.is_empty(),
            "the slots still hold their requests"
        );
        assert_eq!(
            inputs.gather,
            [Some(slot(1)), Some(slot(2)), None],
            "both requests' slots sampled last step; the dummy never"
        );
    }

    #[test]
    fn a_slot_changing_hands_rewrites_its_record_and_gathers_nothing() {
        let mut owners = SlotOwners::new(8);
        SamplerInputs::for_step(&decodes(), &mut owners, Some(3)).unwrap();
        let mut layout = decodes();
        layout.sampling[0].request = RequestId::new(9);
        let inputs = SamplerInputs::for_step(&layout, &mut owners, Some(3)).unwrap();
        assert_eq!(inputs.records, [(slot(1), SlotRecord::new(&drawn(3)))]);
        assert_eq!(
            inputs.gather,
            [None, Some(slot(2)), None],
            "slot 1 changed hands; slot 2 still holds the request it sampled for"
        );
    }

    #[test]
    fn a_caller_that_covers_no_token_rows_gathers_nothing_and_still_claims_its_slots() {
        let mut owners = SlotOwners::new(8);
        let prefill = command(
            vec![
                entry(1, 0, vec![1, 2, 3], &[10], true),
                entry(2, 4, vec![8], &[20, 21], true),
            ],
            0,
        );
        let layout = BatchLayout::lay_out(&prefill, BLOCK_SIZE).unwrap();
        let inputs = SamplerInputs::for_step(&layout, &mut owners, None).unwrap();
        assert_eq!(
            inputs.row_slots,
            [slot(1), slot(2)],
            "batch order: the prefill leads"
        );
        assert_eq!(inputs.records.len(), 2);
        assert!(inputs.gather.is_empty());
    }

    #[test]
    fn two_rows_in_one_slot_and_a_gather_past_the_batch_are_refused_by_name() {
        let mut owners = SlotOwners::new(8);
        let mut layout = decodes();
        layout.sampling[1].slot = layout.sampling[0].slot;
        assert_eq!(
            SamplerInputs::for_step(&layout, &mut owners, None).unwrap_err(),
            SamplerInputsError::SlotTwice {
                slot: slot(1),
                request: RequestId::new(1),
                other: RequestId::new(2),
            }
        );

        assert_eq!(
            SamplerInputs::for_step(&decodes(), &mut SlotOwners::new(8), Some(4)).unwrap_err(),
            SamplerInputsError::GatherPastBatch { rows: 4, tokens: 3 },
            "the batch holds three token rows"
        );
    }

    #[test]
    fn a_slot_past_the_mirror_is_refused() {
        let mut owners = SlotOwners::new(2);
        assert_eq!(
            SamplerInputs::for_step(&decodes(), &mut owners, None).unwrap_err(),
            SamplerInputsError::Owners(OwnersError::SlotOutOfRange {
                slot: slot(2),
                slots: 2
            })
        );
    }
}
