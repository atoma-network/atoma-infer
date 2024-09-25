use std::{
    collections::VecDeque,
    fmt::Debug,
    time::{Duration, Instant},
};

use crate::sequence::SequenceGroup;

/// A trait for defining scheduling policies for sequence groups.
///
/// Implementors of this trait determine the priority of sequence groups
/// for processing in a scheduler.
pub trait Policy: Debug {
    /// Calculates the priority of a sequence group at a given time.
    ///
    /// # Arguments
    ///
    /// * `now` - The current time.
    /// * `sequence_group` - The sequence group to evaluate.
    ///
    /// # Returns
    ///
    /// A `Duration` representing the priority. Larger durations indicate higher priority.
    fn get_priority(now: Instant, sequence_group: &SequenceGroup) -> Duration;

    /// Sorts a collection of sequence groups by their priority.
    ///
    /// # Arguments
    ///
    /// * `now` - The current time.
    /// * `sequence_groups` - A queue of sequence groups to sort.
    ///
    /// # Returns
    ///
    /// A new `VecDeque` of sequence groups sorted by descending priority.
    fn sort_by_priority(
        now: Instant,
        sequence_groups: &VecDeque<SequenceGroup>,
    ) -> VecDeque<SequenceGroup> {
        let mut output: Vec<SequenceGroup> = sequence_groups.iter().cloned().collect::<Vec<_>>();
        output.sort_by(|v1, v2| {
            Self::get_priority(now, v2)
                .partial_cmp(&Self::get_priority(now, v1))
                .unwrap() // DON'T PANIC: `Duration` admits a complete ordering
        });
        output.into()
    }
}

/// First-Come, First-Served (FCFS) scheduling policy.
///
/// This policy prioritizes sequence groups based on their arrival time,
/// giving higher priority to those that arrived earlier.
#[derive(Debug)]
pub struct FcfsPolicy {}

impl Policy for FcfsPolicy {
    fn get_priority(now: Instant, sequence_group: &SequenceGroup) -> Duration {
        now - sequence_group.arrival_time()
    }
}
