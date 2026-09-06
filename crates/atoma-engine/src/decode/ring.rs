//! The staging ring: `depth` staging entries, each guarded by a fence, handed out in turn once
//! the copy that last read the staging entry has finished.
//!
//! An upload reads a staging entry asynchronously and signals the staging entry's fence behind
//! the copy, so the host must not write the staging entry again until the fence is passed. The
//! staging ring keeps the fences and a cursor at the staging entry handed out next.
//! [`StagingRing::acquire`] waits on that staging entry's fence, blocking, hands the staging entry
//! out and moves the cursor on; [`StagingRing::try_acquire`] asks the fence without blocking and
//! leaves the cursor where it is while the copy is still in flight. Whoever owns the staging
//! memory keeps each staging entry's memory indexed by the staging entry, reaches the fence the
//! upload signals through [`StagingRing::fence`], and waits on every fence through
//! [`StagingRing::wait_all`] before letting the memory go.
//!
//! A fence nobody has signaled is passed, and so is one whose copy has finished, so an acquire
//! that is not overtaking a copy returns at once: it costs one wait on a passed fence, and blocks
//! only when the host has run ahead of the device by the whole depth.
//!
//! [`EntryFence`] is what the staging ring asks of a fence, so the protocol runs on a host with
//! no GPU over a fake; [`StagingFence`] is the fence in serving.

use std::fmt;
use std::iter;
use std::num::NonZeroUsize;

use atoma_runtime::error::RuntimeError;
use atoma_runtime::fence::StagingFence;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// How many staging entries the staging ring holds, between one and [`StagingDepth::MAX`] by
/// construction: two unless configured, so the host writes one staging entry while the device is
/// still reading the other. Read from configuration as a plain integer, through
/// [`StagingDepth::new`], so a depth out of range refuses the configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(into = "usize", try_from = "usize")]
pub struct StagingDepth(NonZeroUsize);

impl StagingDepth {
    /// Eight, the deepest staging ring. Each staging entry pins a block of the largest bucket's
    /// packed size, and the staging ring exists so the host can run one step or a few ahead of
    /// the device, so a deeper staging ring pins memory nobody reaches: a depth above eight is a
    /// typo, refused as zero is.
    pub const MAX: Self = Self(NonZeroUsize::new(8).unwrap());

    /// The depth `depth` names, or `None` at zero or above [`StagingDepth::MAX`]: a staging ring
    /// with no staging entry stages nothing, and a deeper one pins memory nobody reaches.
    #[must_use]
    pub const fn new(depth: usize) -> Option<Self> {
        if depth > Self::MAX.get() {
            return None;
        }
        match NonZeroUsize::new(depth) {
            Some(depth) => Some(Self(depth)),
            None => None,
        }
    }

    #[must_use]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

impl Default for StagingDepth {
    /// Two: the host writes one staging entry while the device is still reading the other.
    fn default() -> Self {
        Self::new(2).expect("two is nonzero")
    }
}

impl fmt::Display for StagingDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl TryFrom<usize> for StagingDepth {
    type Error = StagingDepthError;

    fn try_from(depth: usize) -> Result<Self, Self::Error> {
        Self::new(depth).ok_or(StagingDepthError { depth })
    }
}

impl From<StagingDepth> for usize {
    fn from(depth: StagingDepth) -> Self {
        depth.get()
    }
}

/// A staging depth no staging ring can be built at: zero, or above [`StagingDepth::MAX`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error(
    "staging depth {depth} is out of range; a staging ring holds between 1 and {max} staging \
     entries",
    max = StagingDepth::MAX
)]
pub struct StagingDepthError {
    /// The depth that was asked for.
    pub depth: usize,
}

/// What the staging ring asks of the fence guarding one staging entry: a blocking wait and a
/// non-blocking one, each passed once the copy behind the fence's last signal has finished.
pub trait EntryFence {
    type Error;

    /// Waits until the fence is passed, blocking the calling thread.
    ///
    /// # Errors
    ///
    /// Returns the fence's error when the wait fails.
    fn wait(&self) -> Result<(), Self::Error>;

    /// Whether the fence is passed, without waiting.
    ///
    /// # Errors
    ///
    /// Returns the fence's error when the query fails for any reason but not ready.
    fn try_wait(&self) -> Result<bool, Self::Error>;
}

impl EntryFence for StagingFence {
    type Error = RuntimeError;

    fn wait(&self) -> Result<(), RuntimeError> {
        StagingFence::wait(self)
    }

    fn try_wait(&self) -> Result<bool, RuntimeError> {
        StagingFence::try_wait(self)
    }
}

/// One staging entry the staging ring has handed out: which staging memory the host may write,
/// and which fence the copy that reads it signals. Minted by the staging ring alone, and neither
/// `Copy` nor `Clone`: one acquire hands out one staging entry, and whoever uploads through it
/// takes the only one.
#[derive(Debug, PartialEq, Eq)]
pub struct StagingEntry {
    index: usize,
}

impl StagingEntry {
    /// The staging entry's position in the staging ring: the index of its staging memory and of
    /// its fence.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }
}

/// The staging ring: one fence per staging entry, and a cursor at the staging entry handed out
/// next.
#[derive(Debug)]
pub struct StagingRing<F> {
    fences: Vec<F>,
    cursor: usize,
}

impl<F: EntryFence> StagingRing<F> {
    /// A staging ring of `depth` staging entries, each guarded by a fence `fence` creates, with
    /// the cursor at the first.
    ///
    /// # Errors
    ///
    /// Returns the first error `fence` returns.
    pub fn new(
        depth: StagingDepth,
        fence: impl FnMut() -> Result<F, F::Error>,
    ) -> Result<Self, F::Error> {
        let fences = iter::repeat_with(fence)
            .take(depth.get())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { fences, cursor: 0 })
    }

    /// The staging entry at the cursor, once the copy that last read it has finished: waits on
    /// its fence, blocking, then moves the cursor on. A fence nobody has signaled is passed, so
    /// the wait returns at once.
    ///
    /// # Errors
    ///
    /// Returns the fence's error when the wait fails; the cursor stays where it was.
    pub fn acquire(&mut self) -> Result<StagingEntry, F::Error> {
        self.fences[self.cursor].wait()?;
        Ok(self.take())
    }

    /// The staging entry at the cursor if the copy that last read it has finished, without
    /// waiting: `None` leaves the cursor where it is, so the next call asks about the same
    /// staging entry.
    ///
    /// # Errors
    ///
    /// Returns the fence's error when the query fails; the cursor stays where it was.
    pub fn try_acquire(&mut self) -> Result<Option<StagingEntry>, F::Error> {
        if !self.fences[self.cursor].try_wait()? {
            return Ok(None);
        }
        Ok(Some(self.take()))
    }

    /// The fence guarding `entry`: what the copy that reads the staging entry signals once it is
    /// done.
    #[must_use]
    pub fn fence(&self, entry: &StagingEntry) -> &F {
        &self.fences[entry.index]
    }

    /// Waits on every staging entry's fence in turn, blocking, so no copy reads any staging
    /// entry once this returns: what the owner of the staging memory calls before freeing it.
    ///
    /// # Errors
    ///
    /// Returns the first fence's error; the fences after it are not waited on.
    pub fn wait_all(&self) -> Result<(), F::Error> {
        self.fences.iter().try_for_each(EntryFence::wait)
    }

    /// Hands out the staging entry at the cursor and moves the cursor to the next, wrapping.
    fn take(&mut self) -> StagingEntry {
        let entry = StagingEntry { index: self.cursor };
        // The cursor is below the depth, so the increment cannot overflow.
        self.cursor = (self.cursor + 1) % self.fences.len();
        entry
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use thiserror::Error;

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
    #[error("the fake fence was told to fail")]
    struct FakeFenceError;

    /// A fence the test steers by hand and shares with the staging ring through a clone: whether
    /// the copy it guards is still in flight, whether it has been told to fail, and how many
    /// blocking waits it has served.
    #[derive(Debug, Clone, Default)]
    struct FakeFence {
        in_flight: Rc<Cell<bool>>,
        failing: Rc<Cell<bool>>,
        waits: Rc<Cell<usize>>,
    }

    impl FakeFence {
        fn waits(&self) -> usize {
            self.waits.get()
        }

        fn copy_in_flight(&self) {
            self.in_flight.set(true);
        }

        fn copy_done(&self) {
            self.in_flight.set(false);
        }

        fn fail(&self) {
            self.failing.set(true);
        }

        fn recover(&self) {
            self.failing.set(false);
        }

        /// Whether `other` is this fence, shared through a clone.
        fn is(&self, other: &FakeFence) -> bool {
            Rc::ptr_eq(&self.waits, &other.waits)
        }
    }

    impl EntryFence for FakeFence {
        type Error = FakeFenceError;

        fn wait(&self) -> Result<(), FakeFenceError> {
            if self.failing.get() {
                return Err(FakeFenceError);
            }
            // A blocking wait outlasts the copy: the fence is passed once the wait returns.
            self.waits.set(self.waits.get() + 1);
            self.in_flight.set(false);
            Ok(())
        }

        fn try_wait(&self) -> Result<bool, FakeFenceError> {
            if self.failing.get() {
                return Err(FakeFenceError);
            }
            Ok(!self.in_flight.get())
        }
    }

    /// A staging ring of `depth` fake fences, with the fences handed back in staging-entry order.
    fn staging_ring(depth: usize) -> (StagingRing<FakeFence>, Vec<FakeFence>) {
        let fences: Vec<FakeFence> = (0..depth).map(|_| FakeFence::default()).collect();
        let mut handed = fences.iter().cloned();
        let ring = StagingRing::new(StagingDepth::new(depth).unwrap(), || {
            Ok(handed
                .next()
                .expect("the staging ring asks for exactly depth fences"))
        })
        .unwrap();
        (ring, fences)
    }

    fn waits(fences: &[FakeFence]) -> Vec<usize> {
        fences.iter().map(FakeFence::waits).collect()
    }

    #[test]
    fn the_staging_depth_is_two_unless_configured_and_never_zero() {
        assert_eq!(StagingDepth::default().get(), 2);
        assert_eq!(StagingDepth::new(0), None);
        assert_eq!(StagingDepth::new(3).unwrap().get(), 3);
        assert_eq!(StagingDepth::default().to_string(), "2");
    }

    #[test]
    fn the_staging_depth_refuses_anything_above_its_maximum() {
        let deepest = StagingDepth::MAX.get();
        assert_eq!(StagingDepth::new(deepest), Some(StagingDepth::MAX));
        assert_eq!(StagingDepth::new(deepest + 1), None, "one past the deepest");
        assert_eq!(StagingDepth::new(usize::MAX), None);
    }

    #[test]
    fn a_plain_integer_becomes_a_staging_depth_only_inside_the_bounds() {
        assert_eq!(StagingDepth::try_from(1).map(usize::from), Ok(1));
        assert_eq!(StagingDepth::try_from(8).map(usize::from), Ok(8));
        assert_eq!(
            StagingDepth::try_from(0),
            Err(StagingDepthError { depth: 0 })
        );
        assert_eq!(
            StagingDepth::try_from(9),
            Err(StagingDepthError { depth: 9 })
        );
        assert_eq!(
            StagingDepthError { depth: 9 }.to_string(),
            "staging depth 9 is out of range; a staging ring holds between 1 and 8 staging entries"
        );
    }

    #[test]
    fn a_depth_two_staging_ring_alternates_and_waits_on_the_staging_entry_it_hands_out() {
        let (mut ring, fences) = staging_ring(2);

        assert_eq!(ring.acquire().unwrap().index(), 0);
        assert_eq!(waits(&fences), [1, 0]);
        assert_eq!(ring.acquire().unwrap().index(), 1);
        assert_eq!(waits(&fences), [1, 1]);
        assert_eq!(ring.acquire().unwrap().index(), 0);
        assert_eq!(waits(&fences), [2, 1]);
        assert_eq!(ring.acquire().unwrap().index(), 1);
        assert_eq!(waits(&fences), [2, 2]);
    }

    #[test]
    fn try_acquire_holds_the_cursor_while_the_staging_entrys_copy_is_in_flight_and_never_blocks() {
        let (mut ring, fences) = staging_ring(2);
        fences[0].copy_in_flight();

        assert_eq!(ring.try_acquire().unwrap(), None);
        assert_eq!(
            ring.try_acquire().unwrap(),
            None,
            "the same staging entry, still in flight"
        );
        assert_eq!(waits(&fences), [0, 0], "nothing blocked");

        fences[0].copy_done();
        let acquired = ring.try_acquire().unwrap().map(|entry| entry.index());
        assert_eq!(
            acquired,
            Some(0),
            "the cursor held on the refused staging entry"
        );
        let acquired = ring.try_acquire().unwrap().map(|entry| entry.index());
        assert_eq!(acquired, Some(1), "and moved on once it was handed out");
        assert_eq!(waits(&fences), [0, 0], "handed out without a blocking wait");

        fences[0].copy_in_flight();
        assert_eq!(ring.try_acquire().unwrap(), None);
        assert_eq!(
            ring.acquire().unwrap().index(),
            0,
            "a blocking acquire takes the same staging entry"
        );
        assert_eq!(waits(&fences), [1, 0]);
    }

    #[test]
    fn an_acquired_staging_entry_names_the_fence_the_copy_reading_it_signals() {
        let (mut ring, fences) = staging_ring(2);
        for expected in [0, 1, 0] {
            let entry = ring.acquire().unwrap();
            assert_eq!(entry.index(), expected);
            assert!(
                ring.fence(&entry).is(&fences[expected]),
                "the fence of staging entry {expected}"
            );
        }
    }

    #[test]
    fn a_depth_one_staging_ring_waits_on_its_one_fence_before_every_reuse() {
        let (mut ring, fences) = staging_ring(1);
        for reuse in 1..=3 {
            fences[0].copy_in_flight();
            assert_eq!(ring.acquire().unwrap().index(), 0);
            assert_eq!(fences[0].waits(), reuse, "acquire number {reuse} waited");
        }
    }

    #[test]
    fn wait_all_waits_on_every_staging_entrys_fence_whatever_the_cursor() {
        let (mut ring, fences) = staging_ring(3);
        ring.acquire().unwrap();
        for fence in &fences {
            fence.copy_in_flight();
        }

        ring.wait_all().unwrap();

        assert_eq!(waits(&fences), [2, 1, 1], "every fence waited on once more");
        assert!(
            fences.iter().all(|fence| fence.try_wait() == Ok(true)),
            "no copy is in flight after the wait"
        );
    }

    #[test]
    fn wait_all_stops_at_the_first_fence_that_cannot_be_waited_on() {
        let (ring, fences) = staging_ring(3);
        fences[1].fail();

        assert_eq!(ring.wait_all().unwrap_err(), FakeFenceError);
        assert_eq!(
            waits(&fences),
            [1, 0, 0],
            "the fences after the failure are not waited on"
        );
    }

    #[test]
    fn a_fence_that_cannot_be_waited_on_fails_the_acquire_and_holds_the_cursor() {
        let (mut ring, fences) = staging_ring(2);
        fences[0].fail();

        assert_eq!(ring.acquire().unwrap_err(), FakeFenceError);
        assert_eq!(ring.try_acquire().unwrap_err(), FakeFenceError);

        fences[0].recover();
        assert_eq!(
            ring.acquire().unwrap().index(),
            0,
            "the failed acquires handed nothing out"
        );
    }

    #[test]
    fn a_fence_that_cannot_be_created_fails_the_staging_ring_at_the_first() {
        let mut created = 0;
        let ring = StagingRing::<FakeFence>::new(StagingDepth::default(), || {
            created += 1;
            Err(FakeFenceError)
        });
        assert_eq!(ring.unwrap_err(), FakeFenceError);
        assert_eq!(created, 1, "the first failure ends the build");
    }
}
