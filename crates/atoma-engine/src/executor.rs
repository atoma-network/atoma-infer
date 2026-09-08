//! The executor loop: step commands off the ring, through the forward, and step results back,
//! until the engine is gone. One pinned thread per rank runs it: rank zero owns the engine's rings
//! and feeds every other rank, which follows.
//!
//! The loop parks with nothing to serve and is woken by the next command or by the engine's ends
//! dropping. Any error ends the loop: the rings drop with it, which the engine sees as the
//! executor being lost, and the cause is logged and returned from the thread.

mod fanout;
#[cfg(feature = "cuda")]
mod spawn;
mod thread;

use std::sync::Arc;

use atoma_core::engine::ExecutorRings;
use atoma_core::step::{StepCommand, StepResult};
use atoma_core::types::TokenCount;
use thiserror::Error;
use tracing::{debug, info};

pub use fanout::{feed, Follower, FollowerFeed, FollowerRings};
#[cfg(feature = "cuda")]
pub use spawn::{spawn_ranks, StartupError};
pub use thread::{launch, spawn, wait_all, Cause, ExecutorThread, Launched, SpawnError};

use crate::batch::{BatchLayout, LayoutError};
use crate::config::Rank;
use crate::forward::Forward;

/// Why an executor stopped.
#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error("the forward failed: {0}")]
    Forward(#[source] Cause),
    /// The forward returned tokens for other than the rows the layout selected: its bug.
    #[error("{expected} rows were selected but the forward sampled {got} tokens")]
    SampledMismatch { expected: usize, got: usize },
    /// The engine keeps one step in flight, so a full result ring means it broke the protocol.
    #[error("the result ring is full: more than one step is in flight")]
    ResultRingFull,
    /// A follower rank dropped its ends of its feed: it is over, so every rank is.
    #[error("follower rank {rank} is gone")]
    FollowerLost { rank: Rank },
    /// The executor thread panicked, with what the panic said.
    #[error("the executor thread panicked: {message}")]
    Panicked { message: String },
}

/// What one rank's thread runs until the ranks are done: the leader's loop or a follower's.
pub trait ExecutorLoop {
    /// Runs until the far side is gone.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError`] for the first step that could not be served; the loop is over
    /// once it does.
    fn run(self) -> Result<(), ExecutorError>;
}

/// The executor of the rank that owns the engine's rings: it serves every step command, feeds it
/// to every follower, and answers the engine with the step result.
pub struct Executor<F> {
    rings: ExecutorRings,
    forward: F,
    block_size: TokenCount,
    followers: Vec<FollowerFeed>,
}

impl<F: Forward> Executor<F> {
    /// An executor serving `rings` through `forward`, over `block_size`-token KV blocks, with
    /// no followers yet.
    pub fn new(rings: ExecutorRings, forward: F, block_size: TokenCount) -> Self {
        Self {
            rings,
            forward,
            block_size,
            followers: Vec::new(),
        }
    }

    /// Takes the leader's end of a follower's feed, opened with [`feed`] on this executor's
    /// unparker. The follower's ends exist before its thread does, so the feed is opened by
    /// whoever spawns both.
    pub fn follow(&mut self, feed: FollowerFeed) {
        self.followers.push(feed);
    }

    /// Feeds `command` to every follower, then serves it here.
    fn serve(&mut self, command: StepCommand) -> Result<(), ExecutorError> {
        let command = Arc::new(command);
        self.feed_followers(&command)?;
        let layout = BatchLayout::lay_out(&command, self.block_size)?;
        let by_row = self
            .forward
            .forward(&layout)
            .map_err(|cause| ExecutorError::Forward(Box::new(cause)))?;
        if by_row.len() != layout.selected.len() {
            return Err(ExecutorError::SampledMismatch {
                expected: layout.selected.len(),
                got: by_row.len(),
            });
        }
        let sampled: Vec<u32> = layout
            .sampling_rows()
            .iter()
            .map(|&row| by_row[row])
            .collect();
        debug!(
            step = command.step.get(),
            entries = command.entries.len(),
            sampled = sampled.len(),
            "step served"
        );
        self.rings
            .push_result(StepResult {
                step: command.step,
                sampled,
            })
            .map_err(|_| ExecutorError::ResultRingFull)
    }

    /// Feeds `command` to every follower, waiting on a full feed until its follower takes a
    /// command. A follower seen to be gone stops the step before it starts: a forward missing a
    /// rank could not complete.
    fn feed_followers(&mut self, command: &Arc<StepCommand>) -> Result<(), ExecutorError> {
        for feed in &mut self.followers {
            let mut handed = Arc::clone(command);
            loop {
                if feed.follower_gone() {
                    return Err(ExecutorError::FollowerLost { rank: feed.rank() });
                }
                match feed.push(handed) {
                    Ok(()) => break,
                    Err(back) => {
                        handed = back;
                        self.rings.park();
                    }
                }
            }
        }
        Ok(())
    }

    /// The first follower that is gone, if any is.
    fn lost_follower(&self) -> Option<Rank> {
        self.followers
            .iter()
            .find(|feed| feed.follower_gone())
            .map(FollowerFeed::rank)
    }
}

impl<F: Forward> ExecutorLoop for Executor<F> {
    /// Serves step commands as they come, parking between them, until the engine is gone or a
    /// follower is.
    fn run(mut self) -> Result<(), ExecutorError> {
        info!(followers = self.followers.len(), "executor running");
        loop {
            if let Some(rank) = self.lost_follower() {
                return Err(ExecutorError::FollowerLost { rank });
            }
            if let Some(command) = self.rings.pop_command() {
                self.serve(command)?;
            } else if self.rings.engine_gone() {
                info!("engine gone; executor returning");
                return Ok(());
            } else {
                self.rings.park();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use atoma_core::dispatch::{DispatchDecision, EagerReason};
    use atoma_core::engine::{
        Control, Engine, EngineHandle, EngineThread, ExecutorHandoff, ExecutorRings,
    };
    use atoma_core::request::{FinishReason, RequestEvent};

    use super::{feed, Executor, ExecutorError, ExecutorLoop, Follower};
    use crate::config::Rank;
    use crate::test_support::{
        contract, engine_config, submit, FakeForward, FakeForwardError, BLOCK_SIZE, WAIT,
    };

    fn engine() -> (EngineHandle, ExecutorHandoff, EngineThread) {
        Engine::spawn(&engine_config(), &contract()).unwrap()
    }

    fn executor(rings: ExecutorRings, forward: FakeForward) -> Executor<FakeForward> {
        Executor::new(rings, forward, BLOCK_SIZE)
    }

    #[test]
    fn a_request_flows_through_the_executor_to_its_finish() {
        let (handle, handoff, engine) = engine();
        let rings = handoff.rings;
        let forward = FakeForward::constant(5);
        let served = forward.served();
        let executor = thread::spawn(move || executor(rings, forward).run());
        let client = submit(&handle, 3, 2);

        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Token { token: 5, .. }
        ));
        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Token { token: 5, .. }
        ));
        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Finished {
                reason: FinishReason::MaxNewTokens,
                ..
            }
        ));
        let served = served.lock().clone();
        assert_eq!(served.len(), 2, "one prefill, one decode");
        assert_eq!(served[0].tokens, [1, 2, 3]);
        assert!(
            matches!(
                served[0].dispatch,
                DispatchDecision::Eager(EagerReason::NotUniformDecode { .. })
            ),
            "the prefill runs eagerly: {:?}",
            served[0].dispatch
        );
        assert_eq!(served[1].tokens, [5], "decoding what was sampled");
        assert_eq!(served[1].context_lengths, [3]);
        assert!(
            matches!(
                served[1].dispatch,
                DispatchDecision::FullReplay(key) if key.padded_token_count().get() == 1
            ),
            "the one-token decode is keyed to the bucket of one: {:?}",
            served[1].dispatch
        );
        assert_eq!(served[1].padding_count, 0);

        handle.control.try_send(Control::Shutdown).unwrap();
        engine.join();
        executor
            .join()
            .unwrap()
            .expect("the executor returns cleanly once the engine is gone");
    }

    #[test]
    fn a_failing_forward_ends_the_executor_with_the_cause_and_the_engine_fails_the_request() {
        let (handle, handoff, engine) = engine();
        let rings = handoff.rings;
        let forward = FakeForward::constant(5).failing_on_command(2);
        let executor = thread::spawn(move || executor(rings, forward).run());
        let client = submit(&handle, 3, 16);

        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Token { token: 5, .. }
        ));
        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Finished {
                reason: FinishReason::ExecutorLost,
                ..
            }
        ));
        let error = executor.join().unwrap().unwrap_err();
        assert!(
            matches!(&error, ExecutorError::Forward(cause) if cause.is::<FakeForwardError>()),
            "the forward's own error is the cause: {error}"
        );
        assert!(error.to_string().contains("command number 2"), "{error}");
        engine.join();
    }

    type RankThread = thread::JoinHandle<Result<(), ExecutorError>>;

    /// A leader with one follower, each on a thread of its own, over fakes that record what
    /// they ran.
    fn two_ranks(
        leader_forward: FakeForward,
        follower_forward: FakeForward,
    ) -> (EngineHandle, EngineThread, RankThread, RankThread) {
        let (handle, handoff, engine) = engine();
        let rings = handoff.rings;
        let (leader_end, follower_rings) = feed(Rank::new(1), rings.unparker());
        let mut leader = executor(rings, leader_forward);
        leader.follow(leader_end);
        let follower = Follower::new(follower_rings, follower_forward, BLOCK_SIZE);
        let follower_thread = thread::spawn(move || follower.run());
        let leader_thread = thread::spawn(move || leader.run());
        (handle, engine, leader_thread, follower_thread)
    }

    #[test]
    fn a_follower_runs_every_command_the_leader_serves_and_produces_nothing() {
        let leader_forward = FakeForward::constant(5);
        let follower_forward = FakeForward::constant(9);
        let (led, followed) = (leader_forward.served(), follower_forward.served());
        let (handle, engine, leader, follower) = two_ranks(leader_forward, follower_forward);
        let client = submit(&handle, 3, 2);

        for _ in 0..2 {
            assert!(
                matches!(
                    client.recv_timeout(WAIT).unwrap(),
                    RequestEvent::Token { token: 5, .. }
                ),
                "the leader's tokens are the step's, never the follower's"
            );
        }
        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Finished {
                reason: FinishReason::MaxNewTokens,
                ..
            }
        ));
        handle.control.try_send(Control::Shutdown).unwrap();
        engine.join();
        leader.join().unwrap().expect("the leader returns cleanly");
        follower
            .join()
            .unwrap()
            .expect("the follower returns cleanly once the leader is gone");
        let (led, followed) = (led.lock().clone(), followed.lock().clone());
        assert_eq!(led.len(), 2);
        assert_eq!(
            led, followed,
            "the follower ran the same steps in the same order"
        );
    }

    #[test]
    fn a_follower_dying_ends_the_leader_and_fails_the_live_request() {
        let (handle, engine, leader, follower) = two_ranks(
            FakeForward::constant(5),
            FakeForward::constant(5).failing_on_command(2),
        );
        let client = submit(&handle, 3, 16);
        // The leader serves the step the follower died in before it can see the loss, and may
        // serve one more before the follower's thread has dropped its ends; the request ends as
        // lost either way.
        let mut tokens = 0;
        let reason = loop {
            match client.recv_timeout(WAIT).unwrap() {
                RequestEvent::Token { token: 5, .. } => tokens += 1,
                RequestEvent::Token { token, .. } => panic!("sampled {token}, not the leader's 5"),
                RequestEvent::Finished { reason, .. } => break reason,
            }
        };
        assert_eq!(reason, FinishReason::ExecutorLost);
        assert!(tokens >= 1, "the first step, at least, was served");
        let cause = follower.join().unwrap().unwrap_err();
        assert!(matches!(cause, ExecutorError::Forward(_)), "{cause}");
        let error = leader.join().unwrap().unwrap_err();
        assert!(
            matches!(error, ExecutorError::FollowerLost { rank } if rank == Rank::new(1)),
            "{error}"
        );
        engine.join();
    }

    #[test]
    fn the_leader_dying_ends_a_parked_follower() {
        let (handle, engine, leader, follower) = two_ranks(
            FakeForward::constant(5).failing_on_command(1),
            FakeForward::constant(5),
        );
        let client = submit(&handle, 3, 16);
        let event = client.recv_timeout(WAIT).unwrap();
        assert!(
            matches!(
                event,
                RequestEvent::Finished {
                    reason: FinishReason::ExecutorLost,
                    ..
                }
            ),
            "{event:?}"
        );
        assert!(matches!(
            leader.join().unwrap().unwrap_err(),
            ExecutorError::Forward(_)
        ));
        follower
            .join()
            .unwrap()
            .expect("woken by the leader going, and returns cleanly");
        engine.join();
    }

    #[test]
    fn the_executor_returns_once_the_engine_is_gone_with_nothing_to_serve() {
        let (handle, handoff, engine) = engine();
        let rings = handoff.rings;
        let executor = thread::spawn(move || executor(rings, FakeForward::constant(1)).run());
        thread::sleep(Duration::from_millis(10));
        assert!(!executor.is_finished(), "parked, waiting for a command");
        handle.control.try_send(Control::Shutdown).unwrap();
        engine.join();
        assert!(executor.join().unwrap().is_ok());
    }
}
