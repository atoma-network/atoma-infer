//! The engine thread: one thread owning all engine state, driving the scheduler and the executor.
//!
//! Every pass drains control first, applies the executor's result, drains ingress while the
//! slab has room, schedules and issues the next step command, answers a drain once every request
//! that has run is over, and publishes the heartbeat. All state is owned outright by the thread
//! that runs the pass; no lock sits on the step path.

use std::io::ErrorKind;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_utils::sync::Parker;
use flume::{SendError, Sender};
use thiserror::Error;
use tracing::{debug, error, info};

use crate::attention::CaptureContract;
use crate::dispatch::{Dispatcher, PaddingLookup};
use crate::engine::{
    build_command, control, heartbeat, ingress, rings, Control, ControlReceiver, ControlSender,
    EngineConfig, EngineRings, EngineState, ExecutorRings, HeartbeatPublisher, HeartbeatReader,
    IngressReceiver, IngressSender,
};
use crate::kv::{BlockPool, PaddingError, PaddingReservation};
use crate::request::FinishReason;
use crate::scheduler::{Scheduled, Scheduler, SchedulerError};
use crate::step::StepResult;
use crate::types::{RequestCount, StepId, TokenCount};

/// A configuration the engine refuses to start under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum EngineError {
    #[error(transparent)]
    Scheduler(#[from] SchedulerError),
    #[error(transparent)]
    Padding(#[from] PaddingError),
    /// The maximum batch pads to a bucket above itself, whose rows outnumber the one dummy per
    /// entry the reservation holds.
    #[error(
        "scheduler.max_batch of {max_batch} pads to the bucket of {bucket} in \
         dispatch.bucket_ladder, whose rows outnumber the one dummy per entry the reservation \
         holds; add a bucket of {max_batch} to dispatch.bucket_ladder, or set \
         scheduler.max_batch to a bucket dispatch.bucket_ladder holds"
    )]
    PaddingCannotCoverBucket {
        max_batch: RequestCount,
        bucket: TokenCount,
    },
    /// The operating system refused the thread.
    #[error("the engine thread could not be spawned: {0:?}")]
    ThreadSpawn(ErrorKind),
}

/// What the engine's clients hold: ingress, control and the heartbeat.
#[derive(Debug, Clone)]
pub struct EngineHandle {
    pub ingress: IngressSender,
    pub control: ControlSender,
    pub heartbeat: HeartbeatReader,
}

/// The running engine thread. It returns on shutdown or when the executor is gone, after every
/// live request has been told.
#[derive(Debug)]
pub struct EngineThread {
    join: JoinHandle<()>,
}

impl EngineThread {
    /// Waits for the thread to return.
    ///
    /// # Panics
    ///
    /// Panics when the engine thread panicked, carrying its panic on.
    pub fn join(self) {
        self.join.join().expect("the engine thread panicked");
    }

    /// Whether the thread has returned.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.join.is_finished()
    }
}

/// Whether the loop goes on after a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// Park, then pass again.
    Continue,
    /// The thread returns: shut down, the executor gone, or a step out past its deadline.
    Exit,
}

/// The step out with the executor: what was scheduled, and when its command was pushed.
#[derive(Debug)]
struct InFlight {
    scheduled: Scheduled,
    issued: Instant,
}

/// The engine: the scheduler, dispatcher, rings, channels and heartbeat, owned by one thread.
#[derive(Debug)]
pub struct Engine {
    scheduler: Scheduler,
    dispatcher: Dispatcher,
    rings: EngineRings,
    ingress: IngressReceiver,
    control: ControlReceiver,
    heartbeat: HeartbeatPublisher,
    parker: Parker,
    idle_deadline: Duration,
    step_deadline: Duration,
    in_flight: Option<InFlight>,
    /// The drain waiting for every live request to be over, if one is.
    draining: Option<Sender<EngineState>>,
    passes: u64,
}

impl Engine {
    /// Builds the engine, its clients' handle and the executor's ends of the rings.
    ///
    /// `contract` is what the active backends and the model settled before anything was
    /// captured: the level every captured routine is valid at, and the sites the pass leaves the
    /// graph. The dispatcher is built from it, so nothing at runtime can raise either.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] when the pool cannot cover the padding reservation, what it
    /// leaves cannot hold one maximum-length request, or the bucket ladder pads the maximum
    /// batch above itself.
    pub fn new(
        config: &EngineConfig,
        contract: &CaptureContract,
    ) -> Result<(Self, EngineHandle, ExecutorRings), EngineError> {
        let max_batch = config.scheduler.max_batch;
        let lookup = PaddingLookup::new(&config.dispatch.bucket_ladder);
        let max_batch_bucket =
            TokenCount::new(max_batch.get()).and_then(|tokens| lookup.bucket_for(tokens));
        if let Some(bucket) = max_batch_bucket {
            if bucket.get() > max_batch.get() {
                return Err(EngineError::PaddingCannotCoverBucket { max_batch, bucket });
            }
        }
        let mut pool = BlockPool::new(config.block_count);
        let reservation = PaddingReservation::reserve(&mut pool, max_batch)?;
        let scheduler = Scheduler::with_padding(config.scheduler.clone(), pool, reservation)?;

        let parker = Parker::new();
        let (ingress_sender, ingress_receiver) =
            ingress(config.ingress_capacity.get(), parker.unparker().clone());
        let (control_sender, control_receiver) = control(parker.unparker().clone());
        let (heartbeat_publisher, heartbeat_reader) = heartbeat();
        let (engine_rings, executor_rings) = rings(parker.unparker().clone());
        let engine = Self {
            scheduler,
            dispatcher: Dispatcher::new(&config.dispatch, contract),
            rings: engine_rings,
            ingress: ingress_receiver,
            control: control_receiver,
            heartbeat: heartbeat_publisher,
            parker,
            idle_deadline: config.idle_deadline,
            step_deadline: config.step_deadline,
            in_flight: None,
            draining: None,
            passes: 0,
        };
        let handle = EngineHandle {
            ingress: ingress_sender,
            control: control_sender,
            heartbeat: heartbeat_reader,
        };
        Ok((engine, handle, executor_rings))
    }

    /// Builds the engine and runs it on its own thread, named `atoma-engine`.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] for the same configurations [`Engine::new`] refuses, and when the
    /// thread cannot be spawned.
    pub fn spawn(
        config: &EngineConfig,
        contract: &CaptureContract,
    ) -> Result<(EngineHandle, ExecutorRings, EngineThread), EngineError> {
        let (engine, handle, executor_rings) = Self::new(config, contract)?;
        let join = thread::Builder::new()
            .name("atoma-engine".to_owned())
            .spawn(move || engine.run())
            .map_err(|error| EngineError::ThreadSpawn(error.kind()))?;
        Ok((handle, executor_rings, EngineThread { join }))
    }

    /// Passes until shutdown, the executor gone or a step out past its deadline, parking between
    /// passes until ingress, control, the executor or a deadline wakes the thread.
    pub fn run(mut self) {
        info!("engine thread running");
        while self.pass() == Pass::Continue {
            self.park();
        }
        info!(passes = self.passes, "engine thread returning");
    }

    /// One pass: control first, then the executor's result, then ingress — so a slot the result
    /// freed is refilled in the same pass — then the next step, the drain check and the
    /// heartbeat.
    pub fn pass(&mut self) -> Pass {
        if self.drain_control() == Pass::Exit {
            return Pass::Exit;
        }
        if self.rings.executor_gone() {
            error!("executor gone; failing every live request");
            self.fail_all(FinishReason::ExecutorLost);
            return Pass::Exit;
        }
        if let Some(result) = self.rings.pop_result() {
            if self.apply_result(&result) == Pass::Exit {
                return Pass::Exit;
            }
        } else if let Some(step) = self.overdue_step() {
            error!(
                step = step.get(),
                deadline = ?self.step_deadline,
                "no result within the step deadline; the executor is wedged, failing every live \
                 request"
            );
            self.fail_all(FinishReason::ExecutorLost);
            return Pass::Exit;
        }
        self.drain_ingress();
        if self.in_flight.is_none() {
            self.issue_step();
        }
        self.answer_drain();
        self.passes += 1;
        self.heartbeat.publish(self.passes);
        Pass::Continue
    }

    /// Parks until ingress, control or the executor wakes the thread, or a deadline passes —
    /// the idle deadline, or what is left of the step deadline while a step is out — whichever
    /// is first. A spurious wake costs one pass and nothing else.
    fn park(&self) {
        let timeout = self
            .in_flight
            .as_ref()
            .map_or(self.idle_deadline, |in_flight| {
                self.step_deadline
                    .saturating_sub(in_flight.issued.elapsed())
                    .min(self.idle_deadline)
            });
        self.parker.park_timeout(timeout);
    }

    /// The step in flight, once it has been out for the step deadline without a result.
    fn overdue_step(&self) -> Option<StepId> {
        let in_flight = self.in_flight.as_ref()?;
        (in_flight.issued.elapsed() >= self.step_deadline).then_some(in_flight.scheduled.step)
    }

    /// The engine's state as of now.
    #[must_use]
    pub fn state(&self) -> EngineState {
        EngineState {
            step: self.scheduler.step(),
            live_requests: self.scheduler.live_request_count(),
            waiting: self.scheduler.waiting().len(),
            running: self.scheduler.running().len(),
            preempted: self.scheduler.preempted().len(),
            step_in_flight: self.in_flight.is_some(),
            draining: !self.scheduler.is_admission_open(),
            free_blocks: self.scheduler.pool().free_count(),
            available_blocks: self.scheduler.pool().available(),
        }
    }

    #[cfg(test)]
    pub(crate) fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Every control message, before anything else.
    fn drain_control(&mut self) -> Pass {
        while let Some(control) = self.control.try_recv() {
            match control {
                Control::Drain { reply } => {
                    info!("draining: admission closed to waiting requests");
                    self.scheduler.close_admission();
                    self.draining = Some(reply);
                }
                Control::Shutdown => {
                    info!("shutting down; finishing every live request");
                    self.fail_all(FinishReason::Shutdown);
                    return Pass::Exit;
                }
                Control::State { reply } => send_reply(&reply, self.state()),
            }
        }
        Pass::Continue
    }

    /// Ingress, only while the slab has room: once it is full, ingress backs up and refuses.
    fn drain_ingress(&mut self) {
        while self.scheduler.has_room() {
            let Some(request) = self.ingress.try_recv() else {
                return;
            };
            self.scheduler.intake(request);
        }
    }

    /// Applies the result of the step in flight. A result that does not match the step is the
    /// executor breaking the protocol, which is unrecoverable.
    fn apply_result(&mut self, result: &StepResult) -> Pass {
        let Some(InFlight { scheduled, .. }) = self.in_flight.take() else {
            error!(step = result.step.get(), "result with no step in flight");
            self.fail_all(FinishReason::ExecutorLost);
            return Pass::Exit;
        };
        let expected = scheduled.sampling_entries().count();
        if result.step != scheduled.step || result.sampled.len() != expected {
            error!(
                step = scheduled.step.get(),
                result_step = result.step.get(),
                expected,
                got = result.sampled.len(),
                "step result does not match the step in flight"
            );
            self.fail_all(FinishReason::ExecutorLost);
            return Pass::Exit;
        }
        self.scheduler.apply(&scheduled, &result.sampled);
        Pass::Continue
    }

    /// Schedules and, when the pass is not empty, issues its command.
    fn issue_step(&mut self) {
        let scheduled = self.scheduler.schedule();
        if scheduled.is_empty() {
            return;
        }
        let command = build_command(&self.scheduler, &scheduled, &mut self.dispatcher);
        debug!(
            step = command.step.get(),
            entries = command.entries.len(),
            padding = command.padding_count,
            "step issued"
        );
        if self.rings.push_command(command).is_err() {
            unreachable!("one step in flight always leaves a slot in a ring of two");
        }
        self.in_flight = Some(InFlight {
            scheduled,
            issued: Instant::now(),
        });
    }

    /// Reports the drain once every live request is over: nothing running, nothing preempted
    /// waiting to re-enter, and no step in flight.
    fn answer_drain(&mut self) {
        let live = !self.scheduler.running().is_empty() || !self.scheduler.preempted().is_empty();
        if self.in_flight.is_some() || live {
            return;
        }
        if let Some(reply) = self.draining.take() {
            info!("drained: nothing running, nothing preempted, nothing in flight");
            send_reply(&reply, self.state());
        }
    }

    /// Finishes every live request for `reason`, including those still in ingress: those are
    /// taken in only so they can be told, and a prompt refused at intake hears its own reason
    /// instead.
    fn fail_all(&mut self, reason: FinishReason) {
        self.in_flight = None;
        while let Some(request) = self.ingress.try_recv() {
            self.scheduler.intake(request);
        }
        self.scheduler.retire_all(reason);
        if let Some(reply) = self.draining.take() {
            send_reply(&reply, self.state());
        }
    }
}

/// Answers a query; a querier that stopped listening loses nothing anyone can act on.
fn send_reply(reply: &Sender<EngineState>, state: EngineState) {
    match reply.send(state) {
        Ok(()) | Err(SendError(_)) => {}
    }
}
