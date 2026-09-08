//! The executor thread of one rank: spawned pinned to its core, its loop built on the thread and
//! reported as running before the spawner goes on, and joined for why it stopped.

use std::any::Any;
use std::error::Error;
use std::io::ErrorKind;
use std::panic::resume_unwind;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use core_affinity::{get_core_ids, set_for_current, CoreId as AffinityCoreId};
use flume::{Receiver, SendError, Sender, TryRecvError};
use thiserror::Error;
use tracing::{error, info};

use crate::config::{CoreId, Rank};
use crate::executor::{ExecutorError, ExecutorLoop};

/// An error whose type the executor does not name: the forward's own, or whatever building an
/// executor failed with.
pub type Cause = Box<dyn Error + Send + Sync + 'static>;

/// Why an executor thread could not be started.
#[derive(Debug, Error)]
pub enum SpawnError {
    /// The operating system refused the thread.
    #[error("the executor thread for rank {rank} could not be spawned: {kind:?}")]
    Thread { rank: Rank, kind: ErrorKind },
    /// The core is not one this process may run on, or the affinity could not be set.
    #[error(
        "rank {rank} cannot be pinned to core {core}; this process may run on cores {allowed:?}"
    )]
    Pin {
        rank: Rank,
        core: CoreId,
        allowed: Vec<usize>,
    },
    /// Building the executor on its thread failed.
    #[error("rank {rank} could not be built: {source}")]
    Build {
        rank: Rank,
        #[source]
        source: Cause,
    },
}

/// The running executor thread of one rank.
#[derive(Debug)]
pub struct ExecutorThread {
    rank: Rank,
    join: JoinHandle<Result<(), ExecutorError>>,
}

impl ExecutorThread {
    #[must_use]
    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// Waits for the thread to return, handing back why it stopped if it failed. A thread that
    /// panicked stopped on [`ExecutorError::Panicked`], so a rank's panic is a cause to report
    /// rather than a panic of whoever joins.
    ///
    /// # Errors
    ///
    /// Returns the [`ExecutorError`] the loop stopped on.
    pub fn join(self) -> Result<(), ExecutorError> {
        match self.join.join() {
            Ok(outcome) => outcome,
            Err(payload) => Err(ExecutorError::Panicked {
                message: panic_message(payload.as_ref()),
            }),
        }
    }

    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.join.is_finished()
    }
}

/// An executor thread that is starting: pinning itself, building its loop and reporting.
#[derive(Debug)]
pub struct Launched {
    rank: Rank,
    join: JoinHandle<Result<(), ExecutorError>>,
    readiness: Receiver<Result<(), SpawnError>>,
}

/// How often a wait on several launched threads looks again at those still starting.
const STARTING_POLL: Duration = Duration::from_millis(10);

/// What one look at a starting thread found.
enum Starting {
    /// Still pinning or building.
    NotYet,
    /// Built and running.
    Running,
    /// Could not start.
    Failed(SpawnError),
    /// Dropped its report channel without reporting, which only a panic while starting does.
    Panicked,
}

impl Launched {
    /// Waits until the thread's loop is built and running, or it failed to start.
    ///
    /// # Errors
    ///
    /// Returns [`SpawnError`] when the thread could not be pinned or its loop could not be built.
    ///
    /// # Panics
    ///
    /// Panics when the thread panicked while starting, carrying its panic on.
    pub fn wait(self) -> Result<ExecutorThread, SpawnError> {
        let Self {
            rank,
            join,
            readiness,
        } = self;
        match readiness.recv() {
            Ok(Ok(())) => Ok(ExecutorThread { rank, join }),
            Ok(Err(error)) => Err(error),
            Err(_) => Self::resume_panic(join),
        }
    }

    /// Looks once at whether the thread has reported.
    fn look(&self) -> Starting {
        match self.readiness.try_recv() {
            Ok(Ok(())) => Starting::Running,
            Ok(Err(error)) => Starting::Failed(error),
            Err(TryRecvError::Empty) => Starting::NotYet,
            Err(TryRecvError::Disconnected) => Starting::Panicked,
        }
    }

    /// The thread as running, once a look found it so.
    fn running(self) -> ExecutorThread {
        ExecutorThread {
            rank: self.rank,
            join: self.join,
        }
    }

    /// The thread reports before it returns, so a report that never came is a panic while
    /// starting; carries it on.
    fn resume_panic(join: JoinHandle<Result<(), ExecutorError>>) -> ! {
        let panic = join
            .join()
            .expect_err("the executor thread returned without reporting");
        resume_unwind(panic)
    }
}

/// Waits for every launched thread to be running, or for the first to fail. The threads are
/// looked at together rather than one after another: ranks whose start is a rendezvous only
/// finish starting together, and one that fails before it joins would leave the rest waiting
/// for it forever.
///
/// # Errors
///
/// Returns the first [`SpawnError`] any thread failed with; the others are dropped still
/// starting, and end with the process.
///
/// # Panics
///
/// Panics when a thread panicked while starting, carrying its panic on.
pub fn wait_all(launched: Vec<Launched>) -> Result<Vec<ExecutorThread>, SpawnError> {
    let mut starting: Vec<Option<Launched>> = launched.into_iter().map(Some).collect();
    let mut running: Vec<Option<ExecutorThread>> = (0..starting.len()).map(|_| None).collect();
    while starting.iter().any(Option::is_some) {
        let mut progressed = false;
        for (slot, thread) in starting.iter_mut().zip(running.iter_mut()) {
            let Some(launched) = slot else {
                continue;
            };
            match launched.look() {
                Starting::NotYet => {}
                Starting::Failed(error) => return Err(error),
                Starting::Panicked => {
                    if let Some(panicked) = slot.take() {
                        Launched::resume_panic(panicked.join);
                    }
                }
                Starting::Running => {
                    progressed = true;
                    *thread = slot.take().map(Launched::running);
                }
            }
        }
        if !progressed {
            thread::sleep(STARTING_POLL);
        }
    }
    Ok(running.into_iter().flatten().collect())
}

/// Spawns rank `rank`'s executor thread pinned to `core`, builds its loop there with `build` and
/// runs it until the far side is gone. Returns once the loop is built and running, so whatever
/// stops it from starting is this call's error.
///
/// # Errors
///
/// Returns [`SpawnError`] when the thread cannot be spawned, cannot be pinned to `core`, or
/// `build` fails.
pub fn spawn<L, B>(rank: Rank, core: CoreId, build: B) -> Result<ExecutorThread, SpawnError>
where
    L: ExecutorLoop + 'static,
    B: FnOnce() -> Result<L, Cause> + Send + 'static,
{
    launch(rank, core, build)?.wait()
}

/// Launches rank `rank`'s executor thread pinned to `core`, building its loop there with `build`,
/// without waiting for it to start: what [`Launched::wait`] does. Ranks whose start is a
/// rendezvous — every NCCL rank must join before any communicator opens — are all launched
/// before any is waited on.
///
/// The thread is named `atoma-executor-{rank}`. The loop is built on the thread because what it
/// holds — the device, the session — belongs to that thread alone.
///
/// # Errors
///
/// Returns [`SpawnError::Thread`] when the operating system refuses the thread.
pub fn launch<L, B>(rank: Rank, core: CoreId, build: B) -> Result<Launched, SpawnError>
where
    L: ExecutorLoop + 'static,
    B: FnOnce() -> Result<L, Cause> + Send + 'static,
{
    let (ready, readiness) = flume::bounded::<Result<(), SpawnError>>(1);
    let join = thread::Builder::new()
        .name(format!("atoma-executor-{rank}"))
        .spawn(move || {
            let executor_loop = match start(rank, core, build) {
                Ok(executor_loop) => executor_loop,
                Err(error) => {
                    report(&ready, Err(error));
                    return Ok(());
                }
            };
            report(&ready, Ok(()));
            executor_loop
                .run()
                .inspect_err(|cause| error!(%rank, %cause, "executor failed"))
        })
        .map_err(|error| SpawnError::Thread {
            rank,
            kind: error.kind(),
        })?;
    Ok(Launched {
        rank,
        join,
        readiness,
    })
}

/// Pins the current thread and builds its loop.
fn start<L, B>(rank: Rank, core: CoreId, build: B) -> Result<L, SpawnError>
where
    L: ExecutorLoop,
    B: FnOnce() -> Result<L, Cause>,
{
    pin(rank, core)?;
    build().map_err(|source| SpawnError::Build { rank, source })
}

/// Pins the current thread to `core`, refusing a core this process may not run on.
fn pin(rank: Rank, core: CoreId) -> Result<(), SpawnError> {
    let allowed: Vec<usize> = get_core_ids()
        .unwrap_or_default()
        .into_iter()
        .map(|id| id.id)
        .collect();
    let pinned =
        allowed.contains(&core.get()) && set_for_current(AffinityCoreId { id: core.get() });
    if !pinned {
        return Err(SpawnError::Pin {
            rank,
            core,
            allowed,
        });
    }
    info!(%rank, %core, "executor thread pinned");
    Ok(())
}

/// Reports whether starting succeeded; the spawner is waiting, so the send cannot fail.
fn report(ready: &Sender<Result<(), SpawnError>>, outcome: Result<(), SpawnError>) {
    match ready.send(outcome) {
        Ok(()) | Err(SendError(_)) => {}
    }
}

/// What a panic said, for reporting it as a cause.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_owned();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "a panic with no message".to_owned()
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;

    use atoma_core::engine::{Control, Engine, EngineHandle, EngineThread, ExecutorHandoff};
    use atoma_core::request::RequestEvent;
    use core_affinity::{get_core_ids, CoreId as AffinityCoreId};

    use super::{launch, spawn, wait_all, SpawnError};
    use crate::config::{CoreId, Rank};
    use crate::executor::{Executor, ExecutorError, ExecutorLoop};
    use crate::test_support::{contract, engine_config, submit, FakeForward, BLOCK_SIZE, WAIT};

    fn engine() -> (EngineHandle, ExecutorHandoff, EngineThread) {
        Engine::spawn(&engine_config(), &contract()).unwrap()
    }

    /// A core this process may run on.
    fn a_core() -> AffinityCoreId {
        get_core_ids()
            .and_then(|cores| cores.first().copied())
            .expect("the process may run on at least one core")
    }

    #[test]
    fn spawn_pins_the_thread_names_it_and_runs_the_executor_on_it() {
        let core = a_core();
        let (handle, handoff, engine) = engine();
        let rings = handoff.rings;
        let (observed, observations) = mpsc::channel();
        let thread = spawn(Rank::new(3), CoreId::new(core.id), move || {
            let name = thread::current().name().map(str::to_owned);
            let pinned_to: Vec<usize> = get_core_ids()
                .unwrap_or_default()
                .into_iter()
                .map(|id| id.id)
                .collect();
            observed.send((name, pinned_to)).unwrap();
            Ok(Executor::new(rings, FakeForward::constant(7), BLOCK_SIZE))
        })
        .unwrap();
        assert_eq!(thread.rank(), Rank::new(3));
        let (name, pinned_to) = observations.recv_timeout(WAIT).unwrap();
        assert_eq!(name.as_deref(), Some("atoma-executor-3"));
        assert_eq!(
            pinned_to,
            [core.id],
            "the thread may run on that core alone"
        );

        let client = submit(&handle, 2, 1);
        assert!(matches!(
            client.recv_timeout(WAIT).unwrap(),
            RequestEvent::Token { token: 7, .. }
        ));
        handle.control.try_send(Control::Shutdown).unwrap();
        engine.join();
        assert!(thread.join().is_ok());
    }

    #[test]
    fn spawn_refuses_a_core_the_process_may_not_run_on() {
        let allowed = get_core_ids().unwrap_or_default();
        let beyond = allowed
            .iter()
            .map(|id| id.id)
            .max()
            .map_or(0, |max| max + 1);
        let (_handle, handoff, _engine) = engine();
        let rings = handoff.rings;
        let error = spawn(Rank::ZERO, CoreId::new(beyond), move || {
            Ok(Executor::new(rings, FakeForward::constant(7), BLOCK_SIZE))
        })
        .unwrap_err();
        let SpawnError::Pin { rank, core, .. } = &error else {
            panic!("a pin failure, not {error}");
        };
        assert_eq!(*rank, Rank::ZERO);
        assert_eq!(core.get(), beyond);
        assert!(error.to_string().contains("may run on cores"), "{error}");
    }

    #[test]
    fn spawn_returns_a_build_failure_from_the_thread() {
        let core = a_core();
        let error = spawn::<Executor<FakeForward>, _>(Rank::ZERO, CoreId::new(core.id), || {
            Err("no model at /nowhere".into())
        })
        .unwrap_err();
        assert!(matches!(&error, SpawnError::Build { .. }), "{error}");
        assert!(
            error.to_string().contains("no model at /nowhere"),
            "{error}"
        );
    }

    /// A build that panics is carried on to whoever waits, not swallowed as a failure to start.
    #[test]
    #[should_panic(expected = "no device today")]
    fn a_panic_while_starting_is_carried_on_to_the_waiter() {
        let core = a_core();
        let _ = spawn::<Executor<FakeForward>, _>(Rank::ZERO, CoreId::new(core.id), || {
            panic!("no device today")
        });
    }

    /// A loop that panics stops the thread on a reported cause, not a panic of the joiner.
    #[test]
    fn a_panic_in_the_loop_is_the_cause_the_join_returns() {
        struct Panics;
        impl ExecutorLoop for Panics {
            fn run(self) -> Result<(), ExecutorError> {
                give_up();
                Ok(())
            }
        }
        fn give_up() {
            panic!("the loop gave up")
        }
        let thread = spawn(Rank::ZERO, CoreId::new(a_core().id), || Ok(Panics)).unwrap();
        let error = thread.join().unwrap_err();
        assert!(
            matches!(&error, ExecutorError::Panicked { message } if message == "the loop gave up"),
            "{error}"
        );
    }

    /// Ranks are waited on together: a later rank failing to start is reported even while an
    /// earlier one is still starting.
    #[test]
    fn wait_all_reports_the_first_failure_while_others_are_still_starting() {
        let core = a_core();
        let (release, held) = mpsc::channel::<()>();
        let (_handle, handoff, _engine) = engine();
        let rings = handoff.rings;
        let slow = launch(Rank::ZERO, CoreId::new(core.id), move || {
            held.recv().expect("released once the wait has returned");
            Ok(Executor::new(rings, FakeForward::constant(1), BLOCK_SIZE))
        })
        .unwrap();
        let failing =
            launch::<Executor<FakeForward>, _>(Rank::new(1), CoreId::new(core.id), || {
                Err("no model".into())
            })
            .unwrap();
        let error = wait_all(vec![slow, failing]).unwrap_err();
        assert!(
            matches!(&error, SpawnError::Build { rank, .. } if *rank == Rank::new(1)),
            "{error}"
        );
        release.send(()).unwrap();
    }
}
