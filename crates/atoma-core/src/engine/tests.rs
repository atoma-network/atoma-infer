//! Engine tests: every one drives the engine with a mock executor on the far side of the rings —
//! in lockstep from the test thread for the pass, and on threads of their own for the loop. No
//! device.

use std::thread;
use std::time::{Duration, Instant, SystemTime};

use validator::Validate;

use crate::attention::{CaptureContract, SupportLevel};
use crate::dispatch::{BucketLadder, DispatchConfig};
use crate::engine::mock::MockExecutor;
use crate::engine::{
    Control, Engine, EngineConfig, EngineError, EngineHandle, EngineThread, IngressRefused, Pass,
};
use crate::kv::HashAlgorithm;
use crate::request::{
    egress, EgressReceiver, FinishReason, NewRequest, Priority, RequestEvent, SamplingParams,
    StopCriteria, Usage,
};
use crate::scheduler::{AdmissionPolicy, SchedulerConfig, SchedulerError};
use crate::step::StepResult;
use crate::test_support::{contract as capture_contract, requests, tokens};
use crate::types::RequestId;

const BLOCK_SIZE: usize = 4;
const MAX_BATCH: usize = 4;
const BLOCKS: usize = 16;
const EOS: u32 = 99;

/// The idle deadline a spawned engine parks under. Longer than every wait a test can spend, so a
/// test that finishes proves the thread was woken rather than that it passed at its deadline.
/// Bounded rather than endless, so a lost wake fails a test instead of hanging the suite.
const LONG_DEADLINE: Duration = Duration::from_mins(5);

/// How long a test waits on the engine thread before calling it wedged. Generous on purpose: a
/// loaded runner is slow, not broken, and nothing here measures how quickly a wake arrives — only
/// that it arrives. Every wait a single test can spend still fits inside [`LONG_DEADLINE`].
const WAIT: Duration = Duration::from_secs(30);

/// How often a wait re-examines what it is waiting for.
const POLL: Duration = Duration::from_millis(1);

fn config(max_requests: usize, ingress_capacity: usize) -> EngineConfig {
    EngineConfig {
        scheduler: SchedulerConfig {
            token_budget: tokens(64),
            max_batch: requests(MAX_BATCH),
            max_model_len: tokens(32),
            block_size: tokens(BLOCK_SIZE),
            window: requests(8),
            admission: AdmissionPolicy::Fcfs,
            max_requests: requests(max_requests),
            max_client_backlog: tokens(1024),
            eos_token_ids: vec![EOS],
            hash_algorithm: HashAlgorithm::Sha256V1,
        },
        dispatch: DispatchConfig {
            bucket_ladder: BucketLadder::new(vec![1, 2, 4]).unwrap(),
            captured_max_requests: requests(MAX_BATCH),
        },
        block_count: u32::try_from(BLOCKS).unwrap(),
        ingress_capacity: requests(ingress_capacity),
        idle_deadline: Duration::from_millis(1),
        step_deadline: WAIT,
    }
}

/// One backend that captures anything, and a model that breaks nothing: the engine tests are
/// about the pass, not about what the backends can capture.
fn contract() -> CaptureContract {
    capture_contract(SupportLevel::Always, &[])
}

fn engine(max_requests: usize, ingress_capacity: usize) -> (Engine, EngineHandle, MockExecutor) {
    let (engine, handle, rings) =
        Engine::new(&config(max_requests, ingress_capacity), &contract()).unwrap();
    (engine, handle, MockExecutor::constant(rings, 1))
}

fn new_request(prompt_len: usize, max_new_tokens: usize) -> (NewRequest, EgressReceiver) {
    let (sender, receiver) = egress();
    let request = NewRequest {
        prompt: (1..=u32::try_from(prompt_len).unwrap()).collect(),
        sampling: SamplingParams::default(),
        stop: StopCriteria {
            max_new_tokens: tokens(max_new_tokens),
            ignore_eos: false,
        },
        priority: Priority::default(),
        egress: sender,
    };
    (request, receiver)
}

fn submit(handle: &EngineHandle, prompt_len: usize, max_new_tokens: usize) -> EgressReceiver {
    let (request, receiver) = new_request(prompt_len, max_new_tokens);
    handle.ingress.try_send(request).unwrap();
    receiver
}

fn events(receiver: &EgressReceiver) -> Vec<RequestEvent> {
    receiver.try_iter().collect()
}

fn finish_reason(receiver: &EgressReceiver) -> Option<FinishReason> {
    events(receiver).into_iter().find_map(|event| match event {
        RequestEvent::Finished { reason, .. } => Some(reason),
        RequestEvent::Token { .. } => None,
    })
}

/// Asserts the engine thread has exited: ingress refuses a request as a gone engine rather
/// than as overload.
fn assert_engine_gone(handle: &EngineHandle) {
    let (request, _client) = new_request(1, 1);
    assert!(matches!(
        handle.ingress.try_send(request),
        Err(IngressRefused::EngineGone(_))
    ));
}

/// Polls `ready` until it holds, failing once the waiting budget is spent. What is being waited
/// for is named, so a failure says which wake never came rather than which line ran out of time.
fn wait_until(what: &str, mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + WAIT;
    while !ready() {
        assert!(Instant::now() < deadline, "waited {WAIT:?} for {what}");
        thread::sleep(POLL);
    }
}

/// One engine pass followed by the executor serving whatever it issued.
fn turn(engine: &mut Engine, executor: &mut MockExecutor) -> Pass {
    let pass = engine.pass();
    executor.serve_one();
    pass
}

#[test]
fn a_request_flows_from_ingress_through_the_executor_to_its_finish() {
    let (mut engine, handle, mut executor) = engine(8, 8);
    let client = submit(&handle, 5, 2);

    assert_eq!(engine.pass(), Pass::Continue);
    assert_eq!(engine.state().running, 1, "taken in and admitted");
    assert!(engine.state().step_in_flight);
    assert!(executor.serve_one(), "the prefill command was issued");
    assert_eq!(executor.served[0].entries[0].input_tokens, [1, 2, 3, 4, 5]);

    assert_eq!(engine.pass(), Pass::Continue);
    assert!(matches!(
        events(&client).as_slice(),
        [RequestEvent::Token { token: 1, .. }]
    ));
    assert!(executor.serve_one(), "the decode command was issued");
    assert_eq!(executor.served[1].entries[0].context_len, 5);
    assert_eq!(executor.served[1].entries[0].input_tokens, [1]);

    assert_eq!(engine.pass(), Pass::Continue);
    assert_eq!(finish_reason(&client), Some(FinishReason::MaxNewTokens));
    assert!(!executor.serve_one(), "nothing left to run");
    assert_eq!(engine.state().live_requests, 0);
    assert_eq!(handle.heartbeat.read().pass, 3, "one beat per pass");
}

/// Control is drained before ingress: a state query sent behind a burst of requests is
/// answered before any of the burst is taken in.
#[test]
fn control_is_drained_before_ingress() {
    let (mut engine, handle, _executor) = engine(8, 8);
    let clients: Vec<_> = (0..4).map(|_| submit(&handle, 2, 1)).collect();
    let (reply, answer) = flume::bounded(1);
    handle.control.try_send(Control::State { reply }).unwrap();

    engine.pass();
    let state = answer.recv().unwrap();
    assert_eq!(
        state.live_requests, 0,
        "answered before the burst was taken in"
    );
    assert_eq!(
        engine.state().live_requests,
        clients.len(),
        "and then the burst was"
    );
}

#[test]
fn ingress_is_drained_only_while_the_slab_has_room_and_then_refuses() {
    let (mut engine, handle, mut executor) = engine(2, 2);
    let mut clients: Vec<_> = (0..2).map(|_| submit(&handle, 2, 1)).collect();
    engine.pass();
    assert_eq!(engine.state().live_requests, 2, "two slots, two taken in");

    // The slab is full and the step is still out; ingress fills up behind it and refuses.
    clients.extend((0..2).map(|_| submit(&handle, 2, 1)));
    let (fifth, _fifth_client) = new_request(2, 1);
    let IngressRefused::Overload(fifth) = handle.ingress.try_send(fifth).unwrap_err() else {
        panic!("a full ingress behind a full slab is overload");
    };
    engine.pass();
    assert_eq!(engine.state().live_requests, 2, "no room, nothing drained");
    assert!(
        matches!(
            handle.ingress.try_send(fifth),
            Err(IngressRefused::Overload(_))
        ),
        "still overloaded"
    );

    // The two in the slab finish on their first token; the same pass takes the next two in.
    executor.serve_one();
    engine.pass();
    assert_eq!(engine.state().live_requests, 2, "the next two came in");
    assert!(clients[..2]
        .iter()
        .all(|client| finish_reason(client) == Some(FinishReason::MaxNewTokens)));
    turn(&mut engine, &mut executor);
    turn(&mut engine, &mut executor);
    assert!(clients[2..]
        .iter()
        .all(|client| finish_reason(client) == Some(FinishReason::MaxNewTokens)));
}

#[test]
fn a_disconnected_client_retires_its_request_and_the_rest_of_the_batch_still_goes_out() {
    let (mut engine, handle, mut executor) = engine(8, 8);
    let mut clients: Vec<_> = (0..3).map(|_| submit(&handle, 2, 16)).collect();
    turn(&mut engine, &mut executor);
    turn(&mut engine, &mut executor);
    assert_eq!(executor.served[1].live_entries().len(), 3, "three decodes");

    drop(clients.remove(1));
    turn(&mut engine, &mut executor);
    let batch = executor.served.last().unwrap();
    assert_eq!(
        batch.live_entries().len(),
        2,
        "one retired, two still go out"
    );
    assert_eq!(engine.state().live_requests, 2);
}

#[test]
fn a_drain_stops_admission_and_reports_once_nothing_is_in_flight() {
    let (mut engine, handle, mut executor) = engine(8, 8);
    let running = submit(&handle, 2, 2);
    turn(&mut engine, &mut executor);
    let waiting = submit(&handle, 2, 2);
    let (reply, drained) = flume::bounded(1);
    handle.control.try_send(Control::Drain { reply }).unwrap();

    turn(&mut engine, &mut executor);
    assert!(drained.try_recv().is_err(), "a step is still in flight");
    assert_eq!(engine.state().running, 1);
    assert_eq!(engine.state().waiting, 1, "nothing new is admitted");
    assert!(engine.state().draining);

    turn(&mut engine, &mut executor);
    let state = drained.recv().unwrap();
    assert_eq!(state.running, 0);
    assert!(!state.step_in_flight);
    assert_eq!(state.waiting, 1);
    assert_eq!(finish_reason(&running), Some(FinishReason::MaxNewTokens));
    assert_eq!(
        finish_reason(&waiting),
        None,
        "still waiting, never admitted"
    );
    assert_eq!(turn(&mut engine, &mut executor), Pass::Continue);
    assert_eq!(engine.state().waiting, 1);
}

/// A preempted request is a running request the pool displaced, so a drain waits for it: it
/// re-enters, finishes and tells its client before the drain is answered.
#[test]
fn a_drain_waits_for_a_preempted_request_to_finish() {
    // Two one-block requests and no block to grow either: the second decode preempts.
    let mut config = config(8, 8);
    config.scheduler.max_model_len = tokens(2 * BLOCK_SIZE);
    config.block_count = u32::try_from(MAX_BATCH + 2).unwrap();
    let (mut engine, handle, rings) = Engine::new(&config, &contract()).unwrap();
    let mut executor = MockExecutor::constant(rings, 1);
    let first = submit(&handle, BLOCK_SIZE, 16);
    let second = submit(&handle, BLOCK_SIZE, 16);

    turn(&mut engine, &mut executor);
    assert_eq!(engine.state().running, 2, "both prefill together");
    turn(&mut engine, &mut executor);
    assert_eq!(engine.state().preempted, 1, "the pool cannot grow both");

    let (reply, drained) = flume::bounded(1);
    handle.control.try_send(Control::Drain { reply }).unwrap();
    turn(&mut engine, &mut executor);
    assert!(engine.state().draining);
    assert!(
        drained.try_recv().is_err(),
        "a preempted request is still live"
    );

    let mut answer = None;
    for _ in 0..16 {
        turn(&mut engine, &mut executor);
        if let Ok(state) = drained.try_recv() {
            answer = Some(state);
            break;
        }
    }
    let state = answer.expect("the preempted request re-entered and finished");
    assert_eq!(state.running, 0);
    assert_eq!(state.preempted, 0);
    assert!(!state.step_in_flight);
    assert_eq!(state.live_requests, 0);
    assert_eq!(finish_reason(&first), Some(FinishReason::MaxModelLength));
    assert_eq!(
        finish_reason(&second),
        Some(FinishReason::MaxModelLength),
        "the preempted request was told, not stranded"
    );
}

#[test]
fn shutdown_finishes_every_live_request_and_exits() {
    let (mut engine, handle, mut executor) = engine(2, 8);
    let running = submit(&handle, 2, 16);
    turn(&mut engine, &mut executor);
    let waiting = submit(&handle, 2, 16);
    engine.pass();
    let in_ingress = submit(&handle, 2, 16);
    handle.control.try_send(Control::Shutdown).unwrap();

    assert_eq!(engine.pass(), Pass::Exit);
    for client in [&running, &waiting, &in_ingress] {
        assert_eq!(finish_reason(client), Some(FinishReason::Shutdown));
    }
    assert_eq!(engine.state().live_requests, 0);
    assert_eq!(engine.state().available_blocks, BLOCKS - MAX_BATCH);
    drop(engine);
    assert!(executor.engine_gone());
}

#[test]
fn a_gone_executor_fails_every_pending_request_and_exits() {
    let (mut engine, handle, executor) = engine(8, 8);
    let running = submit(&handle, 2, 16);
    engine.pass();
    let waiting = submit(&handle, 2, 16);
    drop(executor);

    assert_eq!(engine.pass(), Pass::Exit);
    assert_eq!(finish_reason(&running), Some(FinishReason::ExecutorLost));
    assert_eq!(finish_reason(&waiting), Some(FinishReason::ExecutorLost));
    assert_eq!(engine.state().live_requests, 0);
}

#[test]
fn a_result_that_does_not_match_the_step_in_flight_is_fatal() {
    let (mut engine, handle, mut executor) = engine(8, 8);
    let client = submit(&handle, 2, 16);
    engine.pass();
    let command = executor.served.first().cloned();
    assert!(command.is_none(), "not served through the mock");
    executor.push_raw(StepResult {
        step: engine.scheduler().step(),
        sampled: vec![1, 2],
    });

    assert_eq!(engine.pass(), Pass::Exit);
    assert_eq!(finish_reason(&client), Some(FinishReason::ExecutorLost));
}

#[test]
fn the_deadlines_are_durations_written_in_milliseconds() {
    let config = config(8, 8);
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains(r#""idle_deadline_millis":1"#), "{json}");
    assert!(json.contains(r#""step_deadline_millis":30000"#), "{json}");
    let back: EngineConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.idle_deadline, Duration::from_millis(1));
    assert_eq!(back.step_deadline, WAIT);
    assert_eq!(back, config);
}

#[test]
fn a_zero_step_deadline_is_refused() {
    let mut config = config(8, 8);
    config.step_deadline = Duration::ZERO;
    let errors = config.validate().unwrap_err().to_string();
    assert!(errors.contains("step_deadline_millis is 0"), "{errors}");
}

#[test]
fn a_step_out_past_the_step_deadline_fails_every_live_request_and_exits() {
    let mut config = config(8, 8);
    config.step_deadline = Duration::from_millis(20);
    // The executor's ends are held, unanswered, for the whole test: the exit is the deadline's.
    let (mut engine, handle, rings) = Engine::new(&config, &contract()).unwrap();
    let running = submit(&handle, 2, 16);
    engine.pass();
    let waiting = submit(&handle, 2, 16);
    assert_eq!(
        engine.pass(),
        Pass::Continue,
        "the step is within its deadline"
    );
    assert!(engine.state().step_in_flight);

    thread::sleep(Duration::from_millis(30));
    assert_eq!(engine.pass(), Pass::Exit);
    assert_eq!(finish_reason(&running), Some(FinishReason::ExecutorLost));
    assert_eq!(finish_reason(&waiting), Some(FinishReason::ExecutorLost));
    assert_eq!(engine.state().live_requests, 0);
    drop(rings);
}

#[test]
fn the_heartbeat_advances_every_pass() {
    let (mut engine, handle, _executor) = engine(8, 8);
    assert_eq!(handle.heartbeat.read().pass, 0);
    engine.pass();
    engine.pass();
    assert_eq!(handle.heartbeat.read().pass, 2);
}

#[test]
fn the_engine_pads_a_replayed_decode_with_the_dummies_it_reserved() {
    let (mut engine, handle, mut executor) = engine(8, 8);
    assert_eq!(
        engine.state().free_blocks,
        BLOCKS - MAX_BATCH,
        "the dummies' blocks are held from the start"
    );
    let _clients: Vec<_> = (0..3).map(|_| submit(&handle, 2, 16)).collect();
    turn(&mut engine, &mut executor);
    turn(&mut engine, &mut executor);
    let decode = executor.served.last().unwrap();
    assert_eq!(
        decode.padding_count, 1,
        "three decodes pad to the bucket of four"
    );
    assert_eq!(decode.entries.len(), 4);
}

#[test]
fn a_bucket_ladder_the_reservation_cannot_pad_to_is_refused() {
    let mut unpaddable = config(8, 8);
    unpaddable.dispatch.bucket_ladder = BucketLadder::new(vec![8]).unwrap();
    let refused = Engine::new(&unpaddable, &contract()).unwrap_err();
    assert_eq!(
        refused,
        EngineError::PaddingCannotCoverBucket {
            max_batch: requests(MAX_BATCH),
            bucket: tokens(8),
        }
    );
    assert_eq!(
        refused.to_string(),
        "scheduler.max_batch of 4 pads to the bucket of 8 in dispatch.bucket_ladder, whose rows \
         outnumber the one dummy per entry the reservation holds; add a bucket of 4 to \
         dispatch.bucket_ladder, or set scheduler.max_batch to a bucket dispatch.bucket_ladder \
         holds",
        "the refusal names both configuration fields and what to do about it"
    );

    let mut too_small = config(8, 8);
    too_small.block_count = 2;
    assert!(matches!(
        Engine::new(&too_small, &contract()).unwrap_err(),
        EngineError::Padding(_)
    ));
}

/// The pool covers the reservation, but what the reservation leaves cannot hold one request of
/// the maximum model length. The engine refuses with the scheduler's error rather than dropping
/// the reservation's leases: they go back to the pool before the error returns.
#[test]
fn a_pool_the_reservation_leaves_too_small_is_refused() {
    // Blocks a maximum-length request needs; the pool holds the dummies' blocks and one fewer.
    let needed = 2;
    let mut starved = config(8, 8);
    starved.scheduler.max_model_len = tokens(needed * BLOCK_SIZE);
    starved.block_count = u32::try_from(MAX_BATCH + needed - 1).unwrap();

    assert_eq!(
        Engine::new(&starved, &contract()).unwrap_err(),
        EngineError::Scheduler(SchedulerError::PoolTooSmallForMaxModelLength {
            needed,
            free: needed - 1,
        }),
        "counted over what the dummies leave, not over the pool as configured"
    );

    let mut covered = starved.clone();
    covered.block_count += 1;
    assert!(
        Engine::new(&covered, &contract()).is_ok(),
        "one more block covers the dummies and a maximum-length request together"
    );
}

/// Spawns the engine with a long idle deadline and the mock executor on its own thread.
fn spawn_with_executor(idle_deadline: Duration) -> (EngineHandle, EngineThread) {
    let mut config = config(8, 8);
    config.idle_deadline = idle_deadline;
    let (handle, rings, engine) = Engine::spawn(&config, &contract()).unwrap();
    let executor = MockExecutor::constant(rings, 1);
    thread::spawn(move || executor.run_until_engine_gone());
    (handle, engine)
}

/// A request runs to its finish on the thread with nothing but wakes: ingress wakes the engine
/// to issue the step, and the executor's result wakes it to apply it, long before the deadline.
#[test]
fn the_thread_wakes_on_ingress_and_on_the_executor() {
    let (handle, engine) = spawn_with_executor(LONG_DEADLINE);
    let started = Instant::now();
    let client = submit(&handle, 3, 2);

    assert!(matches!(
        client.recv_timeout(WAIT).unwrap(),
        RequestEvent::Token { token: 1, .. }
    ));
    assert!(matches!(
        client.recv_timeout(WAIT).unwrap(),
        RequestEvent::Token { token: 1, .. }
    ));
    assert!(matches!(
        client.recv_timeout(WAIT).unwrap(),
        RequestEvent::Finished {
            reason: FinishReason::MaxNewTokens,
            ..
        }
    ));
    assert!(started.elapsed() < LONG_DEADLINE);

    handle.control.try_send(Control::Shutdown).unwrap();
    engine.join();
}

/// With nothing to do the thread still passes at its deadline, so an empty schedule can never
/// wedge it: the heartbeat keeps advancing with no wake at all.
#[test]
fn the_thread_never_wedges_on_an_empty_schedule() {
    let (handle, engine) = spawn_with_executor(Duration::from_millis(1));
    wait_until("five passes with nothing to do", || {
        handle.heartbeat.read().pass >= 5
    });
    let beat = handle.heartbeat.read();
    assert!(SystemTime::now().duration_since(beat.at).unwrap() < WAIT);

    handle.control.try_send(Control::Shutdown).unwrap();
    engine.join();
    assert_engine_gone(&handle);
}

#[test]
fn a_drain_is_answered_from_the_thread_and_shutdown_returns_it() {
    let (handle, engine) = spawn_with_executor(LONG_DEADLINE);
    let client = submit(&handle, 3, 2);
    assert!(matches!(
        client.recv_timeout(WAIT).unwrap(),
        RequestEvent::Token { .. }
    ));
    let (reply, drained) = flume::bounded(1);
    handle.control.try_send(Control::Drain { reply }).unwrap();

    let state = drained.recv_timeout(WAIT).unwrap();
    assert!(state.draining);
    assert_eq!(state.running, 0);
    assert!(!state.step_in_flight);
    assert_eq!(
        client.try_iter().last(),
        Some(RequestEvent::Finished {
            // The dummies took the first ids, so the one live request has the next.
            request: RequestId::new(4),
            reason: FinishReason::MaxNewTokens,
            usage: Usage {
                prompt_tokens: 3,
                generated_tokens: 2
            },
        }),
        "the running request finished before the drain was answered"
    );

    let late = submit(&handle, 3, 2);
    handle.control.try_send(Control::Shutdown).unwrap();
    engine.join();
    assert_eq!(finish_reason(&late), Some(FinishReason::Shutdown));
    assert_engine_gone(&handle);
    assert!(handle.control.try_send(Control::Shutdown).is_err());
}

/// An executor that holds its rings and never answers — a leader kept inside a collective by a
/// rank that died mid-step — is given up on at the step deadline, not waited on to the idle
/// deadline.
#[test]
fn a_wedged_executor_fails_every_live_request_and_returns_the_thread_at_the_step_deadline() {
    let mut config = config(8, 8);
    config.idle_deadline = LONG_DEADLINE;
    config.step_deadline = Duration::from_millis(50);
    let (handle, rings, engine) = Engine::spawn(&config, &contract()).unwrap();
    let client = submit(&handle, 3, 16);
    wait_until("the thread to give the step up", || engine.is_finished());
    engine.join();
    assert_eq!(finish_reason(&client), Some(FinishReason::ExecutorLost));
    assert_engine_gone(&handle);
    drop(rings);
}

#[test]
fn a_dead_executor_fails_every_pending_request_and_returns_the_thread() {
    let mut config = config(8, 8);
    config.idle_deadline = LONG_DEADLINE;
    let (handle, rings, engine) = Engine::spawn(&config, &contract()).unwrap();
    let client = submit(&handle, 3, 16);
    let mut executor = MockExecutor::constant(rings, 1);
    wait_until("the first step to be issued", || executor.serve_one());
    assert!(matches!(
        client.recv_timeout(WAIT).unwrap(),
        RequestEvent::Token { .. }
    ));

    drop(executor);
    wait_until("the thread to notice its executor is gone", || {
        engine.is_finished()
    });
    engine.join();
    assert_eq!(finish_reason(&client), Some(FinishReason::ExecutorLost));
}
