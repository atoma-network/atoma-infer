//! The step-path latency gate: schedule, command build and result apply, measured on the host
//! with no GPU, over two batches that between them cover every branch of a pass.
//!
//! In the decode scenario sixty-four requests decode at a 1024-token context while eight prompts
//! wait in the admission window under longest prefix match. In the churn scenario prompts share
//! one of a few prefixes, so admission claims cached blocks, and they finish after a handful of
//! decodes, so every measured pass also retires requests and leases against a pool with nothing
//! free — where a block comes only from evicting cache.
//!
//! Each measured pass applies the previous step's result, schedules the next step, builds its
//! command and pushes it to the ring; the mock executor answers between passes, untimed. The gate
//! fails when either scenario's p99 pass reaches 200 microseconds.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use atoma_core::attention::{BackendDeclaration, CaptureContract, ModelDeclaration, SupportLevel};
use atoma_core::dispatch::{BucketLadder, DispatchConfig};
use atoma_core::engine::{Engine, EngineConfig, EngineHandle, ExecutorRings, Pass};
use atoma_core::kv::HashAlgorithm;
use atoma_core::request::{
    egress, EgressReceiver, NewRequest, Priority, RequestEvent, SamplingParams, StopCriteria,
};
use atoma_core::scheduler::{AdmissionPolicy, SchedulerConfig};
use atoma_core::step::StepResult;
use atoma_core::types::{RequestCount, TokenCount};
use hdrhistogram::Histogram;
use tracing::{error, info};

const BLOCK_SIZE: usize = 16;
const WARMUP_PASSES: usize = 200;
const MEASURED_PASSES: usize = 2000;
const P99_LIMIT_MICROS: u64 = 200;
/// Both scenarios drain their clients every pass, so no backlog ever approaches this.
const MAX_CLIENT_BACKLOG: usize = 1024;

const DECODE_BATCH: usize = 64;
const DECODE_CONTEXT_TOKENS: usize = 1024;
const DECODE_WAITING: usize = 8;
/// More than the run can generate, so no decode ever finishes and the window stays full.
const DECODE_NEW_TOKENS: usize = 1_000_000;

const CHURN_BATCH: usize = 32;
const CHURN_WAITING: usize = 8;
const CHURN_PROMPT_TOKENS: usize = 128;
/// Tokens a churn prompt shares with every other prompt of its family: the cached prefix
/// admission claims.
const CHURN_SHARED_TOKENS: usize = 64;
const CHURN_FAMILIES: u32 = 8;
const CHURN_NEW_TOKENS: usize = 16;
/// The entry cap: a bucket of the ladder, so the reservation can pad a full batch up to it.
const CHURN_MAX_BATCH: usize = 64;
const CHURN_MAX_MODEL_TOKENS: usize = CHURN_PROMPT_TOKENS + CHURN_NEW_TOKENS + 1;

fn tokens(value: usize) -> TokenCount {
    TokenCount::new(value).expect("bench counts are nonzero")
}

fn requests(value: usize) -> RequestCount {
    RequestCount::new(value).expect("bench counts are nonzero")
}

fn blocks_for(tokens: usize) -> usize {
    tokens.div_ceil(BLOCK_SIZE)
}

fn decode_config() -> EngineConfig {
    let decode_headroom = MEASURED_PASSES + WARMUP_PASSES + BLOCK_SIZE;
    let live = DECODE_BATCH + DECODE_WAITING;
    EngineConfig {
        scheduler: SchedulerConfig {
            token_budget: tokens(DECODE_BATCH * DECODE_CONTEXT_TOKENS),
            max_batch: requests(DECODE_BATCH),
            max_model_len: tokens(DECODE_CONTEXT_TOKENS + decode_headroom),
            block_size: tokens(BLOCK_SIZE),
            window: requests(DECODE_WAITING),
            admission: AdmissionPolicy::LongestPrefixMatch,
            max_requests: requests(live),
            max_client_backlog: tokens(MAX_CLIENT_BACKLOG),
            eos_token_ids: Vec::new(),
            hash_algorithm: HashAlgorithm::Sha256V1,
        },
        dispatch: dispatch_config(DECODE_BATCH),
        block_count: u32::try_from(
            live * blocks_for(DECODE_CONTEXT_TOKENS + decode_headroom) + DECODE_BATCH,
        )
        .expect("fits u32"),
        ingress_capacity: requests(live),
        idle_deadline: Duration::from_millis(1),
        step_deadline: Duration::from_mins(1),
    }
}

/// A pool that holds the running batch and little more, so cached blocks are evicted to lease.
fn churn_config() -> EngineConfig {
    let live = CHURN_BATCH + CHURN_WAITING;
    let max_batch = CHURN_MAX_BATCH;
    EngineConfig {
        scheduler: SchedulerConfig {
            token_budget: tokens(2048),
            max_batch: requests(max_batch),
            max_model_len: tokens(CHURN_MAX_MODEL_TOKENS),
            block_size: tokens(BLOCK_SIZE),
            window: requests(CHURN_WAITING),
            admission: AdmissionPolicy::LongestPrefixMatch,
            max_requests: requests(live),
            max_client_backlog: tokens(MAX_CLIENT_BACKLOG),
            eos_token_ids: Vec::new(),
            hash_algorithm: HashAlgorithm::Sha256V1,
        },
        dispatch: dispatch_config(max_batch),
        block_count: u32::try_from(
            max_batch + (CHURN_BATCH + 4) * blocks_for(CHURN_MAX_MODEL_TOKENS),
        )
        .expect("fits u32"),
        ingress_capacity: requests(live),
        idle_deadline: Duration::from_millis(1),
        step_deadline: Duration::from_mins(1),
    }
}

fn dispatch_config(max_batch: usize) -> DispatchConfig {
    DispatchConfig {
        bucket_ladder: BucketLadder::new(vec![1, 2, 4, 8, 16, 32, 64]).expect("nonzero"),
        captured_max_requests: requests(max_batch),
    }
}

/// One backend that captures anything, breaking nothing: the bench measures the step path, not
/// what the backends can capture.
fn contract() -> CaptureContract {
    CaptureContract::resolve(
        &[BackendDeclaration::new(
            "bench-backend",
            SupportLevel::Always,
        )],
        &ModelDeclaration::new("bench-model"),
    )
}

fn new_request(prompt: Vec<u32>, max_new_tokens: usize) -> (NewRequest, EgressReceiver) {
    let (sender, receiver) = egress();
    let request = NewRequest {
        prompt,
        sampling: SamplingParams::default(),
        stop: StopCriteria {
            max_new_tokens: tokens(max_new_tokens),
            ignore_eos: true,
        },
        priority: Priority::default(),
        egress: sender,
    };
    (request, receiver)
}

/// A prompt of `len` tokens unique to request `index`, so nothing hits the prefix cache.
fn unique_prompt(index: u32, len: usize) -> Vec<u32> {
    let base = index * 1_000_000;
    (0..u32::try_from(len).expect("fits u32"))
        .map(|offset| base + offset)
        .collect()
}

/// A prompt whose leading blocks are its family's, shared with every other prompt of that
/// family, and whose tail is its own.
fn family_prompt(index: u32) -> Vec<u32> {
    let family = index % CHURN_FAMILIES;
    let shared = (0..u32::try_from(CHURN_SHARED_TOKENS).expect("fits u32"))
        .map(|offset| family * 100_000 + offset);
    let own = (0..u32::try_from(CHURN_PROMPT_TOKENS - CHURN_SHARED_TOKENS).expect("fits u32"))
        .map(|offset| 10_000_000 + index * 1_000 + offset);
    shared.chain(own).collect()
}

/// Answers the step in flight, if one is, with one token per sampling entry.
fn answer(rings: &mut ExecutorRings) {
    if let Some(command) = rings.pop_command() {
        let sampled = vec![7; command.sampling_count()];
        rings
            .push_result(StepResult {
                step: command.step,
                sampled,
            })
            .expect("one step in flight");
    }
}

fn new_histogram() -> Histogram<u64> {
    Histogram::new_with_bounds(1, 10_000_000, 3).expect("valid histogram bounds")
}

fn record(histogram: &mut Histogram<u64>, micros: u128) {
    histogram
        .record(u64::try_from(micros).expect("fits u64"))
        .expect("within bounds");
}

/// Drains every client and drops those whose request finished, returning how many did.
fn take_finished(clients: &mut Vec<EgressReceiver>) -> usize {
    let before = clients.len();
    clients.retain(|client| {
        let mut running = true;
        while let Ok(event) = client.try_recv() {
            if matches!(event, RequestEvent::Finished { .. }) {
                running = false;
            }
        }
        running
    });
    before - clients.len()
}

fn submit(handle: &EngineHandle, request: NewRequest) {
    handle.ingress.try_send(request).expect("ingress has room");
}

/// Sixty-four requests decoding at a long context, with a full admission window behind them.
fn measure_decode() -> Histogram<u64> {
    let (mut engine, handle, handoff) =
        Engine::new(&decode_config(), &contract()).expect("the bench configuration is valid");
    let mut rings = handoff.rings;
    let mut clients = Vec::with_capacity(DECODE_BATCH + DECODE_WAITING);
    for index in 0..DECODE_BATCH {
        let index = u32::try_from(index).expect("fits u32");
        let (request, receiver) = new_request(
            unique_prompt(index, DECODE_CONTEXT_TOKENS),
            DECODE_NEW_TOKENS,
        );
        submit(&handle, request);
        clients.push(receiver);
    }
    // The first pass takes every prompt in and prefills them all under the wide budget.
    assert_eq!(engine.pass(), Pass::Continue);
    answer(&mut rings);
    for index in DECODE_BATCH..DECODE_BATCH + DECODE_WAITING {
        let index = u32::try_from(index).expect("fits u32");
        let (request, receiver) = new_request(
            unique_prompt(index, DECODE_CONTEXT_TOKENS),
            DECODE_NEW_TOKENS,
        );
        submit(&handle, request);
        clients.push(receiver);
    }

    let mut histogram = new_histogram();
    for pass in 0..WARMUP_PASSES + MEASURED_PASSES {
        let started = Instant::now();
        assert_eq!(engine.pass(), Pass::Continue);
        let elapsed = started.elapsed();
        answer(&mut rings);
        for client in &clients {
            while client.try_recv().is_ok() {}
        }
        if pass >= WARMUP_PASSES {
            record(&mut histogram, elapsed.as_micros());
        }
    }
    let state = engine.state();
    assert_eq!(state.running, DECODE_BATCH, "every decode is still running");
    assert_eq!(state.waiting, DECODE_WAITING, "the window is still full");
    histogram
}

/// Prompts that share a family prefix, run a handful of decodes and finish, replaced as they go:
/// every pass admits over prefix hits, evicts to lease, and retires what finished.
fn measure_churn() -> Histogram<u64> {
    let (mut engine, handle, handoff) =
        Engine::new(&churn_config(), &contract()).expect("the bench configuration is valid");
    let mut rings = handoff.rings;
    let mut clients = Vec::with_capacity(CHURN_BATCH + CHURN_WAITING);
    let mut next: u32 = 0;
    let mut finished = 0;
    for _ in 0..CHURN_BATCH + CHURN_WAITING {
        let (request, receiver) = new_request(family_prompt(next), CHURN_NEW_TOKENS);
        submit(&handle, request);
        clients.push(receiver);
        next += 1;
    }

    let mut histogram = new_histogram();
    for pass in 0..WARMUP_PASSES + MEASURED_PASSES {
        let started = Instant::now();
        assert_eq!(engine.pass(), Pass::Continue);
        let elapsed = started.elapsed();
        answer(&mut rings);
        let done = take_finished(&mut clients);
        finished += done;
        for _ in 0..done {
            let (request, receiver) = new_request(family_prompt(next), CHURN_NEW_TOKENS);
            submit(&handle, request);
            clients.push(receiver);
            next += 1;
        }
        if pass >= WARMUP_PASSES {
            record(&mut histogram, elapsed.as_micros());
        }
    }
    assert!(
        finished > MEASURED_PASSES / CHURN_NEW_TOKENS,
        "requests finished and were replaced throughout: {finished}"
    );
    let state = engine.state();
    assert_eq!(
        state.free_blocks, 0,
        "the pool is committed, so a block comes only from evicting cache"
    );
    assert!(
        state.available_blocks > 0,
        "finished requests left cache behind for the next prompt of their family to claim"
    );
    histogram
}

/// Reports one scenario, returning whether its p99 reached the limit.
fn over_limit(scenario: &'static str, histogram: &Histogram<u64>) -> bool {
    let p99 = histogram.value_at_quantile(0.99);
    info!(
        scenario,
        passes = MEASURED_PASSES,
        p50_us = histogram.value_at_quantile(0.5),
        p90_us = histogram.value_at_quantile(0.9),
        p99_us = p99,
        max_us = histogram.max(),
        "step path: schedule, command build and result apply"
    );
    if p99 >= P99_LIMIT_MICROS {
        error!(
            scenario,
            p99_us = p99,
            limit_us = P99_LIMIT_MICROS,
            "step-path p99 is over the limit"
        );
        return true;
    }
    false
}

fn main() -> ExitCode {
    tracing_subscriber::fmt().with_target(false).init();
    let decode = measure_decode();
    let churn = measure_churn();
    // Both are reported before either can fail the run.
    let decode_over = over_limit("decode", &decode);
    let churn_over = over_limit("prefix churn", &churn);
    if decode_over || churn_over {
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
