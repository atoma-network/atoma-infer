//! `RuntimeContext::free_memory` against a driver, with a stub `libcuda.so` standing in for one.
//!
//! cudarc resolves the driver by name through `dlopen` under the workspace's
//! `fallback-dynamic-loading` pin, so `tests/support/libcuda_stub.c` — eight entry points over two
//! imaginary devices — answers a context open and a reading on a machine with no GPU. The loader
//! reads its search path once at process start, so this compiles the stub beside the test binary
//! and re-runs the binary as a child with the path set; the child, marked by `CUDA_STUB_CHILD`,
//! is where the assertions run, and both scenarios share the one child. Where the stub does not
//! build — no C compiler — the test says so and passes, since the crate's own gates must run on a
//! host that has neither driver nor compiler.
#![cfg(target_os = "linux")]

use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

use atoma_runtime::context::RuntimeContext;
use atoma_runtime::error::RuntimeError;
use cudarc::driver::sys::CUresult;
use cudarc::driver::DriverError;

/// Marks the child process, the one whose loader search path finds the stub.
const CHILD: &str = "CUDA_STUB_CHILD";

/// What the stub reports free on each of its devices; see `tests/support/libcuda_stub.c`.
const FREE_ON_DEVICE_ZERO: usize = 1000;
const FREE_ON_DEVICE_ONE: usize = 2000;

#[test]
fn a_reading_asks_the_driver_through_this_context() {
    if env::var_os(CHILD).is_some() {
        a_reading_is_this_contexts_device();
        a_deferred_status_keeps_the_classification_it_carries();
        return;
    }

    let Some(stub) = compile_the_stub() else {
        eprintln!(
            "skipped: tests/support/libcuda_stub.c did not build, so there is no driver to read"
        );
        return;
    };

    let mut search_path = stub.into_os_string();
    if let Some(inherited) = env::var_os("LD_LIBRARY_PATH") {
        search_path.push(":");
        search_path.push(inherited);
    }
    let child = Command::new(env::current_exe().expect("this test binary's own path"))
        .args(["--test-threads=1", "--nocapture"])
        .env(CHILD, "1")
        .env("LD_LIBRARY_PATH", search_path)
        .output()
        .expect("re-running this test binary as a child");

    assert!(
        child.status.success(),
        "the reading against the stub driver failed:\n{}{}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
}

/// The count is the device of the context asked, not of whichever context the thread had current.
fn a_reading_is_this_contexts_device() {
    let device_zero = RuntimeContext::new(0).expect("the stub's device zero");
    // Opening device one binds its context to this thread, so a reading that skipped the bind
    // cudarc runs ahead of it would answer with device one's count for both.
    let device_one = RuntimeContext::new(1).expect("the stub's device one");

    let zero = device_zero.free_memory().expect("device zero's reading");
    let one = device_one.free_memory().expect("device one's reading");

    assert_eq!(zero.get(), FREE_ON_DEVICE_ZERO);
    assert_eq!(one.get(), FREE_ON_DEVICE_ONE);
}

/// A status carried over from an earlier operation keeps the classification the status earns.
fn a_deferred_status_keeps_the_classification_it_carries() {
    let context = RuntimeContext::new(0).expect("the stub's device zero");
    // What cudarc's CudaSlice, CudaEvent and CudaStream drops do with a driver error they cannot
    // return: record it on the context for the next call that checks. The bind cudarc runs ahead
    // of the read is that call, so the reading fails on a status the driver was never asked for.
    context.cuda().record_err::<()>(Err(DriverError(
        CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
    )));

    assert!(matches!(
        context.free_memory(),
        Err(RuntimeError::CaptureUnsupported(_))
    ));
}

/// Builds the stub beside the test binary and answers where it is, or `None` where it did not
/// build.
fn compile_the_stub() -> Option<PathBuf> {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/support/libcuda_stub.c");
    let directory = env::current_exe().ok()?.parent()?.join("libcuda-stub");
    fs::create_dir_all(&directory).ok()?;
    let built = Command::new(env::var("CC").unwrap_or_else(|_| "cc".to_owned()))
        .args(["-shared", "-fPIC", "-Wall", "-Wextra"])
        .arg("-o")
        .arg(directory.join("libcuda.so"))
        .arg(source)
        .status();
    matches!(built, Ok(status) if status.success()).then_some(directory)
}
