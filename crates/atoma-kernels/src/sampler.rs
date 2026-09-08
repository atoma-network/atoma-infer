//! The sampler's kernels: the gather that takes a row's token from what the device last sampled
//! for its request slot, and the sample that draws each live row's next token and leaves it
//! there.
//!
//! The sources are in-house (`kernels/sampler.cu`), compiled by nvcc under the `cuda` feature into
//! the in-house kernel library, apart from the vendored flash-attention build and its fast-math
//! flags: the weights are exponentials, and those must be the precise ones for the device to
//! stay within an ulp of the host reference. Each launcher returns its launch status and takes
//! the caller's stream. Without the feature the same functions return
//! [`KernelError::NotCompiled`](crate::error::KernelError::NotCompiled), so the crates built on
//! this one compile and test without a toolkit.

use core::ffi::c_void;

#[cfg(not(feature = "cuda"))]
use crate::error::KernelError;

/// Overwrites each row's token id with the token last sampled for its slot: `token_ids[r] =
/// sampled[gather_slots[r]]` for every row whose gather slot is not negative.
#[derive(Debug, Clone, Copy)]
pub struct GatherCall {
    /// u32 `[n_rows]`, read and written.
    pub token_ids: u64,
    /// i32 `[n_rows]`: the slot each row gathers from, or a negative value to leave the row.
    pub gather_slots: u64,
    /// u32 `[slots]`: the token last sampled for each slot.
    pub sampled: u64,
    pub n_rows: usize,
    pub stream: *mut c_void,
}

/// Samples one token per live row from its logits under its slot's record, writes it to the
/// slot's `sampled` entry and the row's `out` entry, and advances the slot's draw counter. The
/// live rows are the leading rows the device-resident `live_rows` counts: a row at or past that
/// count is left as it is, so a launch over more rows than the batch has live samples the live
/// rows and no others.
#[derive(Debug, Clone, Copy)]
pub struct SampleCall {
    /// f32 `[n_rows, vocab]`, row-major.
    pub logits: u64,
    /// i32 `[n_rows]`: the slot each live row samples under.
    pub row_slots: u64,
    /// The slot records, 24 bytes each, `[slots]`; each sampling row's is advanced.
    pub records: u64,
    /// u32 `[slots]`: written for each sampling row's slot.
    pub sampled: u64,
    /// u32 `[n_rows]`: written with the token sampled for each live row.
    pub out: u64,
    /// One u32 on the device: how many of the launch's leading rows are live and sample.
    pub live_rows: u64,
    pub vocab: usize,
    pub n_rows: usize,
    pub stream: *mut c_void,
}

#[cfg(feature = "cuda")]
mod compiled {
    use core::ffi::c_void;

    use super::{GatherCall, SampleCall};
    use crate::error::{arg_i64, KernelError};
    use crate::ffi;

    /// Enqueues the gather on `call.stream`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError`] when the row count does not fit the kernel's argument or the
    /// launch fails.
    ///
    /// # Safety
    /// Every address in `call` must be live on the stream's device and match its documented
    /// shape; every non-negative gather slot must index `sampled`.
    pub unsafe fn gather(call: &GatherCall) -> Result<(), KernelError> {
        // SAFETY: the caller's contract; the FFI returns the launch status.
        let status = unsafe {
            ffi::sampler_gather_u32(
                call.token_ids as *mut c_void,
                call.gather_slots as *const c_void,
                call.sampled as *const c_void,
                arg_i64("n_rows", call.n_rows)?,
                call.stream,
            )
        };
        ffi::check_launch("sampler_gather_u32", status)
    }

    /// Enqueues the sample on `call.stream`.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError`] when a count does not fit the kernel's argument or the launch
    /// fails.
    ///
    /// # Safety
    /// Every address in `call` must be live on the stream's device and match its documented
    /// shape; every live row's slot must index `records` and `sampled`, and no two live rows may
    /// share one.
    pub unsafe fn sample(call: &SampleCall) -> Result<(), KernelError> {
        let args = ffi::SampleArgs {
            logits: call.logits as *const c_void,
            row_slots: call.row_slots as *const c_void,
            records: call.records as *mut c_void,
            sampled: call.sampled as *mut c_void,
            out: call.out as *mut c_void,
            live_rows: call.live_rows as *const c_void,
            vocab: arg_i64("vocab", call.vocab)?,
            n_rows: arg_i64("n_rows", call.n_rows)?,
        };
        // SAFETY: the caller's contract; the FFI reads the arguments for the launch and returns
        // its status.
        let status = unsafe { ffi::sampler_sample_f32(&raw const args, call.stream) };
        ffi::check_launch("sampler_sample_f32", status)
    }
}

#[cfg(feature = "cuda")]
pub use compiled::{gather, sample};

/// Named refusal: this build carries no kernels.
///
/// # Errors
///
/// Always returns [`KernelError::NotCompiled`].
///
/// # Safety
/// Dereferences nothing; the signature matches the real launch.
#[cfg(not(feature = "cuda"))]
pub unsafe fn gather(_call: &GatherCall) -> Result<(), KernelError> {
    Err(KernelError::NotCompiled {
        kernel: "sampler_gather_u32",
    })
}

/// Named refusal: this build carries no kernels.
///
/// # Errors
///
/// Always returns [`KernelError::NotCompiled`].
///
/// # Safety
/// Dereferences nothing; the signature matches the real launch.
#[cfg(not(feature = "cuda"))]
pub unsafe fn sample(_call: &SampleCall) -> Result<(), KernelError> {
    Err(KernelError::NotCompiled {
        kernel: "sampler_sample_f32",
    })
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "cuda"))]
    use core::ptr;

    #[cfg(not(feature = "cuda"))]
    use super::*;

    /// The kernel sources, so the launcher list and the sources are held to the same names.
    const KERNEL_SOURCE: &str = include_str!("../kernels/sampler.cu");

    /// Every launcher the Rust side declares, as the sources must export it.
    const LAUNCHERS: [&str; 2] = ["sampler_sample_f32", "sampler_gather_u32"];

    /// The record as the host lays it out, field for field.
    const RECORD_FIELDS: [&str; 5] = [
        "float temperature;",
        "float top_p;",
        "uint32_t top_k;",
        "uint32_t draws;",
        "uint64_t seed;",
    ];

    /// The sample launch's arguments as the host lays them out: each field as the sources
    /// declare it, its name, and the byte offset `ffi::SampleArgs` holds it to.
    const SAMPLE_ARGS_FIELDS: [(&str, &str, usize); 8] = [
        ("const float* logits;", "logits", 0),
        ("const int32_t* row_slots;", "row_slots", 8),
        ("SlotRecord* records;", "records", 16),
        ("uint32_t* sampled;", "sampled", 24),
        ("uint32_t* out;", "out", 32),
        ("const uint32_t* live_rows;", "live_rows", 40),
        ("int64_t vocab;", "vocab", 48),
        ("int64_t n_rows;", "n_rows", 56),
    ];

    /// The fields of the struct the sources declare as `name`, in order.
    fn declared_fields(name: &str) -> Vec<&'static str> {
        let (_, rest) = KERNEL_SOURCE
            .split_once(&format!("struct {name} {{"))
            .unwrap_or_else(|| panic!("the sources declare {name}"));
        let body = rest.split_once("};").expect("the struct closes").0;
        body.lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect()
    }

    #[test]
    fn the_sources_export_every_launcher_with_a_status_and_a_stream() {
        for launcher in LAUNCHERS {
            let declaration = format!("extern \"C\" cudaError_t {launcher}(");
            let Some((_, rest)) = KERNEL_SOURCE.split_once(&declaration) else {
                panic!("sampler.cu does not export {launcher}");
            };
            let parameters = rest.split_once(')').expect("a parameter list closes").0;
            assert!(
                parameters.ends_with("cudaStream_t stream"),
                "{launcher} must take the caller's stream last, got: {parameters}"
            );
        }
    }

    #[test]
    fn no_kernel_launch_ignores_its_status() {
        let launches = KERNEL_SOURCE.matches("<<<").count();
        let checks = KERNEL_SOURCE.matches("return cudaGetLastError();").count();
        assert_eq!(launches, LAUNCHERS.len());
        assert_eq!(checks, LAUNCHERS.len());
    }

    #[test]
    fn the_record_and_the_arguments_are_declared_field_for_field_at_the_hosts_offsets() {
        assert_eq!(declared_fields("SlotRecord"), RECORD_FIELDS);
        assert!(
            KERNEL_SOURCE.contains("static_assert(sizeof(SlotRecord) == 24"),
            "the sources hold the record to its size"
        );
        assert_eq!(
            declared_fields("SampleArgs"),
            SAMPLE_ARGS_FIELDS.map(|(declaration, _, _)| declaration)
        );
        assert!(
            KERNEL_SOURCE.contains("static_assert(sizeof(SampleArgs) == 64"),
            "the sources hold the arguments to their size"
        );
        // Each side is held to the same absolute offsets, so a reordering of either alone stops
        // that side's build; the names and the order alone would not catch one.
        for (_, field, offset) in SAMPLE_ARGS_FIELDS {
            let held = format!("static_assert(offsetof(SampleArgs, {field}) == {offset}");
            assert!(
                KERNEL_SOURCE.contains(&held),
                "the sources hold {field} to offset {offset}"
            );
        }
    }

    #[test]
    fn the_philox_constants_are_random123s() {
        for constant in ["0xD2511F53u", "0xCD9E8D57u", "0x9E3779B9u", "0xBB67AE85u"] {
            assert!(KERNEL_SOURCE.contains(constant), "{constant}");
        }
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn a_build_without_kernels_refuses_each_by_name() {
        let stream = ptr::null_mut();
        // SAFETY: the stubs dereference nothing.
        let refusals = unsafe {
            [
                sample(&SampleCall {
                    logits: 0,
                    row_slots: 0,
                    records: 0,
                    sampled: 0,
                    out: 0,
                    live_rows: 0,
                    vocab: 8,
                    n_rows: 1,
                    stream,
                })
                .unwrap_err(),
                gather(&GatherCall {
                    token_ids: 0,
                    gather_slots: 0,
                    sampled: 0,
                    n_rows: 1,
                    stream,
                })
                .unwrap_err(),
            ]
        };
        for (refusal, launcher) in refusals.iter().zip(LAUNCHERS) {
            assert_eq!(refusal, &KernelError::NotCompiled { kernel: launcher });
        }
    }
}
