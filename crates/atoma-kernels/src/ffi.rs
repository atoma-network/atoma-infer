use crate::error::KernelError;
use core::ffi::{c_char, c_int, c_void};
use std::ffi::CStr;
use std::mem::{offset_of, size_of};

/// The sample launch's arguments, as `kernels/sampler.cu` lays them out.
#[repr(C)]
pub(crate) struct SampleArgs {
    /// f32 `[n_rows, vocab]`, row-major.
    pub(crate) logits: *const c_void,
    /// i32 `[n_rows]`: the slot each row samples under.
    pub(crate) row_slots: *const c_void,
    /// The slot records, 24 bytes each, `[slots]`.
    pub(crate) records: *mut c_void,
    /// u32 `[slots]`: the token last sampled for each slot.
    pub(crate) sampled: *mut c_void,
    /// u32 `[n_rows]`: the token sampled for each row.
    pub(crate) out: *mut c_void,
    /// One u32: how many of the launch's leading rows are live and sample.
    pub(crate) live_rows: *const c_void,
    pub(crate) vocab: i64,
    pub(crate) n_rows: i64,
}

/// The size the sources assert for the arguments.
const SAMPLE_ARGS_BYTES: usize = 64;

/// The layout the sources declare, checked at compile time: the size the launch reads, and every
/// field at the byte the sources hold it to, so a reordering on this side stops the build rather
/// than handing the kernel one argument as another.
const _: () = {
    assert!(size_of::<SampleArgs>() == SAMPLE_ARGS_BYTES);
    assert!(offset_of!(SampleArgs, logits) == 0);
    assert!(offset_of!(SampleArgs, row_slots) == 8);
    assert!(offset_of!(SampleArgs, records) == 16);
    assert!(offset_of!(SampleArgs, sampled) == 24);
    assert!(offset_of!(SampleArgs, out) == 32);
    assert!(offset_of!(SampleArgs, live_rows) == 40);
    assert!(offset_of!(SampleArgs, vocab) == 48);
    assert!(offset_of!(SampleArgs, n_rows) == 56);
};

extern "C" {
    /// Records any failure for [`flash_last_error`] rather than returning it: the vendored dispatch
    /// templates this walks return nothing, and threading a status through them would fork all 66
    /// kernel instantiation files.
    pub(crate) fn run_mha(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        alibi_slopes_ptr: *const c_void,

        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,

        is_seqlens_k_cumulative: bool,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,
        alibi_slopes_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,

        num_splits: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,
        scale_softmax_log2: f32,

        block_table: *const c_int,
        block_table_batch_stride: u32,
        page_block_size: c_int,

        seqused_k: *const c_int,
        seqlen_q: u32,
        seqlen_k: u32,
        total_q: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,

        is_bf16: c_int,
        is_causal: c_int,

        window_size_left: c_int,
        window_size_right: c_int,
        softcap: f32,
        unpadded_lse: bool,
        force_split_kernel: bool,

        softmax_lseaccum_ptr: *const c_void,
        oaccum_ptr: *const c_void,

        stream: *mut c_void,
    );

    /// The `cudaError_t` recorded during the most recent [`run_mha`].
    pub(crate) fn flash_last_error() -> c_int;

    /// Returns the `cudaError_t` of the launch.
    pub(crate) fn copy_blocks_cache(
        cache: *mut c_void,
        block_mapping: *const c_void,
        num_pairs: i64,
        numel_per_block: i64,
        stream: *mut c_void,
    ) -> c_int;

    /// Returns the `cudaError_t` of the launch.
    pub(crate) fn reshape_and_cache_flash_cache(
        source: *const c_void,
        cache: *mut c_void,
        slot_mapping: *const i64,
        block_stride: i64,
        num_tokens: i64,
        num_heads: i64,
        head_size: i64,
        block_size: i64,
        source_stride: i64,
        dtype: u32,
        stream: *mut c_void,
    ) -> c_int;

    /// Returns the `cudaError_t` of the launch.
    pub(crate) fn decode_embedding_gather_bf16(
        table: *const c_void,
        token_ids: *const c_void,
        out: *mut c_void,
        hidden: i64,
        n_tokens: i64,
        stream: *mut c_void,
    ) -> c_int;

    /// Returns the `cudaError_t` of the launch.
    pub(crate) fn decode_rmsnorm_bf16(
        x: *const c_void,
        gain: *const c_void,
        out: *mut c_void,
        hidden: i64,
        n_tokens: i64,
        eps: f32,
        stream: *mut c_void,
    ) -> c_int;

    /// Returns the `cudaError_t` of the launch.
    pub(crate) fn decode_rope_bf16(
        qkv: *mut c_void,
        positions: *const c_void,
        cos_table: *const c_void,
        sin_table: *const c_void,
        n_tokens: i64,
        rot_heads: i64,
        head_dim: i64,
        row_width: i64,
        stream: *mut c_void,
    ) -> c_int;

    /// Returns the `cudaError_t` of the launch.
    pub(crate) fn decode_silu_mul_bf16(
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        len: i64,
        stream: *mut c_void,
    ) -> c_int;

    /// Returns the `cudaError_t` of the launch.
    pub(crate) fn decode_add_bf16(
        lhs: *const c_void,
        rhs: *const c_void,
        out: *mut c_void,
        len: i64,
        stream: *mut c_void,
    ) -> c_int;

    /// Returns the `cudaError_t` of the launch.
    pub(crate) fn sampler_sample_f32(args: *const SampleArgs, stream: *mut c_void) -> c_int;

    /// Returns the `cudaError_t` of the launch.
    pub(crate) fn sampler_gather_u32(
        token_ids: *mut c_void,
        gather_slots: *const c_void,
        sampled: *const c_void,
        n_rows: i64,
        stream: *mut c_void,
    ) -> c_int;

    /// `cudaGetErrorString` for a `cudaError_t` value, as a static NUL-terminated string.
    fn flash_cuda_error_string(code: c_int) -> *const c_char;
}

/// Turns the status returned by an FFI launcher into a typed error.
///
/// A launch is asynchronous, so a success here means the kernel was accepted by the driver, not
/// that it has run; faults raised during execution surface on a later synchronization.
///
/// # Arguments
///
/// * `kernel` - Name of the FFI entry point, used to identify the failure.
/// * `status` - The `cudaError_t` value the entry point returned.
pub(crate) fn check_launch(kernel: &'static str, status: c_int) -> Result<(), KernelError> {
    if status == 0 {
        return Ok(());
    }
    // SAFETY: `cudaGetErrorString` returns a pointer to a static string for every input, including
    // unrecognized error codes.
    let message = unsafe { CStr::from_ptr(flash_cuda_error_string(status)) }
        .to_string_lossy()
        .into_owned();
    Err(KernelError::LaunchFailed {
        kernel,
        code: status,
        message,
    })
}
