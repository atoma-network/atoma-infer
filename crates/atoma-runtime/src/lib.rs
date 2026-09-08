//! CUDA graph capture substrate: context, streams, capture and graph lifetime, and the capture
//! arena.
//!
//! This crate owns device execution — the CUDA context, stream topology, graph capture, graph
//! lifetime, the staging fence the host waits on before writing a staging entry again, and the
//! arena from which every captured step's activations are addressed. It knows nothing about
//! models, attention, or kernels; the layer whose allocation-freedom must be provable stays small
//! enough to prove.
//!
//! The crate links cudarc unconditionally under the workspace's `fallback-dynamic-loading` pin, so
//! it compiles, links, and runs `cargo test` on a machine with no CUDA toolkit, driver, or GPU.
//! Only paths that call the driver require one, and they fail loudly at the call.
//!
//! Layout rule: `#[repr(C)]` (or `#[repr(transparent)]`) is required for exactly the types whose
//! bytes cross the driver or device boundary — kernel-argument structs, anything serialized into
//! a device buffer. No such type exists in this crate yet: every struct the driver reads or
//! writes is cudarc's bindgen-generated type, buffers are opaque `CudaSlice<u8>` whose contents
//! the caller formats, and the load-bearing field order in [`graph_entry::GraphEntry`] is drop
//! order — a source-declaration guarantee independent of memory layout — so default `repr(Rust)`
//! is correct for every type defined here.
//!
//! Module map:
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`context`] | Device context: construction, global event-tracking disable, free memory, loud no-driver failure |
//! | [`stream`] | The dedicated capture stream, whose surface has no synchronize and no allocate |
//! | [`fence`] | The staging fence: one event, signaled through the seam, waited on with or without blocking |
//! | [`capture`] | Capture lifecycle, end-capture instantiate/discard, the captured graph |
//! | [`graph_entry`] | Graph-lifetime ownership with load-bearing teardown order |
//! | [`session`] | The capture session: phase-typed Allocation, Capture and Replay of one graph set |
//! | `communicator` | The NCCL communicator behind a surface that reaches no stream (`nccl` feature only) |
//! | [`arena`] | Activation addresses as a pure function of (bucket, layer, role) |
//! | [`tensor`] | A tensor as a view over runtime-owned device memory: address, shape, strides, dtype |
//! | [`error`] | Driver statuses classified into named errors with remediation text |

pub mod arena;
pub mod capture;
#[cfg(feature = "nccl")]
pub mod communicator;
pub mod context;
pub mod error;
pub mod fence;
pub mod graph_entry;
pub mod session;
pub mod stream;
pub mod tensor;
