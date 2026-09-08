# atoma-infer

<img src="assets/atoma-cover.png" alt="Atoma Logo" width="800"/>

`atoma-infer` is a Rust and CUDA project for large-language-model inference.

> **Learn more about Atoma:** Visit [atoma.ai](https://atoma.ai/) for information about Atoma's secure AI infrastructure platform.

## Status

The current implementation is under revival and is **not production-ready**. The repository does not yet have a verified build, test, or serving baseline. OpenAI API compatibility, model support, and single-node or distributed GPU topologies are not verified capabilities of this checkout. Do not rely on it for production workloads.

Rung 0 will restore a trustworthy build and test baseline before feature or performance claims are reintroduced. See the canonical [rung-0 specification](https://github.com/AtomaAI/atoma-infer/issues/147) for its bounded scope.

## Launch-gate parity target

Launch-gate parity is a future measured target, not a description of the current implementation. The gate requires DeepSeek-class goodput within 10% of the better of vLLM or SGLang, plus at least two outright headline benchmark wins. Measurements cover 2×8×H100 with FP8 and 8×B200 with NVFP4, in aggregated and prefill/decode-disaggregated topologies.

The canonical [revival decision map](https://github.com/AtomaAI/atoma-infer/issues/138) records the decisions and measurement destination. The [rung-0 specification](https://github.com/AtomaAI/atoma-infer/issues/147) defines the initial recovery work. These GitHub issues are the public sources of truth for the revival plan.

## Contributor setup

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/atoma-infer.git`.
3. Enter the checkout: `cd atoma-infer`.
4. Install Rust using [rustup](https://www.rust-lang.org/tools/install). The repository's toolchain file selects the required Rust version.
5. Initialize dependencies: `git submodule update --init --recursive`.

Run the GPU-less workspace checks with:

```shell
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

Install [prek](https://prek.j178.dev/), then install and run the repository hooks with `prek install` and `prek run --all-files`.

## CUDA builds

The `cuda` feature compiles the flash-attention and cache-manager kernels in `crates/atoma-kernels`, which include CUTLASS headers, and the decode step's own kernels beside them. CUTLASS is a Git submodule at `crates/atoma-kernels/cutlass` and is not part of a plain `git clone`. Step 5 of the contributor setup checks it out; in an existing checkout, or in CI that clones without submodules, fetch it with:

```shell
git submodule update --init --depth 1 crates/atoma-kernels/cutlass
```

The build fails with missing `cutlass/...` headers if the submodule directory is empty.

Building the `cuda` feature requires a CUDA toolkit providing `nvcc`, but not a GPU — `cargo check --workspace --features cuda` succeeds on a machine with no NVIDIA device. Without a visible GPU the build script cannot probe the compute capability, so set it explicitly for the target architecture:

```shell
CUDA_COMPUTE_CAP=80 cargo check --workspace --features cuda
```

Use a CUDA 12.x toolkit. The workspace pins `cudarc`'s `cuda-12000` feature as its API baseline, and 12.x is the range GPU verification runs against; newer toolkits are unverified. NVIDIA's `cuda-toolkit` metapackage currently installs 13.3, so request a specific version instead — for example `cuda-toolkit-12-9` on Ubuntu 24.04.

Compiling the kernels takes considerably longer than the Rust build. Set `FLASH_ATTN_BUILD_DIR` to an absolute path outside `target/` to cache the compiled kernel archive across builds:

```shell
export FLASH_ATTN_BUILD_DIR="$HOME/.cache/atoma-flash-attn-build"
```

Multi-GPU builds additionally enable `nccl`, which is compile-checked with `cargo check --workspace --features cuda,nccl`. Running tensor-parallel inference is not verified in this checkout.

## GPU verification

GPU verification is manual by design: no GPU CI runner is registered, and GPU hardware is rented
for scheduled stints rather than kept standing. `scripts/gpu-verify.sh` is the entry point — run it from a
checkout on a CUDA rig. It preflights the machine (driver, CUDA toolkit, NCCL, OpenSSL build
dependencies, CUTLASS submodule), builds the `cuda` feature, runs the `cuda` and `cuda,nccl` test
suites without fail-fast, runs clippy over all features, and prints an evidence block to paste
into the tracking ticket.

`scripts/decode-parity.sh` runs the decode step over runtime-owned tensors beside the candle
forward on the same weights and KV cache, records the step under capture, and compares the two
forwards' logits over decode steps of varying ids, lengths and block tables. It needs a device, the
toolkit and a Llama checkpoint loadable in bf16, and prints its own evidence block.

`scripts/sampler-parity.sh` runs the device sampler over synthetic logits against the host
reference it is written to: every row's token, a seeded request's tokens across batches, rows and
slots, the draw frequencies against the distribution the filters leave, and the gather of a
decoding row's token from its slot. It needs a device and the toolkit, no checkpoint and no model,
and prints its own evidence block.

`scripts/copy-in.sh` runs the per-step copy-in: a step's eight arrays staged into a staging
entry's pinned block and copied in with one copy, the device block read back and every array
compared, the same for a second step through the other staging entry and for a dummy run, a
staging entry's fence asked eight times over while the copy that reads it is still in flight
and reading not passed every time, and `acquire` timed over a thousand copy-ins the host runs
ahead of. It needs a device and the toolkit, no checkpoint and no model, and prints the fence
and acquire evidence blocks.

## Benchmarks

`crates/bench` holds the benchmark harness (`atoma-bench`). It offers an open-loop Poisson workload
over the OpenAI surface, records TTFT, inter-token and end-to-end latency as hdrhistogram
distributions, reports goodput at a fixed SLO, samples the engine's available-KV-block gauge for
leaks, and drives a pinned vLLM baseline over the same workload on the same host for comparison. Copy
`crates/bench/bench.example.toml`, then run `cargo run --release -p atoma-bench -- --help`. The
protocol and the procedure are in
[docs/benchmarks/rung0-baseline.md](docs/benchmarks/rung0-baseline.md), and the tables it produced
on one H100 PCIe against vLLM `v0.26.0` are
[docs/benchmarks/rung0-baseline-table.md](docs/benchmarks/rung0-baseline-table.md) (ShareGPT) and
[docs/benchmarks/rung0-baseline-table-long-context.md](docs/benchmarks/rung0-baseline-table-long-context.md)
(8k input tokens).

## Configuration and running

The example configuration at the repository root holds everything the server is built from: the
engine, the executor's ranks, the model and the server itself. Copy it and set the model and the
chat template it is served under, the device and core of each rank, and where to listen:

```shell
cp config.example.toml config.toml
```

`config.toml` is ignored by Git. Every field can be overridden by an environment variable under
`ATOMA_`, nesting with `__`, and a Hugging Face token is read from `HF_TOKEN`. The prefix is the
configuration's alone: a variable carrying it that names no field refuses the whole configuration,
so a build-time setting such as `FLASH_ATTN_BUILD_DIR` stays outside it. The server requires
the opt-in `cuda` feature, `nccl` as well for more than one rank, and accepts the configuration
path through `--config-path`:

```shell
RUST_LOG=info cargo run --release --features cuda --bin atoma-api -- --config-path config.toml
```

CUDA compilation and live GPU execution remain unverified until the remaining rung-0 CUDA checkpoints are complete.

## Contributing

Keep each pull request focused on one purpose, such as one bug fix, feature, or performance improvement. Unrelated changes belong in separate pull requests so each change can be reviewed independently.

A narrow exception applies to canonical roadmap integration pull requests. Such a pull request may integrate multiple planned changes only when it identifies their canonical tickets and preserves reviewable commit ranges for each ticket. This exception does not apply to unrelated cleanup or opportunistic changes.

Pull request descriptions should state the problem, the chosen approach, and the verification performed. Bug fixes and features should include tests at a public behavior seam. Performance changes should identify one bottleneck, describe the benchmark and hardware, and report speed and memory results.

## License

Licensed under the [Apache License 2.0](LICENSE).
