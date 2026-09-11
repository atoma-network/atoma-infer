#!/usr/bin/env bash
#
# The correctness gates over the full bucket ladder on a CUDA rig: runs the ignored integration
# tests that capture every bucket the decode step serves and report the capture matrix, hold the
# replay of each bucket's graph to the same decode step run eagerly bit for bit over decode steps
# of varying ids, lengths and block tables, read every baked address again after each replay and
# across a thousand replays that must leave free memory where they found it, and hold the greedy
# arena layout to the no-reuse reference bit for bit, with the gates shown to bite over a role
# table that declares one lifetime one op short.
#
# Usage, from a checkout on the rig with a CUDA 12.x toolkit on PATH:
#   HF_TOKEN=... scripts/ladder-gates.sh [model-id]
#
# The model defaults to Llama 3.2 1B Instruct; any Llama loadable in bf16 that fits the device
# works. LADDER_GATE_MAX_BATCH cuts the Hopper bucket ladder at a maximum batch, 128 unless set
# and 512 for the whole ladder, and must be a bucket of that ladder; LADDER_GATE_STEPS raises the
# identity steps, which are never fewer than 32 or twice the bucket count. The flash-attention
# kernels are a long nvcc build the first time; set FLASH_ATTN_BUILD_DIR to keep the build across
# checkouts.
#
# Both gates open the device, load the checkpoint and hold free device memory still around their
# steps, so they run one at a time: run concurrently, each would read the other's allocations as
# its own leak.
#
# The tests print two evidence blocks, the bucket ladder gates' and the arena layout gates';
# paste both into the pull request.

set -euo pipefail

export CARGO_TERM_COLOR=never

die() {
	echo "ladder-gates: error: $*" >&2
	exit 1
}

command -v nvidia-smi >/dev/null || die "nvidia-smi not found — NVIDIA driver is not installed"
nvidia-smi >/dev/null || die "nvidia-smi failed — no visible GPU"
command -v nvcc >/dev/null || die "nvcc not on PATH — install a CUDA 12.x toolkit"
command -v cargo >/dev/null ||
	die "cargo not found — install rustup; rust-toolchain.toml pins the version"
[[ -n ${HF_TOKEN:-} ]] ||
	echo "ladder-gates: HF_TOKEN is unset; a gated checkpoint will fail to fetch" >&2

cutlass_header=crates/atoma-kernels/cutlass/include/cutlass/cutlass.h
if [[ ! -f $cutlass_header ]]; then
	echo "==> CUTLASS submodule is empty; fetching"
	git submodule update --init --depth 1 crates/atoma-kernels/cutlass
fi

export LADDER_GATE_MODEL="${1:-${LADDER_GATE_MODEL:-unsloth/Llama-3.2-1B-Instruct}}"
echo "==> model: $LADDER_GATE_MODEL"
echo "==> maximum batch: ${LADDER_GATE_MAX_BATCH:-default}"
echo "==> identity steps: ${LADDER_GATE_STEPS:-default}"
echo "==> commit: $(git rev-parse HEAD)"
echo "==> gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"

# test-support is what lets the layout gate state a role table that is not the model's, and one
# test thread is what keeps the two gates off each other's device readings.
cargo test -p atoma-engine --locked --features cuda,test-support --test ladder_gates -- \
	--ignored --nocapture --test-threads=1
