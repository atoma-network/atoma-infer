#!/usr/bin/env bash
#
# The per-step copy-in on a CUDA rig: runs the ignored integration tests that stage a step's
# seven arrays into a staging entry's pinned block, copy the bucket's packed length in through
# the staging entry's fence, read the device block back and compare every array; do the same
# for a second step through the other staging entry and for a dummy run through the first;
# ask a staging entry's fence, eight times over, while the copy that reads it is still in
# flight, which it must answer not passed every time; and time `acquire` over a thousand
# staged copy-ins the host runs ahead of, printing the max and the p99.
#
# Needs a device and a CUDA 12.x toolkit; no checkpoint and no model.
#
# Usage, from a checkout on the rig:
#   scripts/copy-in.sh
#
# COPY_IN_ACQUIRE_P99_MICROS sets the p99 the timed acquires must stay under, in microseconds;
# it defaults to 20. The flash-attention kernels are a long nvcc build the first time; set
# FLASH_ATTN_BUILD_DIR to keep the build across checkouts.

set -euo pipefail

export CARGO_TERM_COLOR=never

die() {
	echo "copy-in: error: $*" >&2
	exit 1
}

command -v nvidia-smi >/dev/null || die "nvidia-smi not found — NVIDIA driver is not installed"
nvidia-smi >/dev/null || die "nvidia-smi failed — no visible GPU"
command -v nvcc >/dev/null || die "nvcc not on PATH — install a CUDA 12.x toolkit"
command -v cargo >/dev/null || die "cargo not found — install rustup; rust-toolchain.toml pins it"

cutlass_header=crates/atoma-kernels/cutlass/include/cutlass/cutlass.h
if [[ ! -f $cutlass_header ]]; then
	echo "==> CUTLASS submodule is empty; fetching"
	git submodule update --init --depth 1 crates/atoma-kernels/cutlass
fi

echo "==> commit: $(git rev-parse HEAD)"
echo "==> gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"

# One test at a time: all three open the device, and the acquire timing is a latency the other
# tests would inflate by running beside it.
cargo test -p atoma-engine --locked --features cuda --test copy_in -- \
	--ignored --nocapture --test-threads=1
