#!/usr/bin/env bash
#
# Decode parity, the capture of the bucket ladder and its replay on a CUDA rig: runs the ignored
# integration test that builds the decode step over runtime tensors beside the candle forward on
# the same weights and KV cache, captures every bucket into a graph, compares the eager step's
# logits with candle's over decode steps of varying ids, lengths and block tables, compares each
# replay's sampled token and cache writes with the eager step's, and reads free device memory
# around every step and across a soak of replays.
#
# Usage, from a checkout on the rig with a CUDA 12.x toolkit on PATH:
#   HF_TOKEN=... scripts/decode-parity.sh [model-id]
#
# The model defaults to the Llama 3.1 8B Instruct checkpoint; any Llama loadable in bf16 that
# fits the device works. The flash-attention kernels are a long nvcc build the first time; set
# FLASH_ATTN_BUILD_DIR to keep the build across checkouts.
#
# The test prints an evidence block with the argmax agreement, the largest absolute difference
# on the f32 logits, what capturing the bucket ladder cost, and the replay's agreement with the
# eager step; paste it into the pull request.

set -euo pipefail

export CARGO_TERM_COLOR=never

die() {
	echo "decode-parity: error: $*" >&2
	exit 1
}

command -v nvidia-smi >/dev/null || die "nvidia-smi not found — NVIDIA driver is not installed"
nvidia-smi >/dev/null || die "nvidia-smi failed — no visible GPU"
command -v nvcc >/dev/null || die "nvcc not on PATH — install a CUDA 12.x toolkit"
command -v cargo >/dev/null || die "cargo not found — install rustup; rust-toolchain.toml pins the version"
[[ -n ${HF_TOKEN:-} ]] || echo "decode-parity: HF_TOKEN is unset; a gated checkpoint will fail to fetch" >&2

cutlass_header=crates/atoma-kernels/cutlass/include/cutlass/cutlass.h
if [[ ! -f $cutlass_header ]]; then
	echo "==> CUTLASS submodule is empty; fetching"
	git submodule update --init --depth 1 crates/atoma-kernels/cutlass
fi

export PARITY_MODEL="${1:-${PARITY_MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}}"
echo "==> model: $PARITY_MODEL"
echo "==> commit: $(git rev-parse HEAD)"
echo "==> gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"

cargo test -p atoma-engine --locked --features cuda --test decode_parity -- --ignored --nocapture
