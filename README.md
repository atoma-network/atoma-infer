# Atoma Paged Attention

![Atoma Logo](https://github.com/atoma-network/atoma-paged-attention/blob/main/assets/atoma-symbol.jpg)

The Atoma Node Inference repository is a collection of optimized infrastructure for Large Language Models (LLMs) compute. We rely on highly optimized KV cache memory management software, through block pagination, such as
[PagedAttention](https://github.com/vllm-project/vllm) and [FlashAttention2](https://github.com/Dao-AILab/flash-attention). The codebase is mostly written in the Rust programming language, which leads to safe and highly optimized inference request scheduling, enhancing LLM inference serving.

## Features

- Implements Paged Attention for efficient KV cache management
- Supports Llama3.1 models
- Optimized for inference serving in distributed systems
- Integrates with the Candle ML framework for high-performance Rust-based machine learning
- Scalable architecture for handling multiple concurrent requests
- Efficient memory management for improved performance

## Paged Attention

Paged Attention is an innovative technique for managing KV cache memory in LLMs. It significantly improves inference efficiency, especially for long-context scenarios. For more details, see the [original paper](https://arxiv.org/pdf/2309.06180).

## Flash Attention 2

Flash Attention 2 is a highly optimized algorithm for efficient attention computation in transformers. It mostly relies on the observation that writing in and out of HBM GPU memory presents the main bottleneck to compute attention efficiently. The latter is especially relevant when computing attention softmax intermediate values. The algorithm exploits custom CUDA kernels that minimize memory HBM writes, by computing the attention scores in a block-wise fashion, requiring only shared memory reads and writes. For more details, see the [original paper](https://arxiv.org/abs/2307.08691).

## Integration with Candle

This project leverages [Candle](https://github.com/huggingface/candle), HuggingFace's Rust-based ML framework. Candle offers several advantages:

- Blazing fast performance of Rust
- Memory safety guarantees
- Seamless integration with AI inference distributed systems
- Fully open-source and community-driven development

## Getting Started

1. Fork and star the repository.
2. Clone your forked repository: `git clone https://github.com/your-username/atoma-node-inference.git`
3. Install Rust: Follow the instructions at [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)
4. Navigate to the project directory: `cd atoma-node-inference`
5. Initialize the git submodules: `git submodule init` and then `git pull --recurse-submodules`
6. Setup env variables: `cp .env.development .env` (development) or `cp .env.production` (production)
7. Build the project: `cargo build --release`
8. Run tests: `cargo test`

For more detailed instructions, please refer to our [documentation](docs/README.md).

## Contributing

### General guidance for your PR

Under no circumstances should a single PR mix different purposes: Your
PR is either a bug fix, a new feature, or a performance improvement,
never a combination. Nor should you include, for example, two
unrelated performance improvements in one PR. Please just submit
separate PRs. The goal is to make reviewing your PR as simple as
possible, and you should be thinking about how to compose the PR to
minimise the burden on the reviewer.

Here are a few specific guidelines for the three main categories of
PRs that we expect:

#### The PR fixes a bug

In the PR description, please clearly but briefly describe

1. the bug (could be a reference to a GH issue; if it is from a
   discussion (on Discord/email/etc. for example), please copy in the
   relevant parts of the discussion);
2. what turned out to the cause the bug; and
3. how the PR fixes the bug.

Wherever possible, PRs that fix bugs should include additional tests
that (i) trigger the original bug and (ii) pass after applying the PR.

#### The PR implements a new feature

In the PR description, please clearly but briefly describe

1. what the feature does
2. the approach taken to implement it

All PRs for new features must include a suitable test suite.

#### The PR improves performance

Performance improvements are particularly welcome!

1. The target bottleneck (only one per PR to avoid confusing things!)
2. How performance is measured
3. Characteristics of the machine used (CPU, OS, GPU, etc.)
4. Performance gains in terms of speedups and memory usage (e.g. 2x speedup and 50% memory reduction).

#### Report bugs

If you find a bug, please open an issue in our GitHub repository with a clear description and steps to reproduce.

#### Documentation

Help us enhance our documentation by fixing typos, clarifying explanations, or adding examples.

#### Share knowledge

Participate in discussions, answer questions, and share your expertise with the community.
