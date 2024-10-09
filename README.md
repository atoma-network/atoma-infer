# Atoma Paged Attention

<img src="https://github.com/atoma-network/atoma-node-inference/blob/readm-updates/assets/atoma-pfp.jpg" alt="Logo" height="500"/>

The Atoma Node Inference repository is a collection of optimized infrastructure for Large Language Models (LLMs) compute. We rely on highly optimized KV cache memory management software, through block pagination, such as 
[PagedAttention](https://github.com/vllm-project/vllm) and [FlashAttention2](https://github.com/Dao-AILab/flash-attention). The codebase is mostly written in the Rust programming language, which leads to safe and highly optimized inference request scheduling, enhancing LLM inference serving.

## Features

- Fully compatible OpenAI API, providing a seamless experience for developers to serve open source LLM models.
- Supports Paged Attention for efficient KV cache management (see [paper](https://arxiv.org/pdf/2309.06180)).
- Supports FlashAttention2 for efficient attention computation, by minimizing HBM writes (see [paper](https://arxiv.org/abs/2307.08691)).
- Supports Llama3.1 and Llama3.2 models text generation models.
- Optimized for serverless inference serving.
- Supports multi-GPU Tensor parallelism inference, using multiple NVIDIA's GPU devices, by leveraging Cuda's NCCL library. This allows for running any LLM, provided the user's machine has enough GPU cards.
- The repository is mainly written in Rust and it integrates with the Candle ML framework for high-performance Rust-based LLM inference, making it ideal to deploy in serverless environments.
- Avoids dependencies of very large Machine Learning frameworks such as PyTorch. Our repository can be deployed through lightweight binaries.
- Avoids Python overhead from production workloads. 

## Status

This project is currently in the early stages of development. We are looking for open source contributors to help us expand the reach of the current project. Currently, we support:

- [x] - Fully compatible OpenAI API.
- [x] - Paged Attention for efficient KV cache management.
- [x] - FlashAttention2 for efficient attention computation.
- [x] - Llama3.1 and Llama3.2 models, of any size. 
- [x] - Multi-GPU Tensor parallelism inference, using NCCL.
- [x] - Highly-optimized inference request scheduler.
- [x] - Continuous batching of inference requests.
- [x] - CPU offloading of inference requests, through CPU/GPU request swapping.
- [x] - Streaming responses through SSE.
- [ ] - Support for different quantization techniques.
- [ ] - FlashAttention3.
- [x] - Utoipa OpenAPI documentation.
- [ ] - Support for multi-modal models.
- [ ] - Support backends other than vLLM.
- [ ] - Prompt caching.
- [ ] - Speculative decoding.
- [ ] - Parallel sampling for multiple completions.
- [ ] - Support for asymmetric signature server authentication.

## Getting Started

1. Fork and star the repository.
2. Clone your forked repository: `git clone https://github.com/your-username/atoma-infer.git`
3. Install Rust: Follow the instructions at [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)
4. Navigate to the project directory: `cd atoma-infer`
5. Initialize the git submodules: `git submodule init` and then `git pull --recurse-submodules`
6. Setup env variables: `cp .env.development .env` (development) or `cp .env.production` (production)
7. Build the project with the vLLM backend (required): `cargo build --release --features vllm`
8. Run tests: `cargo test --features vllm`
9. Build the project with multi-GPU inference support (optional): `cargo build --release --features nccl`

## Start the OpenAI-compatible jRPC server

In order to start the OpenAI-compatible jRPC server, and start listening for new inference requests, start by setting up a configuration file. An example of such configuration file is the following:

```toml
[inference]
api_key = "YOUR_HUGGINGFACE_API_KEY" # Your HuggingFace API key
cache_dir = "CACHE_DIRECTORY_FOR_STORING_MODEL_WEIGHTS" # Directory to store the model weights
flush_storage = true # Whether to flush the storage after the model has been loaded
model_name = "HUGGING_FACE_MODEL_ID" # HuggingFace model ID, e.g. "meta-llama/Llama-3.1-405B-Instruct"
device_ids = [0] # List of GPU IDs to use, if you have multiple GPUs, please provide the list of IDs available (e.g. [0, 1, 2, 3, ...])
dtype = "bf16" # Data type to use for inference (we recommend using either bf16 or f16)
num_tokenizer_workers = 4 # Number of workers to use for tokenizing incoming inference requests
revision = "main" # Revision of the model to use, e.g. "main" or "refs/tags/v1.0"

[cache]
block_size = 16 # Block size to use for the vLLM cache memory management
cache_dtype = "bf16" # Most often, it agrees with inference.dtype
gpu_memory_utilization = 0.5 # Fraction of the GPU memory to use for storing the KV cache
swap_space_fraction = 0.1 # Fraction of the GPU memory to use for storing the KV cache during inference

[scheduler]
max_num_batched_tokens = 1024 # Maximum number of total batched tokens to use for the vLLM scheduler
max_num_sequences = 32 # Maximum number of batched sequences that the vLLM scheduler should handle
max_model_len = 1024 # Maximum length of a model sequence can have to use for the vLLM scheduler
delay_factor = 0.0 # Delay factor to use for the vLLM scheduler
enable_chunked_prefill = false # Whether to use the chunked prefill feature for the vLLM scheduler
block_size = 16 # Block size to use for the vLLM cache memory management

[validation]
best_of = 1 # Best of n value to use for the vLLM scheduler
max_stop_sequences = 1 # Maximum number of stop sequences to use for the vLLM scheduler
max_top_n_tokens = 1 # Maximum number of top n tokens to use for the vLLM scheduler
max_input_length = 4096 # Maximum input length to use for the vLLM scheduler  
max_total_tokens = 8192 # Maximum total tokens to use for the vLLM scheduler
```

1. Start the OpenAI-compatible jRPC server: 

- In development:

`$ RUST_LOG=info cargo run --features vllm -- --config_path CONFIGURATION_FILE_PATH` 

- In production:

`$ RUST_LOG=info cargo run --release --features vllm -- --config_path CONFIGURATION_FILE_PATH`.

2. If multi-GPU inference support is enabled, you can start the server with NCCL support: 

- In development:

`$ RUST_LOG=info cargo run --features nccl -- --config_path CONFIGURATION_FILE_PATH`.

- In production:

`$ RUST_LOG=info cargo run --release --features nccl -- --config_path CONFIGURATION_FILE_PATH`.

## Build considerations

1. When building the project with NCCL support, ensure that the CUDA toolkit version is compatible with the NVIDIA driver version. You can check your CUDA version by running `nvcc --version`.
2. Current flash-attention2 version is only compatible with sm_8x/sm_90 GPUs or newer.
3. Nccl is currently not supported on Windows.

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
