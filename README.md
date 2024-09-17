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
2. Clone your forked repository: `git clone https://github.com/your-username/atoma-paged-attention.git`
3. Install Rust: Follow the instructions at [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)
4. Navigate to the project directory: `cd atoma-paged-attention`
5. Build the project: `cargo build --release`
6. Run tests: `cargo test`

For more detailed instructions, please refer to our [documentation](docs/README.md).

## Contributing

We welcome contributions from both the ML and Rust communities. Here's how you can contribute:

1. **Report bugs**: If you find a bug, please open an issue in our GitHub repository with a clear description and steps to reproduce.

2. **Suggest enhancements**: Have ideas for improvements? Open an issue to discuss your proposal.

3. **Submit pull requests**: Ready to contribute code? Here's the process:
   - Fork the repository
   - Create a new branch for your feature: `git checkout -b feature-name`
   - Make your changes and commit them: `git commit -m 'Add some feature'`
   - Push to the branch: `git push origin feature-name`
   - Submit a pull request through GitHub

4. **Improve documentation**: Help us enhance our documentation by fixing typos, clarifying explanations, or adding examples.

5. **Share knowledge**: Participate in discussions, answer questions, and share your expertise with the community.
