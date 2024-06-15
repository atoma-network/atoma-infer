# Atoma Paged Attention

![Atoma Logo](https://github.com/atoma-network/atoma-paged-attention/blob/main/assets/atoma-symbol.jpg)

A collection of Large Language Models (LLMs) optimized for KV cache memory management, through Paged Attention see [here](https://arxiv.org/pdf/2309.06180). In particular, Paged Attention allows for optimized inference serving, which is crucial for Atoma nodes.

## Integration with Candle

We plan to integrate with the HuggingFace [candle](https://github.com/huggingface/candle) ML framework, a fully Rust based ML framework. Candle allows to rely on the blazing Rust performance and its memory safety. Moreover, it eases the process of integrating Machine Learning pipelines with AI inference distributed systems. For these reasons, we believe this repository can be of great value to both the ML and Rust communities. 
