# SoS Core Library

This directory contains the core implementation of the SoS (Sum-of-Squares) library.

## Structure

- `evals/`: Evaluation scripts and utilities
- `train/`: Training scripts and configurations

Our training process is simple - we use [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory/) for model training. You only need to download the [Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M) model and use our dataset and training scripts to complete the training process.
