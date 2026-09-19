# Getting Started

Welcome to the Privacy and Grokking repository. This guide covers the basic setup and how to start a training run.

## Installation & Setup

We use `uv` for fast dependency management.

To install dependencies with CPU support:
```bash
uv sync --extra cpu
```

To install dependencies with CUDA 12.1+ support (for GPU):
```bash
uv sync --extra cu130
```

## Command Line Interface (CLI)

The main entry point for the project is `uv run pag`. 

You can view all available commands and arguments using the help flag:
```bash
uv run pag --help
```

### Starting a Training Run

To start a training run, you typically use the `train` command along with the desired version and models:
```bash
uv run pag train v1.0.0 --models
```

### Running Attacks

To execute a privacy attack (e.g., MIA):
```bash
uv run pag attack mia_threshold_probs v2.3.0 MNIST_MLP_GROK_TRAIN_NOCAN 250000
```

### Evaluation

To run evaluations:
```bash
uv run pag evaluate v2.3.0
```
