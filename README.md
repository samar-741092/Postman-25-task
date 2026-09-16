[README (2).md](https://github.com/user-attachments/files/32265667/README.2.md)
# Sparse vs. Dense Attention in Transformer Architectures

An empirical implementation, correctness validation, benchmarking, and language-modeling quality assessment comparing **Dense Causal Attention**, **Sliding Window Attention**, and **BigBird Sparse Attention** using PyTorch.

---

## Project Overview

This repository implements and evaluates sparse attention mechanisms versus standard full dense attention from first principles:
1. **Manual Scaled Dot-Product Attention**: Baseline dense attention implementation with support for arbitrary binary masks and numerical stabilization (handling masked `-inf` softmax rows with `torch.nan_to_num`).
2. **Sparsity Mask Generators**:
   - **Sliding Window Mask**: Constrains queries to attend only to local keys within window distance `w`. Supports causal and bidirectional variants.
   - **BigBird Mask**: Combines local sliding window (`w`), global tokens (`g`), and random key selections (`r`), strictly enforcing causality.
3. **Correctness Harness**: Validates numerical equivalence between masked dense attention and baseline dense attention across active (unmasked) positions under tight floating-point tolerances.
4. **Latency & Memory Benchmarking**: Measures wall-clock execution time (via CUDA events) and peak allocated GPU memory across sequence lengths N in [512, 1024, 2048, 4096, 8192].
5. **Quality Evaluation on TinyShakespeare**: Trains a 2-layer character-level decoder-only GPT to evaluate convergence, train/val loss, and trade-offs between dense, sliding window, and BigBird attention patterns.

---

## Features & Implementation Details

### 1. Manual Attention Functions
- `manual_dense_attention(Q, K, V, mask=None)`: Computes `Softmax((Q @ K.T) / sqrt(d_k) + mask) @ V` over tensors of shape `(batch_size, num_heads, seq_len, head_dim)`.
- `manual_dense_attention_NaN_Handled(Q, K, V, mask=None)`: Guards against complete row masking (where an entire row evaluates to `-inf`, yielding `NaN` in standard softmax) by replacing `NaN` probabilities with `0.0`.

### 2. Sparsity Mask Construction
- **Sliding Window (`create_sliding_window_mask`)**:
  - Query-to-key index distance: `dist = i - j`
  - Causal constraint: `(dist >= 0) & (dist <= w)`
  - Symmetric non-causal constraint: `abs(dist) <= w`
- **BigBird (`create_bigbird_mask`)**:
  - Local window: tokens within distance `w`.
  - Global tokens: first `g` tokens attend to and are attended by all tokens.
  - Random connections: `r` randomly sampled key indices per query row via `torch.randperm`.
  - Causal enforcement: `mask = mask & torch.tril(torch.ones(N, N, dtype=torch.bool))`.

---

## Verification & Correctness

The test harness `test_sparse_vs_dense_correctness` runs automated assertions across multiple configurations:
- Config 1: `B=2, H=4, N=64, D=32, w=8, g=1` (4,320 active elements verified) -> **PASSED**
- Config 2: `B=1, H=8, N=128, D=64, w=16, g=2` (16,320 active elements verified) -> **PASSED**
- Config 3: `B=4, H=2, N=256, D=32, w=32, g=4` (63,360 active elements verified) -> **PASSED**

Relative and absolute tolerance assertions (`rtol=1e-4`, `atol=1e-5`) verify exact numerical equivalence on all active attention positions.

---

## Empirical Benchmarks (CUDA Profiling)

Profiling setup: Batch size `B=1`, Heads `H=8`, Head dimension `D=64`, 5 warmup iterations, 20 benchmark iterations on an NVIDIA T4 GPU across sequence lengths `N in [512, 1024, 2048, 4096, 8192]`.

> **Implementation Note:** In this notebook, sparse attention is computed via dense matrix multiplication with sparsity masks applied. Because dense `N x N` attention matrices are still materialized in GPU memory, latency and peak memory scale quadratically in this baseline implementation. Dedicated custom kernels (such as Block-Sparse FlashAttention or Triton kernels) are required to realize sub-quadratic hardware execution.

Benchmark curves are saved directly to `attention_benchmark.png`.

---

## Language Modeling Quality on TinyShakespeare

A 2-layer character-level GPT (`block_size=256`, `n_head=4`, `w=32`, `g=2`, `r=2`) was trained on the TinyShakespeare dataset (1,115,394 characters; 90/10 train-validation split).

### Experiment 1: Baseline Architecture
- **Hyperparameters:** `n_embd = 128`, `dropout = 0.0`, `learning_rate = 3e-4`, `max_iters = 5000`.

| Pattern | Train Loss | Validation Loss | Wall-Clock Time (T4) |
| :--- | :---: | :---: | :---: |
| **Dense** | 1.3593 | 1.6098 | 4.75 min |
| **Sliding Window** | 1.2146 | 1.5867 | 4.64 min |
| **BigBird** | 1.2267 | 1.5783 | 8.88 min |

*(Values from execution logs)*

### Experiment 2: Scaled Capacity & Regularization
- **Hyperparameters:** `n_embd = 256`, `dropout = 0.2`, `learning_rate = 1e-3`, `max_iters = 2500`.

| Pattern | Train Loss | Validation Loss | Wall-Clock Time (T4) |
| :--- | :---: | :---: | :---: |
| **Dense** | 1.2589 | 1.5445 | 4.92 min |
| **Sliding Window** | 1.2469 | 1.5477 | 4.79 min |
| **BigBird** | 1.2528 | 1.5430 | 6.06 min |

*(Values from execution logs)*

### Key Takeaways
1. **Model Convergence:** Sparse attention (Sliding Window and BigBird) matches or slightly outperforms full dense attention validation loss on local character-level modeling tasks, showing that dense quadratic context is unnecessary for short-range text generation.
2. **BigBird Overhead:** Generating dynamic random indices via `torch.randperm` per query row inside Python introduces CPU/dispatch overhead, resulting in higher iteration times compared to static slicing.

---

## Repository Structure

```text
.
|-- attention_benchmark.png       # Latency and peak GPU memory benchmark plots
|-- input.txt                     # TinyShakespeare dataset (auto-downloaded)
|-- notebook.ipynb                # Primary Colab notebook with experiments
`-- README.md                     # Project documentation
```

---

## Requirements & Setup

```bash
pip install torch matplotlib
```

An NVIDIA GPU with CUDA support is recommended for running memory and wall-clock benchmarks.
