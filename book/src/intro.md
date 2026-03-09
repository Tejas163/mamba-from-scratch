# Build Mamba from Scratch

<div align="center">

![Mamba Architecture](images/mamba_arch.png)

**An Interactive Guide to Selective State Space Models**

</div>

---

## What is Mamba?

Mamba is a **selective state space model** (SSM) that achieves state-of-the-art performance on sequence modeling tasks. Unlike Transformers with their O(N²) attention complexity, Mamba processes sequences in **O(N) linear time** while maintaining similar quality.

This tutorial walks you through implementing Mamba from scratch, building up each component step-by-step with full mathematical derivations.

## Why Learn Mamba?

| Feature | Transformer | Mamba |
|---------|-------------|-------|
| Time Complexity | O(N²) | O(N) |
| Memory | O(N²) KV cache | O(N) |
| Inference | Cached attention | Single pass |
| Content Selection | Full attention | Learned selective |
| Parallel Training | Yes | Yes |

## What You'll Build

This tutorial follows a progressive approach:

| Step | Topic | Key Concept |
|------|-------|-------------|
| 1 | SSM Foundations | Math behind state space models |
| 2 | Discrete Time | Converting continuous to discrete |
| 3 | Naive Implementation | Sequential SSM computation |
| 4 | Parallel Scan | The key efficiency algorithm |
| 5 | Selective SSM | Mamba's content-based selection |
| 6 | Full Mamba Block | Complete architecture |
| 7 | Language Model | End-to-end training |
| 8 | Benchmark | Comparing with Transformers |

## Prerequisites

- **PyTorch** fundamentals
- **Linear algebra** basics (matrices, vectors)
- Understanding of **deep learning** concepts
- Familiarity with **Python**

## How to Use This Tutorial

### Interactive Notebooks (Google Colab)

Each step has a corresponding Jupyter notebook that you can run directly in Google Colab:

1. Open the notebook link
2. Click "Run All" or run cells individually
3. Experiment with the code
4. Complete the exercises

### Website (This Book)

The book provides:

- **Detailed explanations** with full math derivations
- **Code walkthroughs** with comments
- **Visualizations** to build intuition
- **Exercises** to test understanding

## Project Structure

```
mamba-from-scratch/
├── notebooks/          # Jupyter notebooks for Colab
│   ├── 01_ssm_foundations.ipynb
│   ├── 02_discrete_time.ipynb
│   ├── 03_naive_ssm.ipynb
│   ├── 04_parallel_scan.ipynb
│   ├── 05_selective_ssm.ipynb
│   ├── 06_mamba_block.ipynb
│   └── 07_language_model.ipynb
├── book/              # mdbook website
│   └── src/
└── README.md
```

## The Journey

We'll start with the basic mathematical foundations of state space models, then progressively add:

1. **Discretization** - Making SSMs work with discrete tokens
2. **Parallelization** - The scan algorithm for efficiency
3. **Selection** - Making parameters input-dependent
4. **Architecture** - Adding gating, normalization, residuals
5. **Training** - End-to-end language modeling

By the end, you'll have a complete understanding of how Mamba works and be able to implement it yourself!

## Next Step

Start with [Step 1: SSM Foundations](./step_01.md) to begin your journey into state space models.
