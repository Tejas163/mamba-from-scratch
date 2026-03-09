# Mamba SSM From Scratch

An educational implementation of Mamba selective state space models from scratch, inspired by minitorch and llm.modular.com.

## Installation

```bash
pip install torch numpy matplotlib
```

## Quick Start

### Colab
Open the notebooks in Google Colab:
- [Part 1: SSM Foundations](notebooks/01_ssm_foundations.ipynb)
- [Part 2: Parallel Scan](notebooks/02_parallel_scan.ipynb)
- [Part 3: Selective SSM](notebooks/03_selective_ssm.ipynb)
- [Part 4: Mamba Block](notebooks/04_mamba_block.ipynb)
- [Part 5: Language Model](notebooks/05_language_model.ipynb)
- [Part 6: Benchmark](notebooks/06_benchmark.ipynb)

### Local
```bash
# Run notebooks with jupyter
jupyter lab notebooks/

# Build the book
pip install mdbook
mdbook build book
mdbook serve book
```

## What You'll Learn

1. **State Space Models** - Mathematical foundations
2. **Parallel Scan** - The key efficiency algorithm
3. **Selective Mechanism** - Mamba's innovation
4. **Full Architecture** - Complete Mamba block
5. **Language Model** - End-to-end training
6. **Benchmarks** - Performance comparison

## Project Structure

```
mamba-from-scratch/
├── nbs/                    # nbdev notebooks (executable in browser!)
├── notebooks/              # Jupyter notebooks for Colab
├── book/                  # mdbook website
├── images/                # Figures
└── README.md
```

## nbdev Integration

These notebooks are **nbdev** compatible and can be:
1. Executed directly in **Google Colab**
2. Edited in **Jupyter/VS Code**
3. Built into a **Python package**

To run locally:
```bash
pip install nbdev
nbdev_preview
```

## License

MIT
