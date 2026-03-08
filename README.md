# vram-constrained-training

Experiment data, analysis code, and key CUDA source files for: **"Memory-Efficient Training Techniques Under Extreme VRAM Constraints: A Controlled Ablation on a 4 GB GPU"**

## Contents

```
├── analysis/
│   └── analyze_results.py       # generates all 9 paper figures from CSVs
├── cuda/                        # key CUDA C++ source (see below)
│   ├── include/ct/
│   │   ├── memory.cuh           # bump memory allocator (persistent + scratch pools)
│   │   ├── layers/block.cuh     # transformer block with gradient checkpointing
│   │   ├── layers/transformer.cuh  # full model: forward/backward, checkpoint mask
│   │   └── training/trainer.cuh    # trainer config, CUDA event timing
│   └── src/
│       ├── memory.cu            # allocator: alloc, reset, save_mark, restore_mark
│       ├── train_main.cu        # CLI entry point, default config (55M params)
│       ├── layers/block.cu      # block forward/backward + backward_with_recompute
│       ├── layers/transformer.cu   # model-level GC orchestration, mixed precision
│       └── training/trainer.cu     # training loop, grad accum, benchmark instrumentation
├── experiments/
│   ├── run_benchmarks.bat       # Stage 1: 200-step profiling (6 configs)
│   ├── run_convergence.bat      # Stage 2: 2000-step convergence (6 configs)
│   ├── run_oom_boundary.bat     # Stage 3: OOM boundary validation
│   └── data/                    # raw experiment CSVs (16 files)
└── requirements.txt
```

## Hardware and Software

- GPU: NVIDIA GeForce GTX 1650 Ti (4 GB GDDR6, Turing SM 7.5, no tensor cores)
- System: 8 GB DDR4 RAM, Windows 11
- CUDA: 11.7
- Model: 55M-parameter transformer, custom CUDA C++ kernels, bump memory allocator

## Regenerating Figures

```bash
pip install -r requirements.txt
cd analysis
python analyze_results.py
```

Figures are written to `../figures/`. The script reads CSVs from `../experiments/data/` using relative paths.

## CUDA Source

The `cuda/` directory contains the key source files from the CUDA C++ transformer that directly support the paper's claims. These are not independently compilable (they depend on kernel and utility headers not included here) but are provided for full transparency on:

- **Memory management** (`memory.cuh/cu`): Bump allocator with pre-reserved persistent and scratch pools. The `save_mark()`/`restore_mark()` mechanism enables per-block activation scoping for gradient checkpointing.
- **Gradient checkpointing** (`block.cu`, `transformer.cu`): `backward_with_recompute()` re-runs the forward pass within a mark/restore scope, freeing activations after each block. The model-level loop applies a per-layer checkpoint mask.
- **Training instrumentation** (`trainer.cu`, `train_main.cu`): CUDA event timing around forward/backward/optimizer phases, dual VRAM reporting (pool accounting vs. `cudaMemGetInfo`), dynamic loss scaling for mixed precision.

The full CUDA codebase (kernels, data loading, checkpointing, inference) is available from the author on request.

## Experiment Scripts

The `.bat` files document the exact CLI flags passed to the CUDA trainer binary (`ct_train.exe`) for each experimental configuration.

**Stage 1** (`run_benchmarks.bat`): 200 steps per config, no validation. Measures steady-state throughput, per-phase CUDA timing, and dual-level VRAM usage.

**Stage 2** (`run_convergence.bat`): 2000 steps per config, validation every 500 steps. Measures convergence behavior and total wall-clock time.

**Stage 3** (`run_oom_boundary.bat`): Tests OOM boundary at S=384 (default scratch pool) and S=256 (reduced scratch pool of 195 MB). Validates the memory model's crossover prediction.

## CSV Columns

Each experiment CSV contains per-step measurements:

| Column | Description |
|--------|-------------|
| `step` | Training step number |
| `loss` | Training loss |
| `lr` | Learning rate |
| `tok_per_sec` | Throughput (tokens/second) |
| `elapsed_s` | Wall-clock time since start |
| `vram_mb` | Pool VRAM: persistent.used() + scratch.peak() |
| `vram_real_mb` | Driver VRAM via cudaMemGetInfo |
| `fwd_ms` | Forward pass time (ms, accumulated over 32 micro-batches) |
| `bwd_ms` | Backward pass time (ms, accumulated over 32 micro-batches) |
| `optim_ms` | Optimizer step time (ms) |
| `val_loss` | Validation loss (NaN when not evaluated) |
| `val_ppl` | Validation perplexity (NaN when not evaluated) |

## Contact

Mahdi Ettehadnejad — mahdi.ettehad85@gmail.com
