# vram-constrained-training

Experiment data and analysis code for: **"Memory-Efficient Training Techniques Under Extreme VRAM Constraints: A Controlled Ablation on a 4 GB GPU"**

## Contents

```
├── analysis/
│   └── analyze_results.py       # generates all 9 paper figures from CSVs
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

## Experiment Scripts

The `.bat` files document the exact CLI flags passed to the CUDA trainer binary (`ct_train.exe`) for each experimental configuration. They are not directly runnable without the full CUDA codebase but serve as a complete record of the experimental procedure.

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

## Source Code

The CUDA C++ transformer implementation is available from the author on request. Contact: mahdi.ettehad85@gmail.com
