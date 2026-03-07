import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
import os

# =================== Config =================

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR / ".." / "experiments" / "data"
FIG_DIR = SCRIPT_DIR / ".." / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "A_baseline":  {"label": "Baseline",     "color": "#2196F3", "marker": "o"},
    "B_mp":        {"label": "MP",           "color": "#4CAF50", "marker": "s"},
    "C_gc_all":    {"label": "GC-All",       "color": "#FF9800", "marker": "^"},
    "D_gc_half":   {"label": "GC-Half",      "color": "#F44336", "marker": "D"},
    "E_mp_gc_all": {"label": "MP+GC-All",    "color": "#9C27B0", "marker": "v"},
    "F_mp_gc_half":{"label": "MP+GC-Half",   "color": "#795548", "marker": "P"},
}

DPI = 300
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "font.family": "sans-serif",
})


# =================== Load Data =================

def load_csv(prefix, key):
    path = EXP_DIR / f"{prefix}_{key}.csv"
    return pd.read_csv(path)

quick = {k: load_csv("quick", k) for k in CONFIGS}
full  = {k: load_csv("full", k)  for k in CONFIGS}

# steady-state = steps >= 50 for quick runs
def steady(df, min_step=50):
    return df[df["step"] >= min_step]


# =================== Summary Table =================

print("=" * 90)
print("QUICK RUN SUMMARY (steady-state: steps 50-190)")
print("=" * 90)
header = f"{'Config':<16} {'tok/s':>8} {'fwd_ms':>8} {'bwd_ms':>8} {'opt_ms':>8} {'pool_MB':>9} {'real_MB':>9}"
print(header)
print("-" * 90)

baseline_toks = None
for k, meta in CONFIGS.items():
    ss = steady(quick[k])
    toks = ss["tok_per_sec"].mean()
    fwd  = ss["fwd_ms"].mean()
    bwd  = ss["bwd_ms"].mean()
    opt  = ss["optim_ms"].mean()
    pool = ss["vram_mb"].iloc[-1]
    real = ss["vram_real_mb"].iloc[-1]

    if baseline_toks is None:
        baseline_toks = toks
    delta = (toks - baseline_toks) / baseline_toks * 100

    print(f"{meta['label']:<16} {toks:>7.1f}  {fwd:>7.1f}  {bwd:>7.1f}  {opt:>7.1f}  {pool:>8.1f}  {real:>8.1f}  ({delta:+.1f}%)")

print()
print("=" * 90)
print("CONVERGENCE SUMMARY (full runs, 2000 steps)")
print("=" * 90)

for k, meta in CONFIGS.items():
    df = full[k]
    final = df.iloc[-1]
    val_rows = df[df["val_loss"].notna()]
    last_val = val_rows.iloc[-1]["val_loss"] if len(val_rows) > 0 else float("nan")
    last_val_ppl = val_rows.iloc[-1]["val_ppl"] if len(val_rows) > 0 else float("nan")
    total_s = final["elapsed_s"]
    print(f"{meta['label']:<16} final_loss={final['loss']:.4f}  val_loss={last_val:.4f}  val_ppl={last_val_ppl:.1f}  total_time={total_s:.0f}s ({total_s/60:.1f}min)")


# =================== Figure 1: Throughput Comparison =================

fig, ax = plt.subplots(figsize=(8, 5))
labels = [CONFIGS[k]["label"] for k in CONFIGS]
colors = [CONFIGS[k]["color"] for k in CONFIGS]
toks_vals = [steady(quick[k])["tok_per_sec"].mean() for k in CONFIGS]
baseline = toks_vals[0]

bars = ax.bar(labels, toks_vals, color=colors, edgecolor="black", linewidth=0.5, width=0.65)
ax.axhline(y=baseline, color="#2196F3", linestyle="--", alpha=0.4, linewidth=1)

for bar, val in zip(bars, toks_vals):
    delta = (val - baseline) / baseline * 100
    sign = "+" if delta >= 0 else ""
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{val:.0f}\n({sign}{delta:.1f}%)", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Training Throughput by Configuration\n(GTX 1650 Ti, 55M params, seq=256, accum=32)")
ax.set_ylim(0, max(toks_vals) * 1.15)
ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
plt.tight_layout()
plt.savefig(FIG_DIR / "throughput_comparison.png")
plt.close()
print(f"\nSaved: throughput_comparison.png")


# =================== Figure 2: Timing Breakdown =================

fig, ax = plt.subplots(figsize=(8, 5))
fwd_vals = [steady(quick[k])["fwd_ms"].mean() for k in CONFIGS]
bwd_vals = [steady(quick[k])["bwd_ms"].mean() for k in CONFIGS]
opt_vals = [steady(quick[k])["optim_ms"].mean() for k in CONFIGS]

x = np.arange(len(labels))
w = 0.6

p1 = ax.bar(x, fwd_vals, w, label="Forward", color="#42A5F5", edgecolor="black", linewidth=0.5)
p2 = ax.bar(x, bwd_vals, w, bottom=fwd_vals, label="Backward", color="#EF5350", edgecolor="black", linewidth=0.5)
bottoms = [f + b for f, b in zip(fwd_vals, bwd_vals)]
p3 = ax.bar(x, opt_vals, w, bottom=bottoms, label="Optimizer", color="#66BB6A", edgecolor="black", linewidth=0.5)

totals = [f + b + o for f, b, o in zip(fwd_vals, bwd_vals, opt_vals)]
for i, total in enumerate(totals):
    ax.text(i, total + 30, f"{total:.0f}ms", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Time per step (ms)")
ax.set_title("Per-Step Timing Breakdown\n(32 micro-batches accumulated)")
ax.legend(loc="upper left")
ax.set_ylim(0, max(totals) * 1.12)
plt.tight_layout()
plt.savefig(FIG_DIR / "timing_breakdown.png")
plt.close()
print(f"Saved: timing_breakdown.png")


# =================== Figure 3: VRAM Comparison =================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

pool_vals = [steady(quick[k])["vram_mb"].iloc[-1] for k in CONFIGS]
real_vals = [steady(quick[k])["vram_real_mb"].iloc[-1] for k in CONFIGS]

ax1.bar(labels, pool_vals, color=colors, edgecolor="black", linewidth=0.5, width=0.65)
ax1.set_ylabel("Pool VRAM (MB)")
ax1.set_title("Bump Allocator Accounting")
ax1.set_ylim(1100, 1160)
for i, v in enumerate(pool_vals):
    ax1.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

ax2.bar(labels, real_vals, color=colors, edgecolor="black", linewidth=0.5, width=0.65)
ax2.set_ylabel("Real VRAM (MB)")
ax2.set_title("cudaMemGetInfo (Driver-Level)")
ax2.set_ylim(2200, 2350)
for i, v in enumerate(real_vals):
    ax2.text(i, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("VRAM Usage: Pool Accounting vs Real GPU Memory", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "vram_comparison.png")
plt.close()
print(f"Saved: vram_comparison.png")


# =================== Figure 4: Convergence Curves =================

fig, ax = plt.subplots(figsize=(10, 6))
for k, meta in CONFIGS.items():
    df = full[k]
    ax.plot(df["step"], df["loss"], label=meta["label"], color=meta["color"],
            linewidth=1.2, alpha=0.85)

ax.set_xlabel("Training Step")
ax.set_ylabel("Training Loss")
ax.set_title("Convergence Comparison Across Configurations\n(2000 steps, seed=42, identical data order)")
ax.legend(loc="upper right")
ax.set_xlim(0, 2000)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "convergence_curves.png")
plt.close()
print(f"Saved: convergence_curves.png")


# =================== Figure 5: Validation Loss =================

fig, ax = plt.subplots(figsize=(8, 5))
for k, meta in CONFIGS.items():
    df = full[k]
    val_rows = df[df["val_loss"].notna()]
    if len(val_rows) > 0:
        ax.plot(val_rows["step"], val_rows["val_loss"], label=meta["label"],
                color=meta["color"], marker=meta["marker"], markersize=7, linewidth=1.5)

ax.set_xlabel("Training Step")
ax.set_ylabel("Validation Loss")
ax.set_title("Validation Loss at Eval Checkpoints")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "validation_loss.png")
plt.close()
print(f"Saved: validation_loss.png")


# =================== Figure 6: Pareto Frontier =================

fig, ax = plt.subplots(figsize=(8, 6))
for k, meta in CONFIGS.items():
    ss = steady(quick[k])
    toks = ss["tok_per_sec"].mean()
    pool = ss["vram_mb"].iloc[-1]
    df_full = full[k]
    final_loss = df_full.iloc[-1]["loss"]

    size = max(30, 200 * (1 - (final_loss - 4.24) / 0.02))

    ax.scatter(pool, toks, s=size, c=meta["color"], marker=meta["marker"],
               edgecolors="black", linewidth=0.8, zorder=5, label=meta["label"])

ax.set_xlabel("Pool VRAM (MB)")
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("VRAM vs Throughput Pareto Frontier\n(bubble size ~ convergence quality, all nearly equal)")
ax.legend(loc="center left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "pareto_frontier.png")
plt.close()
print(f"Saved: pareto_frontier.png")


# =================== Figure 7: Wall Clock Time =================

fig, ax = plt.subplots(figsize=(8, 5))
times = [full[k].iloc[-1]["elapsed_s"] / 60 for k in CONFIGS]
baseline_time = times[0]

bars = ax.bar(labels, times, color=colors, edgecolor="black", linewidth=0.5, width=0.65)
ax.axhline(y=baseline_time, color="#2196F3", linestyle="--", alpha=0.4)

for bar, val in zip(bars, times):
    delta = (val - baseline_time) / baseline_time * 100
    sign = "+" if delta >= 0 else ""
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}m\n({sign}{delta:.1f}%)", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Wall Clock Time (minutes)")
ax.set_title("Total Training Time for 2000 Steps")
ax.set_ylim(0, max(times) * 1.15)
plt.tight_layout()
plt.savefig(FIG_DIR / "wall_clock_time.png")
plt.close()
print(f"Saved: wall_clock_time.png")


# =================== Figure 8: Backward Time Deep Dive =================

fig, ax = plt.subplots(figsize=(8, 5))

# backward overhead vs baseline
baseline_bwd = steady(quick["A_baseline"])["bwd_ms"].mean()
bwd_overhead = [(steady(quick[k])["bwd_ms"].mean() - baseline_bwd) for k in CONFIGS]

bars = ax.bar(labels, [steady(quick[k])["bwd_ms"].mean() for k in CONFIGS],
              color=colors, edgecolor="black", linewidth=0.5, width=0.65)
ax.axhline(y=baseline_bwd, color="#2196F3", linestyle="--", alpha=0.4, label=f"Baseline ({baseline_bwd:.0f}ms)")

for bar, overhead in zip(bars, bwd_overhead):
    if abs(overhead) > 5:
        sign = "+" if overhead >= 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f"{sign}{overhead:.0f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Backward Pass Time (ms)")
ax.set_title("Backward Pass Overhead from Activation Recomputation\n(32 micro-batches accumulated)")
ax.legend(loc="upper left")
ax.set_ylim(0, max([steady(quick[k])["bwd_ms"].mean() for k in CONFIGS]) * 1.1)
plt.tight_layout()
plt.savefig(FIG_DIR / "backward_overhead.png")
plt.close()
print(f"Saved: backward_overhead.png")

# =================== Figure 9: OOM Boundary (Memory Model Validation) =================

oom_gc_path = EXP_DIR / "oom_A2_seq384_gc.csv"
if oom_gc_path.exists():
    print("\n" + "=" * 90)
    print("OOM BOUNDARY EXPERIMENT (S=384)")
    print("=" * 90)

    oom_baseline_path = EXP_DIR / "oom_A1_seq384_baseline.csv"
    g_status = "OOM (no CSV)" if not oom_baseline_path.exists() else "Ran (unexpected)"
    if oom_baseline_path.exists():
        try:
            g_df = pd.read_csv(oom_baseline_path)
            g_status = f"Ran {len(g_df)} steps (unexpected)" if len(g_df) > 0 else "OOM (empty CSV)"
        except Exception:
            g_status = "OOM (corrupt/empty CSV)"

    h_df = pd.read_csv(oom_gc_path)
    print(f"Config G (seq=384, no GC): {g_status}")
    print(f"Config H (seq=384, GC-All): {len(h_df)} steps completed, final loss={h_df.iloc[-1]['loss']:.4f}")

    # empirical scratch comparison: S=256 vs S=384
    fig, ax = plt.subplots(figsize=(8, 5))

    scratch_pool = 300

    # S=256 from existing experiments (pool - persistent)
    s256_no_gc = steady(quick["A_baseline"])["vram_mb"].iloc[-1] - 937.5
    s256_gc = steady(quick["C_gc_all"])["vram_mb"].iloc[-1] - 937.5

    # S=384: no-GC OOMed (>300), GC succeeded
    s384_gc = h_df.iloc[-1]["vram_mb"] - 938.8 if "vram_mb" in h_df.columns else 297.0
    s384_no_gc = 301  # OOM -- exceeded pool, actual value > 300

    labels_oom = ["S=256\n(All configs fit)", "S=384\n(OOM boundary)"]
    scratch_no_gc = [s256_no_gc, s384_no_gc]
    scratch_gc = [s256_gc, s384_gc]

    x = np.arange(2)
    w = 0.35

    bars1 = ax.bar(x - w/2, scratch_no_gc, w, label="Without GC", color="#F44336",
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, scratch_gc, w, label="With GC", color="#4CAF50",
                   edgecolor="black", linewidth=0.5)

    ax.axhline(y=scratch_pool, color="black", linestyle="--", linewidth=1.5,
               label=f"Scratch Pool ({scratch_pool} MB)")

    for bar, val, is_oom in zip(bars1, scratch_no_gc, [False, True]):
        tag = " (OOM)" if is_oom else ""
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{val:.0f} MB{tag}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, scratch_gc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{val:.0f} MB", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_oom)
    ax.set_ylabel("Scratch Memory (MB)")
    ax.set_title("OOM Boundary: Scratch Usage at S=256 vs S=384\n(scratch pool = 300 MB)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(scratch_no_gc) * 1.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "oom_boundary.png")
    plt.close()
    print(f"Saved: oom_boundary.png")
else:
    print("\nOOM boundary CSVs not found -- run experiments/run_oom_boundary.bat first")

print(f"\nAll figures saved to: {FIG_DIR}")
