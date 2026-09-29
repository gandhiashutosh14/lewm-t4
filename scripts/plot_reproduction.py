"""Reproduction figure from the committed Kaggle outputs (reports/kaggle-*.json).

Three panels, one scale each: prediction loss and SIGReg over training (both seeds), and planning
success on the official 50-episode TwoRoom protocol with 95% Wilson intervals.

    python scripts/plot_reproduction.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SEED_COLORS = {3072: "#2a78d6", 3073: "#eb6834"}     # categorical slots 1 and 2
OFFICIAL = "#1baf7a"                                 # slot 3; below 3:1 contrast, so every bar is labelled


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - r) / d, 100 * (c + r) / d


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(which="both", colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    curve = json.loads((REP / "kaggle-train-curve.json").read_text())
    train = json.loads((REP / "kaggle-train.json").read_text())
    verify = json.loads((REP / "kaggle-verify.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), dpi=160, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    fig.patch.set_facecolor(SURFACE)
    steps = [r["step"] for r in curve]
    for ax, key, title in ((axes[0], "pred", "next-embedding prediction loss (train)"),
                           (axes[1], "sigreg", "SIGReg (train)")):
        style(ax, title)
        for seed, col in SEED_COLORS.items():
            ys = [r[f"{key}_{seed}"] for r in curve]
            ax.plot(steps, ys, color=col, linewidth=1.4, label=f"seed {seed}")
            ax.annotate(f"seed {seed}", (steps[-1], ys[-1]), xytext=(4, 6 if seed == 3072 else -8),
                        textcoords="offset points", fontsize=7, color=MUTED)
        ax.set_yscale("log")
        ax.set_xlabel("training step (batch 128)", fontsize=7, color=MUTED)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax = axes[2]
    style(ax, "planning success (official protocol)")
    bars = [("official\ncheckpoint", verify["plan_official"]["episode_successes"], OFFICIAL),
            ("this code,\nofficial weights", verify["plan_reimplementation"]["episode_successes"], OFFICIAL),
            ("from scratch\nseed 3072", train["plan_s3072"]["episode_successes"], SEED_COLORS[3072]),
            ("from scratch\nseed 3073", train["plan_s3073"]["episode_successes"], SEED_COLORS[3073])]
    for i, (label, succ, col) in enumerate(bars):
        k, n = sum(succ), len(succ)
        lo, hi = wilson(k, n)
        ax.bar(i, 100 * k / n, width=0.62, color=col, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(i, 100 * k / n, yerr=[[100 * k / n - lo], [hi - 100 * k / n]], color=INK, capsize=3, linewidth=1)
        ax.text(i, hi + 1.5, f"{k}/{n}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=6.5, color=MUTED)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of episodes reaching the goal", fontsize=7, color=MUTED)
    fig.text(0.01, 0.01, "from-scratch runs: Kaggle T4 x2, fp16, 18,498 steps (3.6 epochs) in 9.1 h; curves: one logged training batch every 100 steps; "
             "error bars: 95% Wilson intervals; the same 50 episodes for every bar",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(REP / "reproduction.png", facecolor=SURFACE)


if __name__ == "__main__":
    main()
