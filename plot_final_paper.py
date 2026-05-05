#!/usr/bin/env python3
"""
Generate final paper figures combining all methods and models.

Run from project root:
    python3 plot_final_paper.py
"""

import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from methods.common import (
    ALL_MODELS, MODEL_DISPLAY, ICML_RC, C,
    savefig, set_style,
)

EXCLUDE_MODELS = {"Qwen3-4B", "gemma-2-2b-it"}
MODELS = [m for m in ALL_MODELS if m not in EXCLUDE_MODELS]
LABELS = [MODEL_DISPLAY[m] for m in MODELS]

OUT_DIR = "plots/final"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load summaries ─────────────────────────────────────────────────────────────

def load_summary(path):
    with open(path) as f:
        return json.load(f)

m1 = load_summary("plots/methods/method1_steering/summary.json")
m2 = load_summary("plots/methods/method2_probe/summary.json")
m3 = load_summary("plots/methods/method3_bad/summary.json")

# ── Collect per-model values ───────────────────────────────────────────────────

def get_vals(summary, key):
    return [summary[m][key] for m in MODELS]

baseline_delta = get_vals(m1, "baseline_delta")   # same across all methods
baseline_acc   = get_vals(m1, "baseline_acc")

m1_delta = get_vals(m1, "method_delta")
m2_delta = get_vals(m2, "method_delta")
m3_delta = get_vals(m3, "method_delta")

m1_acc = get_vals(m1, "method_acc")
m2_acc = get_vals(m2, "method_acc")
m3_acc = get_vals(m3, "method_acc")

# ── Figure 1: δ comparison — all methods ──────────────────────────────────────

def annotate(ax, bars, fmt="{:.2f}", threshold=0.02, fontsize=5.5):
    for bar in bars:
        v = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        if v < threshold:
            ax.text(x, threshold * 0.25, fmt.format(v),
                    ha="center", va="bottom", fontsize=fontsize,
                    color="black", rotation=90)
        else:
            ax.text(x, v + 0.005, fmt.format(v),
                    ha="center", va="bottom", fontsize=fontsize,
                    color="black", rotation=90)

set_style()

n = len(MODELS)
x = np.arange(n)
w = 0.18   # bar width for 4 groups

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.0))

# δ panel
offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
colors  = [C["baseline"], C["method1"], C["method2"], C["method3"]]
labels  = ["Baseline", "Activation Steering", "Probe Steering", "BAD"]
data_d  = [baseline_delta, m1_delta, m2_delta, m3_delta]

bars_all = []
for off, col, lbl, dat in zip(offsets, colors, labels, data_d):
    b = ax1.bar(x + off, dat, w, color=col, alpha=0.88, label=lbl)
    annotate(ax1, b)
    bars_all.append(b)

ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7.5)
ax1.set_ylabel(r"Deceptive Behavior Score $\delta$", fontsize=9)
ax1.set_ylim(0, max(max(baseline_delta), max(m1_delta), max(m2_delta), max(m3_delta)) * 1.42)
ax1.legend(loc="upper right", fontsize=7, ncol=2)
ax1.set_title(r"Deceptive Behavior Score $\delta$ — All Methods", fontsize=9)

# Accuracy panel
data_a = [baseline_acc, m1_acc, m2_acc, m3_acc]
for off, col, lbl, dat in zip(offsets, colors, labels, data_a):
    b = ax2.bar(x + off, dat, w, color=col, alpha=0.88, label=lbl)
    annotate(ax2, b, threshold=0.05)

ax2.set_xticks(x)
ax2.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7.5)
ax2.set_ylabel("Initial Accuracy", fontsize=9)
ax2.set_ylim(0, 1.3)
ax2.legend(loc="upper right", fontsize=7, ncol=2)
ax2.set_title("Accuracy Preservation — All Methods", fontsize=9)

fig.tight_layout()
out1 = os.path.join(OUT_DIR, "all_methods_comparison.pdf")
savefig(fig, out1)

# ── Figure 2: per-method δ reduction (Δδ = baseline − method) ─────────────────

set_style()
fig2, ax = plt.subplots(figsize=(7.5, 2.8))

delta_reduction_m1 = [b - m for b, m in zip(baseline_delta, m1_delta)]
delta_reduction_m2 = [b - m for b, m in zip(baseline_delta, m2_delta)]
delta_reduction_m3 = [b - m for b, m in zip(baseline_delta, m3_delta)]

w2 = 0.22
for off, col, lbl, dat in zip(
    [-w2, 0, w2],
    [C["method1"], C["method2"], C["method3"]],
    ["Activation Steering", "Probe Steering", "BAD"],
    [delta_reduction_m1, delta_reduction_m2, delta_reduction_m3],
):
    b = ax.bar(x + off, dat, w2, color=col, alpha=0.88, label=lbl)
    annotate(ax, b, fontsize=5.5)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel(r"$\Delta\delta$ (baseline $-$ method)", fontsize=9)
ax.set_title(r"Deceptive Behavior Reduction $\Delta\delta$ per Method", fontsize=9)
ax.legend(loc="upper right", fontsize=8)

# Shade positive region
ymax = ax.get_ylim()[1]
ax.set_ylim(bottom=min(min(delta_reduction_m1), min(delta_reduction_m2), min(delta_reduction_m3)) - 0.05)
ax.fill_between([-0.5, n - 0.5], 0, ymax, alpha=0.04, color="green", zorder=0)

fig2.tight_layout()
out2 = os.path.join(OUT_DIR, "delta_reduction.pdf")
savefig(fig2, out2)

print(f"\nFinal figures saved to {OUT_DIR}/")
print(f"  {out1}")
print(f"  {out2}")
