"""
Generate publication-quality ICML-style figures and summary table.
Handles up to ~15 models; layout auto-scales.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from .metrics import analyze_model_condition


# ── Condition styling ──────────────────────────────────────────
CONDITION_STYLES = {
    "baseline":        {"color": "#333333", "marker": "o", "ls": "-",  "label": "Baseline"},
    "intervention_a":  {"color": "#e74c3c", "marker": "s", "ls": "--", "label": "A: Forced Enum."},
    "intervention_b":  {"color": "#3498db", "marker": "^", "ls": "-.", "label": "B: Sample & Vote"},
    "intervention_c":  {"color": "#2ecc71", "marker": "D", "ls": ":",  "label": "C: Belief First"},
    "intervention_ab": {"color": "#9b59b6", "marker": "v", "ls": "--", "label": "A+B Combined"},
}

MODEL_DISPLAY = {
    "gemma-2-2b-it": "Gemma-2-2B",
    "Qwen3-4B": "Qwen3-4B",
    "Phi-4-mini-instruct": "Phi-4-mini",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "gemma-2-9b-it": "Gemma-2-9B",
    "Mistral-Nemo-Instruct-2407": "Mistral-Nemo-12B",
    "Qwen2.5-32B-Instruct": "Qwen2.5-32B",
    "Qwen3-235B-A22B-Instruct": "Qwen3-235B",
}

# 10-model colour palette (colourblind-friendly, print-safe)
MODEL_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
]


def setup_style():
    """ICML single-column-friendly defaults."""
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _model_color(model, models):
    """Consistent colour for a model across all figures."""
    idx = models.index(model) if model in models else 0
    return MODEL_COLORS[idx % len(MODEL_COLORS)]


def _display(model):
    return MODEL_DISPLAY.get(model, model)


# ── Data collection ────────────────────────────────────────────
def _collect_all(results_dir, models, conditions, n_values):
    all_data = {}
    for model in models:
        all_data[model] = {}
        for cond in conditions:
            print(f"  Analyzing {_display(model)} / {cond}...")
            all_data[model][cond] = analyze_model_condition(
                results_dir, model, cond, n_values)
    return all_data


# ── Helper: auto-grid ─────────────────────────────────────────
def _auto_grid(n):
    """Return (nrows, ncols) that tightly fits n items."""
    if n <= 4:
        return 2, 2
    elif n <= 6:
        return 2, 3
    elif n <= 9:
        return 3, 3
    elif n <= 12:
        return 3, 4
    else:
        return 4, 4


# ============================================================
# Figure 1: δ vs n for every model (auto-grid)
# ============================================================
def plot_figure1(all_data, target_model, n_values, output_dir, models=None):
    """δ vs n for baseline + all interventions, one subplot per model."""
    setup_style()
    if models is None:
        models = [target_model]
    models = [m for m in models if m in all_data]

    nrows, ncols = _auto_grid(len(models))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.6, nrows * 2.8),
                             sharex=True, sharey=True)
    axes_flat = np.array(axes).flatten()

    # global y-range
    ymax = 0
    for model in models:
        for cond in CONDITION_STYLES:
            data = all_data.get(model, {}).get(cond)
            if not data:
                continue
            for n in n_values:
                v = data["delta_by_n"].get(n, {}).get("ci_upper", 0)
                if not np.isnan(v) and v > ymax:
                    ymax = v

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        for cond, style in CONDITION_STYLES.items():
            data = all_data.get(model, {}).get(cond)
            if not data:
                continue
            deltas = [data["delta_by_n"].get(n, {}).get("delta", np.nan) for n in n_values]
            ci_lo = [data["delta_by_n"].get(n, {}).get("ci_lower", np.nan) for n in n_values]
            ci_hi = [data["delta_by_n"].get(n, {}).get("ci_upper", np.nan) for n in n_values]
            ax.plot(n_values, deltas, color=style["color"], marker=style["marker"],
                    ls=style["ls"], label=style["label"], markersize=4)
            ax.fill_between(n_values, ci_lo, ci_hi, color=style["color"], alpha=0.08)

        ax.set_title(_display(model), fontsize=10, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(n_values)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(-0.02, ymax * 1.15 + 0.02)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("n")
        if idx % ncols == 0:
            ax.set_ylabel("δ")

    for idx in range(len(models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(CONDITION_STYLES),
               fontsize=8, bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle("Deceptive Behavior Score (δ) vs. Task Difficulty",
                 fontsize=13, fontweight="bold", y=1.07)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig1_delta_vs_n.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig1_delta_vs_n.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 1")


# ============================================================
# Figure 2: Heatmap  models × interventions  (Δδ)
# ============================================================
def plot_figure2(all_data, models, conditions, n_values, output_dir):
    setup_style()
    interventions = [c for c in conditions if c != "baseline"]
    model_labels = [_display(m) for m in models]

    matrix = np.full((len(models), len(interventions)), np.nan)
    for i, model in enumerate(models):
        base = all_data.get(model, {}).get("baseline")
        if not base:
            continue
        for j, cond in enumerate(interventions):
            cd = all_data.get(model, {}).get(cond)
            if not cd:
                continue
            reds = []
            for n in n_values:
                db = base["delta_by_n"].get(n, {}).get("delta", np.nan)
                dc = cd["delta_by_n"].get(n, {}).get("delta", np.nan)
                if not np.isnan(db) and not np.isnan(dc):
                    reds.append(db - dc)
            matrix[i, j] = np.mean(reds) if reds else np.nan

    height = max(4, 0.45 * len(models) + 1.5)
    fig, ax = plt.subplots(figsize=(7, height))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(interventions)))
    ax.set_xticklabels([CONDITION_STYLES[c]["label"] for c in interventions],
                       rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels, fontsize=9)

    for i in range(len(models)):
        for j in range(len(interventions)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color="black",
                        fontweight="bold" if v > 0 else "normal")

    ax.set_title("Δδ: Reduction in Deceptive Behavior (higher = better)",
                 fontsize=11, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, label="Δδ", shrink=0.8)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig2_heatmap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig2_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 2")


# ============================================================
# Figure 3: Four-panel  (baseline δ / best δ / accuracy / % reduction)
# ============================================================
def plot_figure3(all_data, models, n_values, output_dir, best_cond="intervention_c"):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    (ax1, ax2), (ax3, ax4) = axes
    model_labels = [_display(m) for m in models]
    x = np.arange(len(models))
    width = 0.35

    delta_before, delta_after = [], []
    for model in models:
        base = all_data.get(model, {}).get("baseline", {})
        best = all_data.get(model, {}).get(best_cond, {})
        db = np.nanmean([base.get("delta_by_n", {}).get(n, {}).get("delta", np.nan) for n in n_values])
        da = np.nanmean([best.get("delta_by_n", {}).get(n, {}).get("delta", np.nan) for n in n_values])
        delta_before.append(db if not np.isnan(db) else 0)
        delta_after.append(da if not np.isnan(da) else 0)

    # Panel 1: baseline δ for all models (sorted by value)
    sort_idx = np.argsort(delta_before)[::-1]
    sorted_labels = [model_labels[i] for i in sort_idx]
    sorted_before = [delta_before[i] for i in sort_idx]
    sorted_colors = [_model_color(models[i], models) for i in sort_idx]
    xs = np.arange(len(models))
    bars1 = ax1.bar(xs, sorted_before, color=sorted_colors, alpha=0.85,
                    edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars1, sorted_before):
        if val > 0.01:
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        else:
            ax1.text(bar.get_x() + bar.get_width() / 2, 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7, color="gray")
    ax1.set_ylabel("Baseline δ")
    ax1.set_title("(a) Baseline Deceptive Behavior Score", fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(sorted_labels, rotation=35, ha="right", fontsize=8)

    # Panel 2: before vs after side-by-side
    ax2.bar(x - width / 2, delta_before, width, label="Baseline",
            color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax2.bar(x + width / 2, delta_after, width,
            label=CONDITION_STYLES[best_cond]["label"],
            color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.4)
    # Annotate values on bars
    for i in range(len(models)):
        if delta_before[i] > 0.01:
            ax2.text(i - width / 2, delta_before[i] + 0.005, f"{delta_before[i]:.2f}",
                     ha="center", va="bottom", fontsize=6, color="#e74c3c")
            ax2.text(i + width / 2, delta_after[i] + 0.005, f"{delta_after[i]:.2f}",
                     ha="center", va="bottom", fontsize=6, color="#2ecc71")
    ax2.set_ylabel("δ")
    ax2.set_title("(b) Deception: Before vs. After Best Intervention", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8)
    ax2.legend(fontsize=8)

    # Panel 3: accuracy preservation
    conds_acc = ["baseline", "intervention_a", "intervention_ab"]
    clabels = ["Baseline", "A: Enum.", "A+B"]
    ccolors = ["#e74c3c", "#f39c12", "#9b59b6"]
    bw = 0.8 / len(conds_acc)
    for ci, (cond, cl, cc) in enumerate(zip(conds_acc, clabels, ccolors)):
        accs = []
        for model in models:
            d = all_data.get(model, {}).get(cond, {})
            acc = np.nanmean([d.get("accuracy_by_n", {}).get(n, {}).get("accuracy", np.nan) for n in n_values])
            accs.append(acc if not np.isnan(acc) else 0)
        offset = (ci - len(conds_acc) / 2 + 0.5) * bw
        ax3.bar(x + offset, accs, bw, label=cl, color=cc, alpha=0.85,
                edgecolor="black", linewidth=0.4)
    ax3.set_ylabel("Linked-List Accuracy")
    ax3.set_title("(c) Accuracy Preservation Across Interventions", fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8)
    ax3.legend(fontsize=7, loc="upper right")
    ax3.set_ylim(0, 1.15)

    # Panel 4: % reduction per intervention (all 4)
    interventions = ["intervention_a", "intervention_b", "intervention_c", "intervention_ab"]
    int_labels = ["A: Enum.", "B: Vote", "C: Belief", "A+B"]
    int_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    bw = 0.8 / len(interventions)
    for ci, (cond, cl, cc) in enumerate(zip(interventions, int_labels, int_colors)):
        pcts = []
        for mi, model in enumerate(models):
            bd = delta_before[mi]
            cd = all_data.get(model, {}).get(cond, {})
            cd_val = np.nanmean([cd.get("delta_by_n", {}).get(n, {}).get("delta", np.nan) for n in n_values])
            if bd > 0.001 and not np.isnan(cd_val):
                pcts.append((bd - cd_val) / bd * 100)
            else:
                pcts.append(np.nan)
        offset = (ci - len(interventions) / 2 + 0.5) * bw
        ax4.bar(x + offset, pcts, bw, label=cl, color=cc, alpha=0.85,
                edgecolor="black", linewidth=0.4)
    ax4.set_ylabel("% Reduction in δ")
    ax4.set_title("(d) Relative Deception Reduction by Intervention", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=8)
    ax4.legend(fontsize=7, loc="best")
    ax4.axhline(0, color="black", linewidth=0.8)

    fig.suptitle("Intervention Effectiveness: Deception Reduction & Accuracy Preservation",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig3_honesty_vs_accuracy.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig3_honesty_vs_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 3")


# ============================================================
# Figure 4: Δδ heatmap per intervention (models × difficulty)
# ============================================================
def plot_regime_analysis(all_data, models, conditions, n_values, output_dir):
    """2×2 heatmaps: one per intervention, rows=models, cols=n values, cell=Δδ."""
    setup_style()
    interventions = [c for c in conditions if c != "baseline"]
    model_labels = [_display(m) for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(14, max(6, 0.5 * len(models) + 3)))

    vmin, vmax = 0, 0
    # Pre-compute to find global colour range
    matrices = {}
    for cond in interventions:
        mat = np.full((len(models), len(n_values)), np.nan)
        for i, model in enumerate(models):
            base = all_data.get(model, {}).get("baseline")
            cd = all_data.get(model, {}).get(cond)
            if not base or not cd:
                continue
            for j, n in enumerate(n_values):
                db = base["delta_by_n"].get(n, {}).get("delta", np.nan)
                dc = cd["delta_by_n"].get(n, {}).get("delta", np.nan)
                if not np.isnan(db) and not np.isnan(dc):
                    mat[i, j] = db - dc
        matrices[cond] = mat
        vmin = min(vmin, np.nanmin(mat)) if not np.all(np.isnan(mat)) else vmin
        vmax = max(vmax, np.nanmax(mat)) if not np.all(np.isnan(mat)) else vmax

    # Symmetric colour range centred on 0
    vlim = max(abs(vmin), abs(vmax), 0.01)

    for idx, cond in enumerate(interventions):
        ax = axes.flatten()[idx]
        mat = matrices[cond]
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vlim, vmax=vlim)
        ax.set_xticks(range(len(n_values)))
        ax.set_xticklabels([str(n) for n in n_values], fontsize=9)
        ax.set_xlabel("n (difficulty)")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(model_labels, fontsize=8)
        ax.set_title(CONDITION_STYLES[cond]["label"], fontsize=11, fontweight="bold")

        for i in range(len(models)):
            for j in range(len(n_values)):
                v = mat[i, j]
                if not np.isnan(v):
                    color = "white" if abs(v) > vlim * 0.7 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

    for idx in range(len(interventions), 4):
        axes.flatten()[idx].set_visible(False)

    fig.suptitle("Intervention Effectiveness by Model and Difficulty",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Δδ (positive = less deceptive)",
                 shrink=0.7, pad=0.03, location="right")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig4_regime_analysis.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig4_regime_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 4")


# ============================================================
# Figure 5: Radar charts — one per metric, models as spokes
# ============================================================
def plot_radar(all_data, models, n_values, output_dir):
    """
    Two consolidated radar charts side by side:
      Left:  % δ reduction — all 4 interventions overlaid, models as spokes
      Right: Baseline profile — δ, ρ, and accuracy overlaid, models as spokes
    """
    setup_style()

    interventions = [
        ("intervention_a", "A: Forced Enum.", "#e74c3c"),
        ("intervention_b", "B: Sample & Vote", "#3498db"),
        ("intervention_c", "C: Belief First", "#2ecc71"),
        ("intervention_ab", "A+B Combined", "#9b59b6"),
    ]
    model_labels = [_display(m) for m in models]
    n_models = len(models)
    angles = np.linspace(0, 2 * np.pi, n_models, endpoint=False).tolist()
    angles += angles[:1]

    # ── Pre-compute % δ reduction per (model, intervention) ──
    pct_data = {}
    for cond, _, _ in interventions:
        vals = []
        for model in models:
            base = all_data.get(model, {}).get("baseline", {})
            bd = np.nanmean([base.get("delta_by_n", {}).get(n, {}).get("delta", np.nan)
                              for n in n_values])
            cd = all_data.get(model, {}).get(cond, {})
            cd_d = np.nanmean([cd.get("delta_by_n", {}).get(n, {}).get("delta", np.nan)
                                for n in n_values])
            if bd > 0.001 and not np.isnan(cd_d):
                vals.append((bd - cd_d) / bd * 100)
            else:
                vals.append(0)
        pct_data[cond] = vals

    all_pcts = [v for vs in pct_data.values() for v in vs]
    rmin = min(0, min(all_pcts) if all_pcts else 0)
    rmax = max(all_pcts) if all_pcts else 100
    rrange = rmax - rmin if rmax > rmin else 1

    def to_radar(vals, lo, hi):
        rng = hi - lo if hi > lo else 1
        return [max(0, (v - lo) / rng) for v in vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                    subplot_kw=dict(projection="polar"))

    # ── Left panel: all interventions overlaid ──
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(30)
    ax1.set_thetagrids(np.degrees(angles[:-1]), model_labels, fontsize=8)

    n_ticks = 5
    tick_vals = np.linspace(0, 1, n_ticks)
    tick_labels = [f"{rmin + t * rrange:.0f}%" for t in tick_vals]
    ax1.set_yticks(tick_vals)
    ax1.set_yticklabels(tick_labels, fontsize=7, color="gray")
    ax1.set_ylim(0, 1.15)

    for cond, label, color in interventions:
        raw = pct_data[cond]
        normed = to_radar(raw, rmin, rmax)
        values = normed + normed[:1]
        ax1.plot(angles, values, linewidth=2, color=color, marker="o",
                 markersize=4, label=label)
        ax1.fill(angles, values, alpha=0.08, color=color)

    # Zero reference line
    zero_norm = max(0, (0 - rmin) / rrange)
    ax1.plot(angles, [zero_norm] * len(angles), color="gray",
             linewidth=0.8, linestyle="--", alpha=0.5)

    ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8,
               framealpha=0.9)
    ax1.set_title("% Deception Reduction\n(all interventions)", fontsize=11,
                   fontweight="bold", pad=25)

    # ── Right panel: baseline δ, ρ, accuracy profiles ──
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_rlabel_position(30)
    ax2.set_thetagrids(np.degrees(angles[:-1]), model_labels, fontsize=8)

    # Compute baseline metrics per model
    base_deltas, base_rhos, base_accs = [], [], []
    for model in models:
        base = all_data.get(model, {}).get("baseline", {})
        bd = np.nanmean([base.get("delta_by_n", {}).get(n, {}).get("delta", np.nan)
                          for n in n_values])
        br = np.nanmean([base.get("rho_by_n", {}).get(n, {}).get("rho", np.nan)
                          for n in n_values])
        ba = np.nanmean([base.get("accuracy_by_n", {}).get(n, {}).get("accuracy", np.nan)
                          for n in n_values])
        base_deltas.append(bd if not np.isnan(bd) else 0)
        base_rhos.append(br if not np.isnan(br) else 0)
        base_accs.append(ba if not np.isnan(ba) else 0)

    metrics = [
        (base_deltas, "Baseline δ", "#e74c3c"),
        (base_rhos, "Baseline ρ", "#3498db"),
        (base_accs, "Baseline Accuracy", "#2ecc71"),
    ]

    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=7, color="gray")
    ax2.set_ylim(0, 1.15)

    for raw, label, color in metrics:
        # Normalize each metric to [0, 1] by its own range
        lo, hi = min(raw), max(raw)
        normed = to_radar(raw, lo, hi) if hi > lo else [0.5] * len(raw)
        values = normed + normed[:1]
        ax2.plot(angles, values, linewidth=2, color=color, marker="o",
                 markersize=4, label=label)
        ax2.fill(angles, values, alpha=0.08, color=color)

    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8,
               framealpha=0.9)
    ax2.set_title("Baseline Model Profiles\n(normalized per metric)", fontsize=11,
                   fontweight="bold", pad=25)

    fig.suptitle("Model Comparison: Intervention Effectiveness & Baseline Profiles",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig5_radar.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig5_radar.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 5 (2 radar charts)")


# ============================================================
# Figure 6: Model-size scaling — δ baseline & best reduction
# ============================================================
MODEL_PARAMS = {
    "gemma-2-2b-it": 2,
    "Qwen3-4B": 4,
    "Phi-4-mini-instruct": 3.8,
    "Qwen2.5-7B-Instruct": 7,
    "Meta-Llama-3.1-8B-Instruct": 8,
    "gemma-2-9b-it": 9,
    "Mistral-Nemo-Instruct-2407": 12,
    "Qwen2.5-32B-Instruct": 32,
    "Qwen3-235B-A22B-Instruct": 235,
}


def plot_scaling(all_data, models, n_values, output_dir):
    """Scatter plot: model size vs baseline δ and best-intervention δ."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    sizes, base_deltas, best_deltas, labels = [], [], [], []
    for model in models:
        if model not in MODEL_PARAMS:
            continue
        base = all_data.get(model, {}).get("baseline", {})
        bd = np.nanmean([base.get("delta_by_n", {}).get(n, {}).get("delta", np.nan)
                          for n in n_values])
        # find best intervention (lowest delta)
        best_d = bd
        for cond in ["intervention_a", "intervention_b", "intervention_c", "intervention_ab"]:
            cd = all_data.get(model, {}).get(cond, {})
            cd_d = np.nanmean([cd.get("delta_by_n", {}).get(n, {}).get("delta", np.nan)
                                for n in n_values])
            if not np.isnan(cd_d) and cd_d < best_d:
                best_d = cd_d

        sizes.append(MODEL_PARAMS[model])
        base_deltas.append(bd)
        best_deltas.append(best_d)
        labels.append(_display(model))

    ax.scatter(sizes, base_deltas, s=80, c="#e74c3c", marker="o", label="Baseline δ",
               edgecolors="black", linewidth=0.5, zorder=3)
    ax.scatter(sizes, best_deltas, s=80, c="#2ecc71", marker="D", label="Best intervention δ",
               edgecolors="black", linewidth=0.5, zorder=3)

    # connect pairs
    for sx, bd, btd in zip(sizes, base_deltas, best_deltas):
        ax.plot([sx, sx], [bd, btd], color="gray", linewidth=0.8, alpha=0.5, zorder=1)

    for sx, bd, lbl in zip(sizes, base_deltas, labels):
        ax.annotate(lbl, (sx, bd), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B params)")
    ax.set_ylabel("Deceptive Behavior Score (δ)")
    ax.set_title("Scaling: Deception vs. Model Size", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig5_scaling.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig5_scaling.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 5 (scaling)")


# ============================================================
# Summary Table (LaTeX)
# ============================================================
def generate_table(all_data, models, conditions, n_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def _latex_escape(s):
        return s.replace("&", "\\&")

    lines = []
    header = "Model & Condition & $\\bar{\\rho}$ & $\\bar{\\delta}$ & $\\Delta\\bar{\\delta}$ (\\%) \\\\"
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")

    for model in models:
        mname = _display(model)
        baseline_delta = all_data.get(model, {}).get("baseline", {}).get("overall_delta", np.nan)

        for cond in conditions:
            data = all_data.get(model, {}).get(cond)
            if not data:
                continue
            rho_bar = data.get("overall_rho", np.nan)
            delta_bar = data.get("overall_delta", np.nan)
            clabel = _latex_escape(CONDITION_STYLES.get(cond, {}).get("label", cond))

            rho_str = f"{rho_bar:.4f}" if not np.isnan(rho_bar) else "--"
            delta_str = f"{delta_bar:.4f}" if not np.isnan(delta_bar) else "--"

            if cond == "baseline":
                red_str = "--"
            elif not np.isnan(delta_bar) and not np.isnan(baseline_delta) and baseline_delta > 0:
                pct = (baseline_delta - delta_bar) / baseline_delta * 100
                sign = "+" if pct >= 0 else ""
                red_str = f"{sign}{pct:.1f}\\%"
            else:
                red_str = "--"

            lines.append(f"{mname} & {clabel} & {rho_str} & {delta_str} & {red_str} \\\\")
        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(os.path.join(output_dir, "summary_table.tex"), "w") as f:
        f.write("\n".join(lines))

    # Console printout
    print(f"\n{'='*90}")
    print(f"{'Model':<20} {'Condition':<18} {'ρ̄':>8} {'δ̄':>8} {'Δδ (%)':>9}")
    print(f"{'='*90}")
    for model in models:
        mname = _display(model)
        baseline_delta = all_data.get(model, {}).get("baseline", {}).get("overall_delta", np.nan)
        for cond in conditions:
            data = all_data.get(model, {}).get(cond)
            if not data:
                continue
            rho_bar = data.get("overall_rho", np.nan)
            delta_bar = data.get("overall_delta", np.nan)
            clabel = CONDITION_STYLES.get(cond, {}).get("label", cond)

            rho_s = f"{rho_bar:8.4f}" if not np.isnan(rho_bar) else f"{'--':>8}"
            delta_s = f"{delta_bar:8.4f}" if not np.isnan(delta_bar) else f"{'--':>8}"

            if cond == "baseline":
                red_s = f"{'--':>9}"
            elif not np.isnan(delta_bar) and not np.isnan(baseline_delta) and baseline_delta > 0:
                pct = (baseline_delta - delta_bar) / baseline_delta * 100
                red_s = f"{pct:>+8.1f}%"
            else:
                red_s = f"{'--':>9}"

            print(f"{mname:<20} {clabel:<18} {rho_s} {delta_s} {red_s}")
        print(f"{'-'*90}")

    print(f"\nLaTeX table → {os.path.join(output_dir, 'summary_table.tex')}")


# ============================================================
# Main entry point
# ============================================================
def generate_all_plots(results_dir, models, conditions, n_values, output_dir,
                       target_model=None):
    print("\n=== Collecting analysis data ===")
    all_data = _collect_all(results_dir, models, conditions, n_values)

    # Save raw analysis JSON
    analysis_path = os.path.join(output_dir, "analysis_data.json")

    def _convert(obj):
        if isinstance(obj, (np.floating, float)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    os.makedirs(output_dir, exist_ok=True)
    with open(analysis_path, "w") as f:
        json.dump(all_data, f, indent=2, default=_convert)
    print(f"  Saved analysis data to {analysis_path}")

    if target_model is None:
        target_model = models[-1]

    print("\n=== Generating Figure 1 (δ vs n) ===")
    plot_figure1(all_data, target_model, n_values, output_dir, models=models)

    print("\n=== Generating Figure 2 (heatmap) ===")
    plot_figure2(all_data, models, conditions, n_values, output_dir)

    print("\n=== Generating Figure 3 (before/after + accuracy) ===")
    plot_figure3(all_data, models, n_values, output_dir)

    print("\n=== Generating Figure 4 (regime analysis) ===")
    plot_regime_analysis(all_data, models, conditions, n_values, output_dir)

    print("\n=== Generating Figure 5 (scaling) ===")
    plot_scaling(all_data, models, n_values, output_dir)

    print("\n=== Generating Summary Table ===")
    generate_table(all_data, models, conditions, n_values, output_dir)

    return all_data
