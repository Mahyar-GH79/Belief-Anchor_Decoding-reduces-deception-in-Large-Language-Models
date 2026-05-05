#!/usr/bin/env python3
"""
Method 2: Probe-Guided Activation Steering
==========================================
Mathematical formulation
------------------------
Train a logistic regression probe P_φ: R^d → [0,1] on labelled hidden states:

    L(φ) = −Σᵢ [yᵢ log P_φ(hᵢ) + (1−yᵢ) log(1−P_φ(hᵢ))]

For logistic regression, P_φ(h) = σ(w·h + b).
The weight vector w ∈ R^d is the normal to the decision boundary.

Probe-based steering direction (normalised):
    d_probe = w / ‖w‖₂

Steering at inference:
    h̃^(l*) = h^(l*) − λ (h^(l*) · d_probe) d_probe

where l* = argmax_l  AUROC_l  (best layer by probe discrimination).

Key difference from Method 1:
    Method 1 uses the *mean-difference* direction (unsupervised, geometric)
    Method 2 uses the *classification-boundary* normal (supervised, discriminative)
    The probe direction is orthogonal to the decision hyperplane that best
    separates deceptive from consistent representations.

Additionally, probe confidence can gate resampling:
    If P_φ(h^(l*)) > τ (high deception probability), resample once and
    keep the answer with lower deception score.

Usage
-----
python -m methods.method2_probe --device 0
python -m methods.method2_probe --device 1 --models Phi-4-mini-instruct Qwen2.5-7B-Instruct
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from methods.common import (
    set_style, savefig, C, COL1, COL2, ROW,
    ALL_MODELS, MODEL_DISPLAY, MODEL_HF_IDS, N_VALUES,
    load_questions, load_results, save_results,
    load_model_and_tokenizer, get_layers, unload,
    prompt_baseline, chat_prompt,
    greedy_answer, extract_yes_no, get_yes_no_logprobs,
    delta_direct, accuracy_initial, bootstrap_delta,
    plot_delta_comparison,
)
from methods.method1_steering import (
    collect_hidden_states, select_layers,
    SteeringHook, register_hooks,
)

METHOD_TAG = "method2_probe"
RESULTS_DIR   = "results"
OUT_RESULTS   = f"results_methods/{METHOD_TAG}"
OUT_PLOTS     = f"plots/methods/{METHOD_TAG}"
QUESTIONS_DIR = "questions"

LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
TRAIN_FRAC    = 0.5
RESAMPLE_TAU  = 0.7   # resampling threshold on deception probability
BATCH_SIZE    = 2


# ── Probe training ─────────────────────────────────────────────────────────────

def train_probes(embs, labels, n_layers):
    """
    Train one logistic regression probe per layer.
    Returns: list of (scaler, clf, auroc) per layer, plus best_layer index.
    """
    binary = np.array([1 if l == "deceptive" else 0 if l == "consistent" else -1
                       for l in labels])
    mask   = binary >= 0   # keep only deceptive / consistent
    if mask.sum() < 20:
        return None, -1

    probes = []
    for li in range(n_layers):
        X = embs[mask, li, :]
        y = binary[mask]

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_sc, y)

        probs  = clf.predict_proba(X_sc)[:, 1]
        auroc  = roc_auc_score(y, probs) if y.sum() > 0 and (1-y).sum() > 0 else 0.5
        probes.append((scaler, clf, auroc))

    best_layer = int(np.argmax([p[2] for p in probes]))
    return probes, best_layer


# ── Resampling with probe gate ─────────────────────────────────────────────────

def probe_deception_prob(h_vec, scaler, clf):
    """Probability that h_vec is deceptive according to probe."""
    X = scaler.transform(h_vec.reshape(1, -1))
    return clf.predict_proba(X)[0, 1]


# ── Inference with probe-guided steering ──────────────────────────────────────

def run_with_probe_steering(model, tok, questions, results_ref,
                            lam, probe_direction, best_layer_idx, sel_layers,
                            scaler, clf, resample_tau, device):
    """
    Inference with:
    (a) Activation steering using the probe weight direction.
    (b) Optional resampling when probe confidence > resample_tau.
    """
    layers_obj = get_layers(model)
    dir_t      = torch.tensor(probe_direction, dtype=torch.bfloat16,
                               device=device).unsqueeze(0)  # (1, H) → broadcast

    # Build a single-layer directions array for register_hooks
    # We only steer at the best layer
    dirs_t     = torch.tensor(
        probe_direction[np.newaxis, :], dtype=torch.bfloat16, device=device
    )   # (1, H)
    best_abs   = sel_layers[best_layer_idx]   # absolute layer index

    new_results = []
    for idx, (q, ref) in enumerate(zip(questions, results_ref)):
        prompt = prompt_baseline(q, tok)

        # Steering pass
        handles = []
        if lam > 0:
            # Register hook only at best layer
            block_idx = best_abs - 1
            if block_idx < len(layers_obj):
                dir_hook = SteeringHook(
                    torch.tensor(probe_direction, dtype=torch.bfloat16, device=device),
                    lam
                )
                handles.append(layers_obj[block_idx].register_forward_hook(dir_hook))

        ans_text = greedy_answer(model, tok, prompt, device, max_new_tokens=8)
        ans      = extract_yes_no(ans_text)
        for h in handles: h.remove()

        gt     = q.get("answer", "")
        fu_gt  = q.get("followup_problem", {}).get("answer", "")
        fu_ans = ref.get("followup_answer")

        new_results.append({
            "initial_output":        ans_text,
            "initial_answer":        ans,
            "followup_output":       ref.get("followup_output", ""),
            "followup_answer":       fu_ans,
            "ground_truth":          gt,
            "followup_ground_truth": fu_gt,
            "initial_is_correct":    ans == gt if ans else False,
            "followup_is_correct":   fu_ans == fu_gt if fu_ans else False,
        })

        if idx % 20 == 0:
            print(f"    [{idx+1}/{len(questions)}]", flush=True)

    return new_results


# ── Visualizations ─────────────────────────────────────────────────────────────

def plot_probe_accuracy_by_layer(probes, sel_layers, num_layers,
                                  model_display, out_path):
    """Bar chart of probe AUROC at each selected layer."""
    set_style()
    fig, ax = plt.subplots(figsize=(COL1 * 1.4, ROW))
    aurocs  = [p[2] for p in probes]
    x       = np.arange(len(sel_layers))
    labels  = [f"L{li}\n({round(li/num_layers*100)}%)" for li in sel_layers]

    bars = ax.bar(x, aurocs, color=C["method2"], alpha=0.85)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Probe AUROC")
    ax.set_title(f"{model_display} — Deception Probe AUROC per Layer")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Annotate best layer
    best_idx = int(np.argmax(aurocs))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(1.5)

    fig.tight_layout()
    savefig(fig, out_path)


def plot_roc_curves(probes, embs, labels, sel_layers, num_layers,
                     model_display, out_path):
    """ROC curves for each layer's probe."""
    set_style()
    binary = np.array([1 if l == "deceptive" else 0 if l == "consistent" else -1
                        for l in labels])
    mask   = binary >= 0

    fig, ax = plt.subplots(figsize=(COL1, COL1))
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(sel_layers)))

    for li_pos, (li, color) in enumerate(zip(sel_layers, colours)):
        scaler, clf, auroc = probes[li_pos]
        X     = embs[mask, li_pos, :]
        y     = binary[mask]
        probs = clf.predict_proba(scaler.transform(X))[:, 1]
        fpr, tpr, _ = roc_curve(y, probs)
        pct = round(li / num_layers * 100)
        ax.plot(fpr, tpr, color=color, lw=1.2,
                label=f"L{li} ({pct}%)  AUC={auroc:.2f}")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_display} — Deception Probe ROC")
    ax.legend(fontsize=6)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_lambda_sweep(lambda_vals, delta_vals, acc_vals, model_display, out_path):
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, ROW))

    ax1.plot(lambda_vals, delta_vals, marker="o", color=C["method2"])
    ax1.axhline(delta_vals[0], ls="--", color=C["baseline"], lw=1, label="Baseline")
    ax1.set_xlabel(r"Probe steering strength $\lambda$")
    ax1.set_ylabel(r"Deceptive Behavior Score $\delta$")
    ax1.set_title(f"{model_display} — δ vs λ")
    ax1.legend()

    ax2.plot(lambda_vals, acc_vals, marker="s", color=C["method2"])
    ax2.axhline(acc_vals[0], ls="--", color=C["baseline"], lw=1, label="Baseline")
    ax2.set_xlabel(r"Probe steering strength $\lambda$")
    ax2.set_ylabel("Initial Answer Accuracy")
    ax2.set_title(f"{model_display} — Accuracy vs λ")
    ax2.legend()

    fig.tight_layout()
    savefig(fig, out_path)


def plot_pca_with_probe_direction(embs, labels, probe_direction,
                                   best_layer_idx, sel_layers, num_layers,
                                   model_display, out_path):
    """
    PCA scatter at best layer with the probe decision direction overlaid as an arrow.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.9))

    labels_arr = np.array(labels)
    keep       = np.isin(labels_arr, ["deceptive", "consistent"])
    X          = embs[keep, best_layer_idx, :]
    lbl        = labels_arr[keep]

    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    for cat, color in [("consistent", C["consistent"]), ("deceptive", C["deceptive"])]:
        mask = lbl == cat
        if mask.sum():
            ax.scatter(X2[mask, 0], X2[mask, 1],
                       c=color, alpha=0.5, s=12, linewidths=0, label=cat.capitalize())

    # Project probe direction into PCA space and draw as arrow
    d_pca = pca.transform(probe_direction.reshape(1, -1))[0]
    d_pca = d_pca / (np.linalg.norm(d_pca) + 1e-8)
    scale = (X2[:, 0].max() - X2[:, 0].min()) * 0.25
    cx, cy = X2[:, 0].mean(), X2[:, 1].mean()
    ax.annotate("", xy=(cx + d_pca[0]*scale, cy + d_pca[1]*scale),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(cx + d_pca[0]*scale*1.15, cy + d_pca[1]*scale*1.15,
            r"$\mathbf{w}_{probe}$", fontsize=8)

    li = sel_layers[best_layer_idx]
    ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({var[1]:.1%})", fontsize=8)
    ax.set_title(f"{model_display}\nLayer {li} ({round(li/num_layers*100)}%) + probe direction")
    ax.legend(fontsize=7, markerscale=1.5)
    fig.tight_layout()
    savefig(fig, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_model(model_name, device_id, args):
    hf_id  = MODEL_HF_IDS[model_name]
    mdisp  = MODEL_DISPLAY[model_name]
    device = f"cuda:{device_id}"

    print(f"\n{'='*60}\n{mdisp}\n{'='*60}")

    # ── Gather labelled data ──────────────────────────────────────────────────
    train_qs, train_res, test_qs, test_res = [], [], [], []
    for n in N_VALUES:
        qs  = load_questions(QUESTIONS_DIR, "Broken", n)
        res = load_results(RESULTS_DIR, model_name, "baseline", "Broken", n)
        if not qs or not res:
            continue
        m   = len(min(qs, res, key=len))
        mid = m // 2
        train_qs += qs[:mid];   train_res += res[:mid]
        test_qs  += qs[mid:m];  test_res  += res[mid:m]

    if not train_qs:
        print(f"  No data, skipping")
        return None

    model, tok = load_model_and_tokenizer(hf_id, device_id)
    num_layers  = model.config.num_hidden_layers
    sel         = select_layers(num_layers)

    def full_text(q, r):
        p = prompt_baseline(q, tok)
        return p + str(r.get("initial_output") or r.get("initial_answer") or "")

    # ── Collect hidden states ─────────────────────────────────────────────────
    print("  Collecting training hidden states...")
    train_labels = [
        "deceptive" if (not r["initial_is_correct"] and r["followup_is_correct"])
        else "consistent" if (r["initial_is_correct"] and r["followup_is_correct"])
        else "other"
        for r in train_res
    ]
    train_texts = [full_text(q, r) for q, r in zip(train_qs, train_res)]
    train_embs  = collect_hidden_states(model, tok, train_texts, sel,
                                         BATCH_SIZE, device)

    print("  Collecting test hidden states...")
    test_labels = [
        "deceptive" if (not r["initial_is_correct"] and r["followup_is_correct"])
        else "consistent" if (r["initial_is_correct"] and r["followup_is_correct"])
        else "other"
        for r in test_res
    ]
    test_texts = [full_text(q, r) for q, r in zip(test_qs, test_res)]
    test_embs  = collect_hidden_states(model, tok, test_texts, sel,
                                        BATCH_SIZE, device)

    # ── Train probes ──────────────────────────────────────────────────────────
    print("  Training probes per layer...")
    probes, best_layer = train_probes(train_embs, train_labels, len(sel))
    if probes is None:
        print("  Not enough labelled data, skipping")
        unload(model, tok)
        return None

    scaler, clf, best_auroc = probes[best_layer]
    probe_dir = clf.coef_[0]
    probe_dir = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
    print(f"  Best layer: {sel[best_layer]} (AUROC={best_auroc:.3f})")

    # ── λ sweep ───────────────────────────────────────────────────────────────
    print("  Running λ sweep...")
    sweep_n = min(200, len(test_qs))
    sweep_qs, sweep_res = test_qs[:sweep_n], test_res[:sweep_n]

    lam_deltas, lam_accs = [], []
    for lam in LAMBDA_VALUES:
        print(f"    λ={lam:.2f} ...", flush=True)
        res_l = run_with_probe_steering(
            model, tok, sweep_qs, sweep_res,
            lam, probe_dir, best_layer, sel,
            scaler, clf, RESAMPLE_TAU, device
        )
        lam_deltas.append(delta_direct(res_l))
        lam_accs.append(accuracy_initial(res_l))

    baseline_acc = lam_accs[0]
    best_lam     = LAMBDA_VALUES[0]
    best_delta   = lam_deltas[0]
    for la, d, acc in zip(LAMBDA_VALUES, lam_deltas, lam_accs):
        if acc >= baseline_acc - 0.05 and d < best_delta:
            best_delta = d
            best_lam   = la

    print(f"  Best λ={best_lam:.2f}: δ={best_delta:.4f}  (baseline={lam_deltas[0]:.4f})")

    # ── Full evaluation at best λ ─────────────────────────────────────────────
    full_results = run_with_probe_steering(
        model, tok, test_qs, test_res,
        best_lam, probe_dir, best_layer, sel,
        scaler, clf, RESAMPLE_TAU, device
    )

    # Save per-n
    idx = 0
    for n in N_VALUES:
        qs  = load_questions(QUESTIONS_DIR, "Broken", n)
        res = load_results(RESULTS_DIR, model_name, "baseline", "Broken", n)
        if not qs or not res: continue
        m   = len(min(qs, res, key=len)); mid = m // 2; n_test = m - mid
        save_results(full_results[idx:idx+n_test], OUT_RESULTS,
                     model_name, "best_lambda", "Broken", n)
        idx += n_test

    unload(model, tok)

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(OUT_PLOTS, exist_ok=True)

    plot_probe_accuracy_by_layer(
        probes, sel, num_layers, mdisp,
        os.path.join(OUT_PLOTS, f"probe_auroc_{model_name}.pdf")
    )
    plot_roc_curves(
        probes, train_embs, train_labels, sel, num_layers, mdisp,
        os.path.join(OUT_PLOTS, f"roc_curves_{model_name}.pdf")
    )
    plot_lambda_sweep(
        LAMBDA_VALUES, lam_deltas, lam_accs, mdisp,
        os.path.join(OUT_PLOTS, f"lambda_sweep_{model_name}.pdf")
    )
    plot_pca_with_probe_direction(
        test_embs, test_labels, probe_dir, best_layer, sel, num_layers, mdisp,
        os.path.join(OUT_PLOTS, f"pca_probe_direction_{model_name}.pdf")
    )

    return {
        "baseline_delta": lam_deltas[0],
        "method_delta":   best_delta,
        "best_lambda":    best_lam,
        "best_layer":     sel[best_layer],
        "best_auroc":     best_auroc,
        "baseline_acc":   lam_accs[0],
        "method_acc":     lam_accs[LAMBDA_VALUES.index(best_lam)],
        "lambda_deltas":  lam_deltas,
        "lambda_accs":    lam_accs,
        "layer_aurocs":   [p[2] for p in probes],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Method 2: Probe-Guided Activation Steering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device",  type=int, default=0)
    parser.add_argument("--models",  nargs="+", default=ALL_MODELS)
    args = parser.parse_args()

    os.makedirs(OUT_PLOTS,   exist_ok=True)
    os.makedirs(OUT_RESULTS, exist_ok=True)

    all_results = {}
    for model_name in args.models:
        if model_name not in MODEL_HF_IDS:
            print(f"Unknown model: {model_name}, skipping")
            continue
        res = run_model(model_name, args.device, args)
        if res:
            all_results[model_name] = res

    if len(all_results) >= 2:
        set_style()
        # AUROC heatmap across models × layers
        models  = list(all_results.keys())
        n_lay   = len(next(iter(all_results.values()))["layer_aurocs"])
        mat     = np.array([all_results[m]["layer_aurocs"] for m in models])

        fig, ax = plt.subplots(figsize=(COL1 * 1.6, ROW + 0.4))
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.5, vmax=1.0)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_DISPLAY[m] for m in models], fontsize=7)
        ax.set_xlabel("Layer index (selected)")
        ax.set_title("Probe AUROC: model × layer")
        fig.colorbar(im, ax=ax, label="AUROC", shrink=0.8)
        fig.tight_layout()
        savefig(fig, os.path.join(OUT_PLOTS, "auroc_heatmap_all_models.pdf"))

        # λ sweep overlay
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, ROW))
        for model_name, res in all_results.items():
            label = MODEL_DISPLAY[model_name]
            ax1.plot(LAMBDA_VALUES, res["lambda_deltas"], marker="o",
                     label=label, markersize=3)
            ax2.plot(LAMBDA_VALUES, res["lambda_accs"],   marker="s",
                     label=label, markersize=3)
        ax1.set_xlabel(r"$\lambda$"); ax1.set_ylabel(r"$\delta$")
        ax1.set_title(r"$\delta$ vs $\lambda$"); ax1.legend(fontsize=6, ncol=2)
        ax2.set_xlabel(r"$\lambda$"); ax2.set_ylabel("Accuracy")
        ax2.set_title(r"Accuracy vs $\lambda$"); ax2.legend(fontsize=6, ncol=2)
        fig.suptitle("Method 2: Probe-Guided Steering", fontsize=9)
        fig.tight_layout()
        savefig(fig, os.path.join(OUT_PLOTS, "all_models_lambda_sweep.pdf"))

        plot_delta_comparison(
            baseline_deltas={m: r["baseline_delta"] for m, r in all_results.items()},
            method_deltas  ={m: r["method_delta"]   for m, r in all_results.items()},
            method_label="Probe Steering",
            method_color=C["method2"],
            out_path=os.path.join(OUT_PLOTS, "summary_bar.pdf"),
            title="Method 2: Probe-Guided Steering — δ Comparison",
            also_plot_acc=True,
            baseline_accs={m: r["baseline_acc"] for m, r in all_results.items()},
            method_accs  ={m: r["method_acc"]   for m, r in all_results.items()},
        )

        with open(os.path.join(OUT_PLOTS, "summary.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nDone. Plots: {OUT_PLOTS}")


if __name__ == "__main__":
    main()
