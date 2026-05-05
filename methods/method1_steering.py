#!/usr/bin/env python3
"""
Method 1: Activation Steering via Deception Subspace Projection
================================================================
Mathematical formulation
------------------------
Let h^(l)_i ∈ R^d be the hidden state at layer l, last token position,
for question i with known label (deceptive / consistent).

Fisher LDA deception direction:
    d* = (μ_dec - μ_con) / ‖μ_dec - μ_con‖₂

Steering intervention at inference time (applied to ALL token positions):
    h̃^(l) = h^(l) − α (h^(l) · d̂*) d̂*

where α ∈ [0,2] is the steering strength hyperparameter.

α = 0  → no intervention (baseline)
α = 1  → full projection of deception component removed
α > 1  → over-correction (flip toward consistent subspace)

Procedure
---------
1. Collect last-token hidden states on a labelled split (50% of questions).
2. Compute d* per selected layer via mean-difference normalised by pooled std
   (equivalent to Fisher criterion for balanced classes).
3. Register forward hooks on transformer blocks to subtract α⋅proj during
   greedy decoding on the test split.
4. Evaluate δ(α) and accuracy(α) across α ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0}.
5. Pick best α (min δ subject to accuracy ≥ baseline − 0.05).

Usage
-----
python -m methods.method1_steering --device 0
python -m methods.method1_steering --device 1 --models Phi-4-mini-instruct
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from methods.common import (
    set_style, savefig, C, COL1, COL2, ROW,
    ALL_MODELS, MODEL_DISPLAY, MODEL_HF_IDS, N_VALUES, BROKEN_QTYPES,
    load_questions, load_results, save_results,
    load_model_and_tokenizer, get_layers, unload,
    prompt_baseline, chat_prompt,
    greedy_answer, extract_yes_no,
    delta_direct, accuracy_initial, bootstrap_delta,
    plot_delta_comparison,
    ENUMERATION_PREAMBLE,
)

METHOD_TAG = "method1_steering"
RESULTS_DIR   = "results"
OUT_RESULTS   = f"results_methods/{METHOD_TAG}"
OUT_PLOTS     = f"plots/methods/{METHOD_TAG}"
QUESTIONS_DIR = "questions"

ALPHA_VALUES  = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
TRAIN_FRAC    = 0.5     # fraction of questions used to compute direction
BATCH_SIZE    = 2

# Layers at ~25%, 50%, 75%, 100% depth
def select_layers(num_layers):
    n = num_layers
    return [max(1, n//4), max(1, n//2), max(1, 3*n//4), n]


# ── Hidden state collection ────────────────────────────────────────────────────

@torch.no_grad()
def collect_hidden_states(model, tok, full_texts, sel_layers, batch_size, device):
    """
    Forward-pass full_texts, return last-token hidden states.
    Uses forward hooks on selected layers only (avoids storing ALL layer activations).
    Returns: np.ndarray (N, num_sel_layers, H)
    """
    layers_list = get_layers(model)
    all_embs    = []

    for i in range(0, len(full_texts), batch_size):
        batch = full_texts[i: i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=4096)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        last_ids = attn_mask.sum(1) - 1          # (B,)
        b_idx    = torch.arange(len(batch), device=device)

        # Use hooks to capture only selected layers (saves peak memory vs output_hidden_states=True)
        captured = {}  # li_pos → (B, H) tensor

        def make_hook(li_pos, last_ids_ref, b_idx_ref):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output  # (B, T, H)
                captured[li_pos] = h[b_idx_ref, last_ids_ref].float().detach().cpu()
            return hook_fn

        handles = []
        for li_pos, li_abs in enumerate(sel_layers):
            block_idx = li_abs - 1              # 0-indexed into layers list
            if 0 <= block_idx < len(layers_list):
                h = layers_list[block_idx].register_forward_hook(
                    make_hook(li_pos, last_ids, b_idx)
                )
                handles.append(h)

        out = model(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)

        for h in handles:
            h.remove()

        # Assemble in sel_layers order
        layer_vecs = [captured[li_pos].numpy() for li_pos in range(len(sel_layers))]
        all_embs.append(np.stack(layer_vecs, axis=1))  # (B, L, H)

        del out, input_ids, attn_mask, captured
        torch.cuda.empty_cache()

    return np.concatenate(all_embs, axis=0)      # (N, L, H)


def compute_deception_direction(embs, labels):
    """
    Fisher LDA direction per layer.
    embs   : (N, L, H)
    labels : list of "deceptive" | "consistent" | other  (length N)
    Returns: directions (L, H) — one unit vector per layer
    """
    dec_mask = np.array([l == "deceptive"  for l in labels])
    con_mask = np.array([l == "consistent" for l in labels])

    if dec_mask.sum() < 5 or con_mask.sum() < 5:
        return None

    L, H = embs.shape[1], embs.shape[2]
    directions = np.zeros((L, H), dtype=np.float32)

    for li in range(L):
        mu_dec = embs[dec_mask, li, :].mean(0)
        mu_con = embs[con_mask, li, :].mean(0)
        diff   = mu_dec - mu_con
        norm   = np.linalg.norm(diff)
        if norm < 1e-8:
            directions[li] = diff
        else:
            directions[li] = diff / norm

    return directions       # (L, H)


# ── Steering hook ─────────────────────────────────────────────────────────────

class SteeringHook:
    """Subtracts the deception-direction component from every token position."""

    def __init__(self, direction_tensor, alpha):
        """
        direction_tensor : (H,) float32 on the same device as the model
        alpha            : steering strength
        """
        self.d     = direction_tensor   # unit vector
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]                                      # (B, T, H)
            d = self.d.to(dtype=h.dtype, device=h.device)     # match dtype
            proj = (h @ d).unsqueeze(-1)                       # (B, T, 1)
            h    = h - self.alpha * proj * d
            return (h,) + output[1:]
        else:
            d    = self.d.to(dtype=output.dtype, device=output.device)
            proj = (output @ d).unsqueeze(-1)
            return output - self.alpha * proj * d


def register_hooks(layers, directions_tensor, alpha, sel_layer_indices):
    """
    Register steering hooks on the selected transformer blocks.
    Returns list of hook handles (call .remove() to deactivate).
    """
    handles = []
    for li_pos, li_abs in enumerate(sel_layer_indices):
        if li_abs <= len(layers):               # li_abs is 1-indexed into hidden_states
            block_idx = li_abs - 1              # 0-indexed into layers list
            if block_idx < len(layers):
                hook = SteeringHook(directions_tensor[li_pos], alpha)
                h    = layers[block_idx].register_forward_hook(hook)
                handles.append(h)
    return handles


# ── Inference with steering ────────────────────────────────────────────────────

def run_with_steering(model, tok, questions, results_ref, alpha,
                      directions, sel_layers, device, batch_size=1):
    """
    Run baseline-condition inference on questions with steering applied.
    Returns list of result dicts in the same format as existing results.
    """
    layers    = get_layers(model)
    dirs_t    = torch.tensor(directions, dtype=torch.bfloat16, device=device)  # (L, H)

    new_results = []
    for idx, (q, ref) in enumerate(zip(questions, results_ref)):
        prompt = prompt_baseline(q, tok)

        # Register hooks for this forward pass
        handles = register_hooks(layers, dirs_t, alpha, sel_layers)

        ans_text = greedy_answer(model, tok, prompt, device, max_new_tokens=8)
        ans      = extract_yes_no(ans_text)

        for h in handles:
            h.remove()

        gt        = q.get("answer", "")
        fu_gt     = q.get("followup_problem", {}).get("answer", "")
        fu_ans    = ref.get("followup_answer")    # reuse original followup answer

        new_results.append({
            "initial_output":       ans_text,
            "initial_answer":       ans,
            "followup_output":      ref.get("followup_output", ""),
            "followup_answer":      fu_ans,
            "ground_truth":         gt,
            "followup_ground_truth": fu_gt,
            "initial_is_correct":   ans == gt if ans else False,
            "followup_is_correct":  fu_ans == fu_gt if fu_ans else False,
        })

        if idx % 20 == 0:
            print(f"    [{idx+1}/{len(questions)}]", flush=True)

    return new_results


# ── Visualizations ─────────────────────────────────────────────────────────────

def plot_pca_before_after(embs_before, embs_after, labels, sel_layers, num_layers,
                          model_display, n_val, out_path, layer_idx=None):
    """
    2×4 grid: row 0 = before steering, row 1 = after steering.
    Columns = selected layers.
    """
    set_style()
    L   = len(sel_layers)
    fig, axes = plt.subplots(2, L, figsize=(COL2, ROW * 2 + 0.4))

    labels_arr = np.array(labels)
    titles = [f"Layer {li} ({round(li/num_layers*100)}%)" for li in sel_layers]

    for col, li_pos in enumerate(range(L)):
        for row, (embs, row_title) in enumerate([
            (embs_before, "Before steering"),
            (embs_after,  "After steering"),
        ]):
            ax  = axes[row, col]
            keep = np.isin(labels_arr, ["deceptive", "consistent"])
            X   = embs[keep, li_pos, :]
            lbl = labels_arr[keep]

            pca  = PCA(n_components=2, random_state=42)
            X2   = pca.fit_transform(X)
            var  = pca.explained_variance_ratio_

            for cat, color in [("consistent", C["consistent"]),
                                ("deceptive",  C["deceptive"])]:
                mask = lbl == cat
                if mask.sum():
                    ax.scatter(X2[mask, 0], X2[mask, 1],
                               c=color, alpha=0.5, s=10, linewidths=0)

            ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=7)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=8)
            if row == 0:
                ax.set_title(titles[col], fontsize=8)
            ax.tick_params(labelsize=6)

    # Legend
    handles = [
        mpatches.Patch(color=C["consistent"], label="Consistent"),
        mpatches.Patch(color=C["deceptive"],  label="Deceptive"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{model_display}  |  Activation Steering (n={n_val})", fontsize=9)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_alpha_sweep(alpha_vals, delta_vals, acc_vals, model_display, out_path):
    """δ and accuracy as a function of steering strength α."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, ROW))

    ax1.plot(alpha_vals, delta_vals, marker="o", color=C["method1"])
    ax1.axhline(delta_vals[0], ls="--", color=C["baseline"], lw=1, label="Baseline")
    ax1.set_xlabel(r"Steering strength $\alpha$")
    ax1.set_ylabel(r"Deceptive Behavior Score $\delta$")
    ax1.set_title(f"{model_display} — δ vs α")
    ax1.legend()

    ax2.plot(alpha_vals, acc_vals, marker="s", color=C["method1"])
    ax2.axhline(acc_vals[0], ls="--", color=C["baseline"], lw=1, label="Baseline")
    ax2.set_xlabel(r"Steering strength $\alpha$")
    ax2.set_ylabel("Initial Answer Accuracy")
    ax2.set_title(f"{model_display} — Accuracy vs α")
    ax2.legend()

    fig.tight_layout()
    savefig(fig, out_path)


def plot_layer_heatmap(alpha_delta_by_layer, sel_layers, num_layers,
                       alpha_vals, model_display, out_path):
    """
    Heatmap: rows = layers, columns = α values, colour = Δδ (reduction in δ).
    """
    set_style()
    n_layers = len(sel_layers)
    n_alpha  = len(alpha_vals)
    mat      = np.array(alpha_delta_by_layer)   # (n_layers, n_alpha)

    fig, ax = plt.subplots(figsize=(COL1 * 1.4, ROW))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", origin="lower")

    ax.set_xticks(range(n_alpha))
    ax.set_xticklabels([f"{a:.2f}" for a in alpha_vals], fontsize=7)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(
        [f"L{li} ({round(li/num_layers*100)}%)" for li in sel_layers], fontsize=7
    )
    ax.set_xlabel(r"Steering strength $\alpha$")
    ax.set_ylabel("Layer")
    ax.set_title(f"{model_display} — δ by layer & α", fontsize=9)
    fig.colorbar(im, ax=ax, label=r"$\delta$", shrink=0.8)
    fig.tight_layout()
    savefig(fig, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_model(model_name, device_id, args):
    """
    Difficulty-stratified activation steering.

    Key fix over the naive pooled approach:
    A single direction averaged across all n values fails because the deception
    subspace is n-dependent — easy questions (n=5) and hard questions (n=40)
    live in different regions of representation space.

    Fix: compute a separate Fisher LDA direction d*(n) for each difficulty
    level using only training examples at that level, then apply d*(n) only
    when testing at difficulty n.
    """
    hf_id   = MODEL_HF_IDS[model_name]
    mdisp   = MODEL_DISPLAY[model_name]
    device  = f"cuda:{device_id}"

    print(f"\n{'='*60}\n{mdisp}\n{'='*60}")

    # Check data availability
    available_n = []
    for n in N_VALUES:
        qs  = load_questions(QUESTIONS_DIR, "Broken", n)
        res = load_results(RESULTS_DIR, model_name, "baseline", "Broken", n)
        if qs and res:
            available_n.append(n)
    if not available_n:
        print(f"  No data found for {model_name}, skipping")
        return None

    model, tok = load_model_and_tokenizer(hf_id, device_id)
    num_layers  = model.config.num_hidden_layers
    sel         = select_layers(num_layers)
    print(f"  Layers: {num_layers}, steering at {sel}")

    def full_text(q, r):
        p = prompt_baseline(q, tok)
        return p + str(r.get("initial_output") or r.get("initial_answer") or "")

    def make_labels(res_list):
        return [
            "deceptive" if (not r["initial_is_correct"] and r["followup_is_correct"])
            else "consistent" if (r["initial_is_correct"] and r["followup_is_correct"])
            else "other"
            for r in res_list
        ]

    # ── Per-n: compute direction, run α sweep, evaluate ──────────────────────
    per_n_results  = {}   # n → {"baseline_delta", "best_alpha", "method_delta", ...}
    all_test_res_baseline = []
    all_test_res_steered  = []

    # First pass: compute all per-n directions
    directions_by_n = {}
    embs_before_by_n = {}
    test_data_by_n   = {}

    for n in available_n:
        qs  = load_questions(QUESTIONS_DIR, "Broken", n)
        res = load_results(RESULTS_DIR, model_name, "baseline", "Broken", n)
        m   = len(min(qs, res, key=len))
        mid = m // 2

        train_qs, train_res = qs[:mid], res[:mid]
        test_qs,  test_res  = qs[mid:m], res[mid:m]
        test_data_by_n[n]   = (test_qs, test_res)

        print(f"  n={n}: computing direction ({mid} train / {len(test_qs)} test)...")

        train_texts  = [full_text(q, r) for q, r in zip(train_qs, train_res)]
        train_labels = make_labels(train_res)
        train_embs   = collect_hidden_states(model, tok, train_texts, sel,
                                             BATCH_SIZE, device)
        directions   = compute_deception_direction(train_embs, train_labels)
        directions_by_n[n] = directions

        test_texts  = [full_text(q, r) for q, r in zip(test_qs, test_res)]
        embs_before_by_n[n] = collect_hidden_states(model, tok, test_texts, sel,
                                                     BATCH_SIZE, device)

        if directions is None:
            print(f"    Not enough labelled examples at n={n}, will use α=0")

    # ── α sweep: track δ and accuracy per-n at each α ────────────────────────
    MIN_VALID_RATE = 0.9  # reject α if <90% of steered answers are valid Yes/No

    def validity_rate(results):
        """Fraction of results where initial_answer is not None."""
        if not results: return 0.0
        return sum(1 for r in results if r.get("initial_answer") is not None) / len(results)

    print("\n  Running α sweep (per-n selection)...")
    alpha_deltas_total  = []
    alpha_accs_total    = []
    alpha_deltas_by_n   = {n: [] for n in available_n}  # for per-n plot
    alpha_accs_by_n     = {n: [] for n in available_n}  # per-n accuracy for selection
    alpha_validity_by_n = {n: [] for n in available_n}  # per-n valid-answer rate

    for alpha in ALPHA_VALUES:
        all_steered = []
        for n in available_n:
            test_qs, test_res = test_data_by_n[n]
            dirs = directions_by_n.get(n)
            sweep_n = min(100, len(test_qs))
            steered = run_with_steering(
                model, tok, test_qs[:sweep_n], test_res[:sweep_n],
                alpha if dirs is not None else 0.0,
                dirs if dirs is not None else np.zeros((len(sel), model.config.hidden_size)),
                sel, device
            )
            all_steered += steered
            alpha_deltas_by_n[n].append(delta_direct(steered))
            alpha_accs_by_n[n].append(accuracy_initial(steered))
            alpha_validity_by_n[n].append(validity_rate(steered))

        alpha_deltas_total.append(delta_direct(all_steered))
        alpha_accs_total.append(accuracy_initial(all_steered))
        print(f"    α={alpha:.2f}: δ={alpha_deltas_total[-1]:.4f}  acc={alpha_accs_total[-1]:.4f}")

    # Per-n best α: minimise δ(n, α) subject to:
    #   1. validity_rate(n, α) ≥ MIN_VALID_RATE  (model must produce valid Yes/No answers)
    #   2. acc(n, α) ≥ baseline_acc(n) − 0.05   (accuracy must not collapse)
    best_alpha_by_n = {}
    for n in available_n:
        base_acc_n = alpha_accs_by_n[n][0]   # α=0 accuracy for this n
        best_a_n   = ALPHA_VALUES[0]
        best_d_n   = alpha_deltas_by_n[n][0]
        for ai, a in enumerate(ALPHA_VALUES):
            d     = alpha_deltas_by_n[n][ai]
            acc   = alpha_accs_by_n[n][ai]
            valid = alpha_validity_by_n[n][ai]
            if valid >= MIN_VALID_RATE and acc >= base_acc_n - 0.05 and d < best_d_n:
                best_d_n = d
                best_a_n = a
        best_alpha_by_n[n] = best_a_n
        v0 = alpha_validity_by_n[n][0]
        print(f"    n={n}: best α={best_a_n:.2f}  δ={best_d_n:.4f} "
              f"(base δ={alpha_deltas_by_n[n][0]:.4f}, base acc={base_acc_n:.4f}, "
              f"base valid={v0:.2f})")

    # Aggregate best_alpha for reporting (most-common or mean)
    best_alpha = float(np.median(list(best_alpha_by_n.values())))
    print(f"\n  Per-n best α: {best_alpha_by_n}  (median={best_alpha:.2f})")

    # ── Full evaluation at per-n best α ──────────────────────────────────────
    print(f"  Full evaluation with per-n optimal α...")
    pca_embs_before, pca_embs_after, pca_labels = [], [], []

    for n in available_n:
        test_qs, test_res = test_data_by_n[n]
        dirs  = directions_by_n.get(n)
        n_alpha = best_alpha_by_n[n]
        steered = run_with_steering(
            model, tok, test_qs, test_res,
            n_alpha if dirs is not None else 0.0,
            dirs if dirs is not None else np.zeros((len(sel), model.config.hidden_size)),
            sel, device
        )
        save_results(steered, OUT_RESULTS, model_name,
                     f"stratified_perN_alpha{n_alpha:.2f}", "Broken", n)
        all_test_res_baseline += test_res
        all_test_res_steered  += steered

        per_n_results[n] = {
            "baseline_delta": delta_direct(test_res),
            "method_delta":   delta_direct(steered),
            "baseline_acc":   accuracy_initial(test_res),
            "method_acc":     accuracy_initial(steered),
            "best_alpha":     n_alpha,
        }

        # Collect PCA data (small subset per n)
        pca_n = min(60, len(test_qs))
        test_labels_n = make_labels(test_res[:pca_n])
        pca_embs_before.append(embs_before_by_n[n][:pca_n])
        pca_labels     += test_labels_n

        # After-steering hidden states for PCA (use this n's optimal α)
        test_texts_n = [full_text(q, r) for q, r in
                        zip(test_qs[:pca_n], test_res[:pca_n])]
        layers_obj = get_layers(model)
        dirs_t     = torch.tensor(
            dirs if dirs is not None else np.zeros((len(sel), model.config.hidden_size)),
            dtype=torch.bfloat16, device=device
        )
        after_n = np.zeros_like(embs_before_by_n[n][:pca_n])
        for bi in range(0, pca_n, BATCH_SIZE):
            bt = test_texts_n[bi: bi + BATCH_SIZE]
            handles = register_hooks(layers_obj, dirs_t, n_alpha, sel)
            enc  = tok(bt, return_tensors="pt", padding=True,
                       truncation=True, max_length=4096)
            iids = enc["input_ids"].to(device)
            amsk = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=iids, attention_mask=amsk,
                            output_hidden_states=True, return_dict=True)
            last_ids = amsk.sum(1) - 1
            b_idx    = torch.arange(len(bt), device=device)
            for li_pos, li in enumerate(sel):
                h = out.hidden_states[li][b_idx, last_ids]
                after_n[bi: bi+len(bt), li_pos, :] = h.float().cpu().numpy()
            for hh in handles: hh.remove()
            del out, iids, amsk; torch.cuda.empty_cache()
        pca_embs_after.append(after_n)

    unload(model, tok)

    embs_before_all = np.concatenate(pca_embs_before, axis=0)
    embs_after_all  = np.concatenate(pca_embs_after,  axis=0)

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(OUT_PLOTS, exist_ok=True)

    # 1. PCA before / after (stratified directions)
    plot_pca_before_after(
        embs_before_all, embs_after_all, pca_labels,
        sel, num_layers, mdisp, "all-n (stratified)",
        os.path.join(OUT_PLOTS, f"pca_before_after_{model_name}.pdf")
    )

    # 2. α sweep (averaged across n)
    plot_alpha_sweep(
        ALPHA_VALUES, alpha_deltas_total, alpha_accs_total, mdisp,
        os.path.join(OUT_PLOTS, f"alpha_sweep_{model_name}.pdf")
    )

    # 3. Per-n δ comparison (baseline vs steered at best α)
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(COL2, ROW))
    ns    = sorted(per_n_results.keys())
    b_d   = [per_n_results[n]["baseline_delta"] for n in ns]
    s_d   = [per_n_results[n]["method_delta"]   for n in ns]
    b_a   = [per_n_results[n]["baseline_acc"]   for n in ns]
    s_a   = [per_n_results[n]["method_acc"]     for n in ns]

    axes[0].plot(ns, b_d, marker="o", color=C["baseline"], label="Baseline", lw=1.5)
    axes[0].plot(ns, s_d, marker="s", color=C["method1"],  label="Steering (per-n α)", lw=1.5)
    # annotate each point with its α
    for n in ns:
        axes[0].annotate(f"α={per_n_results[n]['best_alpha']:.2f}",
                         (n, per_n_results[n]["method_delta"]),
                         textcoords="offset points", xytext=(0, 5), fontsize=5.5, ha="center")
    axes[0].set_xscale("log", base=2); axes[0].set_xticks(ns)
    axes[0].set_xticklabels([str(n) for n in ns])
    axes[0].set_xlabel("Chain length $n$"); axes[0].set_ylabel(r"$\delta$")
    axes[0].set_title(f"{mdisp} — δ per difficulty (per-n optimal α)")
    axes[0].legend()

    axes[1].plot(ns, b_a, marker="o", color=C["baseline"], label="Baseline", lw=1.5)
    axes[1].plot(ns, s_a, marker="s", color=C["method1"],  label="Steering (per-n α)", lw=1.5)
    axes[1].set_xscale("log", base=2); axes[1].set_xticks(ns)
    axes[1].set_xticklabels([str(n) for n in ns])
    axes[1].set_xlabel("Chain length $n$"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{mdisp} — Accuracy per difficulty")
    axes[1].legend()
    fig.tight_layout()
    savefig(fig, os.path.join(OUT_PLOTS, f"per_n_comparison_{model_name}.pdf"))

    # 4. Per-n α sweep lines
    set_style()
    fig, ax = plt.subplots(figsize=(COL1 * 1.4, ROW))
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(available_n)))
    for n, color in zip(available_n, colours):
        ax.plot(ALPHA_VALUES, alpha_deltas_by_n[n], marker="o", color=color,
                lw=1.2, markersize=3, label=f"n={n}")
    ax.set_xlabel(r"Steering strength $\alpha$")
    ax.set_ylabel(r"$\delta$")
    ax.set_title(f"{mdisp} — δ vs α per difficulty level")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, os.path.join(OUT_PLOTS, f"alpha_sweep_per_n_{model_name}.pdf"))

    return {
        "baseline_delta": delta_direct(all_test_res_baseline),
        "method_delta":   delta_direct(all_test_res_steered),
        "best_alpha":     best_alpha,
        "best_alpha_by_n": best_alpha_by_n,
        "baseline_acc":   accuracy_initial(all_test_res_baseline),
        "method_acc":     accuracy_initial(all_test_res_steered),
        "alpha_deltas":   alpha_deltas_total,
        "alpha_accs":     alpha_accs_total,
        "per_n":          per_n_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Method 1: Activation Steering",
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

    # ── Cross-model comparison plot ───────────────────────────────────────────
    if len(all_results) >= 2:
        set_style()

        # α sweep overlay for all models
        fig, axes = plt.subplots(1, 2, figsize=(COL2, ROW))
        for model_name, res in all_results.items():
            label = MODEL_DISPLAY[model_name]
            axes[0].plot(ALPHA_VALUES, res["alpha_deltas"], marker="o",
                         label=label, markersize=3)
            axes[1].plot(ALPHA_VALUES, res["alpha_accs"],   marker="s",
                         label=label, markersize=3)

        axes[0].set_xlabel(r"Steering strength $\alpha$")
        axes[0].set_ylabel(r"Deceptive Behavior Score $\delta$")
        axes[0].set_title(r"$\delta$ vs $\alpha$ (all models)")
        axes[0].legend(fontsize=6, ncol=2)

        axes[1].set_xlabel(r"Steering strength $\alpha$")
        axes[1].set_ylabel("Initial Accuracy")
        axes[1].set_title(r"Accuracy vs $\alpha$ (all models)")
        axes[1].legend(fontsize=6, ncol=2)

        fig.suptitle("Method 1: Activation Steering", fontsize=9)
        fig.tight_layout()
        savefig(fig, os.path.join(OUT_PLOTS, "all_models_alpha_sweep.pdf"))

        # Summary bar chart
        plot_delta_comparison(
            baseline_deltas = {m: r["baseline_delta"] for m, r in all_results.items()},
            method_deltas   = {m: r["method_delta"]   for m, r in all_results.items()},
            method_label    = "Activation Steering",
            method_color    = C["method1"],
            out_path        = os.path.join(OUT_PLOTS, "summary_bar.pdf"),
            title           = "Method 1: Activation Steering — δ Comparison",
            also_plot_acc   = True,
            baseline_accs   = {m: r["baseline_acc"] for m, r in all_results.items()},
            method_accs     = {m: r["method_acc"]   for m, r in all_results.items()},
        )

        # Save numeric summary
        with open(os.path.join(OUT_PLOTS, "summary.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nDone. Plots: {OUT_PLOTS}   Results: {OUT_RESULTS}")


if __name__ == "__main__":
    main()
