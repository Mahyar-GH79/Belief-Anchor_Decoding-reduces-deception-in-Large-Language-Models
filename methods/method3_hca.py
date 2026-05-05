#!/usr/bin/env python3
"""
Method 3: Hierarchical Consistency Annealing (HCA)
===================================================
Mathematical formulation
------------------------
For a BrokenLinkedList question x at full chain length N, we construct a
difficulty hierarchy by extracting sub-chains of increasing length k:

    k ∈ K = {k_min, ..., N}   where k_min = 3 (src + 1 intermediate + tgt)

For each sub-chain level k, we query the model for its answer y_k ∈ {Yes, No}.
Because shorter chains are easier (less context to track), y_{k_min} is the
model's most reliable answer.

Consistency profile: c = (y_{k_1}, y_{k_2}, ..., y_N)

Consistency penalty (number of answer flips along the hierarchy):
    Φ(c) = Σ_{i=1}^{|K|-1}  𝟙[y_{k_{i+1}} ≠ y_{k_i}]

Re-scored log-probability:
    score(y | x) = log P(y | x_N)  −  λ Φ(c_y)

where c_y is the consistency profile obtained when the full answer is y,
and log P(y | x_N) is obtained via the next-token logprob.

Final answer:
    y* = argmax_{y ∈ {Yes, No}} score(y | x)

Key property: a model that is deceptive will show a flip pattern
(correct at low k, wrong at high k), and the penalty discourages this.
A consistently honest or consistently deceptive model gets Φ = 0.

λ = 0    → pure log-probability (baseline)
λ > 0    → penalise inconsistency along the hierarchy

Visualisations
--------------
1. Consistency profile plot: fraction of questions with each flip pattern.
2. λ sweep: δ and accuracy vs λ.
3. Sankey-style flow diagram showing how answers change across difficulty levels.
4. Per-model comparison bar chart.

Usage
-----
python -m methods.method3_hca --device 0
python -m methods.method3_hca --device 1 --models Phi-4-mini-instruct
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import Counter
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent.parent))
from methods.common import (
    set_style, savefig, C, COL1, COL2, ROW,
    ALL_MODELS, MODEL_DISPLAY, MODEL_HF_IDS, N_VALUES,
    load_questions, load_results, save_results,
    load_model_and_tokenizer, unload,
    prompt_baseline, chat_prompt,
    get_yes_no_logprobs, greedy_answer, extract_yes_no,
    delta_direct, accuracy_initial,
    plot_delta_comparison,
    build_subchain_questions,
)

METHOD_TAG    = "method3_hca"
RESULTS_DIR   = "results"
OUT_RESULTS   = f"results_methods/{METHOD_TAG}"
OUT_PLOTS     = f"plots/methods/{METHOD_TAG}"
QUESTIONS_DIR = "questions"

LAMBDA_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
# Sub-chain lengths to use in the hierarchy (intermediate nodes, not counting src/tgt)
SUB_LENGTHS   = [2, 4, 8]   # + full chain = 4 points in hierarchy


# ── HCA inference ─────────────────────────────────────────────────────────────

def run_hca_on_question(model, tok, question, lam, device):
    """
    Run HCA for a single question.
    Returns result dict with HCA answer + metadata.
    """
    chain  = question["linked_list"]
    broken = question.get("broken_edges", [])
    gt     = question.get("answer", "")
    fu_gt  = question.get("followup_problem", {}).get("answer", "")

    # Build hierarchy: sub-chains at SUB_LENGTHS + full chain
    sub_qs = build_subchain_questions(question, SUB_LENGTHS)
    # Full chain question (use original problem text)
    full_q = {"problem": question["problem"], "answer": gt}

    # Get log P(Yes), log P(No) at each hierarchy level
    hierarchy_answers = []    # per level: "Yes" or "No"
    hierarchy_correct = []    # per level: bool

    for sq in sub_qs + [full_q]:
        p = prompt_baseline(sq, tok)
        lp_yes, lp_no = get_yes_no_logprobs(model, tok, p, device)
        level_ans = "Yes" if lp_yes >= lp_no else "No"
        hierarchy_answers.append(level_ans)
        hierarchy_correct.append(level_ans == sq["answer"])

    # ── Compute consistency penalty for each candidate answer ─────────────────
    # c_yes: profile obtained by fixing full answer = Yes
    # c_no:  profile obtained by fixing full answer = No
    # (sub-chain answers are taken as-is from the model)
    full_level_idx = len(sub_qs)   # index of full chain in hierarchy_answers

    def flip_count(profile):
        return sum(1 for i in range(len(profile)-1)
                   if profile[i] != profile[i+1])

    # For scoring, use all hierarchy answers + candidate
    # The key insight: penalise the candidate whose inclusion creates more flips
    def score(candidate_ans):
        profile = hierarchy_answers[:full_level_idx] + [candidate_ans]
        phi = flip_count(profile)
        # log P(candidate | full question)
        p = prompt_baseline(full_q, tok)
        lp_yes, lp_no = get_yes_no_logprobs(model, tok, p, device)
        lp = lp_yes if candidate_ans == "Yes" else lp_no
        return lp - lam * phi

    score_yes = score("Yes")
    score_no  = score("No")
    hca_ans   = "Yes" if score_yes >= score_no else "No"

    # Also run baseline (greedy from full question, no penalty)
    p_full     = prompt_baseline(full_q, tok)
    lp_yes_f, lp_no_f = get_yes_no_logprobs(model, tok, p_full, device)
    baseline_ans = "Yes" if lp_yes_f >= lp_no_f else "No"

    # Run followup from original reference (reuse)
    fu_q  = question.get("followup_problem", {})
    fu_ans = None
    if fu_q:
        p_fu = prompt_baseline(fu_q, tok)
        fu_lp_y, fu_lp_n = get_yes_no_logprobs(model, tok, p_fu, device)
        fu_ans = "Yes" if fu_lp_y >= fu_lp_n else "No"

    n_flips = flip_count(hierarchy_answers[:full_level_idx] + [hca_ans])

    return {
        "initial_output":        hca_ans,
        "initial_answer":        hca_ans,
        "baseline_answer":       baseline_ans,
        "followup_output":       fu_ans or "",
        "followup_answer":       fu_ans,
        "ground_truth":          gt,
        "followup_ground_truth": fu_gt,
        "initial_is_correct":    hca_ans == gt if hca_ans else False,
        "followup_is_correct":   fu_ans == fu_gt if fu_ans else False,
        "hierarchy_answers":     hierarchy_answers,
        "n_flips":               n_flips,
        "score_yes":             float(score_yes),
        "score_no":              float(score_no),
    }


def run_hca_batch(model, tok, questions, lam, device, max_questions=None):
    n = len(questions) if max_questions is None else min(len(questions), max_questions)
    results = []
    for idx, q in enumerate(questions[:n]):
        try:
            r = run_hca_on_question(model, tok, q, lam, device)
        except Exception as e:
            print(f"  Warning: question {idx} failed: {e}")
            r = {
                "initial_answer": None, "followup_answer": None,
                "ground_truth": q.get("answer", ""),
                "followup_ground_truth": q.get("followup_problem", {}).get("answer", ""),
                "initial_is_correct": False, "followup_is_correct": False,
                "hierarchy_answers": [], "n_flips": 0,
            }
        results.append(r)
        if idx % 10 == 0:
            print(f"    [{idx+1}/{n}]", flush=True)
    return results


# ── Visualizations ─────────────────────────────────────────────────────────────

def plot_consistency_profiles(all_results_by_lambda, lambda_vals, model_display, out_path):
    """
    Stacked bar chart: fraction of questions with 0, 1, 2, 3+ flips per λ.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(COL1 * 1.5, ROW))

    flip_counts = []
    for lam, results in zip(lambda_vals, all_results_by_lambda):
        counts = Counter(min(r.get("n_flips", 0), 3) for r in results)
        total  = len(results)
        flip_counts.append([counts.get(k, 0) / total for k in range(4)])

    flip_counts = np.array(flip_counts).T   # (4, n_lambda)
    colours = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    labels  = ["0 flips", "1 flip", "2 flips", "3+ flips"]
    x = np.arange(len(lambda_vals))
    bottom = np.zeros(len(lambda_vals))

    for i, (color, label) in enumerate(zip(colours, labels)):
        ax.bar(x, flip_counts[i], bottom=bottom, color=color, label=label, alpha=0.85)
        bottom += flip_counts[i]

    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={la}" for la in lambda_vals], fontsize=7)
    ax.set_ylabel("Fraction of questions")
    ax.set_title(f"{model_display} — Consistency profile distribution")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_answer_flow(results, sub_lengths, model_display, out_path):
    """
    Sankey-style flow: how answers change across hierarchy levels.
    Shows fraction Yes / No at each level and transitions between them.
    """
    set_style()
    n_levels = len(sub_lengths) + 1    # sub-chains + full
    level_labels = [f"n={k+2}" for k in sub_lengths] + ["Full"]

    yes_frac = []
    for li in range(n_levels):
        ans_at_level = [
            r["hierarchy_answers"][li]
            for r in results
            if len(r.get("hierarchy_answers", [])) > li
        ]
        yes_frac.append(sum(a == "Yes" for a in ans_at_level) / len(ans_at_level)
                        if ans_at_level else 0.5)

    fig, ax = plt.subplots(figsize=(COL1 * 1.4, ROW))
    x = np.arange(n_levels)
    ax.plot(x, yes_frac,    color="#2ecc71", marker="o", lw=1.5, label="P(Yes)")
    ax.plot(x, [1-y for y in yes_frac], color="#e74c3c", marker="s", lw=1.5, label="P(No)")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels, fontsize=8)
    ax.set_ylabel("Fraction of answers")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_display} — Answer flow across hierarchy")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_path)


def plot_lambda_sweep_hca(lambda_vals, delta_vals, acc_vals, model_display, out_path):
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, ROW))
    ax1.plot(lambda_vals, delta_vals, marker="o", color=C["method3"])
    ax1.axhline(delta_vals[0], ls="--", color=C["baseline"], lw=1, label="Baseline")
    ax1.set_xlabel(r"Penalty weight $\lambda$")
    ax1.set_ylabel(r"Deceptive Behavior Score $\delta$")
    ax1.set_title(f"{model_display} — δ vs λ")
    ax1.legend()
    ax2.plot(lambda_vals, acc_vals, marker="s", color=C["method3"])
    ax2.axhline(acc_vals[0], ls="--", color=C["baseline"], lw=1, label="Baseline")
    ax2.set_xlabel(r"Penalty weight $\lambda$")
    ax2.set_ylabel("Initial Answer Accuracy")
    ax2.set_title(f"{model_display} — Accuracy vs λ")
    ax2.legend()
    fig.tight_layout()
    savefig(fig, out_path)


def plot_hierarchy_delta_by_n(results_by_n, n_values, model_display, out_path):
    """
    Line plot: baseline δ and HCA δ as a function of difficulty n.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(COL1 * 1.2, ROW))
    base_d = [delta_direct(results_by_n[n]["baseline"]) for n in n_values
              if n in results_by_n]
    hca_d  = [delta_direct(results_by_n[n]["hca"])      for n in n_values
              if n in results_by_n]
    ns     = [n for n in n_values if n in results_by_n]

    ax.plot(ns, base_d, marker="o", color=C["baseline"], label="Baseline", lw=1.5)
    ax.plot(ns, hca_d,  marker="s", color=C["method3"],  label="HCA",      lw=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Chain length $n$")
    ax.set_ylabel(r"$\delta$")
    ax.set_title(f"{model_display} — δ across difficulty levels")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_model(model_name, device_id, args):
    hf_id  = MODEL_HF_IDS[model_name]
    mdisp  = MODEL_DISPLAY[model_name]
    device = f"cuda:{device_id}"

    print(f"\n{'='*60}\n{mdisp}\n{'='*60}")

    model, tok = load_model_and_tokenizer(hf_id, device_id)

    # ── λ sweep on n=20 (richest deception signal) ────────────────────────────
    n_sweep = 20
    qs_sw   = load_questions(QUESTIONS_DIR, "Broken", n_sweep)
    if not qs_sw:
        print("  No questions found, skipping")
        unload(model, tok)
        return None

    sweep_n = min(100, len(qs_sw))    # smaller for speed during sweep
    print(f"  λ sweep on {sweep_n} questions (n={n_sweep})...")

    lam_deltas, lam_accs          = [], []
    all_results_by_lam = []

    for lam in LAMBDA_VALUES:
        print(f"  λ={lam:.1f} ...", flush=True)
        res = run_hca_batch(model, tok, qs_sw, lam, device, max_questions=sweep_n)
        all_results_by_lam.append(res)
        lam_deltas.append(delta_direct(res))
        lam_accs.append(accuracy_initial(res))

    baseline_acc = lam_accs[0]
    best_lam  = LAMBDA_VALUES[0]
    best_delta = lam_deltas[0]
    for la, d, acc in zip(LAMBDA_VALUES, lam_deltas, lam_accs):
        if acc >= baseline_acc - 0.05 and d < best_delta:
            best_delta = d
            best_lam   = la

    print(f"  Best λ={best_lam:.1f}: δ={best_delta:.4f}  (baseline={lam_deltas[0]:.4f})")

    # ── Full evaluation at best λ across all n ────────────────────────────────
    results_by_n = {}
    for n in N_VALUES:
        qs  = load_questions(QUESTIONS_DIR, "Broken", n)
        ref = load_results(RESULTS_DIR, model_name, "baseline", "Broken", n)
        if not qs:
            continue
        print(f"  Full eval n={n} ({len(qs)} questions)...")
        hca_res  = run_hca_batch(model, tok, qs, best_lam, device)
        save_results(hca_res, OUT_RESULTS, model_name, f"hca_lam{best_lam:.1f}", "Broken", n)

        results_by_n[n] = {
            "baseline": ref or [],
            "hca":      hca_res,
        }

    unload(model, tok)

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(OUT_PLOTS, exist_ok=True)

    plot_consistency_profiles(
        all_results_by_lam, LAMBDA_VALUES, mdisp,
        os.path.join(OUT_PLOTS, f"consistency_profiles_{model_name}.pdf")
    )
    plot_answer_flow(
        all_results_by_lam[0], SUB_LENGTHS, mdisp,
        os.path.join(OUT_PLOTS, f"answer_flow_{model_name}.pdf")
    )
    plot_lambda_sweep_hca(
        LAMBDA_VALUES, lam_deltas, lam_accs, mdisp,
        os.path.join(OUT_PLOTS, f"lambda_sweep_{model_name}.pdf")
    )
    plot_hierarchy_delta_by_n(
        results_by_n, N_VALUES, mdisp,
        os.path.join(OUT_PLOTS, f"delta_by_n_{model_name}.pdf")
    )

    # Aggregate δ across all n at best λ
    all_hca      = []
    all_baseline = []
    for n, d in results_by_n.items():
        all_hca      += d["hca"]
        all_baseline += d["baseline"]

    return {
        "baseline_delta": delta_direct(all_baseline) if all_baseline else lam_deltas[0],
        "method_delta":   delta_direct(all_hca),
        "best_lambda":    best_lam,
        "baseline_acc":   accuracy_initial(all_baseline) if all_baseline else lam_accs[0],
        "method_acc":     accuracy_initial(all_hca),
        "lambda_deltas":  lam_deltas,
        "lambda_accs":    lam_accs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Method 3: Hierarchical Consistency Annealing",
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
        # λ sweep overlay
        set_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, ROW))
        for model_name, res in all_results.items():
            lab = MODEL_DISPLAY[model_name]
            ax1.plot(LAMBDA_VALUES, res["lambda_deltas"], marker="o",
                     label=lab, markersize=3)
            ax2.plot(LAMBDA_VALUES, res["lambda_accs"],   marker="s",
                     label=lab, markersize=3)
        ax1.set_xlabel(r"$\lambda$"); ax1.set_ylabel(r"$\delta$")
        ax1.legend(fontsize=6, ncol=2); ax1.set_title(r"$\delta$ vs $\lambda$")
        ax2.set_xlabel(r"$\lambda$"); ax2.set_ylabel("Accuracy")
        ax2.legend(fontsize=6, ncol=2); ax2.set_title(r"Accuracy vs $\lambda$")
        fig.suptitle("Method 3: Hierarchical Consistency Annealing", fontsize=9)
        fig.tight_layout()
        savefig(fig, os.path.join(OUT_PLOTS, "all_models_lambda_sweep.pdf"))

        plot_delta_comparison(
            baseline_deltas={m: r["baseline_delta"] for m, r in all_results.items()},
            method_deltas  ={m: r["method_delta"]   for m, r in all_results.items()},
            method_label="HCA",
            method_color=C["method3"],
            out_path=os.path.join(OUT_PLOTS, "summary_bar.pdf"),
            title="Method 3: Hierarchical Consistency Annealing — δ Comparison",
            also_plot_acc=True,
            baseline_accs={m: r["baseline_acc"] for m, r in all_results.items()},
            method_accs  ={m: r["method_acc"]   for m, r in all_results.items()},
        )

        with open(os.path.join(OUT_PLOTS, "summary.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nDone. Plots: {OUT_PLOTS}")


if __name__ == "__main__":
    main()
