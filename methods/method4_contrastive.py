#!/usr/bin/env python3
"""
Method 4: Deception-Contrastive Decoding (DCD)
===============================================
Mathematical formulation
------------------------
Inspired by contrastive decoding (Li et al., 2023), we define:

Expert distribution:
    p(y | x_broken) = next-token probability of y under the full BrokenLinkedList prompt

Amateur distribution:
    q(y | x_plain)  = next-token probability of y under the same question
                      but WITHOUT the broken-edge rule — i.e., presented as a
                      regular LinkedList where all edges appear to exist.
                      The amateur is systematically over-confident about Yes
                      (it cannot see the break), so its log-probability encodes
                      a "naive" bias we wish to subtract.

Contrastive score for answer y:
    s(y | x) = log p(y | x_broken)  −  β · log q(y | x_plain)

subject to the adaptive plausibility constraint:
    p(y | x_broken) ≥ γ · max_{y'} p(y' | x_broken)   (γ = 0.1)

Final answer:
    y* = argmax_{y ∈ {Yes, No}, feasible} s(y | x)

β = 0    → pure expert (baseline)
β = 1    → full contrastive subtraction
β > 1    → over-subtraction (aggressive)

Amateur prompt construction
---------------------------
The plain prompt is built by:
1. Taking the BrokenLinkedList prompt.
2. Removing any mention of broken edges from the Facts section.
3. The path facts are all stated as active contacts.
This makes the path reachable, so the amateur confidently says "Yes";
subtracting this bias corrects the expert toward "No" when appropriate.

Visualisations
--------------
1. Log-ratio distribution: log p/q for deceptive vs consistent questions.
2. β sweep: δ and accuracy vs β.
3. Score scatter: expert vs amateur log-probabilities, coloured by outcome.
4. Per-model comparison bar chart.

Usage
-----
python -m methods.method4_contrastive --device 0
python -m methods.method4_contrastive --device 1 --models Phi-4-mini-instruct
"""

import os
import sys
import json
import re
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))
from methods.common import (
    set_style, savefig, C, COL1, COL2, ROW,
    ALL_MODELS, MODEL_DISPLAY, MODEL_HF_IDS, N_VALUES,
    load_questions, load_results, save_results,
    load_model_and_tokenizer, unload,
    prompt_baseline, chat_prompt,
    get_yes_no_logprobs, extract_yes_no,
    delta_direct, accuracy_initial,
    plot_delta_comparison,
)

METHOD_TAG    = "method4_contrastive"
RESULTS_DIR   = "results"
OUT_RESULTS   = f"results_methods/{METHOD_TAG}"
OUT_PLOTS     = f"plots/methods/{METHOD_TAG}"
QUESTIONS_DIR = "questions"

BETA_VALUES   = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
GAMMA         = 0.1    # plausibility threshold


# ── Amateur prompt construction ────────────────────────────────────────────────

def build_plain_prompt(question: dict, tok) -> str:
    """
    Build the 'amateur' prompt: same question but with all broken edges
    removed from the Facts section, making all edges appear active.
    The amateur model will believe the path is fully connected.
    """
    problem = question["problem"]
    broken  = question.get("broken_edges", [])

    # Remove broken-edge mentions from the Facts section.
    # Broken edges are listed as facts *about contact that does NOT exist*
    # — actually in this dataset they are stated as facts that exist but the
    # path is broken. We reconstruct the full (non-broken) chain facts.
    chain = question["linked_list"]

    # Build plain facts: every consecutive pair in the chain can contact
    plain_facts = "\n".join(f"{chain[i]} can contact {chain[i+1]}"
                             for i in range(len(chain) - 1))

    # Replace the Facts section in the original problem
    # Pattern: "Facts:\n...\n---"
    plain_problem = re.sub(
        r"(Facts:\n)(.+?)(\n---)",
        lambda m: m.group(1) + plain_facts + m.group(3),
        problem,
        flags=re.DOTALL
    )
    # If the regex didn't match (different format), just append plain facts
    if plain_problem == problem:
        plain_problem = problem + f"\n[Plain chain facts]\n{plain_facts}"

    msgs = [{"role": "user", "content": plain_problem}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ── Contrastive decoding ──────────────────────────────────────────────────────

@torch.no_grad()
def contrastive_answer(model, tok, question, beta, device):
    """
    Run DCD for a single question.
    Returns: y*, log_p_expert, log_p_amateur (for Yes and No)
    """
    expert_prompt  = prompt_baseline(question, tok)
    amateur_prompt = build_plain_prompt(question, tok)

    # Expert log-probs
    lp_yes_e, lp_no_e = get_yes_no_logprobs(model, tok, expert_prompt,  device)
    # Amateur log-probs
    lp_yes_a, lp_no_a = get_yes_no_logprobs(model, tok, amateur_prompt, device)

    # Plausibility filter: keep only y where p(y|expert) >= gamma * max_y p(y|expert)
    max_lp_expert = max(lp_yes_e, lp_no_e)
    feasible_yes  = lp_yes_e >= np.log(GAMMA) + max_lp_expert
    feasible_no   = lp_no_e  >= np.log(GAMMA) + max_lp_expert

    # Contrastive scores
    score_yes = lp_yes_e - beta * lp_yes_a if feasible_yes else -1e9
    score_no  = lp_no_e  - beta * lp_no_a  if feasible_no  else -1e9

    answer = "Yes" if score_yes >= score_no else "No"

    # Baseline (β=0): pure expert
    baseline_ans = "Yes" if lp_yes_e >= lp_no_e else "No"

    return {
        "answer":        answer,
        "baseline_ans":  baseline_ans,
        "lp_yes_expert": float(lp_yes_e),
        "lp_no_expert":  float(lp_no_e),
        "lp_yes_amateur":float(lp_yes_a),
        "lp_no_amateur": float(lp_no_a),
        "score_yes":     float(score_yes),
        "score_no":      float(score_no),
        "log_ratio_yes": float(lp_yes_e - lp_yes_a),
        "log_ratio_no":  float(lp_no_e  - lp_no_a),
    }


def run_dcd_batch(model, tok, questions, results_ref, beta, device, max_q=None):
    n = len(questions) if max_q is None else min(len(questions), max_q)
    out = []
    for idx, (q, ref) in enumerate(zip(questions[:n], results_ref[:n])):
        try:
            cd = contrastive_answer(model, tok, q, beta, device)
        except Exception as e:
            print(f"  Warning: q{idx} failed: {e}")
            cd = {
                "answer": None, "baseline_ans": None,
                "lp_yes_expert": 0.0, "lp_no_expert": 0.0,
                "lp_yes_amateur": 0.0, "lp_no_amateur": 0.0,
                "score_yes": 0.0, "score_no": 0.0,
                "log_ratio_yes": 0.0, "log_ratio_no": 0.0,
            }

        gt    = q.get("answer", "")
        fu_gt = q.get("followup_problem", {}).get("answer", "")
        fu_ans = ref.get("followup_answer")

        out.append({
            "initial_output":        cd["answer"],
            "initial_answer":        cd["answer"],
            "followup_output":       ref.get("followup_output", ""),
            "followup_answer":       fu_ans,
            "ground_truth":          gt,
            "followup_ground_truth": fu_gt,
            "initial_is_correct":    cd["answer"] == gt if cd["answer"] else False,
            "followup_is_correct":   fu_ans == fu_gt if fu_ans else False,
            **{k: cd[k] for k in cd if k not in ("answer",)},
        })

        if idx % 10 == 0:
            print(f"    [{idx+1}/{n}]", flush=True)
    return out


# ── Visualizations ─────────────────────────────────────────────────────────────

def plot_log_ratio_distribution(results, model_display, out_path):
    """
    Kernel-smoothed histogram of log(p_expert/p_amateur) for Yes answers,
    split by deceptive vs consistent questions.
    """
    set_style()
    from scipy.stats import gaussian_kde

    dec_ratios = [r["log_ratio_yes"] for r in results
                  if not r.get("initial_is_correct") and r.get("followup_is_correct")
                  and "log_ratio_yes" in r]
    con_ratios = [r["log_ratio_yes"] for r in results
                  if r.get("initial_is_correct") and r.get("followup_is_correct")
                  and "log_ratio_yes" in r]

    fig, ax = plt.subplots(figsize=(COL1 * 1.2, ROW))
    all_vals = dec_ratios + con_ratios
    if not all_vals:
        plt.close(fig)
        return

    x = np.linspace(min(all_vals) - 1, max(all_vals) + 1, 300)

    for data, color, label in [
        (con_ratios, C["consistent"], "Consistent"),
        (dec_ratios, C["deceptive"],  "Deceptive"),
    ]:
        if len(data) < 3:
            continue
        kde = gaussian_kde(data, bw_method=0.3)
        ax.fill_between(x, kde(x), alpha=0.3, color=color)
        ax.plot(x, kde(x), color=color, lw=1.5, label=f"{label} (n={len(data)})")

    ax.axvline(0, ls="--", color="gray", lw=0.8)
    ax.set_xlabel(r"$\log\, p_{\mathrm{expert}}(y) - \log\, q_{\mathrm{amateur}}(y)$")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_display} — Expert vs Amateur log-ratio")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out_path)


def plot_expert_vs_amateur_scatter(results, model_display, out_path):
    """
    Scatter: log p_expert(Yes) vs log q_amateur(Yes), coloured by outcome.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.9))

    for cat, color, label in [
        ("consistent", C["consistent"], "Consistent"),
        ("deceptive",  C["deceptive"],  "Deceptive"),
        ("both_wrong", "#95a5a6",        "Both wrong"),
    ]:
        pts = []
        for r in results:
            i = r.get("initial_is_correct", False)
            f = r.get("followup_is_correct", False)
            if cat == "consistent" and i and f:     pts.append(r)
            if cat == "deceptive"  and not i and f: pts.append(r)
            if cat == "both_wrong" and not i and not f: pts.append(r)
        if not pts: continue
        xe = [p["lp_yes_expert"]  for p in pts]
        ya = [p["lp_yes_amateur"] for p in pts]
        ax.scatter(xe, ya, c=color, alpha=0.4, s=10, linewidths=0, label=label)

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", lw=0.7)
    ax.set_xlabel(r"$\log p_{\rm expert}(\rm Yes)$")
    ax.set_ylabel(r"$\log q_{\rm amateur}(\rm Yes)$")
    ax.set_title(f"{model_display}\nExpert vs Amateur (Yes)")
    ax.legend(fontsize=7, markerscale=1.5)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_beta_sweep(beta_vals, delta_vals, acc_vals, model_display, out_path):
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, ROW))

    ax1.plot(beta_vals, delta_vals, marker="o", color=C["method4"])
    ax1.axhline(delta_vals[0], ls="--", color=C["baseline"], lw=1, label="Baseline (β=0)")
    ax1.set_xlabel(r"Contrastive weight $\beta$")
    ax1.set_ylabel(r"Deceptive Behavior Score $\delta$")
    ax1.set_title(f"{model_display} — δ vs β")
    ax1.legend()

    ax2.plot(beta_vals, acc_vals, marker="s", color=C["method4"])
    ax2.axhline(acc_vals[0], ls="--", color=C["baseline"], lw=1, label="Baseline (β=0)")
    ax2.set_xlabel(r"Contrastive weight $\beta$")
    ax2.set_ylabel("Initial Answer Accuracy")
    ax2.set_title(f"{model_display} — Accuracy vs β")
    ax2.legend()

    fig.tight_layout()
    savefig(fig, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_model(model_name, device_id, args):
    hf_id  = MODEL_HF_IDS[model_name]
    mdisp  = MODEL_DISPLAY[model_name]
    device = f"cuda:{device_id}"

    print(f"\n{'='*60}\n{mdisp}\n{'='*60}")

    model, tok = load_model_and_tokenizer(hf_id, device_id)

    # ── β sweep on n=20 ───────────────────────────────────────────────────────
    n_sweep = 20
    qs_sw   = load_questions(QUESTIONS_DIR, "Broken", n_sweep)
    ref_sw  = load_results(RESULTS_DIR, model_name, "baseline", "Broken", n_sweep)
    if not qs_sw or not ref_sw:
        print("  No data, skipping")
        unload(model, tok)
        return None

    sweep_n  = min(100, len(qs_sw))
    beta_deltas, beta_accs = [], []

    print(f"  β sweep on {sweep_n} questions (n={n_sweep})...")
    for beta in BETA_VALUES:
        print(f"  β={beta:.2f} ...", flush=True)
        res = run_dcd_batch(model, tok, qs_sw, ref_sw, beta, device, max_q=sweep_n)
        beta_deltas.append(delta_direct(res))
        beta_accs.append(accuracy_initial(res))

    baseline_acc = beta_accs[0]
    best_beta  = BETA_VALUES[0]
    best_delta = beta_deltas[0]
    for b, d, acc in zip(BETA_VALUES, beta_deltas, beta_accs):
        if acc >= baseline_acc - 0.05 and d < best_delta:
            best_delta = d
            best_beta  = b

    print(f"  Best β={best_beta:.2f}: δ={best_delta:.4f}  (baseline={beta_deltas[0]:.4f})")

    # ── Full evaluation at best β across all n ────────────────────────────────
    all_hca, all_baseline = [], []
    for n in N_VALUES:
        qs  = load_questions(QUESTIONS_DIR, "Broken", n)
        ref = load_results(RESULTS_DIR, model_name, "baseline", "Broken", n)
        if not qs or not ref: continue
        print(f"  Full eval n={n} ({len(qs)} questions)...")
        res = run_dcd_batch(model, tok, qs, ref, best_beta, device)
        save_results(res, OUT_RESULTS, model_name, f"dcd_beta{best_beta:.2f}", "Broken", n)
        all_hca      += res
        all_baseline += ref

    unload(model, tok)

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(OUT_PLOTS, exist_ok=True)

    # Use n=20 sweep results for distribution plots
    n20_full  = run_dcd_batch  # already saved above; reload for plots
    n20_res_path = os.path.join(OUT_RESULTS, model_name,
                                f"dcd_beta{best_beta:.2f}_Broken_n20.json")
    n20_results = json.load(open(n20_res_path)) if os.path.exists(n20_res_path) else all_hca

    plot_log_ratio_distribution(
        n20_results, mdisp,
        os.path.join(OUT_PLOTS, f"log_ratio_dist_{model_name}.pdf")
    )
    plot_expert_vs_amateur_scatter(
        n20_results, mdisp,
        os.path.join(OUT_PLOTS, f"expert_amateur_scatter_{model_name}.pdf")
    )
    plot_beta_sweep(
        BETA_VALUES, beta_deltas, beta_accs, mdisp,
        os.path.join(OUT_PLOTS, f"beta_sweep_{model_name}.pdf")
    )

    return {
        "baseline_delta": delta_direct(all_baseline) if all_baseline else beta_deltas[0],
        "method_delta":   delta_direct(all_hca),
        "best_beta":      best_beta,
        "baseline_acc":   accuracy_initial(all_baseline) if all_baseline else beta_accs[0],
        "method_acc":     accuracy_initial(all_hca),
        "beta_deltas":    beta_deltas,
        "beta_accs":      beta_accs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Method 4: Deception-Contrastive Decoding",
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, ROW))
        for model_name, res in all_results.items():
            lab = MODEL_DISPLAY[model_name]
            ax1.plot(BETA_VALUES, res["beta_deltas"], marker="o", label=lab, markersize=3)
            ax2.plot(BETA_VALUES, res["beta_accs"],   marker="s", label=lab, markersize=3)
        ax1.set_xlabel(r"$\beta$"); ax1.set_ylabel(r"$\delta$")
        ax1.legend(fontsize=6, ncol=2); ax1.set_title(r"$\delta$ vs $\beta$")
        ax2.set_xlabel(r"$\beta$"); ax2.set_ylabel("Accuracy")
        ax2.legend(fontsize=6, ncol=2); ax2.set_title(r"Accuracy vs $\beta$")
        fig.suptitle("Method 4: Deception-Contrastive Decoding", fontsize=9)
        fig.tight_layout()
        savefig(fig, os.path.join(OUT_PLOTS, "all_models_beta_sweep.pdf"))

        plot_delta_comparison(
            baseline_deltas={m: r["baseline_delta"] for m, r in all_results.items()},
            method_deltas  ={m: r["method_delta"]   for m, r in all_results.items()},
            method_label="Contrastive Decoding",
            method_color=C["method4"],
            out_path=os.path.join(OUT_PLOTS, "summary_bar.pdf"),
            title="Method 4: Deception-Contrastive Decoding — δ Comparison",
            also_plot_acc=True,
            baseline_accs={m: r["baseline_acc"] for m, r in all_results.items()},
            method_accs  ={m: r["method_acc"]   for m, r in all_results.items()},
        )

        with open(os.path.join(OUT_PLOTS, "summary.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nDone. Plots: {OUT_PLOTS}")


if __name__ == "__main__":
    main()
