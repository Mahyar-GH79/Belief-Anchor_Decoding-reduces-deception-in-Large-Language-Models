#!/usr/bin/env python3
"""
Sycophancy / incentivizing-prompt robustness for BAD.

Same Wu §5.4 prefix as runs/run_sycophancy_prompt.py, applied to BAD on all
4 question types at the model's previously selected best λ. The prefix is
prepended to the question text before BAD computes edge beliefs and runs the
guided decode.

Outputs:
    results_methods/method3_bad/<model>/sycophancy_best_lambda_<qtype>_n<n>.json

Usage:
    python -m runs.run_sycophancy_bad --model gemma-2-9b-it --device 0
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from methods.common import (
    ALL_MODELS, MODEL_DISPLAY, MODEL_HF_IDS, N_VALUES,
    load_questions, load_results, save_results,
    load_model_and_tokenizer, unload,
    prompt_baseline, extract_yes_no,
)
from methods.method3_bad import compute_belief_logsum, _get_yes_no_ids
from runs._polarity_bad import is_negative_phrasing, PolarityAwareBeliefProcessor
from runs.run_sycophancy_prompt import SYCOPHANCY_PREFIX


SUMMARY_PATH    = "plots/methods/method3_bad/summary.json"
OUT_RESULTS_DIR = "results_methods/method3_bad"
QUESTIONS_DIR   = "questions"
RESULTS_DIR     = "results"
ALL_QTYPES      = ["Linked", "LinkedReverse", "Broken", "BrokenReverse"]


def add_prefix(q):
    """Return a deep-copied question dict with SYCOPHANCY_PREFIX prepended to the
    main 'problem' string and the followup_problem.problem if present."""
    qp = copy.deepcopy(q)
    qp["problem"] = SYCOPHANCY_PREFIX + qp["problem"]
    if "followup_problem" in qp and "problem" in qp["followup_problem"]:
        qp["followup_problem"]["problem"] = SYCOPHANCY_PREFIX + qp["followup_problem"]["problem"]
    return qp


@torch.no_grad()
def run_one(model, tok, q_with_prefix, device, lam, yes_id, no_id):
    yes_logsum = compute_belief_logsum(model, tok, q_with_prefix, device)
    polarity   = is_negative_phrasing(q_with_prefix["problem"])
    processor  = PolarityAwareBeliefProcessor(
        yes_logsum, yes_id, no_id, lam, polarity_reversed=polarity
    )
    prompt = prompt_baseline(q_with_prefix, tok)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    out_ids = model.generate(
        **inputs, max_new_tokens=8, do_sample=False,
        logits_processor=[processor], pad_token_id=tok.eos_token_id,
    )
    return tok.decode(
        out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--qtypes", nargs="+", default=ALL_QTYPES, choices=ALL_QTYPES)
    ap.add_argument("--n-values", nargs="+", type=int, default=N_VALUES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(SUMMARY_PATH):
        sys.exit(f"ERROR: {SUMMARY_PATH} not found.")
    summary = json.load(open(SUMMARY_PATH))
    if args.model not in summary or "best_lambda" not in summary[args.model]:
        sys.exit(f"ERROR: best_lambda for {args.model} missing in {SUMMARY_PATH}.")
    best_lam = float(summary[args.model]["best_lambda"])
    print(f"=== {MODEL_DISPLAY[args.model]} sycophancy-BAD | best λ = {best_lam:.2f} ===")

    pending = []
    for qt in args.qtypes:
        for n in args.n_values:
            if not os.path.exists(os.path.join(QUESTIONS_DIR, f"{qt}_n{n}.json")):
                continue
            opath = os.path.join(OUT_RESULTS_DIR, args.model,
                                 f"sycophancy_best_lambda_{qt}_n{n}.json")
            if os.path.exists(opath) and not args.force:
                print(f"  skip {qt} n={n} (exists)")
                continue
            pending.append((qt, n))
    if not pending:
        print("Nothing to do.")
        return

    model, tok = load_model_and_tokenizer(MODEL_HF_IDS[args.model], args.device)
    yes_id, no_id = _get_yes_no_ids(tok)
    device = f"cuda:{args.device}"

    for qt, n in pending:
        qs = load_questions(QUESTIONS_DIR, qt, n)
        baseline = load_results(RESULTS_DIR, args.model,
                                "sycophancy_baseline", qt, n)
        # Fall back to non-sycophancy baseline if the sycophancy baseline file
        # hasn't been generated yet — only used to inherit followup answers.
        if not baseline:
            baseline = load_results(RESULTS_DIR, args.model, "baseline", qt, n)
        print(f"\n  → {qt} n={n}: {len(qs)} questions")

        out = []
        for i, q in enumerate(qs):
            qp = add_prefix(q)
            text = run_one(model, tok, qp, device, best_lam, yes_id, no_id)
            ans  = extract_yes_no(text)
            gt   = q.get("answer", "")
            entry = {
                "initial_output":     text,
                "initial_answer":     ans,
                "ground_truth":       gt,
                "initial_is_correct": (ans == gt) if ans else False,
            }
            if qt in ("Broken", "BrokenReverse") and i < len(baseline):
                fu_gt  = q.get("followup_problem", {}).get("answer", "")
                fu_ans = baseline[i].get("followup_answer")
                entry.update({
                    "followup_output":       baseline[i].get("followup_output", ""),
                    "followup_answer":       fu_ans,
                    "followup_ground_truth": fu_gt,
                    "followup_is_correct":   (fu_ans == fu_gt) if fu_ans else False,
                })
            out.append(entry)
            if (i + 1) % 50 == 0: print(f"    [{i+1}/{len(qs)}]", flush=True)

        save_results(out, OUT_RESULTS_DIR, args.model,
                     "sycophancy_best_lambda", qt, n)
        print(f"    saved → {OUT_RESULTS_DIR}/{args.model}/sycophancy_best_lambda_{qt}_n{n}.json")

    unload(model, tok)
    print("\nDone.")


if __name__ == "__main__":
    main()
