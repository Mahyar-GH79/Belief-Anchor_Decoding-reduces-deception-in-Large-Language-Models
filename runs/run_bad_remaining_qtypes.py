#!/usr/bin/env python3
"""
Extend Method 3 (BAD) to Linked, LinkedReverse, BrokenReverse, using the
best-λ already chosen on Broken (read from plots/methods/method3_bad/summary.json).

This fills the protocol gap required to compute Wu's bias-corrected δ (Eq. 2)
and contribute BAD's behavior to ρ (Eq. 1).

Outputs:
    results_methods/method3_bad/<model>/best_lambda_<qtype>_n<n>.json

Usage:
    python -m runs.run_bad_remaining_qtypes --model gemma-2-9b-it --device 0
"""
import argparse
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


SUMMARY_PATH    = "plots/methods/method3_bad/summary.json"
OUT_RESULTS_DIR = "results_methods/method3_bad"
QUESTIONS_DIR   = "questions"
RESULTS_DIR     = "results"
NEW_QTYPES      = ["Linked", "LinkedReverse", "BrokenReverse"]


@torch.no_grad()
def run_bad_on_question(model, tok, q, device, lam, yes_id, no_id):
    """Apply polarity-aware BAD to one question, return decoded text."""
    yes_logsum = compute_belief_logsum(model, tok, q, device)
    polarity   = is_negative_phrasing(q["problem"])
    processor  = PolarityAwareBeliefProcessor(
        yes_logsum, yes_id, no_id, lam, polarity_reversed=polarity
    )
    prompt = prompt_baseline(q, tok)
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
    ap.add_argument("--qtypes", nargs="+", default=NEW_QTYPES)
    ap.add_argument("--n-values", nargs="+", type=int, default=N_VALUES)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output files")
    args = ap.parse_args()

    if not os.path.exists(SUMMARY_PATH):
        sys.exit(f"ERROR: {SUMMARY_PATH} not found. Run methods/method3_bad.py first.")
    summary = json.load(open(SUMMARY_PATH))
    if args.model not in summary or "best_lambda" not in summary[args.model]:
        sys.exit(f"ERROR: best_lambda for {args.model} missing in {SUMMARY_PATH}.")
    best_lam = float(summary[args.model]["best_lambda"])
    print(f"=== {MODEL_DISPLAY[args.model]} | best λ = {best_lam:.2f} ===")

    pending = []
    for qt in args.qtypes:
        for n in args.n_values:
            qpath = os.path.join(QUESTIONS_DIR, f"{qt}_n{n}.json")
            if not os.path.exists(qpath):
                continue
            opath = os.path.join(OUT_RESULTS_DIR, args.model,
                                 f"best_lambda_{qt}_n{n}.json")
            if os.path.exists(opath) and not args.force:
                print(f"  skip {qt} n={n} (exists)")
                continue
            pending.append((qt, n))
    if not pending:
        print("Nothing to do; pass --force to overwrite.")
        return

    model, tok = load_model_and_tokenizer(MODEL_HF_IDS[args.model], args.device)
    yes_id, no_id = _get_yes_no_ids(tok)
    device = f"cuda:{args.device}"
    print(f"  loaded; Yes_id={yes_id}, No_id={no_id}")

    for qt, n in pending:
        qs = load_questions(QUESTIONS_DIR, qt, n)
        baseline = load_results(RESULTS_DIR, args.model, "baseline", qt, n)
        print(f"\n  → {qt} n={n}: {len(qs)} questions")

        out = []
        for i, q in enumerate(qs):
            text = run_bad_on_question(model, tok, q, device, best_lam, yes_id, no_id)
            ans  = extract_yes_no(text)
            gt   = q.get("answer", "")
            entry = {
                "initial_output":     text,
                "initial_answer":     ans,
                "ground_truth":       gt,
                "initial_is_correct": (ans == gt) if ans else False,
            }
            # Carry-through followup for Broken-family qtypes (BAD doesn't intervene there)
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
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(qs)}]", flush=True)

        save_results(out, OUT_RESULTS_DIR, args.model, "best_lambda", qt, n)
        print(f"    saved → {OUT_RESULTS_DIR}/{args.model}/best_lambda_{qt}_n{n}.json")

    unload(model, tok)
    print("\nDone.")


if __name__ == "__main__":
    main()
