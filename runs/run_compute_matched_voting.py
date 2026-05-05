#!/usr/bin/env python3
"""
Compute-matched Sample-and-Vote: k = n + 1 majority vote, all 4 question types.

BAD performs ~n forward passes per question to compute edge beliefs, plus one
generation pass — total n + 1. To isolate BAD's algorithmic contribution from
its compute budget, we run plain Sample-and-Vote at the same compute: k = n + 1
samples at temperature 1.0, majority vote.

Outputs:
    results/<model>/intervention_b_matched_<qtype>_n<n>.json

Usage:
    python -m runs.run_compute_matched_voting --model gemma-2-9b-it --port 8002
"""
import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.common import ALL_MODELS, MODEL_DISPLAY, N_VALUES
from inference.client import (
    make_client, make_async_client, get_model_id,
    chat_single, chat_multi, extract_yes_no,
)


CONFIG_PATH   = "configs/models.yaml"
QUESTIONS_DIR = "questions"
RESULTS_DIR   = "results"
ALL_QTYPES    = ["Linked", "LinkedReverse", "Broken", "BrokenReverse"]
LINKED_FAMILY = {"Linked", "LinkedReverse"}
BROKEN_FAMILY = {"Broken", "BrokenReverse"}


def vote(answers):
    valid = [a for a in answers if a]
    if not valid:
        return None, valid
    return Counter(valid).most_common(1)[0][0], valid


def run_linked_cell(client, model_id, qs, k, temperature):
    out = []
    for idx, q in enumerate(qs):
        raw_outputs, votes = [], []
        for _ in range(k):
            r = chat_single(client, model_id, q["problem"],
                            temperature=temperature, max_tokens=10)
            raw_outputs.append(r)
            a = extract_yes_no(r)
            if a:
                votes.append(a)
        majority, _ = vote(votes)
        gt = q.get("answer", "")
        out.append({
            "raw_outputs":        raw_outputs,
            "answer":             majority,
            "votes":              votes,
            "ground_truth":       gt,
            "is_correct":         (majority == gt) if majority else False,
        })
        if (idx + 1) % 25 == 0:
            print(f"      [{idx+1}/{len(qs)}]", flush=True)
    return out


def run_broken_cell(client, model_id, qs, k, temperature):
    out = []
    for idx, q in enumerate(qs):
        i_raw, f_raw, i_votes, f_votes = [], [], [], []
        for _ in range(k):
            messages = []
            i_r = chat_multi(client, model_id, messages, q["problem"],
                             temperature=temperature, max_tokens=10)
            f_r = chat_multi(client, model_id, messages,
                             q["followup_problem"]["problem"],
                             temperature=temperature, max_tokens=10)
            i_raw.append(i_r); f_raw.append(f_r)
            i_a = extract_yes_no(i_r); f_a = extract_yes_no(f_r)
            if i_a: i_votes.append(i_a)
            if f_a: f_votes.append(f_a)
        i_maj, _ = vote(i_votes)
        f_maj, _ = vote(f_votes)
        gt    = q.get("answer", "")
        fu_gt = q.get("followup_problem", {}).get("answer", "")
        out.append({
            "initial_output":        i_raw,
            "initial_answer":        i_maj,
            "initial_votes":         i_votes,
            "followup_output":       f_raw,
            "followup_answer":       f_maj,
            "followup_votes":        f_votes,
            "ground_truth":          gt,
            "followup_ground_truth": fu_gt,
            "initial_is_correct":    (i_maj == gt) if i_maj else False,
            "followup_is_correct":   (f_maj == fu_gt) if f_maj else False,
        })
        if (idx + 1) % 25 == 0:
            print(f"      [{idx+1}/{len(qs)}]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--qtypes", nargs="+", default=ALL_QTYPES, choices=ALL_QTYPES)
    ap.add_argument("--n-values", nargs="+", type=int, default=N_VALUES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG_PATH))
    model_cfg = dict(cfg["models"][args.model])
    if args.port:
        model_cfg["port"] = args.port
    client   = make_client(model_cfg)
    model_id = get_model_id(model_cfg)
    print(f"=== {MODEL_DISPLAY[args.model]} (port {model_cfg.get('port')}) ===")

    for qt in args.qtypes:
        for n in args.n_values:
            qpath = os.path.join(QUESTIONS_DIR, f"{qt}_n{n}.json")
            if not os.path.exists(qpath):
                continue
            out_dir  = os.path.join(RESULTS_DIR, args.model)
            out_path = os.path.join(out_dir, f"intervention_b_matched_{qt}_n{n}.json")
            if os.path.exists(out_path) and not args.force:
                print(f"  skip {qt} n={n} (exists)")
                continue
            qs = json.load(open(qpath))
            k  = n + 1
            print(f"\n  → {qt} n={n}: {len(qs)} questions × k={k} samples")
            if qt in LINKED_FAMILY:
                results = run_linked_cell(client, model_id, qs, k, args.temperature)
            else:
                results = run_broken_cell(client, model_id, qs, k, args.temperature)
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f)
            print(f"    saved → {out_path}")


if __name__ == "__main__":
    main()
