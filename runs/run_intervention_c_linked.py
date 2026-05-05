#!/usr/bin/env python3
"""
Intervention C (Belief-First) on Linked + LinkedReverse via vLLM.

Linked questions have no natural "broken-edge" probe in the dataset, so we
synthesize a single-edge belief probe from the first adjacent pair in the
chain. The probe phrasing matches the main question:

    Linked        → "Can <P_1> contact <P_2>?"        (correct answer: Yes)
    LinkedReverse → "<P_1> cannot contact <P_2>?"     (correct answer: No)

The conversation order is the same as Wu's C protocol:
    Turn 1: probe (the simple sub-question, full facts+rules in context)
    Turn 2: the original long question

Outputs:
    results/<model>/intervention_c_<qtype>_n<n>.json

Usage:
    python -m runs.run_intervention_c_linked --model gemma-2-9b-it --port 8002
"""
import argparse
import json
import os
import re
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.common import ALL_MODELS, MODEL_DISPLAY, N_VALUES
from inference.client import (
    make_client, get_model_id, chat_multi, extract_yes_no,
)


CONFIG_PATH = "configs/models.yaml"
QUESTIONS_DIR = "questions"
RESULTS_DIR   = "results"
QTYPES        = ["Linked", "LinkedReverse"]


def load_config():
    return yaml.safe_load(open(CONFIG_PATH))


def make_probe_question(q, negative_phrasing):
    """Build a single-edge belief probe from the first adjacent pair in the chain.

    The probe shares the facts+rules block of the main prompt and only swaps
    the leading "Derive if ..." line.
    """
    chain = q["linked_list"]
    a, b  = chain[0], chain[1]
    verb  = "cannot" if negative_phrasing else "can"
    new_first_line = (
        f"Derive if {a} {verb} contact {b} based on the following rules and facts, "
        f"answer with a single word 'Yes' or 'No':"
    )
    full = q["problem"]
    lines = full.split("\n")
    lines[0] = new_first_line
    probe_text = "\n".join(lines)
    # Correct answer: positive probe → Yes (edge in facts);
    # negative probe → No (chain edge does exist, so "cannot" is false).
    probe_answer = "No" if negative_phrasing else "Yes"
    return probe_text, probe_answer, (a, b)


def run_one(client, model_id, q, negative_phrasing, temperature):
    probe_text, probe_gt, (a, b) = make_probe_question(q, negative_phrasing)
    messages = []
    probe_resp = chat_multi(client, model_id, messages, probe_text,
                            temperature=temperature, max_tokens=10)
    main_resp  = chat_multi(client, model_id, messages, q["problem"],
                            temperature=temperature, max_tokens=10)
    probe_ans  = extract_yes_no(probe_resp)
    main_ans   = extract_yes_no(main_resp)
    return {
        "initial_output":        main_resp,
        "initial_answer":        main_ans,
        "followup_output":       probe_resp,
        "followup_answer":       probe_ans,
        "ground_truth":          q.get("answer", ""),
        "followup_ground_truth": probe_gt,
        "probe_pair":            [a, b],
        "initial_is_correct":    (main_ans == q.get("answer", "")) if main_ans else False,
        "followup_is_correct":   (probe_ans == probe_gt) if probe_ans else False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--port", type=int, default=None,
                    help="Override vLLM port (default: from configs/models.yaml)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--qtypes", nargs="+", default=QTYPES, choices=QTYPES)
    ap.add_argument("--n-values", nargs="+", type=int, default=N_VALUES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    model_cfg = dict(cfg["models"][args.model])
    if args.port:
        model_cfg["port"] = args.port
    client   = make_client(model_cfg)
    model_id = get_model_id(model_cfg)
    print(f"=== {MODEL_DISPLAY[args.model]} (port {model_cfg.get('port')}) ===")

    for qt in args.qtypes:
        negative = (qt == "LinkedReverse")
        for n in args.n_values:
            qpath = os.path.join(QUESTIONS_DIR, f"{qt}_n{n}.json")
            if not os.path.exists(qpath):
                continue
            out_dir  = os.path.join(RESULTS_DIR, args.model)
            out_path = os.path.join(out_dir, f"intervention_c_{qt}_n{n}.json")
            if os.path.exists(out_path) and not args.force:
                print(f"  skip {qt} n={n} (exists)")
                continue
            qs = json.load(open(qpath))
            print(f"\n  → {qt} n={n}: {len(qs)} questions")
            results = []
            for i, q in enumerate(qs):
                results.append(run_one(client, model_id, q, negative, args.temperature))
                if (i + 1) % 50 == 0:
                    print(f"    [{i+1}/{len(qs)}]", flush=True)
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f)
            print(f"    saved → {out_path}")


if __name__ == "__main__":
    main()
