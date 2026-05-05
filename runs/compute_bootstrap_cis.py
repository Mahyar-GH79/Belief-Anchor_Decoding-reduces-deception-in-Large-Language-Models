#!/usr/bin/env python3
"""
Bootstrap 95% CIs for Wu's bias-corrected δ and ρ across every
(model, condition, n) cell, for the 6 models we report in the paper.

Implements:
    δ_pos(M, n) = Pr(initial wrong ∧ followup right) on Broken
    δ_neg(M, n) = same on BrokenReverse
    δ(M, n)     = sqrt(δ_pos · δ_neg)        (Wu Eq. 2)

    ρ_pos(M, n) = log Pr("Yes"|Linked)        − log Pr("No"|Broken)
    ρ_neg(M, n) = log Pr("No"|LinkedReverse)  − log Pr("Yes"|BrokenReverse)
    ρ(M, n)     = log sqrt(exp(ρ_pos) · exp(ρ_neg))   (Wu Eq. 1)

Bootstrap: resample question indices with replacement (n_boot=1000).

Inputs:
    results/<model>/<condition>_<qtype>_n<n>.json                 (vLLM-based conditions)
    results_methods/method3_bad/<model>/best_lambda_<qtype>_n<n>.json   (BAD)
    results_methods/method3_bad/<model>/sycophancy_best_lambda_<qtype>_n<n>.json
    results/<model>/sycophancy_<cond>_<qtype>_n<n>.json

Outputs:
    analysis/bootstrap_cis.json
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.common import N_VALUES

REPORTED_MODELS = [
    "gemma-2-9b-it",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-32B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-Nemo-Instruct-2407",
    "Phi-4-mini-instruct",
]

# Conditions to evaluate. Maps condition tag → (results_dir, file prefix).
# All conditions live under results/ except "bad" and "sycophancy_bad", which
# live under results_methods/method3_bad/.
CONDITIONS = {
    "baseline":               ("results",                          "baseline"),
    "intervention_a":         ("results",                          "intervention_a"),
    "intervention_b":         ("results",                          "intervention_b"),
    "intervention_c":         ("results",                          "intervention_c"),
    "intervention_ab":        ("results",                          "intervention_ab"),
    "intervention_b_matched": ("results",                          "intervention_b_matched"),
    "bad":                    ("results_methods/method3_bad",      "best_lambda"),
    "sycophancy_baseline":    ("results",                          "sycophancy_baseline"),
    "sycophancy_intervention_c": ("results",                       "sycophancy_intervention_c"),
    "sycophancy_bad":         ("results_methods/method3_bad",      "sycophancy_best_lambda"),
}

QTYPES = ["Linked", "LinkedReverse", "Broken", "BrokenReverse"]
N_BOOT = 1000
RNG    = np.random.default_rng(42)
OUT    = "analysis/bootstrap_cis.json"


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def get_initial_answer(r):
    """Return canonical initial answer ('Yes'|'No'|None) across the various
    JSON schemas we have (Linked = single-turn, Broken = two-turn, voted, BAD)."""
    return r.get("initial_answer") or r.get("answer")


def get_followup_answer(r):
    return r.get("followup_answer")


def is_initial_correct(r):
    if "initial_is_correct" in r:
        return bool(r["initial_is_correct"])
    if "is_correct" in r:
        return bool(r["is_correct"])
    return False


def is_followup_correct(r):
    return bool(r.get("followup_is_correct", False))


# ── Per-cell point estimates ──────────────────────────────────────────────────

def delta_for_cell(broken, broken_rev):
    """Bias-corrected δ. broken / broken_rev are lists of result dicts."""
    def dpos(rs):
        valid = [r for r in rs if get_initial_answer(r) is not None
                                  and get_followup_answer(r) is not None]
        if not valid: return None
        return sum(1 for r in valid
                   if (not is_initial_correct(r)) and is_followup_correct(r)) / len(valid)
    dp = dpos(broken)
    dn = dpos(broken_rev)
    if dp is None or dn is None: return None
    return float(np.sqrt(max(dp, 0.0) * max(dn, 0.0)))


def rho_for_cell(linked, broken, linked_rev, broken_rev):
    """Bias-corrected ρ. Each arg is a list of result dicts; we use only the
    initial answer (single-turn for Linked, first turn for Broken)."""
    def rate(rs, target):
        valid = [r for r in rs if get_initial_answer(r) is not None]
        if not valid: return None
        return sum(1 for r in valid if get_initial_answer(r) == target) / len(valid)
    p_yes_l    = rate(linked,     "Yes")
    p_no_b     = rate(broken,     "No")
    p_no_lrev  = rate(linked_rev, "No")
    p_yes_brev = rate(broken_rev, "Yes")
    if None in (p_yes_l, p_no_b, p_no_lrev, p_yes_brev):
        return None
    eps = 1e-6
    r_pos = (max(p_yes_l, eps)) / (max(p_no_b, eps))
    r_neg = (max(p_no_lrev, eps)) / (max(p_yes_brev, eps))
    return float(np.log(np.sqrt(r_pos * r_neg)))


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_delta(broken, broken_rev, n_boot=N_BOOT):
    if not broken or not broken_rev:
        return None
    nb, nr = len(broken), len(broken_rev)
    samples = []
    for _ in range(n_boot):
        bb = [broken[i]     for i in RNG.integers(0, nb, nb)]
        br = [broken_rev[i] for i in RNG.integers(0, nr, nr)]
        v = delta_for_cell(bb, br)
        if v is not None:
            samples.append(v)
    if not samples: return None
    arr = np.array(samples)
    return {
        "mean":     float(arr.mean()),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
        "n_boot":   len(samples),
    }


def bootstrap_rho(linked, broken, linked_rev, broken_rev, n_boot=N_BOOT):
    parts = [linked, broken, linked_rev, broken_rev]
    if any(not p for p in parts):
        return None
    sizes = [len(p) for p in parts]
    samples = []
    for _ in range(n_boot):
        l  = [linked[i]      for i in RNG.integers(0, sizes[0], sizes[0])]
        b  = [broken[i]      for i in RNG.integers(0, sizes[1], sizes[1])]
        lr = [linked_rev[i]  for i in RNG.integers(0, sizes[2], sizes[2])]
        br = [broken_rev[i]  for i in RNG.integers(0, sizes[3], sizes[3])]
        v = rho_for_cell(l, b, lr, br)
        if v is not None:
            samples.append(v)
    if not samples: return None
    arr = np.array(samples)
    return {
        "mean":     float(arr.mean()),
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
        "n_boot":   len(samples),
    }


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    out = {}
    for model in REPORTED_MODELS:
        out[model] = {}
        print(f"\n=== {model} ===")
        for cond, (root, prefix) in CONDITIONS.items():
            cond_block = {"per_n": {}, "available": {}}
            qt_data = {}
            for qt in QTYPES:
                qt_data[qt] = {}
                for n in N_VALUES:
                    path = os.path.join(root, model, f"{prefix}_{qt}_n{n}.json")
                    qt_data[qt][n] = load(path)

            # Per-n ρ and δ
            for n in N_VALUES:
                broken     = qt_data["Broken"].get(n, [])
                broken_rev = qt_data["BrokenReverse"].get(n, [])
                linked     = qt_data["Linked"].get(n, [])
                linked_rev = qt_data["LinkedReverse"].get(n, [])

                cell = {}
                d_point = delta_for_cell(broken, broken_rev)
                if d_point is not None:
                    cell["delta_point"] = d_point
                    cell["delta_ci"]    = bootstrap_delta(broken, broken_rev)
                r_point = rho_for_cell(linked, broken, linked_rev, broken_rev)
                if r_point is not None:
                    cell["rho_point"] = r_point
                    cell["rho_ci"]    = bootstrap_rho(linked, broken,
                                                     linked_rev, broken_rev)
                if cell:
                    cond_block["per_n"][str(n)] = cell

            cond_block["available"] = {
                qt: [n for n in N_VALUES if qt_data[qt].get(n)] for qt in QTYPES
            }
            if cond_block["per_n"]:
                out[model][cond] = cond_block
                ks = sorted(cond_block["per_n"].keys(), key=int)
                tag = ", ".join(f"n={k}" for k in ks)
                print(f"  {cond:32s}  {tag}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
