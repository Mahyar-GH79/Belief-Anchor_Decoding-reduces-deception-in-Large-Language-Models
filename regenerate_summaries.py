#!/usr/bin/env python3
"""
Patch summary.json files with updated Qwen3-4B results and regenerate summary_bar.pdf.
Run from the project root: python3 regenerate_summaries.py
"""

import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__))

from methods.common import (
    delta_direct, accuracy_initial, plot_delta_comparison,
    ALL_MODELS, C
)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_per_n_stats(result_dir, model_name, tag_glob):
    """
    Aggregate delta and accuracy across all n from per-n result JSON files.
    tag_glob e.g. 'stratified_perN_alpha*' or 'best_lambda*'
    """
    base_dir    = f"results/{model_name}"
    method_dir  = f"{result_dir}/{model_name}"

    all_baseline, all_method = [], []
    per_n = {}

    for n in [5, 10, 20, 40, 80]:
        bfile = f"{base_dir}/baseline_Broken_n{n}.json"
        if not os.path.exists(bfile):
            continue
        base = json.load(open(bfile))

        # find the matching method file
        mfiles = glob.glob(f"{method_dir}/{tag_glob}_Broken_n{n}.json")
        if not mfiles:
            continue
        method = json.load(open(mfiles[0]))

        # use the test split (second half) to match what run_model used
        mid = len(base) // 2
        base_test   = base[mid:]
        all_baseline += base_test
        all_method   += method

        per_n[str(n)] = {
            "baseline_delta": delta_direct(base_test),
            "method_delta":   delta_direct(method),
            "baseline_acc":   accuracy_initial(base_test),
            "method_acc":     accuracy_initial(method),
        }

    return {
        "baseline_delta": delta_direct(all_baseline),
        "method_delta":   delta_direct(all_method),
        "baseline_acc":   accuracy_initial(all_baseline),
        "method_acc":     accuracy_initial(all_method),
        "per_n":          per_n,
    }


EXCLUDE_MODELS = {"Qwen3-4B", "gemma-2-2b-it"}   # Qwen3: CoT reasoning model; Gemma-2-2B: "No" bias → trivially near-zero δ

def patch_and_replot(summary_json_path, model_name, new_stats, method_label, method_color, title):
    summary = json.load(open(summary_json_path))
    old = summary.get(model_name, {})

    # Update only the keys we can recompute; keep alpha/lambda sweep data if present
    for k in ["baseline_delta", "method_delta", "baseline_acc", "method_acc", "per_n"]:
        if k in new_stats:
            old[k] = new_stats[k]
    summary[model_name] = old

    # Save updated summary.json (full, including excluded models)
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Updated {summary_json_path}")

    # For plotting, drop models that are excluded
    plot_summary = {m: r for m, r in summary.items() if m not in EXCLUDE_MODELS}

    # Regenerate summary_bar.pdf
    out_dir  = os.path.dirname(summary_json_path)
    bar_path = os.path.join(out_dir, "summary_bar.pdf")
    plot_delta_comparison(
        baseline_deltas = {m: r["baseline_delta"] for m, r in plot_summary.items()},
        method_deltas   = {m: r["method_delta"]   for m, r in plot_summary.items()},
        method_label    = method_label,
        method_color    = method_color,
        out_path        = bar_path,
        title           = title,
        also_plot_acc   = True,
        baseline_accs   = {m: r["baseline_acc"] for m, r in plot_summary.items()},
        method_accs     = {m: r["method_acc"]   for m, r in plot_summary.items()},
    )
    print(f"  Regenerated {bar_path}")
    return summary


# ── Method 1 ─────────────────────────────────────────────────────────────────
print("=== Method 1: Activation Steering ===")
m1_stats = load_per_n_stats(
    result_dir = "results_methods/method1_steering",
    model_name = "Qwen3-4B",
    tag_glob   = "stratified_perN_alpha*",
)
print(f"  Qwen3-4B: baseline_δ={m1_stats['baseline_delta']:.4f}  "
      f"method_δ={m1_stats['method_delta']:.4f}  "
      f"baseline_acc={m1_stats['baseline_acc']:.4f}  "
      f"method_acc={m1_stats['method_acc']:.4f}")

m1_summary = patch_and_replot(
    summary_json_path = "plots/methods/method1_steering/summary.json",
    model_name        = "Qwen3-4B",
    new_stats         = m1_stats,
    method_label      = "Activation Steering",
    method_color      = C["method1"],
    title             = "Method 1: Activation Steering — δ Comparison",
)

# ── Method 2 ─────────────────────────────────────────────────────────────────
print("\n=== Method 2: Probe-Guided Steering ===")
m2_stats = load_per_n_stats(
    result_dir = "results_methods/method2_probe",
    model_name = "Qwen3-4B",
    tag_glob   = "best_lambda*",
)
print(f"  Qwen3-4B: baseline_δ={m2_stats['baseline_delta']:.4f}  "
      f"method_δ={m2_stats['method_delta']:.4f}  "
      f"baseline_acc={m2_stats['baseline_acc']:.4f}  "
      f"method_acc={m2_stats['method_acc']:.4f}")

# Load best_lambda and best_auroc from the per-model summary if available
m2_model_json = "plots/methods/method2_probe/summary.json"
m2_extra = {}
if os.path.exists(m2_model_json):
    existing = json.load(open(m2_model_json)).get("Qwen3-4B", {})
    m2_extra = {k: existing[k] for k in ["best_lambda","best_layer","best_auroc","layer_aurocs"]
                if k in existing}
m2_stats.update(m2_extra)

m2_summary = patch_and_replot(
    summary_json_path = m2_model_json,
    model_name        = "Qwen3-4B",
    new_stats         = m2_stats,
    method_label      = "Probe Steering",
    method_color      = C["method2"],
    title             = "Method 2: Probe-Guided Steering — δ Comparison",
)


# ── Add Qwen2.5-32B results to both summaries ────────────────────────────────
print("\n=== Adding Qwen2.5-32B to Method 1 ===")
qwen32b_m1_stats = load_per_n_stats(
    result_dir = "results_methods/method1_steering",
    model_name = "Qwen2.5-32B-Instruct",
    tag_glob   = "stratified_perN_alpha*",
)
print(f"  Qwen2.5-32B: baseline_δ={qwen32b_m1_stats['baseline_delta']:.4f}  "
      f"method_δ={qwen32b_m1_stats['method_delta']:.4f}  "
      f"baseline_acc={qwen32b_m1_stats['baseline_acc']:.4f}  "
      f"method_acc={qwen32b_m1_stats['method_acc']:.4f}")

# Also pull sweep data from the per-model plots summary if available
m1_existing = json.load(open("plots/methods/method1_steering/summary.json")).get("Qwen2.5-32B-Instruct", {})
for k in ["best_alpha", "alpha_deltas", "alpha_accs"]:
    if k in m1_existing:
        qwen32b_m1_stats[k] = m1_existing[k]

patch_and_replot(
    summary_json_path = "plots/methods/method1_steering/summary.json",
    model_name        = "Qwen2.5-32B-Instruct",
    new_stats         = qwen32b_m1_stats,
    method_label      = "Activation Steering",
    method_color      = C["method1"],
    title             = "Method 1: Activation Steering — δ Comparison",
)

print("\n=== Adding Qwen2.5-32B to Method 2 ===")
qwen32b_m2_stats = load_per_n_stats(
    result_dir = "results_methods/method2_probe",
    model_name = "Qwen2.5-32B-Instruct",
    tag_glob   = "best_lambda*",
)
print(f"  Qwen2.5-32B: baseline_δ={qwen32b_m2_stats['baseline_delta']:.4f}  "
      f"method_δ={qwen32b_m2_stats['method_delta']:.4f}  "
      f"baseline_acc={qwen32b_m2_stats['baseline_acc']:.4f}  "
      f"method_acc={qwen32b_m2_stats['method_acc']:.4f}")

m2_existing_32b = json.load(open(m2_model_json)).get("Qwen2.5-32B-Instruct", {})
for k in ["best_lambda", "best_layer", "best_auroc", "layer_aurocs"]:
    if k in m2_existing_32b:
        qwen32b_m2_stats[k] = m2_existing_32b[k]

patch_and_replot(
    summary_json_path = m2_model_json,
    model_name        = "Qwen2.5-32B-Instruct",
    new_stats         = qwen32b_m2_stats,
    method_label      = "Probe Steering",
    method_color      = C["method2"],
    title             = "Method 2: Probe-Guided Steering — δ Comparison",
)


# ── Fix Method 2 baselines to match canonical split ──────────────────────────
print("\n=== Fixing Method 2 baselines to canonical split ===")
m2_summary = json.load(open(m2_model_json))

for model_name in m2_summary:
    canon_all = []
    per_n_canon = {}
    for n in [5, 10, 20, 40, 80]:
        bfile = f"results/{model_name}/baseline_Broken_n{n}.json"
        if not os.path.exists(bfile):
            continue
        b = json.load(open(bfile))
        mid = len(b) // 2
        test = b[mid:]
        canon_all += test
        per_n_canon[str(n)] = {
            "baseline_delta": delta_direct(test),
            "baseline_acc":   accuracy_initial(test),
        }
    if not canon_all:
        continue

    old_bd = m2_summary[model_name].get("baseline_delta", "?")
    new_bd = delta_direct(canon_all)
    m2_summary[model_name]["baseline_delta"] = new_bd
    m2_summary[model_name]["baseline_acc"]   = accuracy_initial(canon_all)

    # Update per_n baseline fields without touching method fields
    if "per_n" in m2_summary[model_name]:
        for n_str, canon in per_n_canon.items():
            if n_str in m2_summary[model_name]["per_n"]:
                m2_summary[model_name]["per_n"][n_str]["baseline_delta"] = canon["baseline_delta"]
                m2_summary[model_name]["per_n"][n_str]["baseline_acc"]   = canon["baseline_acc"]

    print(f"  {model_name}: baseline_delta {old_bd:.4f} → {new_bd:.4f}")

with open(m2_model_json, "w") as f:
    json.dump(m2_summary, f, indent=2)
print(f"  Saved {m2_model_json}")

# Replot with fixed baselines (excluding same models)
plot_summary = {m: r for m, r in m2_summary.items() if m not in EXCLUDE_MODELS}
bar_path = os.path.join(os.path.dirname(m2_model_json), "summary_bar.pdf")
plot_delta_comparison(
    baseline_deltas = {m: r["baseline_delta"] for m, r in plot_summary.items()},
    method_deltas   = {m: r["method_delta"]   for m, r in plot_summary.items()},
    method_label    = "Probe Steering",
    method_color    = C["method2"],
    out_path        = bar_path,
    title           = "Method 2: Probe-Guided Steering — δ Comparison",
    also_plot_acc   = True,
    baseline_accs   = {m: r["baseline_acc"] for m, r in plot_summary.items()},
    method_accs     = {m: r["method_acc"]   for m, r in plot_summary.items()},
)
print(f"  Regenerated {bar_path}")

print("\nDone. Both summary_bar.pdf files regenerated with consistent baselines.")
