#!/usr/bin/env python3
"""
Main experiment runner for:
  "Inference-Time Deception Reduction via Self-Consistency Prompting"

Usage examples:
  # Step 1: Generate questions
  python main.py generate --n-questions 200

  # Step 2: Run experiments (one model at a time)
  python main.py run --model Meta-Llama-3.1-8B-Instruct --condition baseline
  python main.py run --model Meta-Llama-3.1-8B-Instruct --condition intervention_a
  python main.py run --model Meta-Llama-3.1-8B-Instruct --condition all

  # Step 3: Analyze and plot
  python main.py analyze

  # Or run everything:
  python main.py run_all
"""
import argparse
import json
import os
import sys
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent
QUESTIONS_DIR = BASE_DIR / "questions"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
CONFIG_PATH = BASE_DIR / "configs" / "models.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ============================================================
# Command: generate
# ============================================================
def cmd_generate(args):
    """Generate CSQ questions."""
    from questions.generate import generate_all_questions
    config = load_config()
    exp = config["experiment"]
    generate_all_questions(
        n_questions=args.n_questions or exp["n_questions"],
        difficulty_levels=args.difficulty or exp["difficulty_levels"],
        seed=args.seed or exp["seed"],
        output_dir=str(QUESTIONS_DIR),
    )


# ============================================================
# Command: run
# ============================================================
def cmd_run(args):
    """Run experiment for specified model(s) and condition(s)."""
    config = load_config()
    exp = config["experiment"]
    models_cfg = config["models"]

    # Determine which models to run
    if args.model == "all":
        model_names = list(models_cfg.keys())
    else:
        model_names = [args.model]

    # Determine conditions
    if args.condition == "all":
        conditions = exp["conditions"]
    else:
        conditions = [args.condition]

    # Determine difficulty levels
    n_values = args.difficulty or exp["difficulty_levels"]

    from inference.runner import run_all_cells

    for model_name in model_names:
        if model_name not in models_cfg:
            print(f"[Error] Model '{model_name}' not in config. "
                  f"Available: {list(models_cfg.keys())}")
            continue

        model_cfg = models_cfg[model_name]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for condition in conditions:
            print(f"\n--- Condition: {condition} ---")
            run_all_cells(
                model_name=model_name,
                model_cfg=model_cfg,
                condition=condition,
                n_values=n_values,
                questions_dir=str(QUESTIONS_DIR),
                results_dir=str(RESULTS_DIR),
                temperature=exp["temperature"],
            )


# ============================================================
# Command: analyze
# ============================================================
def cmd_analyze(args):
    """Run analysis and generate all plots."""
    config = load_config()
    exp = config["experiment"]
    models_cfg = config["models"]

    models = args.models or list(models_cfg.keys())
    conditions = args.conditions or exp["conditions"]
    n_values = args.difficulty or exp["difficulty_levels"]

    from analysis.plots import generate_all_plots
    generate_all_plots(
        results_dir=str(RESULTS_DIR),
        models=models,
        conditions=conditions,
        n_values=n_values,
        output_dir=str(PLOTS_DIR),
        target_model=args.target_model,
    )


# ============================================================
# Command: run_all
# ============================================================
def cmd_run_all(args):
    """Generate questions, run all experiments, then analyze."""
    print("=== Step 1: Generating questions ===")
    cmd_generate(args)

    print("\n=== Step 2: Running experiments ===")
    args.model = "all"
    args.condition = "all"
    cmd_run(args)

    print("\n=== Step 3: Analyzing results ===")
    args.models = None
    args.conditions = None
    args.target_model = None
    cmd_analyze(args)


# ============================================================
# Command: serve
# ============================================================
def cmd_serve(args):
    """Print vLLM server launch commands for local models."""
    config = load_config()
    models_cfg = config["models"]

    print("=== vLLM Server Launch Commands ===\n")
    for name, cfg in models_cfg.items():
        if cfg["type"] != "local":
            print(f"# {name}: remote model, no server needed")
            continue

        tp = cfg.get("tensor_parallel", 1)
        port = cfg["port"]
        hf_id = cfg["hf_id"]
        max_len = cfg.get("max_model_len", 8192)
        quant = cfg.get("quantization")

        cmd = (f"python3 -m vllm.entrypoints.openai.api_server "
               f"--model {hf_id} "
               f"--port {port} "
               f"--tensor-parallel-size {tp} "
               f"--max-model-len {max_len} "
               f"--dtype auto "
               f"--trust-remote-code")
        if quant:
            cmd += f" --quantization {quant}"

        print(f"# {name} (port {port}, TP={tp})")
        print(cmd)
        print()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Inference-Time Deception Reduction Experiments")
    subparsers = parser.add_subparsers(dest="command")

    # generate
    gen_p = subparsers.add_parser("generate", help="Generate CSQ questions")
    gen_p.add_argument("--n-questions", type=int)
    gen_p.add_argument("--difficulty", type=int, nargs="+")
    gen_p.add_argument("--seed", type=int)

    # run
    run_p = subparsers.add_parser("run", help="Run experiments")
    run_p.add_argument("--model", type=str, required=True,
                       help="Model name from config, or 'all'")
    run_p.add_argument("--condition", type=str, default="all",
                       help="Condition name or 'all'")
    run_p.add_argument("--difficulty", type=int, nargs="+")

    # analyze
    ana_p = subparsers.add_parser("analyze", help="Analyze results and plot")
    ana_p.add_argument("--models", type=str, nargs="+")
    ana_p.add_argument("--conditions", type=str, nargs="+")
    ana_p.add_argument("--difficulty", type=int, nargs="+")
    ana_p.add_argument("--target-model", type=str,
                       help="Model for Figure 1 headline plot")

    # run_all
    all_p = subparsers.add_parser("run_all", help="Full pipeline: generate + run + analyze")
    all_p.add_argument("--n-questions", type=int)
    all_p.add_argument("--difficulty", type=int, nargs="+")
    all_p.add_argument("--seed", type=int)

    # serve
    subparsers.add_parser("serve", help="Print vLLM server commands")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "run_all":
        cmd_run_all(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
