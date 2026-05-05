"""
Generate CSQ questions for all difficulty levels and question types.
Reuses the paper's ProblemDef classes with our own generation pipeline.
"""
import json
import os
import sys
import numpy as np
from pathlib import Path

# Add paper repo to path so we can import their problem definitions
PAPER_REPO = Path(__file__).parent.parent / "paper_repo"
sys.path.insert(0, str(PAPER_REPO))
os.chdir(PAPER_REPO)  # needed because ProblemDef.py loads names.txt relative to cwd

from src.problem.ProblemDef import LinkedListProblem, BrokenLinkedListProblem


def generate_all_questions(n_questions=200, difficulty_levels=None, seed=42, output_dir=None):
    """Generate all question types needed for the experiment."""
    if difficulty_levels is None:
        difficulty_levels = [5, 10, 20, 40, 80]
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for n in difficulty_levels:
        print(f"\n=== Generating questions for n={n} ===")
        broken_pos = n // 2  # break at midpoint, same as paper

        # 1. Linked (answer=Yes)
        linked = []
        for i in range(n_questions):
            p = LinkedListProblem(n, seed=seed + i)
            linked.append({
                "problem": p.problem,
                "answer": p.answer,
                "linked_list": p.nodes,
            })
        save_json(linked, output_dir / f"Linked_n{n}.json")

        # 2. LinkedReverse (answer=No)
        linked_rev = []
        for i in range(n_questions):
            p = LinkedListProblem(n, seed=seed + i)
            linked_rev.append({
                "problem": p.problem_reverse,
                "answer": p.answer_reverse,
                "linked_list": p.nodes,
            })
        save_json(linked_rev, output_dir / f"LinkedReverse_n{n}.json")

        # 3. Broken (answer=No) with followup
        broken = []
        for i in range(n_questions):
            p = BrokenLinkedListProblem(n, seed=seed + n_questions + i,
                                        broken_positions=[broken_pos])
            followup_n = max(n // 2, 2)
            fp = p.followup_problem(followup_n)
            broken.append({
                "problem": p.problem,
                "answer": p.answer,
                "linked_list": p.nodes,
                "broken_edges": [f"{u} -> {v}" for u, v in p.broken_edges],
                "followup_problem": {
                    "problem": fp.problem_question_only,
                    "answer": fp.answer,
                    "linked_list": fp.nodes,
                    "broken_edges": [f"{u} -> {v}" for u, v in fp.broken_edges],
                }
            })
        save_json(broken, output_dir / f"Broken_n{n}.json")

        # 4. BrokenReverse (answer=Yes) with followup
        broken_rev = []
        for i in range(n_questions):
            p = BrokenLinkedListProblem(n, seed=seed + 2 * n_questions + i,
                                        broken_positions=[broken_pos])
            followup_n = max(n // 2, 2)
            fp = p.followup_problem(followup_n)
            broken_rev.append({
                "problem": p.problem_reverse,
                "answer": p.answer_reverse,
                "linked_list": p.nodes,
                "broken_edges": [f"{u} -> {v}" for u, v in p.broken_edges],
                "followup_problem": {
                    "problem": fp.problem_reverse_question_only,
                    "answer": fp.answer_reverse,
                    "linked_list": fp.nodes,
                    "broken_edges": [f"{u} -> {v}" for u, v in fp.broken_edges],
                }
            })
        save_json(broken_rev, output_dir / f"BrokenReverse_n{n}.json")

        print(f"  Generated: Linked({len(linked)}), LinkedReverse({len(linked_rev)}), "
              f"Broken({len(broken)}), BrokenReverse({len(broken_rev)})")


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path} ({len(data)} questions)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-questions", type=int, default=200)
    parser.add_argument("--difficulty", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out = args.output_dir or str(Path(__file__).parent)
    generate_all_questions(args.n_questions, args.difficulty, args.seed, out)
