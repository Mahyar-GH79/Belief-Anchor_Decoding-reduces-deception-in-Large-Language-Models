"""
Run experiments for one (model, difficulty, condition) cell.
Saves results as JSON to the results/ directory.

Uses async batched inference for baseline and intervention_b
(single-turn prompts that can be parallelized).
Falls back to sequential for multi-turn conditions (broken baseline, C).
"""
import json
import os
import time
from pathlib import Path
from tqdm import tqdm

from .client import (make_client, get_model_id,
                     extract_yes_no, batch_chat_single, chat_single, chat_multi)
from .interventions import LINKED_RUNNERS, BROKEN_RUNNERS, ENUMERATION_PREAMBLE

# Max concurrent requests — high for local vLLM, lower for remote APIs
MAX_CONCURRENT_LOCAL = 64
MAX_CONCURRENT_REMOTE = 16  # avoid rate limits on Together.ai etc.

# Reasoning models (DeepSeek-R1, Qwen3) need more tokens to think before answering
REASONING_SHORT_TOKENS = 2048   # replaces max_tokens=10 for yes/no answers
REASONING_LONG_TOKENS = 3072    # replaces max_tokens=1024 for enumeration


def _effective_max_tokens(base_max_tokens, model_cfg):
    """Bump max_tokens for reasoning models that need to think first."""
    if model_cfg.get("reasoning"):
        if base_max_tokens <= 10:
            return REASONING_SHORT_TOKENS
        return REASONING_LONG_TOKENS
    return base_max_tokens


def result_path(results_dir, model_name, condition, qtype, n):
    """Construct standardized result file path."""
    return os.path.join(results_dir, model_name,
                        f"{condition}_{qtype}_n{n}.json")


# ============================================================
# Batched runners for single-turn conditions
# ============================================================

def _run_linked_batched(questions, model_cfg, model_id, temperature, condition,
                        max_concurrent=64):
    """Batch all linked questions in one async call."""
    # Build prompts
    if condition in ("intervention_a", "intervention_ab"):
        prompts = [ENUMERATION_PREAMBLE + "\n\n" + q["problem"] for q in questions]
        max_tokens = _effective_max_tokens(1024, model_cfg)
    else:  # baseline, intervention_b
        prompts = [q["problem"] for q in questions]
        max_tokens = _effective_max_tokens(10, model_cfg)

    # For intervention_b and intervention_ab, we need n_samples per question
    if condition in ("intervention_b", "intervention_ab"):
        n_samples = 5
        expanded_prompts = prompts * n_samples  # repeat all prompts n_samples times
        responses = batch_chat_single(model_cfg, model_id, expanded_prompts,
                                       temperature=temperature, max_tokens=max_tokens,
                                       max_concurrent=max_concurrent)
        # Reshape: [n_samples * n_questions] -> [n_questions][n_samples]
        n_q = len(questions)
        results = []
        for i in range(n_q):
            sample_resps = [responses[s * n_q + i] for s in range(n_samples)]
            votes = [extract_yes_no(r) for r in sample_resps]
            votes = [v for v in votes if v is not None]
            from collections import Counter
            majority = Counter(votes).most_common(1)[0][0] if votes else None
            results.append({
                "raw_outputs": sample_resps,
                "answer": majority,
                "votes": votes,
            })
        return results
    else:
        # Single sample per question
        responses = batch_chat_single(model_cfg, model_id, prompts,
                                       temperature=temperature, max_tokens=max_tokens,
                                       max_concurrent=max_concurrent)
        results = []
        for resp in responses:
            results.append({
                "raw_output": resp,
                "answer": extract_yes_no(resp),
            })
        return results


def _run_broken_batched(questions, model_cfg, sync_client, model_id, temperature, condition,
                        max_concurrent=64):
    """
    For broken-list questions, we need two-turn conversations.

    Strategy:
    - Batch turn 1 (initial questions) concurrently
    - Then batch turn 2 (followup questions) concurrently
    - For multi-turn context: we include turn 1's response in turn 2's messages

    For intervention_b/ab: we need n_samples independent conversations.
    For intervention_c: turn order is reversed (belief first).
    """
    n_q = len(questions)

    if condition == "intervention_c":
        # Intervention C: belief (simple) first, then expression (hard)
        # Turn 1: ask simple followup question with full facts context
        turn1_prompts = []
        for q in questions:
            full_problem = q["problem"]
            followup_q = q["followup_problem"]["problem"]
            lines = full_problem.split("\n")
            lines[0] = followup_q
            turn1_prompts.append("\n".join(lines))

        short_tok = _effective_max_tokens(10, model_cfg)
        turn1_responses = batch_chat_single(model_cfg, model_id, turn1_prompts,
                                             temperature=temperature, max_tokens=short_tok,
                                             max_concurrent=max_concurrent)

        # Turn 2: ask the hard question with turn 1 in context (must be sequential per question
        # but we can still batch them since each is independent)
        turn2_prompts_with_context = []
        for i, q in enumerate(questions):
            # Build the messages list with turn 1 context
            turn2_prompts_with_context.append({
                "messages": [
                    {"role": "user", "content": turn1_prompts[i]},
                    {"role": "assistant", "content": turn1_responses[i] or ""},
                ],
                "prompt": q["problem"],
            })

        # Batch turn 2 — create fresh client inside the event loop
        import asyncio
        import os as _os
        from openai import AsyncOpenAI as _AsyncOpenAI

        async def _batch_turn2(cfg, mid, items, temp, _max_tokens):
            if cfg["type"] == "local":
                _base_url = f"http://localhost:{cfg['port']}/v1"
                _api_key = "dummy"
            else:
                _api_key = _os.environ.get(cfg.get("api_key_env", ""), "")
                _base_url = cfg["base_url"]
            async with _AsyncOpenAI(base_url=_base_url, api_key=_api_key) as aclient:
                sem = asyncio.Semaphore(max_concurrent)
                async def _one(item):
                    async with sem:
                        msgs = item["messages"] + [{"role": "user", "content": item["prompt"]}]
                        try:
                            resp = await aclient.chat.completions.create(
                                model=mid, messages=msgs, temperature=temp, max_tokens=_max_tokens)
                            return resp.choices[0].message.content.strip()
                        except:
                            return None
                return await asyncio.gather(*[_one(it) for it in items])

        turn2_responses = asyncio.run(
            _batch_turn2(model_cfg, model_id, turn2_prompts_with_context, temperature, short_tok))

        results = []
        for i in range(n_q):
            results.append({
                "initial_output": turn2_responses[i],
                "initial_answer": extract_yes_no(turn2_responses[i]),
                "followup_output": turn1_responses[i],
                "followup_answer": extract_yes_no(turn1_responses[i]),
            })
        return results

    # For baseline, intervention_a, intervention_b, intervention_ab
    n_samples = 5 if condition in ("intervention_b", "intervention_ab") else 1
    use_enum = condition in ("intervention_a", "intervention_ab")
    max_tok = _effective_max_tokens(1024 if use_enum else 10, model_cfg)

    # Build turn 1 prompts
    if use_enum:
        turn1_prompts = [ENUMERATION_PREAMBLE + "\n\n" + q["problem"] for q in questions]
    else:
        turn1_prompts = [q["problem"] for q in questions]

    # Expand for n_samples
    expanded_t1 = turn1_prompts * n_samples
    turn1_responses = batch_chat_single(model_cfg, model_id, expanded_t1,
                                         temperature=temperature, max_tokens=max_tok,
                                         max_concurrent=max_concurrent)

    # Build turn 2 prompts (followup) with turn 1 context
    if use_enum:
        followup_prompts = [ENUMERATION_PREAMBLE + "\n\n" + q["followup_problem"]["problem"]
                           for q in questions]
    else:
        followup_prompts = [q["followup_problem"]["problem"] for q in questions]

    expanded_fp = followup_prompts * n_samples

    import asyncio
    import os as _os
    from openai import AsyncOpenAI as _AsyncOpenAI

    async def _batch_turn2_multi(cfg, mid, t1_prompts, t1_resps, t2_prompts, temp, max_t):
        if cfg["type"] == "local":
            _base_url = f"http://localhost:{cfg['port']}/v1"
            _api_key = "dummy"
        else:
            _api_key = _os.environ.get(cfg.get("api_key_env", ""), "")
            _base_url = cfg["base_url"]
        async with _AsyncOpenAI(base_url=_base_url, api_key=_api_key) as aclient:
            sem = asyncio.Semaphore(max_concurrent)
            async def _one(t1p, t1r, t2p):
                async with sem:
                    msgs = [
                        {"role": "user", "content": t1p},
                        {"role": "assistant", "content": t1r or ""},
                        {"role": "user", "content": t2p},
                    ]
                    try:
                        resp = await aclient.chat.completions.create(
                            model=mid, messages=msgs, temperature=temp, max_tokens=max_t)
                        return resp.choices[0].message.content.strip()
                    except:
                        return None
            return await asyncio.gather(*[_one(a, b, c) for a, b, c in
                                           zip(t1_prompts, t1_resps, t2_prompts)])

    turn2_responses = asyncio.run(
        _batch_turn2_multi(model_cfg, model_id, expanded_t1, turn1_responses,
                           expanded_fp, temperature, max_tok))

    # Reshape results
    results = []
    if n_samples == 1:
        for i in range(n_q):
            results.append({
                "initial_output": turn1_responses[i],
                "initial_answer": extract_yes_no(turn1_responses[i]),
                "followup_output": turn2_responses[i],
                "followup_answer": extract_yes_no(turn2_responses[i]),
            })
    else:
        from collections import Counter
        for i in range(n_q):
            i_resps = [turn1_responses[s * n_q + i] for s in range(n_samples)]
            f_resps = [turn2_responses[s * n_q + i] for s in range(n_samples)]
            i_votes = [extract_yes_no(r) for r in i_resps if extract_yes_no(r)]
            f_votes = [extract_yes_no(r) for r in f_resps if extract_yes_no(r)]
            i_maj = Counter(i_votes).most_common(1)[0][0] if i_votes else None
            f_maj = Counter(f_votes).most_common(1)[0][0] if f_votes else None
            results.append({
                "initial_output": i_resps,
                "initial_answer": i_maj,
                "initial_votes": i_votes,
                "followup_output": f_resps,
                "followup_answer": f_maj,
                "followup_votes": f_votes,
            })

    return results


# ============================================================
# Main cell runner
# ============================================================

def run_cell(model_name, model_cfg, condition, n, questions_dir, results_dir,
             temperature=1.0):
    """
    Run one experimental cell: all question types for a given
    (model, condition, difficulty) combination.
    Uses batched async inference for massive speedup.
    """
    sync_client = make_client(model_cfg)
    model_id = get_model_id(model_cfg)
    is_remote = model_cfg.get("type") == "remote"
    if is_remote:
        max_concurrent = MAX_CONCURRENT_REMOTE
    elif model_cfg.get("reasoning"):
        max_concurrent = 16  # reasoning models generate many more tokens per request
    else:
        max_concurrent = MAX_CONCURRENT_LOCAL
    os.makedirs(os.path.join(results_dir, model_name), exist_ok=True)

    qtypes = {
        "Linked": {"runner_key": "linked", "file": f"Linked_n{n}.json"},
        "LinkedReverse": {"runner_key": "linked", "file": f"LinkedReverse_n{n}.json"},
        "Broken": {"runner_key": "broken", "file": f"Broken_n{n}.json"},
        "BrokenReverse": {"runner_key": "broken", "file": f"BrokenReverse_n{n}.json"},
    }

    if condition == "intervention_c":
        qtypes = {k: v for k, v in qtypes.items() if v["runner_key"] == "broken"}

    for qtype, qinfo in qtypes.items():
        out_path = result_path(results_dir, model_name, condition, qtype, n)

        if os.path.exists(out_path):
            print(f"  [skip] {condition}|{qtype}|n={n} already done")
            continue

        q_path = os.path.join(questions_dir, qinfo["file"])
        if not os.path.exists(q_path):
            print(f"  [warn] {q_path} not found, skipping")
            continue
        with open(q_path) as f:
            questions = json.load(f)

        n_samples = 5 if condition in ("intervention_b", "intervention_ab") else 1
        total_calls = len(questions) * n_samples
        if qinfo["runner_key"] == "broken":
            total_calls *= 2  # two turns

        desc = f"{condition}|{qtype}|n={n}"
        print(f"  {desc} ({len(questions)}q × {n_samples}s = {total_calls} API calls)...",
              end=" ", flush=True)
        t0 = time.time()

        # Use batched inference
        if qinfo["runner_key"] == "linked":
            results = _run_linked_batched(questions, model_cfg, model_id,
                                          temperature, condition,
                                          max_concurrent=max_concurrent)
        else:
            results = _run_broken_batched(questions, model_cfg, sync_client,
                                          model_id, temperature, condition,
                                          max_concurrent=max_concurrent)

        elapsed = time.time() - t0

        # Add ground truth and correctness
        for i, (r, q) in enumerate(zip(results, questions)):
            r["ground_truth"] = q["answer"]
            if "followup_problem" in q:
                r["followup_ground_truth"] = q["followup_problem"]["answer"]

            if qinfo["runner_key"] == "linked":
                ans = r.get("answer")
                r["is_correct"] = (ans == q["answer"]) if ans else False
            else:
                i_ans = r.get("initial_answer")
                f_ans = r.get("followup_answer")
                r["initial_is_correct"] = (i_ans == q["answer"]) if i_ans else False
                fp_ans = q["followup_problem"]["answer"]
                r["followup_is_correct"] = (f_ans == fp_ans) if f_ans else False

        # Quick stats
        if qinfo["runner_key"] == "linked":
            acc = sum(1 for r in results if r["is_correct"]) / len(results)
            print(f"done in {elapsed:.0f}s ({total_calls/elapsed:.0f} calls/s) acc={acc:.0%}")
        else:
            i_acc = sum(1 for r in results if r["initial_is_correct"]) / len(results)
            f_acc = sum(1 for r in results if r["followup_is_correct"]) / len(results)
            delta = sum(1 for r in results
                       if not r["initial_is_correct"] and r["followup_is_correct"]) / len(results)
            print(f"done in {elapsed:.0f}s ({total_calls/elapsed:.0f} calls/s) "
                  f"init={i_acc:.0%} fup={f_acc:.0%} δ={delta:.2f}")

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)


def run_all_cells(model_name, model_cfg, condition, n_values, questions_dir,
                  results_dir, temperature=1.0):
    """
    Run all difficulty levels for one (model, condition) pair.
    """
    total_cells = 0
    for n in n_values:
        qtypes = ["Linked", "LinkedReverse", "Broken", "BrokenReverse"]
        if condition == "intervention_c":
            qtypes = ["Broken", "BrokenReverse"]
        total_cells += len(qtypes)

    print(f"\n  [{model_name}] {condition}: {total_cells} cells across n={n_values}")
    t0 = time.time()

    for n in n_values:
        run_cell(model_name, model_cfg, condition, n, questions_dir,
                 results_dir, temperature)

    elapsed = time.time() - t0
    print(f"  [{model_name}] {condition}: completed in {elapsed/60:.1f} min")
