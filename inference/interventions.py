"""
Implements the three inference-time interventions plus baseline.

Each function takes a question dict + client/model_id and returns a result dict
with the model's answers and metadata.
"""
from collections import Counter
from .client import chat_single, chat_multi, extract_yes_no


# ============================================================
# Intervention A: Forced Enumeration
# ============================================================

ENUMERATION_PREAMBLE = (
    "Before answering, you MUST write out each person mentioned in the facts "
    "as a chain. For each consecutive pair, state whether a direct contact link "
    "exists in the facts. Only after checking every link should you give your "
    "final answer as a single word 'Yes' or 'No' on the last line."
)


def run_baseline(question, client, model_id, temperature=1.0):
    """Paper's original protocol: single-turn, direct question."""
    resp = chat_single(client, model_id, question["problem"],
                       temperature=temperature, max_tokens=10)
    return {"raw_output": resp, "answer": extract_yes_no(resp)}


def run_baseline_broken(question, client, model_id, temperature=1.0):
    """
    Paper's original protocol for broken-list questions:
    two-turn conversation (initial + followup) in same context.
    """
    messages = []
    # Turn 1: initial (hard) question
    initial_resp = chat_multi(client, model_id, messages,
                              question["problem"],
                              temperature=temperature, max_tokens=10)
    # Turn 2: followup (easy) question
    followup_resp = chat_multi(client, model_id, messages,
                               question["followup_problem"]["problem"],
                               temperature=temperature, max_tokens=10)
    return {
        "initial_output": initial_resp,
        "initial_answer": extract_yes_no(initial_resp),
        "followup_output": followup_resp,
        "followup_answer": extract_yes_no(followup_resp),
    }


# ============================================================
# Intervention A: Forced Enumeration
# ============================================================

def run_intervention_a(question, client, model_id, temperature=1.0):
    """Forced enumeration for linked-list questions."""
    prompt = ENUMERATION_PREAMBLE + "\n\n" + question["problem"]
    resp = chat_single(client, model_id, prompt,
                       temperature=temperature, max_tokens=1024)
    return {"raw_output": resp, "answer": extract_yes_no(resp)}


def run_intervention_a_broken(question, client, model_id, temperature=1.0):
    """Forced enumeration for broken-list questions (two-turn)."""
    messages = []
    prompt_initial = ENUMERATION_PREAMBLE + "\n\n" + question["problem"]
    initial_resp = chat_multi(client, model_id, messages, prompt_initial,
                              temperature=temperature, max_tokens=1024)

    # Followup: also with enumeration preamble
    prompt_followup = (ENUMERATION_PREAMBLE + "\n\n" +
                       question["followup_problem"]["problem"])
    followup_resp = chat_multi(client, model_id, messages, prompt_followup,
                               temperature=temperature, max_tokens=1024)
    return {
        "initial_output": initial_resp,
        "initial_answer": extract_yes_no(initial_resp),
        "followup_output": followup_resp,
        "followup_answer": extract_yes_no(followup_resp),
    }


# ============================================================
# Intervention B: Sample-then-Vote (self-consistency)
# ============================================================

def run_intervention_b(question, client, model_id, temperature=1.0, n_samples=5):
    """Query the same question n_samples times, majority vote."""
    votes = []
    raw_outputs = []
    for _ in range(n_samples):
        resp = chat_single(client, model_id, question["problem"],
                           temperature=temperature, max_tokens=10)
        raw_outputs.append(resp)
        ans = extract_yes_no(resp)
        if ans:
            votes.append(ans)

    if not votes:
        return {"raw_outputs": raw_outputs, "answer": None, "votes": votes}

    counter = Counter(votes)
    majority = counter.most_common(1)[0][0]
    return {"raw_outputs": raw_outputs, "answer": majority, "votes": votes}


def run_intervention_b_broken(question, client, model_id, temperature=1.0, n_samples=5):
    """
    Sample-then-vote for broken-list questions.
    Each sample is an independent two-turn conversation.
    """
    initial_votes = []
    followup_votes = []
    raw_initial = []
    raw_followup = []

    for _ in range(n_samples):
        messages = []
        i_resp = chat_multi(client, model_id, messages,
                            question["problem"],
                            temperature=temperature, max_tokens=10)
        f_resp = chat_multi(client, model_id, messages,
                            question["followup_problem"]["problem"],
                            temperature=temperature, max_tokens=10)
        raw_initial.append(i_resp)
        raw_followup.append(f_resp)
        i_ans = extract_yes_no(i_resp)
        f_ans = extract_yes_no(f_resp)
        if i_ans:
            initial_votes.append(i_ans)
        if f_ans:
            followup_votes.append(f_ans)

    i_majority = Counter(initial_votes).most_common(1)[0][0] if initial_votes else None
    f_majority = Counter(followup_votes).most_common(1)[0][0] if followup_votes else None

    return {
        "initial_output": raw_initial,
        "initial_answer": i_majority,
        "initial_votes": initial_votes,
        "followup_output": raw_followup,
        "followup_answer": f_majority,
        "followup_votes": followup_votes,
    }


# ============================================================
# Intervention C: Belief Elicitation Before Expression
# ============================================================

def run_intervention_c_broken(question, client, model_id, temperature=1.0):
    """
    Belief elicitation before expression.
    Turn 1: Present the full problem (facts+rules) but ask the SIMPLE sub-question.
            This commits the model to a belief about the broken edge.
    Turn 2: Now ask the HARD question (the original full-length query).
            The model's own belief answer is in context.
    """
    messages = []

    # Turn 1: Present facts/rules via the full problem text, but swap in the
    # simpler followup question. We extract the facts/rules section from the
    # full problem and attach the followup question to it.
    full_problem = question["problem"]
    followup_q = question["followup_problem"]["problem"]

    # The full problem format is:
    #   "Derive if X can contact Y based on ...\n---\nRules:\n...\nFacts:\n...\n---\n..."
    # We replace the first line (the question) with the simpler followup question,
    # keeping all rules and facts intact.
    lines = full_problem.split("\n")
    # First line is the question; replace it with the followup question text
    lines[0] = followup_q
    belief_prompt = "\n".join(lines)

    followup_resp = chat_multi(client, model_id, messages, belief_prompt,
                               temperature=temperature, max_tokens=10)

    # Turn 2: Ask the original hard question (model's belief answer is now in context)
    initial_resp = chat_multi(client, model_id, messages,
                              question["problem"],
                              temperature=temperature, max_tokens=10)
    return {
        "initial_output": initial_resp,
        "initial_answer": extract_yes_no(initial_resp),
        "followup_output": followup_resp,
        "followup_answer": extract_yes_no(followup_resp),
    }


# ============================================================
# Intervention A+B Combined
# ============================================================

def run_intervention_ab(question, client, model_id, temperature=1.0, n_samples=5):
    """Forced enumeration + majority vote for linked-list questions."""
    prompt = ENUMERATION_PREAMBLE + "\n\n" + question["problem"]
    votes = []
    raw_outputs = []
    for _ in range(n_samples):
        resp = chat_single(client, model_id, prompt,
                           temperature=temperature, max_tokens=1024)
        raw_outputs.append(resp)
        ans = extract_yes_no(resp)
        if ans:
            votes.append(ans)

    if not votes:
        return {"raw_outputs": raw_outputs, "answer": None, "votes": votes}
    majority = Counter(votes).most_common(1)[0][0]
    return {"raw_outputs": raw_outputs, "answer": majority, "votes": votes}


def run_intervention_ab_broken(question, client, model_id, temperature=1.0, n_samples=5):
    """Forced enumeration + majority vote for broken-list questions."""
    initial_votes = []
    followup_votes = []
    raw_initial = []
    raw_followup = []

    for _ in range(n_samples):
        messages = []
        prompt_i = ENUMERATION_PREAMBLE + "\n\n" + question["problem"]
        i_resp = chat_multi(client, model_id, messages, prompt_i,
                            temperature=temperature, max_tokens=1024)
        prompt_f = (ENUMERATION_PREAMBLE + "\n\n" +
                    question["followup_problem"]["problem"])
        f_resp = chat_multi(client, model_id, messages, prompt_f,
                            temperature=temperature, max_tokens=1024)

        raw_initial.append(i_resp)
        raw_followup.append(f_resp)
        i_ans = extract_yes_no(i_resp)
        f_ans = extract_yes_no(f_resp)
        if i_ans:
            initial_votes.append(i_ans)
        if f_ans:
            followup_votes.append(f_ans)

    i_majority = Counter(initial_votes).most_common(1)[0][0] if initial_votes else None
    f_majority = Counter(followup_votes).most_common(1)[0][0] if followup_votes else None

    return {
        "initial_output": raw_initial,
        "initial_answer": i_majority,
        "initial_votes": initial_votes,
        "followup_output": raw_followup,
        "followup_answer": f_majority,
        "followup_votes": followup_votes,
    }


# ============================================================
# Dispatcher
# ============================================================

LINKED_RUNNERS = {
    "baseline": run_baseline,
    "intervention_a": run_intervention_a,
    "intervention_b": run_intervention_b,
    "intervention_ab": run_intervention_ab,
}

BROKEN_RUNNERS = {
    "baseline": run_baseline_broken,
    "intervention_a": run_intervention_a_broken,
    "intervention_b": run_intervention_b_broken,
    "intervention_c": run_intervention_c_broken,
    "intervention_ab": run_intervention_ab_broken,
}
