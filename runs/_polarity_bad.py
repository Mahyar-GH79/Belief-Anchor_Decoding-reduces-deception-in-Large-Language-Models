"""
Polarity-aware Belief-Anchored Decoding helper.

The original BeliefLogitsProcessor in methods/method3_bad.py assumes the
question uses positive ("can A contact Z?") phrasing. For LinkedReverse and
BrokenReverse, the phrasing is negative ("A cannot contact Z?"), which inverts
the relationship between the model's belief about chain connectivity and the
correct output token.

We add a polarity-aware variant: detect "cannot" in the first line of the
prompt; if present, swap the chain-Yes / chain-No log-probabilities applied to
the Yes / No output tokens.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import LogitsProcessor

from methods.method3_bad import chain_belief_logprobs


def is_negative_phrasing(problem_text: str) -> bool:
    """Return True iff the question uses 'cannot' instead of 'can' in its first line."""
    first = problem_text.split("\n", 1)[0].lower()
    return "cannot" in first


class PolarityAwareBeliefProcessor(LogitsProcessor):
    """
    Adds  λ · log P_chain(y_token)  to each output logit, where the mapping
    chain-Yes / chain-No  →  Yes-token / No-token  is flipped when the question
    is negatively phrased.
    """
    def __init__(self, yes_logsum, yes_id, no_id, lam, polarity_reversed):
        lp_yes_chain, lp_no_chain = chain_belief_logprobs(yes_logsum)
        if polarity_reversed:
            self.lp_yes_token = lp_no_chain
            self.lp_no_token  = lp_yes_chain
        else:
            self.lp_yes_token = lp_yes_chain
            self.lp_no_token  = lp_no_chain
        self.yes_id = yes_id
        self.no_id  = no_id
        self.lam    = lam

    def __call__(self, input_ids, scores):
        if self.lam == 0.0:
            return scores
        if self.yes_id is not None:
            scores[:, self.yes_id] += self.lam * self.lp_yes_token
        if self.no_id is not None:
            scores[:, self.no_id]  += self.lam * self.lp_no_token
        return scores
