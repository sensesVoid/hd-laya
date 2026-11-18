"""High-K choice data generator for hd-laya v1.0 (T11 data side).

Produces synthetic choice questions with K options (K in {77,150,500,1000,2000})
in laya's question format, so the trained model can be evaluated on
cardinalities far beyond laya's ~20-option ceiling. Pure numpy, no torch, no
hub: runs anywhere, including Kaggle smoke mode.

**The task is well-posed by construction.** Each option carries a unique order
key, and the state names exactly one of them, so precisely one option is the
right answer and the attainable accuracy is 1.0 at every K. An earlier version
filled every option from the same 12-topic vocabulary, which made ~K/12 options
equally consistent with the state; accuracy could never exceed 12/K and the
high-K rows measured that ceiling instead of retrieval. The topic is still
present as distracting overlap -- it is what a lexical index has to see past.

Output item shape (matches laya's predict_batch input):
    {"state": str, "questions": {qid: {"type":"choice","instructions":str,
     "criteria": {label: description}}}, "expected": label}
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

__all__ = ["gen_choice_item", "gen_dataset", "K_VALUES"]

K_VALUES = (20, 77, 150, 500, 1000, 2000)

_TOPICS = (
    "refund order", "cancel subscription", "track package", "reset password",
    "update address", "change plan", "report damage", "request invoice",
    "dispute charge", "schedule pickup", "return item", "apply coupon",
)

_KEY_LO = 1_000_000
_KEY_HI = 9_999_999


def gen_choice_item(
    rng: np.random.Generator,
    K: int,
    topic_vocab: "list[str] | None" = None,
) -> Dict[str, Any]:
    """One choice item: state + K uniquely-keyed options + expected label."""
    if topic_vocab is None:
        topic_vocab = list(_TOPICS)
    # The correct option's topic appears in the state; distractor topics are
    # drawn from the same vocabulary so topic alone never identifies the answer.
    correct_topic = str(rng.choice(topic_vocab))
    labels = [f"opt_{i}" for i in range(K)]
    expected_idx = int(rng.integers(0, K))
    keys = rng.choice(np.arange(_KEY_LO, _KEY_HI + 1), size=K, replace=False)
    key_text = [f"ORD-{int(k)}" for k in keys]

    criteria: Dict[str, str] = {}
    for i in range(K):
        topic = correct_topic if i == expected_idx else str(rng.choice(topic_vocab))
        criteria[labels[i]] = f"{topic}; order {key_text[i]}"

    state = f"customer wants to {correct_topic}; order {key_text[expected_idx]}"
    return {
        "state": state,
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Choose the option that matches the customer request.",
                "criteria": criteria,
            }
        },
        "expected": labels[expected_idx],
    }


def gen_dataset(n: int, K: int, seed: int = 0) -> List[Dict[str, Any]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if K < 2:
        raise ValueError("K must be >= 2")
    rng = np.random.default_rng(seed)
    return [gen_choice_item(rng, K) for _ in range(n)]
