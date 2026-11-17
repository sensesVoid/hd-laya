"""High-K choice data generator for hd-laya v1.0 (T11 data side).

Produces synthetic choice questions with K options (K in {77,150,500,1000,2000})
in laya's question format, so the trained model can be evaluated on
cardinalities far beyond laya's ~20-option ceiling. Pure numpy, no torch, no
hub: runs anywhere, including Kaggle smoke mode.

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


def gen_choice_item(
    rng: np.random.Generator,
    K: int,
    topic_vocab: "list[str] | None" = None,
) -> Dict[str, Any]:
    """One choice item: state + K-option criteria + expected label."""
    if topic_vocab is None:
        topic_vocab = list(_TOPICS)
    # Pick the correct option's topic; distractors are other topics (possibly
    # with light wording variation) so the answer is unambiguous by topic.
    correct_topic = str(rng.choice(topic_vocab))
    labels = [f"opt_{i}" for i in range(K)]
    criteria: Dict[str, str] = {}
    expected_idx = int(rng.integers(0, K))
    for i in range(K):
        if i == expected_idx:
            t = correct_topic
        else:
            t = str(rng.choice(topic_vocab))
        criteria[labels[i]] = f"{t} (variant {i})"
    state = f"customer wants to {correct_topic}"
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
