"""Benchmark runner for hd-laya (T7).

Reuses laya's fixed-question shape (questions dict of id -> definition) but is
engine-agnostic: it takes any object exposing ``predict(state, questions)`` and
a list of items, so it runs with a real laya Agent, an HDLayaAgent, or a fake
in tests. The real BANKING77/CLINC150 runs are documented manual steps (see
README) because they need a laya checkpoint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["BenchmarkItem", "run_benchmark"]


@dataclass
class BenchmarkItem:
    state: Any
    expected: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)


def _predicted_label(result: Dict[str, Any], qid: str) -> Any:
    ans = (result.get("answers") or {}).get(qid, {})
    if not isinstance(ans, dict):
        return ans
    if ans.get("refused"):
        return None
    return ans.get("choice", ans.get("score", ans.get("noul")))


def run_benchmark(
    agent: Any,
    items: List[BenchmarkItem],
    make_questions: Callable[[BenchmarkItem], Dict[str, Any]],
    question_id: str = "q",
    refused_token: Any = None,
) -> Dict[str, Any]:
    """Run ``agent`` over ``items`` and return per-row predictions + summary.

    ``make_questions(item)`` builds the questions dict for that item (so labels
    can vary per item). ``question_id`` is the key whose answer is scored.
    """
    rows: List[Dict[str, Any]] = []
    n = 0
    correct = 0
    refused = 0
    answered = 0
    answered_correct = 0
    for item in items:
        questions = make_questions(item)
        result = agent.predict(item.state, questions)
        pred = _predicted_label(result, question_id)
        is_refused = pred is None and isinstance(result.get("answers", {}).get(question_id, {}), dict) \
            and result["answers"][question_id].get("refused") is True
        if is_refused:
            refused += 1
        else:
            answered += 1
        ok = (pred == item.expected) and not is_refused
        if ok:
            correct += 1
            answered_correct += 1
        n += 1
        rows.append({
            "state": item.state,
            "expected": item.expected,
            "predicted": pred,
            "refused": is_refused,
            "correct": ok,
            "meta": item.meta,
            "agent_meta": (result.get("meta") if isinstance(result, dict) else None),
        })
    summary = {
        "n": n,
        "accuracy": (correct / n) if n else 0.0,
        "refusal_rate": (refused / n) if n else 0.0,
        "answered": answered,
        "accuracy_on_answered": (answered_correct / answered) if answered else 0.0,
    }
    return {"rows": rows, "summary": summary}


def dump_json(path: str, result: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
