"""Tests for hd_laya.agent: HDC-aware adapter over a laya Agent (T6)."""

import json

import numpy as np
import pytest

from hd_laya.agent import HDLayaAgent


class FakeAgent:
    """Minimal laya-like agent: records the questions it was called with."""

    def __init__(self, mode="clear"):
        self.mode = mode
        self.last_questions = None
        self.call_count = 0

    def predict(self, state, questions):
        self.last_questions = {k: v for k, v in questions.items()}
        self.call_count += 1
        answers = {}
        for qid, qdef in questions.items():
            crit = qdef.get("criteria", {})
            labels = list(crit.keys()) if isinstance(crit, dict) else (list(crit) if isinstance(crit, list) else [])
            if self.mode == "clear":
                sc = {lbl: (10.0 if i == 0 else 0.1) for i, lbl in enumerate(labels)}
            elif self.mode == "tie":
                sc = {lbl: 0.5 for lbl in labels}
            else:
                sc = {lbl: 0.1 for lbl in labels}
            best = max(sc, key=lambda k: sc[k]) if sc else (labels[0] if labels else None)
            answers[qid] = {
                "choice": best,
                "score": float(sc.get(best, 0.0)) if sc else 0.0,
                "confidence": 0.9,
                "scores": sc,
            }
        return {"model": "fake-laya", "answers": answers, "usage": {"input_tokens": 0}}


def _make_questions(n_opts):
    crit = {f"opt_{i}": f"description number {i}" for i in range(n_opts)}
    return {"q1": {"type": "choice", "instructions": "pick one", "criteria": crit}}


def test_predict_narrows_over_ceiling_and_calls_agent():
    fa = FakeAgent(mode="clear")
    agent = HDLayaAgent(fa, max_options=5, z_bar=0.0)  # z_bar 0 -> never refuse
    qs = _make_questions(30)
    res = agent.predict("state text here", qs)
    assert fa.call_count == 1
    narrowed = fa.last_questions["q1"]["criteria"]
    assert len(narrowed) == 5
    assert set(narrowed).issubset(set(qs["q1"]["criteria"]))
    ans = res["answers"]["q1"]
    assert "choice" in ans and "score" in ans and "confidence" in ans
    assert "scores" in ans
    assert "meta" in res
    hd = res["meta"]["hd_index"]["q1"]
    assert hd["n_options"] == 30
    assert hd["narrowed"] is True
    assert hd["narrowed_to"] == 5
    assert res["meta"]["refused"] == []


def test_predict_z_gate_refuses_near_tie():
    fa = FakeAgent(mode="tie")
    agent = HDLayaAgent(fa, max_options=20, z_bar=5.0)
    qs = _make_questions(30)
    res = agent.predict("state", qs)
    ans = res["answers"]["q1"]
    assert ans.get("refused") is True
    assert ans["choice"] == "REFUSE"
    assert "q1" in res["meta"]["refused"]


def test_predict_z_gate_answers_clear_winner():
    fa = FakeAgent(mode="clear")
    agent = HDLayaAgent(fa, max_options=20, z_bar=5.0)
    qs = _make_questions(30)
    res = agent.predict("state", qs)
    ans = res["answers"]["q1"]
    assert ans.get("refused") is False
    assert ans["choice"] != "REFUSE"
    assert res["meta"]["refused"] == []


def test_predict_small_choice_not_narrowed():
    fa = FakeAgent(mode="clear")
    agent = HDLayaAgent(fa, max_options=20, z_bar=5.0)
    qs = {"q2": {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}}
    res = agent.predict("s", qs)
    hd = res["meta"]["hd_index"]["q2"]
    assert hd["narrowed"] is False
    assert hd["n_options"] == 2
    assert len(fa.last_questions["q2"]["criteria"]) == 2


def test_predict_non_choice_passthrough():
    fa = FakeAgent(mode="clear")
    agent = HDLayaAgent(fa, max_options=5, z_bar=5.0)
    qs = {"q3": {"type": "score", "instructions": "score", "criteria": ["low", "high"]}}
    res = agent.predict("s", qs)
    assert "q3" in res["answers"]
    hd = res["meta"]["hd_index"]["q3"]
    assert hd["narrowed"] is False


def test_predict_state_text_conversion():
    fa = FakeAgent(mode="clear")
    agent = HDLayaAgent(fa, max_options=5, z_bar=0.0)
    qs = _make_questions(12)
    res = agent.predict({"user": "hi", "msg": "hello"}, qs)
    assert fa.call_count == 1
    assert "answers" in res


def test_narrow_choice_duplicate_texts_dont_crash():
    """Duplicate generated option texts must not crash (F2)."""
    fa = FakeAgent(mode="clear")
    agent = HDLayaAgent(fa, max_options=3, z_bar=0.0)
    crit = {}
    crit["a"] = "b: c"
    crit["a: b"] = "c"  # generates same text as above
    for i in range(2, 25):
        crit[f"opt_{i}"] = f"desc {i}"
    qs = {"q1": {"type": "choice", "instructions": "pick", "criteria": crit}}
    res = agent.predict("state text", qs)
    assert fa.call_count == 1
    assert "answers" in res
    hd = res["meta"]["hd_index"]["q1"]
    assert hd["narrowed"] is True


def test_calib_skips_non_finite_z():
    """Non-finite z is not folded into calib (F4)."""
    from hd_laya.zgate import StreamingCalib
    fa = FakeAgent(mode="clear")
    calib = StreamingCalib()
    agent = HDLayaAgent(fa, max_options=20, z_bar=0.0, calib=calib)
    qs = {"q1": {"type": "choice", "instructions": "x", "criteria": {"only": "desc"}}}
    agent.predict("state", qs)
    # single score -> z = inf (non-finite) -> not fed
    assert calib.count() == 0
