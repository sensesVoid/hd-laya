"""Tests for hd_laya.laya_agent: trained-checkpoint adapter (T14)."""

import pytest
import torch

from hd_laya.laya_agent import (
    TrainedLayaAgent,
    apply_head_state_dict,
    load_checkpoint_state,
)


class FakeModel(torch.nn.Module):
    """Model with a frozen 'encoder' and a trainable 'head'."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 3)


class LayaShapedAgent:
    """Records budgets; returns laya's real choice-answer shape (probabilities)."""

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, max_len=None, head_max_len=None):
        self.calls.append({"state": state, "questions": questions,
                           "max_len": max_len, "head_max_len": head_max_len})
        return {
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.7, "b": 0.3},
                    "confidence": 0.7,
                }
            },
            "usage": {},
        }


def _head_only_state_dict():
    model = FakeModel()
    return {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("encoder.")}


def test_apply_head_state_dict_allows_only_encoder_missing():
    model = FakeModel()
    missing, unexpected = apply_head_state_dict(model, _head_only_state_dict())
    assert unexpected == []
    assert missing and all(k.startswith("encoder.") for k in missing)


def test_apply_head_state_dict_rejects_missing_head_weight():
    sd = _head_only_state_dict()
    sd.pop("head.bias")
    with pytest.raises(RuntimeError, match="missing"):
        apply_head_state_dict(FakeModel(), sd)


def test_apply_head_state_dict_rejects_unexpected_weight():
    sd = _head_only_state_dict()
    sd["bogus.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="does not define"):
        apply_head_state_dict(FakeModel(), sd)


def test_load_checkpoint_state_requires_state_dict(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"weights": {}}, path)
    with pytest.raises(ValueError, match="state_dict"):
        load_checkpoint_state(str(path))


def test_load_checkpoint_state_reads_head_only_checkpoint(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": _head_only_state_dict(), "losses": [1.0], "K": 77}, path)
    ckpt, sd = load_checkpoint_state(str(path))
    assert ckpt["K"] == 77
    assert all(not k.startswith("encoder.") for k in sd)


def test_predict_mirrors_probabilities_into_scores_and_forwards_budgets():
    inner = LayaShapedAgent()
    agent = TrainedLayaAgent(inner, max_len=512, head_max_len=192)
    out = agent.predict("state text", {"q": {"type": "choice"}})

    assert out["answers"]["q"]["scores"] == {"a": 0.7, "b": 0.3}
    assert out["answers"]["q"]["score"] == pytest.approx(0.7)
    assert inner.calls[0]["max_len"] == 512
    assert inner.calls[0]["head_max_len"] == 192


def test_predict_leaves_existing_scores_untouched():
    class WithScores(LayaShapedAgent):
        def predict(self, state, questions, max_len=None, head_max_len=None):
            out = super().predict(state, questions, max_len, head_max_len)
            out["answers"]["q"]["scores"] = {"a": 9.0, "b": 1.0}
            return out

    out = TrainedLayaAgent(WithScores()).predict("s", {"q": {"type": "choice"}})
    assert out["answers"]["q"]["scores"] == {"a": 9.0, "b": 1.0}
