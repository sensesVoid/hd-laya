"""Trained-checkpoint adapter for laya (hd-laya v1.0).

A v1.0 checkpoint holds the *trainable* weights only (``head``, ``scorer``,
``act_head``, ``type_emb``, ``temperature``); the encoder stays frozen in the
published laya model. Loading such a checkpoint into a fresh ``laya.Agent`` is
therefore a ``strict=False`` load, and the one thing that must not happen is a
non-encoder weight silently failing to load: the agent would then run a random
head and answer confidently, which is worse than crashing.

``apply_head_state_dict`` makes that failure loud. ``TrainedLayaAgent`` wraps
the resulting agent so it matches hd-laya's ``predict(state, questions)``
contract: laya reports per-option ``probabilities``, while hd-laya's z-gate
reads a ``scores`` distribution, so the adapter mirrors one onto the other and
passes the token budgets training used.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

__all__ = ["ENCODER_PREFIX", "TrainedLayaAgent", "apply_head_state_dict", "load_checkpoint_state"]

ENCODER_PREFIX = "encoder."


def load_checkpoint_state(path: str, map_location: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a training checkpoint and return ``(checkpoint, state_dict)``."""
    import torch

    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # Older torch builds lack `weights_only`, and a checkpoint carrying plain
        # Python metadata is refused by newer ones. Retry unrestricted.
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("state_dict"), dict):
        raise ValueError(f"{path!r} is not a hd-laya training checkpoint (no 'state_dict')")
    return ckpt, ckpt["state_dict"]


def apply_head_state_dict(model: Any, state_dict: Dict[str, Any]) -> Tuple[list, list]:
    """Load a head-only checkpoint into ``model``; only encoder weights may be absent.

    Returns ``(missing, unexpected)``. Raises when any non-encoder weight is
    missing or any weight in the checkpoint is unknown to the model, because
    either means the checkpoint does not belong to this architecture.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)
    non_encoder_missing = [k for k in missing if not k.startswith(ENCODER_PREFIX)]
    if non_encoder_missing:
        raise RuntimeError(
            "checkpoint is missing %d non-encoder weight(s), so the head would run "
            "random: %s" % (len(non_encoder_missing), non_encoder_missing[:5])
        )
    if unexpected:
        raise RuntimeError(
            "checkpoint has %d weight(s) this model does not define: %s"
            % (len(unexpected), unexpected[:5])
        )
    return missing, unexpected


class TrainedLayaAgent:
    """A laya ``Agent`` with a trained head applied, in hd-laya's predict shape.

    ``max_len`` / ``head_max_len`` must be the budgets training used: the head
    is trained on marker positions that move when the sequence layout changes.
    """

    def __init__(self, agent: Any, *, max_len: int = 512, head_max_len: int = 192) -> None:
        self.agent = agent
        self.max_len = int(max_len)
        self.head_max_len = int(head_max_len)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        model_id: str = "convaiinnovations/laya",
        device: Any = "cpu",
        max_len: int = 512,
        head_max_len: int = 192,
        agent: Any = None,
    ) -> "TrainedLayaAgent":
        """Load ``model_id`` and apply ``checkpoint`` to it.

        ``agent`` may be passed to reuse an already-loaded laya Agent (the bench
        loads the model once for every K in one process).
        """
        if agent is None:
            from laya.agent import Agent

            agent = Agent(model_id, device=device)
        agent.model.eval()
        _, state_dict = load_checkpoint_state(checkpoint)
        apply_head_state_dict(agent.model, state_dict)
        return cls(agent, max_len=max_len, head_max_len=head_max_len)

    def predict(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        result = self.agent.predict(
            state,
            questions,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
        )
        if not isinstance(result, dict):
            return {"answers": result or {}, "usage": {}}
        answers = result.get("answers")
        if isinstance(answers, dict):
            for answer in answers.values():
                if not isinstance(answer, dict) or answer.get("scores") is not None:
                    continue
                probabilities = answer.get("probabilities")
                if isinstance(probabilities, dict) and probabilities:
                    scores = {k: float(v) for k, v in probabilities.items()}
                    answer["scores"] = scores
                    answer.setdefault("score", max(scores.values()))
        return result
