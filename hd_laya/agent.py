"""HDC-aware adapter over an upstream laya Agent (hd-laya v0.1).

Wraps a laya Agent (or any object exposing ``predict(state, questions)``) and
adds two things laya cannot do on its own:

1. Option cardinality beyond laya's ~20 ceiling: choice questions with more
   than ``max_options`` criteria are narrowed with the two-stage HDC option
   index (OptionIndex) before the underlying agent sees them; the answer is
   mapped back to the original label set.
2. Post-hoc z-gate abstention (NTWD-1 heritage): each choice answer's score
   distribution is converted to a z-score; answers below ``z_bar`` are refused
   (REFUSE) instead of being returned as confident guesses.

The encoder is NEVER touched here; only the option set and the final answer
are affected, so this layer is engine-agnostic (works with any laya Agent).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from hd_laya.index import OptionIndex
from hd_laya.zgate import REFUSE, StreamingCalib, decide, z_from_scores

__all__ = ["HDLayaAgent"]


def _state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, (list, tuple)):
        return " ".join(str(x) for x in state)
    try:
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(state)


class HDLayaAgent:
    def __init__(
        self,
        agent: Any,
        *,
        max_options: int = 20,
        z_bar: float = 5.0,
        index_seed: str = "hd-laya-v1",
        calib: Optional[StreamingCalib] = None,
    ) -> None:
        if max_options < 1:
            raise ValueError("max_options must be >= 1")
        self.agent = agent
        self.max_options = int(max_options)
        self.z_bar = float(z_bar)
        self.index_seed = index_seed
        self.calib = calib if calib is not None else StreamingCalib()

    def _narrow_choice(
        self, qid: str, qdef: Dict[str, Any], state_text: str
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        criteria = qdef.get("criteria")
        if not (isinstance(criteria, dict) and len(criteria) > self.max_options):
            return qdef, {
                "n_options": len(criteria) if isinstance(criteria, (dict, list)) else 0,
                "narrowed": False,
            }
        labels = list(criteria.keys())
        texts = [f"{lbl}: {criteria[lbl]}" for lbl in labels]
        idx = OptionIndex.fit(texts, seed=self.index_seed)
        top_texts, _scores, meta = idx.rank(state_text, k=1024, M=self.max_options)
        text_to_label = {t: lbl for t, lbl in zip(texts, labels)}
        narrowed_labels = [text_to_label[t] for t in top_texts]
        narrowed_criteria = {lbl: criteria[lbl] for lbl in narrowed_labels}
        new_qdef = dict(qdef)
        new_qdef["criteria"] = narrowed_criteria
        hd_meta = {
            "n_options": len(labels),
            "narrowed": True,
            "narrowed_to": len(narrowed_labels),
            "top_labels": narrowed_labels,
            "index_meta": meta,
        }
        return new_qdef, hd_meta

    def predict(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        state_text = _state_text(state)
        narrowed: Dict[str, Any] = {}
        hd_index: Dict[str, Any] = {}
        for qid, qdef in (questions or {}).items():
            qdef = dict(qdef) if isinstance(qdef, dict) else qdef
            if isinstance(qdef, dict) and qdef.get("type", "choice") == "choice":
                nq, meta = self._narrow_choice(qid, qdef, state_text)
                narrowed[qid] = nq
                hd_index[qid] = meta
            else:
                narrowed[qid] = qdef
                crit = qdef.get("criteria") if isinstance(qdef, dict) else None
                hd_index[qid] = {
                    "n_options": len(crit) if isinstance(crit, (dict, list)) else 0,
                    "narrowed": False,
                }

        result = self.agent.predict(state, narrowed)
        result = dict(result) if isinstance(result, dict) else {"answers": result}
        answers = result.setdefault("answers", {})
        z_map: Dict[str, float] = {}
        refused: List[str] = []
        for qid, qdef in (questions or {}).items():
            qtype = qdef.get("type", "choice") if isinstance(qdef, dict) else "choice"
            if qtype != "choice":
                continue
            ans = answers.get(qid)
            if not isinstance(ans, dict):
                continue
            scores = ans.get("scores")
            if not isinstance(scores, dict) or not scores:
                # Fall back to a single score/confidence if provided.
                val = ans.get("score", ans.get("confidence"))
                scores = {"_single": float(val)} if val is not None else {}
            if not scores:
                z_map[qid] = float("inf")
                continue
            try:
                z = z_from_scores(list(scores.values()))
            except ValueError:
                z = float("nan")
            z_map[qid] = z
            if decide(z, self.z_bar) == REFUSE:
                ans["choice"] = REFUSE
                ans["refused"] = True
                refused.append(qid)
            else:
                ans["refused"] = False
                # Keep the calibration stream fed only from answered decisions.
                try:
                    self.calib.update(float(z) if np.isfinite(z) else 0.0)
                except Exception:
                    pass

        result["meta"] = {
            "hd_index": hd_index,
            "z": z_map,
            "refused": refused,
            "z_bar": self.z_bar,
        }
        return result
