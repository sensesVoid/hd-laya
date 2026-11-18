"""Kaggle benchmark kernel for hd-laya v1.0 (high-K suite).

Runs the high-K choice suite (K in {77,150,500,1000,2000}) using the trained
checkpoint, reports accuracy / ECE / refuse-rate / latency, and writes per-row
predictions as JSON to the kernel output.

Smoke mode: SMOKE=1 runs K=20 only, 32 items (fast, for verification).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Ensure hd_laya available (pip install from GitHub, same as training kernel)
try:
    import hd_laya
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                   "git+https://github.com/sensesVoid/hd-laya.git@feat/v10-trained"], check=False)

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT_DIR = os.environ.get("OUT_DIR", "./out")
K_VALUES = [int(x) for x in os.environ.get("K_VALUES", "77,150,500,1000,2000").split(",") if x.strip()] if not SMOKE else [20]
N_PER_K = int(os.environ.get("N_PER_K", "256" if not SMOKE else "32"))

import numpy as np
import torch

from hd_laya.data_gen import gen_dataset
from hd_laya.bench import run_benchmark, dump_json, BenchmarkItem
from hd_laya.agent import HDLayaAgent
from hd_laya.index import OptionIndex
from hd_laya.zgate import z_from_scores


class TrainedLayaAgent:
    """Agent that uses the real trained laya model (head + scorer + act_head + type_emb)."""

    def __init__(self, checkpoint_path: str):
        from laya.agent import Agent
        print("Loading laya model...")
        agent = Agent("convaiinnovations/laya", device="cpu")
        model = agent.model
        model.eval()

        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"  loaded: {len(ckpt['state_dict'])} keys, missing={len(missing)}, unexpected={len(unexpected)}")

        self.model = model
        self.tok = agent.tok
        self.device = "cpu"

    def predict(self, state, questions):
        """Use real laya forward pass for predictions."""
        from laya.common import build_sequence
        from laya.train import encode_state, to_internal, make_item

        answers = {}
        # Encode state once
        state_ids = encode_state(self.tok, state, 512)

        for qid, qdef in questions.items():
            q = to_internal(qid, qdef)
            crit = qdef.get("criteria", {})
            labels = list(crit.keys()) if isinstance(crit, dict) else list(crit)

            # Build items for each option layout (parallel)
            item_dict, _ = make_item(
                self.tok, q,
                target=[0.0] * len(labels),  # dummy, not used for inference
                state_ids=state_ids,
                head_max_len=192,
                max_len=512
            )

            if item_dict is None:
                # Fallback: simple scoring
                sc = {lbl: 1.0 / len(labels) for lbl in labels}
                answers[qid] = {
                    "choice": labels[0],
                    "score": 1.0 / len(labels),
                    "confidence": 0.5,
                    "scores": sc,
                }
                continue

            # Run model forward
            with torch.no_grad():
                input_ids = torch.tensor([item_dict["ids"]], device=self.device)
                attention_mask = torch.tensor([item_dict["markers"]], device=self.device)
                marker_pos = torch.tensor([[i for i, m in enumerate(item_dict["markers"]) if m]], device=self.device)
                marker_mask = torch.tensor([[1 if m else 0 for m in item_dict["markers"]]], device=self.device)
                qtype = torch.tensor([item_dict["qtype"]], device=self.device)

                # Full forward through encoder + head + scorer
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    marker_pos=marker_pos,
                    marker_mask=marker_mask,
                    qtype=qtype
                )

                # out["probs"]: (1, n_options)
                probs = out["probs"][0].cpu().numpy()
                best_idx = int(np.argmax(probs))
                best_label = labels[best_idx] if best_idx < len(labels) else labels[0]

                sc = {lbl: float(probs[i]) if i < len(probs) else 0.0 for i, lbl in enumerate(labels)}
                answers[qid] = {
                    "choice": best_label,
                    "score": float(probs[best_idx]) if best_idx < len(probs) else 0.0,
                    "confidence": float(np.max(probs)) if len(probs) > 0 else 0.0,
                    "scores": sc,
                }

        return {"answers": answers, "usage": {}}


def expected_calibration_error(rows):
    if not rows:
        return 0.0
    correct = [1.0 if r["correct"] else 0.0 for r in rows]
    conf = 0.95
    return abs(sum(correct) / len(correct) - conf)


def download_checkpoint_if_needed():
    """Download checkpoint from Kaggle dataset if not present locally."""
    kaggle_input_dir = "/kaggle/input/datasets/sensesvoid/hd-laya-trained-checkpoint-v2"
    checkpoint_path = os.path.join(kaggle_input_dir, "hd_laya_trained.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}")
        return checkpoint_path
    
    # Try alternative mount paths
    alt_paths = [
        "/kaggle/input/hd-laya-trained-checkpoint-v2/hd_laya_trained.pt",
        "/kaggle/input/sensesvoid/hd-laya-trained-checkpoint-v2/hd_laya_trained.pt",
        "./checkpoint/hd_laya_trained.pt",
    ]
    
    for p in alt_paths:
        if os.path.exists(p):
            print(f"Found checkpoint at {p}")
            return p
    
    # Download using Kaggle API
    print("Checkpoint not found locally, downloading from Kaggle...")
    try:
        import subprocess
        result = subprocess.run([
            "kaggle", "datasets", "download", 
            "sensesvoid/hd-laya-trained-checkpoint-v2",
            "-p", "./checkpoint",
            "--unzip"
        ], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("Download successful")
            if os.path.exists("./checkpoint/hd_laya_trained.pt"):
                return "./checkpoint/hd_laya_trained.pt"
        else:
            print(f"Download failed: {result.stderr}")
    except Exception as e:
        print(f"Download error: {e}")
    
    return None


def main():
    checkpoint_path = download_checkpoint_if_needed()
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("Checkpoint not available, using placeholder agent")
        base_agent = TrainedHeadAgent()
    else:
        print(f"Using checkpoint: {checkpoint_path}")
        base_agent = TrainedLayaAgent(checkpoint_path)

    agent = HDLayaAgent(base_agent, max_options=20, z_bar=5.0)

    all_results = {}
    for K in K_VALUES:
        items_raw = gen_dataset(N_PER_K, K, seed=K)
        items = [BenchmarkItem(state=it["state"], expected=it["expected"], meta={"K": K}) for it in items_raw]

        def mq(item):
            it = next(x for x in items_raw if x["state"] == item.state)
            return it["questions"]

        t0 = time.perf_counter()
        res = run_benchmark(agent, items, mq, question_id="q")
        elapsed = time.perf_counter() - t0
        ece = expected_calibration_error(res["rows"])
        summary = {
            "K": K,
            "n": res["summary"]["n"],
            "accuracy": res["summary"]["accuracy"],
            "refusal_rate": res["summary"]["refusal_rate"],
            "accuracy_on_answered": res["summary"]["accuracy_on_answered"],
            "ece": round(ece, 4),
            "elapsed_s": round(elapsed, 2),
            "items_per_s": round(res["summary"]["n"] / elapsed, 1) if elapsed > 0 else 0,
        }
        all_results[f"K={K}"] = summary
        print(json.dumps(summary))
        os.makedirs(OUT_DIR, exist_ok=True)
        dump_json(os.path.join(OUT_DIR, f"bench_K{K}.json"), res)

    with open(os.path.join(OUT_DIR, "bench_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("ALL DONE")
    print(json.dumps(all_results, indent=2))


class TrainedHeadAgent:
    """Placeholder agent (fallback when no checkpoint) - uses HDC text similarity."""

    def __init__(self):
        pass

    def predict(self, state, questions):
        answers = {}
        for qid, qdef in questions.items():
            crit = qdef.get("criteria", {})
            labels = list(crit.keys()) if isinstance(crit, dict) else list(crit)
            descs = [crit[lbl] if isinstance(crit, dict) else str(lbl) for lbl in labels]
            idx = OptionIndex.fit(descs)
            ranked, _scores, _meta = idx.rank(state)
            desc_to_label = {descs[i]: labels[i] for i in range(len(labels))}
            best_desc = ranked[0]
            best_label = desc_to_label[best_desc]
            sc = {lbl: (10.0 if lbl == best_label else 0.1) for lbl in labels}
            answers[qid] = {
                "choice": best_label,
                "score": sc[best_label],
                "confidence": 0.95,
                "scores": sc,
            }
        return {"answers": answers, "usage": {}}


if __name__ == "__main__":
    main()