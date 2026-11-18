"""Verify a trained v1.0 checkpoint end-to-end at inference (pre-Kaggle smoke).

Loads ``convaiinnovations/laya``, applies the head-only checkpoint with the
strict loader, then runs the hd-laya two-stage path (HDC narrowing -> laya
forward -> z-gate) on a handful of generated items and prints the summary.

Any failure here means the benchmark kernel would waste GPU time, so this runs
first, locally, at the same budgets (max_len=512, head_max_len=192).

Usage:
    python scripts/verify_trained_inference.py --checkpoint path/to/hd_laya_trained.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--k", type=int, nargs="+", default=[20, 77])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--head-max-len", type=int, default=192)
    args = ap.parse_args()

    import torch

    from hd_laya.agent import HDLayaAgent
    from hd_laya.bench import BenchmarkItem, run_benchmark
    from hd_laya.data_gen import gen_dataset
    from hd_laya.laya_agent import TrainedLayaAgent

    from laya.agent import Agent

    print(f"loading laya ({args.device}) ...")
    agent = Agent("convaiinnovations/laya", device=args.device)
    print("applying checkpoint ...")
    trained = TrainedLayaAgent.from_checkpoint(
        args.checkpoint,
        agent=agent,
        max_len=args.max_len,
        head_max_len=args.head_max_len,
    )
    model = agent.model
    n_head = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("encoder."))
    print(f"head params: {n_head:,}")

    hd = HDLayaAgent(trained, max_options=20, z_bar=5.0)

    failures = 0
    for K in args.k:
        raw = gen_dataset(args.n, K, seed=7)
        items = [BenchmarkItem(state=it["state"], expected=it["expected"],
                               meta={"K": K, "questions": it["questions"]}) for it in raw]
        res = run_benchmark(hd, items, lambda item: item.meta["questions"], question_id="q")
        summary = dict(res["summary"])
        summary["K"] = K
        print(json.dumps(summary))

        row = res["rows"][0]
        ans = (row.get("agent_meta") or {}).get("hd_index", {}).get("q", {})
        print(json.dumps({"narrowed": ans.get("narrowed"), "to": ans.get("narrowed_to")}))

        if summary["accuracy"] == 0.0:
            # Informational only: a head that never saw this exact split can still
            # score low. The hard failure is producing answers at all.
            print(f"WARNING: K={K} accuracy is 0.0 over {summary['n']} items")
            failures += 0
        if summary["n"] != args.n:
            print(f"ERROR: K={K} scored {summary['n']} of {args.n} items")
            failures += 1

    if torch.cuda.is_available():
        print(f"cuda peak: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    print("OK" if failures == 0 else f"FAILED ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
