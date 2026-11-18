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
    ap.add_argument("--baseline", action="store_true",
                    help="skip the checkpoint and measure the frozen base model")
    ap.add_argument("--compare", action="store_true",
                    help="measure base, then apply the checkpoint and measure again; "
                         "prints prediction agreement so training gain is not confused "
                         "with base-model skill")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--k", type=int, nargs="+", default=[20, 77])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7, help="generator seed; training used 0")
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

    def evaluate(label, K):
        hd = HDLayaAgent(TrainedLayaAgent(agent, max_len=args.max_len,
                                          head_max_len=args.head_max_len),
                         max_options=20, z_bar=5.0)
        raw = gen_dataset(args.n, K, seed=args.seed)
        items = [BenchmarkItem(state=it["state"], expected=it["expected"],
                               meta={"K": K, "questions": it["questions"]}) for it in raw]
        res = run_benchmark(hd, items, lambda item: item.meta["questions"], question_id="q")
        summary = dict(res["summary"])
        summary["K"] = K
        summary["arm"] = label
        print(json.dumps(summary))
        return res

    if args.compare:
        base = evaluate("base", args.k[0])
        print("applying checkpoint ...")
        TrainedLayaAgent.from_checkpoint(
            args.checkpoint, agent=agent,
            max_len=args.max_len, head_max_len=args.head_max_len,
        )
        tuned = evaluate("trained", args.k[0])
        agree = sum(1 for a, b in zip(base["rows"], tuned["rows"])
                    if a["predicted"] == b["predicted"])
        print(json.dumps({"agreement": round(agree / len(base["rows"]), 4),
                          "n": len(base["rows"]),
                          "accuracy_delta": round(tuned["summary"]["accuracy"]
                                                  - base["summary"]["accuracy"], 4)}))
        return 0

    if args.baseline:
        print("baseline: base checkpoint, no trained weights applied")
    else:
        print("applying checkpoint ...")
        TrainedLayaAgent.from_checkpoint(
            args.checkpoint,
            agent=agent,
            max_len=args.max_len,
            head_max_len=args.head_max_len,
        )

    failures = 0
    for K in args.k:
        res = evaluate("trained" if not args.baseline else "base", K)
        row = res["rows"][0]
        ans = (row.get("agent_meta") or {}).get("hd_index", {}).get("q", {})
        print(json.dumps({"narrowed": ans.get("narrowed"), "to": ans.get("narrowed_to")}))

        if res["summary"]["n"] != args.n:
            print(f"ERROR: K={K} scored {res['summary']['n']} of {args.n} items")
            failures += 1

    if torch.cuda.is_available():
        print(f"cuda peak: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    print("OK" if failures == 0 else f"FAILED ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
