"""Measure HDC index recall on the well-posed high-K generator.

The index is the part of hd-laya that removes laya's option ceiling, so its
recall is the number that matters: if the correct option is not in the M
survivors, no head can answer. Runs on numpy alone, no model, no GPU.

Usage:
    python scripts/measure_index_recall.py --n 32 --m 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hd_laya.data_gen import gen_dataset
from hd_laya.index import OptionIndex


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--k", type=int, nargs="+", default=[77, 150, 500, 1000, 2000])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for K in args.k:
        items = gen_dataset(args.n, K, seed=args.seed + K)
        hits = 0
        ranks = []
        t0 = time.perf_counter()
        for item in items:
            criteria = item["questions"]["q"]["criteria"]
            labels = list(criteria)
            texts = [f"{lbl}: {criteria[lbl]}" for lbl in labels]
            expected_text = f"{item['expected']}: {criteria[item['expected']]}"

            idx = OptionIndex.fit(texts)
            top_texts, _scores, _meta = idx.rank(item["state"], k=1024, M=args.m)

            if expected_text in top_texts:
                hits += 1
            ranks.append(1 + list(texts).index(expected_text) if expected_text in top_texts else None)
        elapsed = time.perf_counter() - t0

        recall = hits / len(items)
        rows.append({"K": K, "n": len(items), "m": args.m, "recall": round(recall, 4),
                     "seconds": round(elapsed, 2), "ms_per_item": round(1000 * elapsed / len(items), 1)})
        print(json.dumps(rows[-1]), flush=True)

    print(json.dumps({"recall_by_k": {r["K"]: r["recall"] for r in rows}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
