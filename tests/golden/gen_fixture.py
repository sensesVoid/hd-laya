"""Generate tests/golden/two_stage_fixture.json (run once, then freeze)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from hd_laya.hv import DEFAULT_SEED
from hd_laya.index import OptionIndex

_WORDS = ("refund cancel track package order ship late item wrong size color "
          "return label email code verify account login password reset store "
          "pickup ready delivery address billing invoice paid charge").split()


def main():
    rng = np.random.default_rng(0)
    opts = []
    seen = set()
    while len(opts) < 50:
        k = int(rng.integers(2, 5))
        words = [rng.choice(_WORDS) for _ in range(k)]
        s = " ".join(words)
        if s not in seen:
            seen.add(s)
            opts.append(s)
    # 5 queries: take 5 options and perturb one char
    queries = []
    for i in range(5):
        src = opts[i * 10 % len(opts)]
        # swap two adjacent distinct chars
        j = 0
        while j < len(src) - 1 and src[j] == src[j + 1]:
            j += 1
        q = src[:j] + src[j + 1] + src[j] + src[j + 2:]
        queries.append(q)

    idx = OptionIndex.fit(opts, seed=DEFAULT_SEED)
    cases = []
    for q in queries:
        labels, scores, meta = idx.rank(q)
        cases.append({
            "query": q,
            "top1": labels[0],
            "top2_scores": [round(float(s), 6) for s in scores[:2]],
            "margin": round(float(meta["margin"]), 6),
        })
    out = {"seed": DEFAULT_SEED, "options": opts, "queries": queries, "cases": cases}
    path = os.path.join(os.path.dirname(__file__), "two_stage_fixture.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
