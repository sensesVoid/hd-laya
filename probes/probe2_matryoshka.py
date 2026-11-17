"""Probe 2 - matryoshka truncation on the lexical BSC tier.

Frozen bars (prereg): P1 bit-identity of prefix truncation; B2 flip rates;
B3 two-stage final-top1 == full-D; near-tie flip conditioning on margin.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import BSCEncoder, build_labels, perturb, hamming_counts  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

N_OPTIONS = 10_000
N_QUERIES_PER_KIND = 200
SEEDS = [0, 1, 2]
DFULL_GRID = [2048, 4096, 8192]
KFRAC_GRID = {  # prefix bits k for each Dfull
    2048: [512, 1024, 1536, 2048],
    4096: [512, 1024, 2048, 3072, 4096],
    8192: [512, 1024, 2048, 4096, 8192],
}
M_GRID = [16, 64, 256]
MARGIN_BUCKETS = [(0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 1.1)]


def bit_identity_check() -> dict:
    """P1: atom(D1) == atom(D2)[:D1/8]; encode(D1) == encode(D2)[:D1/8]."""
    enc1 = BSCEncoder(dims=1024)
    enc2 = BSCEncoder(dims=2048)
    texts = ["top up failed by card", "  pending  TOP UP ", "refund issued twice march"]
    atom_ok = all(enc1.atom(t) == enc2.atom(t)[:128] for t in ["abc", "xyz card", "q"])
    enc_ok = all(enc1.encode_bits(t) == enc2.encode_bits(t)[:128] for t in texts)
    return {"atom_prefix_equal": bool(atom_ok), "encode_prefix_equal": bool(enc_ok)}


def rank_metrics(query: bytes, mat: np.ndarray, dims_used: int):
    counts = hamming_counts(query, mat)
    order = np.argsort(counts, kind="stable")  # ascending distance
    sims = 1.0 - counts.astype(np.float64) / dims_used
    top = order[0]
    second_sim = sims[order[1]] if len(order) > 1 else -1.0
    return order, sims, top, sims[top] - second_sim


def main() -> None:
    ident = bit_identity_check()
    print("bit_identity:", ident, flush=True)

    labels = build_labels(N_OPTIONS, seed=11)
    results = []
    for dfull in DFULL_GRID:
        enc = BSCEncoder(dims=dfull)
        mat_full = enc.encode_batch(labels)          # (N, dfull/8)
        dims_full_bytes = dfull // 8

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            kinds = ["typo", "drop", "swap"]
            queries, golds = [], []
            for kind in kinds:
                idxs = rng.choice(len(labels), size=N_QUERIES_PER_KIND, replace=False)
                for i in idxs:
                    queries.append(perturb(labels[i], kind, rng))
                    golds.append(int(i))
            qmat = enc.encode_batch(queries)         # (Q, dfull/8)

            # full-D ranking (ground truth for this tier)
            full_top, full_margin = [], []
            for qi in range(len(queries)):
                _, _, top, margin = rank_metrics(qmat[qi], mat_full, dfull)
                full_top.append(top)
                full_margin.append(margin)
            full_top = np.array(full_top)
            full_margin_arr = np.array(full_margin)
            golds_arr = np.array(golds)
            full_acc = float((full_top == golds_arr).mean())

            for k in KFRAC_GRID[dfull]:
                kb = k // 8
                mat_k = mat_full[:, :kb]
                qmat_k = qmat[:, :kb]
                k_top = np.empty(len(queries), dtype=np.int64)
                agree = np.empty(len(queries), dtype=bool)
                recalls = {M: 0 for M in M_GRID}
                two_stage_disagree = 0
                for qi in range(len(queries)):
                    counts = hamming_counts(qmat_k[qi], mat_k)
                    order = np.argsort(counts, kind="stable")
                    top = order[0]
                    k_top[qi] = top
                    agree[qi] = top == full_top[qi]
                    for M in M_GRID:
                        recalls[M] += int(full_top[qi] in order[:M])
                    # two-stage: re-rank the top-64 candidates with full-D bits
                    cand = order[:64]
                    counts2 = hamming_counts(qmat[qi], mat_full[cand])
                    pick = cand[np.argsort(counts2, kind="stable")[0]]
                    two_stage_disagree += int(pick != full_top[qi])
                k_acc = float((k_top == golds_arr).mean())
                recalls = {M: hits / len(queries) for M, hits in recalls.items()}
                ts_dis = two_stage_disagree / len(queries)

                # flips conditioned on full-D margin (vectorized, no extra scans)
                flip_by_margin = {}
                for lo, hi in MARGIN_BUCKETS:
                    m = (full_margin_arr >= lo) & (full_margin_arr < hi)
                    flip_by_margin[f"{lo}-{hi}"] = {
                        "n": int(m.sum()),
                        "flip_rate": float(1.0 - agree[m].mean()) if m.sum() else None,
                    }

                row = {
                    "dfull": dfull,
                    "k": k,
                    "seed": seed,
                    "full_acc": full_acc,
                    "k_acc": k_acc,
                    "acc_delta_pp": (full_acc - k_acc) * 100,
                    "top1_agree": float(agree.mean()),
                    "flip_rate": float(1.0 - agree.mean()),
                    "flips_by_margin": flip_by_margin,
                    "recall_at_M": {str(M): v for M, v in recalls.items()},
                    "two_stage_final_disagree": ts_dis,
                    "queries": len(queries),
                }
                results.append(row)
                print(json.dumps(row), flush=True)

    summary = {}
    for dfull in DFULL_GRID:
        for k in KFRAC_GRID[dfull]:
            rows = [r for r in results if r["dfull"] == dfull and r["k"] == k]
            summary[f"{k}of{dfull}"] = {
                "mean_full_acc": float(np.mean([r["full_acc"] for r in rows])),
                "mean_k_acc": float(np.mean([r["k_acc"] for r in rows])),
                "mean_acc_delta_pp": float(np.mean([r["acc_delta_pp"] for r in rows])),
                "mean_flip_rate": float(np.mean([r["flip_rate"] for r in rows])),
                "mean_recall_at_64": float(np.mean([r["recall_at_M"]["64"] for r in rows])),
                "mean_two_stage_disagree": float(np.mean([r["two_stage_final_disagree"] for r in rows])),
            }
    result = {"probe": "2-matryoshka-lexical", "bit_identity": ident, "rows": results, "summary": summary}
    (OUT / "probe2_matryoshka.json").write_text(json.dumps(result, indent=1))
    print("WROTE", OUT / "probe2_matryoshka.json")


if __name__ == "__main__":
    main()
