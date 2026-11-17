"""Probe 3 - semantic tier (MiniLM SimHash) truncation vs float-cosine truth.

The accuracy question that matters for laya's option index: does a k-bit
SimHash ranking preserve the embedder's own retrieval? Ground truth here is
the float-cosine ranking of all-MiniLM-L6-v2 embeddings.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))
from common import build_spans, perturb  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

N_OPTIONS = 2_500
N_QUERIES = 300
N_DUP_QUERIES = 100
DMAX = 8192
KGRID = [512, 1024, 2048, 4096, 8192]
M_GRID = [16, 64, 256]
MARGIN_BUCKETS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.08), (0.08, 1.1)]
SEEDS = [0, 1]

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL)
    mdl.eval()
    return tok, mdl, torch


@torch.no_grad()
def embed(texts, tok, mdl, torch, bs=64):
    embs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt")
        out = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        e = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embs.append(torch.nn.functional.normalize(e, dim=1))
    return torch.cat(embs).numpy().astype(np.float32)


def main() -> None:
    torch_dev = "cpu"
    tok, mdl, torch = load_embedder()
    spans = build_spans(N_OPTIONS, seed=5)
    print(f"spans={len(spans)}", flush=True)

    t0 = time.perf_counter()
    opt_emb = embed(spans, tok, mdl, torch)
    embed_s = time.perf_counter() - t0
    print(f"embedded options in {embed_s:.1f}s", flush=True)

    # projection at DMAX; first-k columns are a valid rademacher 384xk matrix
    proj = np.random.default_rng(42).choice([-1.0, 1.0], size=(384, DMAX)).astype(np.float32)
    opt_bits = np.packbits((opt_emb @ proj) > 0, axis=1, bitorder="big")  # (N, DMAX/8)

    # queries: self-retrieval + near-duplicates
    rows_out = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        q_idx = list(rng.choice(len(spans), size=N_QUERIES, replace=False))
        dup_idx = list(rng.choice(len(spans), size=N_DUP_QUERIES, replace=False))
        queries = [spans[i] for i in q_idx] + [perturb(spans[i], "typo", rng) for i in dup_idx]
        golds = q_idx + dup_idx
        q_emb = embed(queries, tok, mdl, torch)

        # float-cosine truth
        cos = q_emb @ opt_emb.T
        float_order = np.argsort(-cos, axis=1, kind="stable")
        float_top = float_order[:, 0]
        float_margin = cos[np.arange(len(queries)), float_order[:, 0]] - cos[
            np.arange(len(queries)), float_order[:, 1]
        ]
        golds_arr = np.array(golds)
        # gold-hit rate of float top1 (sanity: self-retrieval should be ~1.0)
        float_gold_acc = float((float_top == golds_arr).mean())

        q_bits = np.packbits((q_emb @ proj) > 0, axis=1, bitorder="big")

        for k in KGRID:
            kb = k // 8
            opt_k = opt_bits[:, :kb]
            q_k = q_bits[:, :kb]
            xor = np.bitwise_xor(q_k[:, None, :], opt_k[None, :, :])
            dists = _LUT_count(xor)  # (Q, N)
            bits_order = np.argsort(dists, axis=1, kind="stable")
            bits_top = bits_order[:, 0]
            bit_sims = 1.0 - dists / k
            bits_second = bit_sims[np.arange(len(queries)), bits_order[:, 1]]
            margin = bit_sims[np.arange(len(queries)), bits_top] - bits_second
            agree = bits_top == float_top

            flips_by_margin = {}
            for lo, hi in MARGIN_BUCKETS:
                m = (float_margin >= lo) & (float_margin < hi)
                flips_by_margin[f"{lo}-{hi}"] = {
                    "n": int(m.sum()),
                    "flip_rate": float(1 - agree[m].mean()) if m.sum() else None,
                }

            # recall@M of bit top-M vs float top-1
            recalls = {}
            for M in M_GRID:
                cand = bits_order[:, :M]
                recalls[M] = float(np.mean([float_top[qi] in cand[qi] for qi in range(len(queries))]))

            # two-stage: re-rank top-64 candidates with full DMAX bits
            disagree = 0
            for qi in range(len(queries)):
                cand = bits_order[qi, :64]
                sub = opt_bits[cand]
                d2 = _LUT_count(np.bitwise_xor(q_bits[qi][None, :], sub))
                pick = cand[np.argsort(d2[0], kind="stable")[0]]
                disagree += int(pick != float_top[qi])

            rows_out.append(
                {
                    "k": k,
                    "seed": seed,
                    "float_gold_acc": float_gold_acc,
                    "bit_top1_vs_float_top1_agree": float(agree.mean()),
                    "bit_gold_acc": float((bits_top == golds_arr).mean()),
                    "flips_by_margin": flips_by_margin,
                    "recall_at_M_vs_float": {str(M): v for M, v in recalls.items()},
                    "two_stage_final_disagree_vs_float_top1": disagree / len(queries),
                    "queries": len(queries),
                }
            )
            print(json.dumps(rows_out[-1]), flush=True)

    summary = {}
    for k in KGRID:
        rows = [r for r in rows_out if r["k"] == k]
        summary[f"k{k}"] = {
            "mean_bit_top1_vs_float_agree": float(np.mean([r["bit_top1_vs_float_top1_agree"] for r in rows])),
            "mean_recall_at_64_vs_float": float(np.mean([r["recall_at_M_vs_float"]["64"] for r in rows])),
            "mean_two_stage_disagree": float(np.mean([r["two_stage_final_disagree_vs_float_top1"] for r in rows])),
            "mean_flips_clear_margin": float(
                np.mean(
                    [
                        r["flips_by_margin"]["0.08-1.1"]["flip_rate"] or 0.0
                        for r in rows
                        if r["flips_by_margin"]["0.08-1.1"]["n"]
                    ]
                )
            ),
        }
    result = {
        "probe": "3-semantic-simhash",
        "model": MODEL,
        "embed_seconds": embed_s,
        "rows": rows_out,
        "summary": summary,
    }
    (OUT / "probe3_semantic.json").write_text(json.dumps(result, indent=1))
    print("WROTE", OUT / "probe3_semantic.json")


def _LUT_count(x: np.ndarray) -> np.ndarray:
    from common import _LUT

    return _LUT[x].sum(axis=-1, dtype=np.uint16)


if __name__ == "__main__":
    main()
