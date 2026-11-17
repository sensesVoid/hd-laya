"""Two-stage coarse-to-fine option index for hd_laya (T2).

Exact (BSC n-gram) tier: text -> seeded HV via the bit-exact recipe shared with
probes/common.py (padded lowercase char n-grams -> sha256-chain atoms ->
majority bundle). Ranking is two-stage: a cheap prefix-k scan over ALL options
selects a top-M candidate set, then a full-D re-rank of only those M. Measured
(probe 2): 0-0.056% disagreement vs full-D, only near-ties (margin < 0.02)
may flip; recall@64 = 1.0.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from hd_laya.hv import DEFAULT_SEED, atom, hamming_counts, pack_bits

__all__ = ["OptionIndex"]


def _encode_text(text: str, seed: str, ngram: int, dims: int) -> np.ndarray:
    """Bit-exact text->HV (must match probes/common.py BSCEncoder.encode_bits)."""
    padded = f"  {text.strip().lower()}  "
    if not padded.strip():
        return np.zeros(dims // 8, dtype=np.uint8)
    grams = [padded[i : i + ngram] for i in range(0, max(len(padded) - ngram + 1, 1))]
    vecs = [atom(seed, g, dims) for g in grams]
    if len(vecs) == 1:
        return vecs[0].copy()
    bits = np.unpackbits(np.stack(vecs), axis=1, bitorder="big")  # (n, dims)
    return pack_bits(bits.sum(axis=0) > len(vecs) / 2)


class OptionIndex:
    """Precomputed option HV matrix + two-stage ranker."""

    def __init__(
        self,
        options: list[str],
        matrix: np.ndarray,
        dims_full: int,
        seed: str,
        ngram: int,
    ) -> None:
        self.options = list(options)
        self.matrix = np.ascontiguousarray(matrix, dtype=np.uint8)
        self.dims_full = int(dims_full)
        self.seed = seed
        self.ngram = int(ngram)

    @classmethod
    def fit(
        cls,
        options: list[str],
        dims_full: int = 2048,
        seed: str = DEFAULT_SEED,
        ngram: int = 3,
    ) -> "OptionIndex":
        opts = list(options)
        mat = np.stack([_encode_text(o, seed, ngram, dims_full) for o in opts]) if opts else np.zeros((0, dims_full // 8), dtype=np.uint8)
        return cls(opts, mat, dims_full, seed, ngram)

    def encode(self, text: str) -> np.ndarray:
        return _encode_text(text, self.seed, self.ngram, self.dims_full)

    def rank(
        self,
        query_text: str,
        k: int = 1024,
        M: int = 64,
    ) -> tuple[list[str], np.ndarray, dict]:
        n = len(self.options)
        if n == 0:
            return [], np.zeros(0, dtype=np.float32), {
                "prefix_top1": None,
                "final_top1": None,
                "margin": 0.0,
                "near_tie": True,
                "k": min(k, self.dims_full),
                "M": 0,
                "mode": "exact",
            }
        k = min(int(k), self.dims_full)
        M = min(int(M), n)
        if M < 1:
            M = 1
        query = self.encode(query_text)
        prefix_bytes = k // 8
        # Stage 1: prefix-k scan over ALL options.
        prefix_dists = hamming_counts(query[:prefix_bytes], self.matrix[:, :prefix_bytes])
        stage1_order = np.argsort(prefix_dists, kind="stable")
        cand_idx = stage1_order[:M]
        prefix_top1 = self.options[int(stage1_order[0])]
        # Stage 2: full-D re-rank of the top-M candidates.
        full_dists = hamming_counts(query, self.matrix[cand_idx])
        order2 = np.argsort(full_dists, kind="stable")
        final_idx = cand_idx[order2]
        labels = [self.options[int(i)] for i in final_idx]
        scores = (1.0 - full_dists[order2].astype(np.float64) / self.dims_full).astype(np.float32)
        if scores.size >= 2:
            margin = float(scores[0]) - float(scores[1])
        else:
            margin = 0.0
        near_tie = bool(margin < 0.02)
        meta = {
            "prefix_top1": prefix_top1,
            "final_top1": labels[0] if labels else None,
            "margin": margin,
            "near_tie": near_tie,
            "k": k,
            "M": M,
            "mode": "exact",
        }
        return labels, scores, meta

    # --- Semantic tier (T3) -------------------------------------------------
    @classmethod
    def fit_semantic(
        cls,
        options: list[str],
        embed_fn: "Callable[[list[str]], np.ndarray]",
        dims: int = 2048,
        seed: str = DEFAULT_SEED,
    ) -> "OptionIndex":
        """Build an index whose HVs come from simhash_bits(embed_fn(options))."""
        from hd_laya.hv import simhash_bits

        opts = list(options)
        emb = np.asarray(embed_fn(opts), dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        # L2-normalize rows (guard zero-norm).
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        mat = simhash_bits(emb, dims, seed)
        return OptionIndex(opts, mat, dims, seed, ngram=3)

    def rank_semantic(
        self,
        query_text: str,
        k: int = 1024,
        M: int = 64,
        embed_fn: "Optional[Callable[[list[str]], np.ndarray]]" = None,
    ) -> tuple[list[str], np.ndarray, dict]:
        # Semantic tier needs an embedder for the query side; if none given,
        # fall back to treating the stored matrix as the query's HV source by
        # encoding the query text with the exact recipe (degrades to exact tier).
        if embed_fn is None:
            return self.rank(query_text, k=k, M=M)
        q_emb = np.asarray(embed_fn([query_text]), dtype=np.float32)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q_emb = q_emb / norms
        from hd_laya.hv import simhash_bits

        q_mat = simhash_bits(q_emb, self.dims_full, self.seed)
        # Reuse two-stage logic by temporarily swapping the matrix/query.
        n = len(self.options)
        if n == 0:
            return [], np.zeros(0, dtype=np.float32), {
                "prefix_top1": None,
                "final_top1": None,
                "margin": 0.0,
                "near_tie": True,
                "k": min(k, self.dims_full),
                "M": 0,
                "mode": "semantic",
            }
        k = min(int(k), self.dims_full)
        M = min(int(M), n)
        if M < 1:
            M = 1
        prefix_bytes = k // 8
        prefix_dists = hamming_counts(q_mat[0, :prefix_bytes], self.matrix[:, :prefix_bytes])
        stage1_order = np.argsort(prefix_dists, kind="stable")
        cand_idx = stage1_order[:M]
        prefix_top1 = self.options[int(stage1_order[0])]
        full_dists = hamming_counts(q_mat[0], self.matrix[cand_idx])
        order2 = np.argsort(full_dists, kind="stable")
        final_idx = cand_idx[order2]
        labels = [self.options[int(i)] for i in final_idx]
        scores = (1.0 - full_dists[order2].astype(np.float64) / self.dims_full).astype(np.float32)
        margin = float(scores[0]) - float(scores[1]) if scores.size >= 2 else 0.0
        meta = {
            "prefix_top1": prefix_top1,
            "final_top1": labels[0] if labels else None,
            "margin": margin,
            "near_tie": bool(margin < 0.02),
            "k": k,
            "M": M,
            "mode": "semantic",
        }
        return labels, scores, meta
