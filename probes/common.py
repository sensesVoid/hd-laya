"""Shared machinery for the hd-laya dimension probes.

Faithful re-implementation of SQAC's zero-dependency BSC encoder
(sqac/encoder.py, BSCEncoder) plus corpus builders and packed-scan utils.
Everything is bit-exact with the production encoder:

  atom(token)  = sha256(seed:token:counter) chained to dims/8 bytes
  encode(text) = majority-vote bundle of trigram atoms of "  {text.strip().lower()}  "
  similarity   = 1 - hamming/dims

Key property used by the matryoshka probe: because atoms chain identical
hashes, atom(D1) == atom(D2)[:D1/8] for D1 < D2 (same seed), so a
prefix-truncated 2048-dim bundle is BIT-IDENTICAL to the plain-1024 bundle.
probe2 verifies this mechanically.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- BSC encoder


class BSCEncoder:
    """Bit-exact port of SQAC's BSCEncoder (numpy bundling path)."""

    def __init__(self, dims: int = 1024, seed: str = "sqac-v1", ngram: int = 3):
        if dims % 8 != 0:
            raise ValueError("dims must be a multiple of 8")
        self.dims = dims
        self.seed = seed
        self.ngram = ngram
        self._atom_cache: dict[str, bytes] = {}

    def atom(self, token: str) -> bytes:
        cached = self._atom_cache.get(token)
        if cached is not None:
            return cached
        need = self.dims // 8
        out = bytearray()
        counter = 0
        while len(out) < need:
            h = hashlib.sha256(f"{self.seed}:{token}:{counter}".encode()).digest()
            out.extend(h)
            counter += 1
        hv = bytes(out[:need])
        self._atom_cache[token] = hv
        return hv

    def encode_bits(self, text: str) -> bytes:
        text = f"  {text.strip().lower()}  "
        if not text.strip():
            return bytes(self.dims // 8)
        grams = [
            text[i : i + self.ngram]
            for i in range(0, max(len(text) - self.ngram + 1, 1))
        ]
        vecs = [self.atom(g) for g in grams]
        n = len(vecs)
        nbytes = self.dims // 8
        if n == 1:
            return bytes(vecs[0])
        mat = np.frombuffer(b"".join(vecs), dtype=np.uint8).reshape(n, -1)[:, :nbytes]
        bits = np.unpackbits(mat, axis=1, bitorder="big")  # (n, D) MSB-first
        majority = bits.sum(axis=0) > n / 2
        return np.packbits(majority, bitorder="big").tobytes()

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """(N, dims//8) uint8 packed matrix."""
        return np.frombuffer(
            b"".join(self.encode_bits(t) for t in texts), dtype=np.uint8
        ).reshape(len(texts), self.dims // 8)


# ---------------------------------------------------------------- packed scan

_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_counts(query_row: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Hamming distances of one packed query against (N, nb) matrix."""
    x = np.bitwise_xor(matrix, query_row[None, :])
    return _LUT[x].sum(axis=1, dtype=np.uint16)


def sims_from_counts(counts: np.ndarray, dims: int) -> np.ndarray:
    return 1.0 - counts.astype(np.float64) / dims


def scan_time(matrix: np.ndarray, reps: int = 20, seed: int = 0) -> tuple[float, float]:
    """Median seconds per full scan of matrix; returns (median, p95)."""
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    q_idx = rng.integers(0, n, size=reps)
    times = []
    for i in q_idx:
        q = matrix[i]
        t0 = time.perf_counter()
        hamming_counts(q, matrix)
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    return med, p95


# ---------------------------------------------------------------- corpora

_WORD_RE = re.compile(r"[a-zA-Z]{3,14}")


def collect_vocab(max_words: int = 6000) -> list[str]:
    """Real English vocabulary from local text (sqac docs + stdlib source)."""
    words: set[str] = set()
    roots = [
        Path(r"C:\Users\geral\sqac"),
        Path(r"C:\Users\geral\AppData\Local\Programs\Python\Python312\Lib"),
    ]
    seen_files = 0
    for root in roots:
        for p in root.rglob("*.py"):
            if seen_files >= 400:
                break
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            words.update(w.lower() for w in _WORD_RE.findall(txt))
            seen_files += 1
        for p in root.rglob("*.md"):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            words.update(w.lower() for w in _WORD_RE.findall(txt))
        if seen_files >= 400:
            break
    vocab = sorted(words)
    if len(vocab) > max_words:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(vocab), size=max_words, replace=False)
        vocab = [vocab[i] for i in idx]
    return vocab


def build_labels(n: int, seed: int = 0, vocab: list[str] | None = None) -> list[str]:
    """Option-like short labels: 2-4 real words, unique."""
    if vocab is None:
        vocab = collect_vocab()
    rng = np.random.default_rng(seed)
    out: set[str] = set()
    guard = 0
    while len(out) < n and guard < n * 20:
        k = int(rng.integers(2, 5))
        words = [vocab[i] for i in rng.integers(0, len(vocab), size=k)]
        out.add(" ".join(words))
        guard += 1
    return sorted(out)[:n]


def build_spans(n: int, lo: int = 200, hi: int = 400, seed: int = 0) -> list[str]:
    """Context-like spans of real prose from local md/py files."""
    texts: list[str] = []
    roots = [
        Path(r"C:\Users\geral\sqac"),
        Path(r"C:\Users\geral\AppData\Local\Programs\Python\Python312\Lib"),
    ]
    seen_files = 0
    for root in roots:
        for p in sorted(root.rglob("*.py")) + sorted(root.rglob("*.md")):
            if seen_files >= 600:
                break
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if len(txt) > 400_000:
                continue
            texts.append(txt)
            seen_files += 1
        if seen_files >= 600:
            break
    rng = np.random.default_rng(seed)
    spans: set[str] = set()
    guard = 0
    big = "\n".join(texts)
    while len(spans) < n and guard < n * 30:
        start = int(rng.integers(0, max(len(big) - hi, 1)))
        width = int(rng.integers(lo, hi))
        span = big[start : start + width].strip()
        span = re.sub(r"\s+", " ", span)
        if len(span) >= lo:
            spans.add(span)
        guard += 1
    return sorted(spans)[:n]


# ---------------------------------------------------------------- perturbs

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def perturb(text: str, kind: str, rng: np.random.Generator) -> str:
    """typo | drop | swap; returns a string guaranteed different."""
    for _ in range(8):
        if kind == "typo":
            i = int(rng.integers(0, len(text)))
            c = _ALPHABET[int(rng.integers(0, 26))]
            new = text[:i] + c + text[i + 1 :]
        elif kind == "drop":
            words = text.split()
            if len(words) < 2:
                continue
            i = int(rng.integers(0, len(words)))
            new = " ".join(words[:i] + words[i + 1 :])
        elif kind == "swap":
            words = text.split()
            if len(words) < 3:
                continue
            i = int(rng.integers(0, len(words) - 1))
            words[i], words[i + 1] = words[i + 1], words[i]
            new = " ".join(words)
        else:
            raise ValueError(kind)
        if new != text and new.strip():
            return new
    return text + " x"


def jitter_typo(text: str, rng: np.random.Generator) -> str:
    return perturb(text, "typo", rng)
