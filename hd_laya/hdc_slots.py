"""HDC option-slot projection for hd-laya v1.0 (T9).

Maps a hyperdimensional vector (or any fixed-width feature) into ``k``
pseudo-tokens in the encoder's model dimension, so a single option occupies
``k`` slots in the head's token budget instead of consuming raw text tokens.
This is the mechanism that decouples option count from laya's ~20-option /
192-token ceiling.

Two paths:
- ``project_slots`` (numpy): deterministic, seeded, dependency-light; used by
  the no-train harness and for golden/fixture generation.
- ``SlotProjector`` (torch.nn.Module): trainable counterpart for v1.0
  training; weights are randomly initialised from a seed and learned.

The projection is a plain linear map ``R^(d_in) -> R^(k * d_model)`` reshaped
to ``(k, d_model)``. No non-linearity in v1.0 (YAGNI; add if training
demands it).
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:  # pragma: no cover - torch optional at import time
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH = False

__all__ = ["project_slots", "SlotProjector"] if _TORCH else ["project_slots"]


def _seed_int(seed: str) -> int:
    return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "little")


def project_slots(
    vec: "np.ndarray",
    k: int = 1,
    d_model: int = 1024,
    seed: str = "hd-laya-slots-v1",
    d_in: Optional[int] = None,
) -> np.ndarray:
    """Deterministic seeded projection of a feature vector into k pseudo-tokens.

    Parameters
    ----------
    vec : array-like, shape (d_in,) or (..., d_in)
    k : number of pseudo-tokens per option
    d_model : encoder model dimension (1024 for ModernBERT-large)
    seed : seed string (vocabulary-as-ABI)
    d_in : input dimension; inferred from vec if None

    Returns
    -------
    np.ndarray, shape (..., k, d_model), float32
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if d_model < 1:
        raise ValueError("d_model must be >= 1")
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("vec must be at least 1-D")
    in_dim = int(arr.shape[-1]) if d_in is None else int(d_in)
    if in_dim < 1:
        raise ValueError("d_in must be >= 1")
    rng = np.random.default_rng(_seed_int(seed))
    W = rng.standard_normal(size=(in_dim, k * d_model), dtype=np.float32) * (1.0 / np.sqrt(in_dim))
    b = np.zeros(k * d_model, dtype=np.float32)
    out = arr.reshape(-1, in_dim) @ W + b  # (N, k*d_model)
    out = out.reshape(*arr.shape[:-1], k, d_model)
    return np.ascontiguousarray(out, dtype=np.float32)


if _TORCH:

    class SlotProjector(nn.Module):  # type: ignore[misc]
        """Trainable slot projection (torch). Weights seeded for reproducibility."""

        def __init__(self, d_in: int, d_model: int = 1024, k: int = 1, seed: str = "hd-laya-slots-v1"):
            super().__init__()
            if d_in < 1 or d_model < 1 or k < 1:
                raise ValueError("d_in, d_model, k must be >= 1")
            self.d_in = int(d_in)
            self.d_model = int(d_model)
            self.k = int(k)
            gen = torch.Generator()
            gen.manual_seed(_seed_int(seed))
            w = torch.randn(self.d_in, self.k * self.d_model, generator=gen) * (1.0 / (self.d_in ** 0.5))
            self.weight = nn.Parameter(w)
            self.bias = nn.Parameter(torch.zeros(self.k * self.d_model))

        def forward(self, x):
            # x: (..., d_in) -> (..., k, d_model)
            orig = x.shape[:-1]
            y = x.reshape(-1, self.d_in) @ self.weight + self.bias
            y = y.reshape(*orig, self.k, self.d_model)
            return y
