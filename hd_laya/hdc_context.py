"""HDC context (hypertoken) projection for hd-laya v1.0 (T10).

Projects the decoded superposition of the int16 counter workspace (a
D-dimensional float vector) into ``k`` pseudo-tokens in the encoder's model
dimension. This is the mechanism that keeps the encoder's input bounded by
design: no matter how large the context, the encoder sees at most ``k``
hypertokens instead of thousands of raw tokens.

Reuses :func:`hd_laya.hdc_slots.project_slots` (same seeded linear map) with
a context-specific default seed, so the two projections stay independent but
share one implementation (DRY).
"""
from __future__ import annotations

import numpy as np

from hd_laya.hdc_slots import project_slots

__all__ = ["project_context", "ContextProjector"]

# Re-export the torch module under a context-specific alias for clarity.
try:
    import torch
    import torch.nn as nn
    from hd_laya.hdc_slots import SlotProjector as _SlotProjector

    class ContextProjector(_SlotProjector):  # type: ignore[misc]
        """Trainable context projection (torch). Alias of SlotProjector."""
        pass

    _TORCH = True
except Exception:  # pragma: no cover
    ContextProjector = None  # type: ignore
    _TORCH = False

DEFAULT_CONTEXT_SEED = "hd-laya-context-v1"


def project_context(
    vec: "np.ndarray",
    k: int = 8,
    d_model: int = 1024,
    seed: str = DEFAULT_CONTEXT_SEED,
    d_in: "int | None" = None,
) -> np.ndarray:
    """Deterministic seeded projection of a workspace superposition into hypertokens."""
    return project_slots(vec, k=k, d_model=d_model, seed=seed, d_in=d_in)
