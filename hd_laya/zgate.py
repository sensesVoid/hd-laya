"""Z-score abstention gate plus streaming calibration for the hd-laya harness.

Port of the NTWD-1 z-gate (150/150 exact, 0 wrong) and probe 4's winning
score rule: with ``top`` the maximum score and ``rest`` the remaining
scores, ``z = (top - mean(rest)) / std(rest, ddof=0)``. A z at or above
the bar answers; below it, the caller abstains.

Caller contract for :class:`StreamingCalib` (adaptive bar): feed
``update(z)`` ONLY from answered (non-refused) decisions — never from
refused or ambiguous items. Feeding refused/ambiguous z's drags the
running mean/std toward the noise floor and defeats the calibration.
The adaptive bar may only RAISE the bar above its default, never lower
it: ``bar() = max(default, mean() + k * std())``.
"""
from __future__ import annotations

import math

import numpy as np

ANSWER = "answer"
REFUSE = "REFUSE"

DEFAULT_BAR = 5.0


def z_from_scores(scores: "np.ndarray | list[float]") -> float:
    """Separation of the top score from the rest: (top - mean(rest)) / std(rest).

    ``top`` is the maximum score (one instance, ties keep the other copies in
    ``rest``); ``rest`` is every other score. Descending input is NOT required.
    std uses ddof=0. Edge rules: fewer than 2 scores -> ``float("inf")``;
    std(rest) == 0 -> ``inf`` if top > mean(rest) else ``0.0``. Any non-finite
    input score raises ValueError.
    """
    arr = np.asarray(scores, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("scores must all be finite")
    if arr.size < 2:
        return float("inf")
    top_idx = int(np.argmax(arr))
    top = float(arr[top_idx])
    rest = np.delete(arr, top_idx)
    rest_mean = float(np.mean(rest))
    rest_std = float(np.std(rest, ddof=0))
    if rest_std == 0.0:
        return float("inf") if top > rest_mean else 0.0
    return (top - rest_mean) / rest_std


def decide(z: float, bar: float = DEFAULT_BAR) -> str:
    """ANSWER if z >= bar else REFUSE (NaN z compares False -> REFUSE)."""
    return ANSWER if z >= bar else REFUSE


class StreamingCalib:
    """O(1) Welford running mean/std (ddof=0) with an adaptive, ratcheted bar.

    See the module docstring for the caller contract: update() from answered
    decisions only.
    """

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0  # sum of squared deviations from the current mean

    def update(self, x: float) -> None:
        """Fold one value in with the Welford recurrence (O(1), exact pass)."""
        x = float(x)
        if not (x == x and x != float('inf') and x != float('-inf')):
            raise ValueError('StreamingCalib.update requires a finite value')
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)

    def count(self) -> int:
        """Number of values folded in so far."""
        return self._n

    def mean(self) -> float:
        """Running mean; 0.0 when empty."""
        return self._mean if self._n else 0.0

    def std(self) -> float:
        """Running population std (ddof=0); 0.0 when fewer than 2 values."""
        if self._n < 2:
            return 0.0
        # Cancellation can nudge M2 a hair below zero for near-constant input.
        return math.sqrt(max(self._m2, 0.0) / self._n)

    def bar(self, default: float = DEFAULT_BAR, k: float = 2.0) -> float:
        """max(default, mean + k*std): calibration may only raise the bar."""
        return max(default, self.mean() + k * self.std())
