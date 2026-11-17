"""Int16 counter workspace for hd_laya: exact bundling/undo + z-gated decode.

Clay-style integer counters, separate from hv.py's packed-BSC path:
- buffers are int16[dims] counters, zero-filled on first touch;
- write adds counter_atom(role) * counter_atom(filler) bounds, erase
  subtracts them — integer-exact, so write+erase is a byte-identical undo;
- the max_mag guard refuses (OverflowError, whole call applies nothing)
  any pair that would push a counter past max_mag in magnitude;
- measured law (probe 4 + Herald Clay): capacity = D/64 (role, filler)
  pairs exact-or-refuse — D=65536 decodes 1024 pairs against a V=64
  codebook with 0 wrong across seeds. Decode unbinds per role
  (buf * counter_atom(role) as float64), scores against the codebook
  /dims -> [-1, 1], and accepts the top value only if
  z = (top - mean rest) / std(rest, ddof=0) >= z_thr and
  margin = top - second >= tau; otherwise it abstains with (None, top, z).
"""
from __future__ import annotations

import hashlib

import numpy as np

from hd_laya.hv import DEFAULT_SEED

__all__ = ["Workspace", "counter_atom"]

# Decode rows per float64 batch: (128, 65536) float64 chunk is 64 MB.
_CHUNK = 128


def counter_atom(seed: str, token: str, dims: int) -> np.ndarray:
    """Seeded +/-1 counter atom, int16[dims].

    Chains sha256(f"{seed}:{token}:{counter}") digests (counter 0, 1, ...)
    until dims//8 bytes are filled, then maps the bits MSB-first:
    bit 1 -> +1, bit 0 -> -1. Same chain as hv.atom, so a shorter atom is
    the prefix of a longer one with the same seed and token.
    """
    if dims % 64 != 0:
        raise ValueError("dims must be a multiple of 64 (decode blocks need it)")
    need = dims // 8
    raw = bytearray()
    counter = 0
    while len(raw) < need:
        raw.extend(hashlib.sha256(f"{seed}:{token}:{counter}".encode()).digest())
        counter += 1
    bits = np.unpackbits(
        np.frombuffer(bytes(raw[:need]), dtype=np.uint8), bitorder="big"
    )
    return 2 * bits[:dims].astype(np.int16) - 1


class Workspace:
    """dict[key -> int16[dims]] counter buffers with exact undo and z-gated decode."""

    def __init__(self, dims: int = 65536, seed: str = DEFAULT_SEED, max_mag: int = 32767):
        if dims % 64 != 0:
            raise ValueError("dims must be a multiple of 64 (decode blocks need it)")
        if not 1 <= max_mag <= np.iinfo(np.int16).max:
            raise ValueError(
                f"max_mag must be in [1, {np.iinfo(np.int16).max}] for int16 counters, got {max_mag}"
            )
        self.dims = int(dims)
        self.seed = seed
        self.max_mag = int(max_mag)
        self._buffers: dict[str, np.ndarray] = {}

    def _bound(self, role: str, filler: str) -> np.ndarray:
        """role * filler binding, elementwise int16, entries +/-1."""
        return counter_atom(self.seed, role, self.dims) * counter_atom(
            self.seed, filler, self.dims
        )

    def _apply(self, key: str, roles: list[str], fillers: list[str], sign: int) -> None:
        """Add (sign=+1) or subtract (sign=-1) bounds atomically.

        Each pair is simulated in int32 first; if any counter would exceed
        max_mag in magnitude, OverflowError is raised and NOTHING is applied
        (the call works on a copy, so the stored buffer is never touched and
        newly created buffers are only registered on full success).
        """
        if len(roles) != len(fillers):
            raise ValueError("roles and fillers must have the same length")
        existing = self._buffers.get(key)
        work = existing.copy() if existing is not None else np.zeros(self.dims, dtype=np.int16)
        for role, filler in zip(roles, fillers):
            bound = self._bound(role, filler)
            cand = work.astype(np.int32)
            cand += bound if sign > 0 else -bound
            if int(np.abs(cand).max()) > self.max_mag:
                raise OverflowError(
                    f"counter would exceed max_mag={self.max_mag}: key={key!r} "
                    f"role={role!r} filler={filler!r}; call applied nothing"
                )
            work[:] = cand
        if existing is None:
            self._buffers[key] = work
        else:
            existing[:] = work

    def write(self, key: str, roles: list[str], fillers: list[str]) -> None:
        """Bundle each (role, filler) binding into key's buffer; atomic under max_mag."""
        self._apply(key, roles, fillers, +1)

    def erase(self, key: str, roles: list[str], fillers: list[str]) -> None:
        """Unbundle each (role, filler) binding from key's buffer; same atomic guard.

        Subtracting can exceed max_mag downward too (e.g. -32767 - 1), so the
        guard is identical to write's. A magnitude-decreasing undo at the
        boundary (erase of the pair that saturated the buffer) stays allowed.
        """
        self._apply(key, roles, fillers, -1)

    def activate(self, key: str) -> np.ndarray:
        """Copy of key's int16[dims] buffer (KeyError if the key was never touched)."""
        try:
            return self._buffers[key].copy()
        except KeyError:
            raise KeyError(f"no buffer for key {key!r} (never written)") from None

    def decode(
        self,
        key: str,
        roles: list[str],
        codebook: list[str],
        z_thr: float = 5.0,
        tau: float = 0.5,
    ) -> list[tuple[str | None, float, float]]:
        """Cleanup each role against the codebook, exact-or-refuse.

        Unbinds buf * counter_atom(role) as float64, scores against the
        (V, dims) float64 codebook matrix /dims -> [-1, 1] (batched matmul,
        exact up to 2^53), and returns (top token, top score, z) per role —
        top = max score, z = (top - mean rest) / std(rest, ddof=0). The token
        is returned only if z >= z_thr and margin = top - second >= tau,
        otherwise (None, top score, z). Empty roles -> empty list; KeyError
        on an unwritten key.
        """
        if key not in self._buffers:
            raise KeyError(f"no buffer for key {key!r} (never written)")
        if not roles:
            return []
        if not codebook:
            raise ValueError("codebook must be non-empty")
        buf = self._buffers[key]
        codebook_matrix = np.stack(
            [counter_atom(self.seed, token, self.dims) for token in codebook]
        ).astype(np.float64)  # (V, dims)
        cbt = np.ascontiguousarray(codebook_matrix.T)  # (dims, V)
        results: list[tuple[str | None, float, float]] = []
        for start in range(0, len(roles), _CHUNK):
            chunk = roles[start : start + _CHUNK]
            unbound = np.stack(
                [buf * counter_atom(self.seed, role, self.dims) for role in chunk]
            )
            scores = unbound.astype(np.float64) @ cbt / self.dims  # (c, V) in [-1, 1]
            scores = np.clip(scores, -1.0, 1.0)  # superposition cross-talk can push slightly past +/-1
            top_idx = scores.argmax(axis=1)
            for j in range(len(chunk)):
                row = scores[j]
                ti = int(top_idx[j])
                top = float(row[ti])
                rest = np.delete(row, ti)
                if rest.size == 0:
                    z = float("nan")
                    margin = float("inf")
                else:
                    mean_rest = float(rest.mean())
                    std_rest = float(rest.std(ddof=0))
                    if std_rest > 0.0:
                        z = (top - mean_rest) / std_rest
                    elif top > mean_rest:
                        z = float("inf")
                    elif top < mean_rest:
                        z = float("-inf")
                    else:
                        z = float("nan")
                    margin = top - float(rest.max())
                if z >= z_thr and margin >= tau:
                    results.append((codebook[ti], top, z))
                else:
                    results.append((None, top, z))
        return results
