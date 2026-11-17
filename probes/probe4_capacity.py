"""Probe 4 - superposition capacity (bundling) + verdict on the 64K question.

Frozen bar (prereg B5): measure cleanup error vs (D, n) and compare
qualitatively to the Clarkson et al. capacity law D >= 2n*ln(1/eps).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import BSCEncoder  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

N_GRID = [32, 64, 128, 256, 512, 1024]
D_GRID = [1024, 2048, 4096, 8192, 16384]
V = 64          # value codebook size
SEEDS = [0, 1, 2]


def run_bundling(dims: int, n: int, seed: int) -> float:
    """Bundle n key(xor)value pairs, cleanup each key against V-atom codebook."""
    rng = np.random.default_rng(seed * 1_000 + dims)
    enc = BSCEncoder(dims=dims, seed=f"cap-{seed}")

    def atom(name: str) -> np.ndarray:
        return np.unpackbits(
            np.frombuffer(enc.atom(name), dtype=np.uint8), bitorder="big"
        ).astype(np.uint8)

    keys = [atom(f"key-{i}") for i in range(n)]
    vals = [atom(f"val-{int(j)}") for j in rng.integers(0, V, size=n)]

    pairs = [k ^ v for k, v in zip(keys, vals)]
    bits = np.stack(pairs)                       # (n, D)
    bundle = (bits.sum(axis=0) > n / 2).astype(np.uint8)

    codebook = np.stack([atom(f"val-{j}") for j in range(V)])  # (V, D)
    wrong = 0
    for i in range(n):
        noisy = keys[i] ^ bundle                 # unbind
        agree = (codebook != noisy).sum(axis=1)  # hamming to each value
        pick = int(np.argmin(agree))
        true_val = int(np.where((codebook == vals[i]).all(axis=1))[0][0])
        wrong += int(pick != true_val)
    return wrong / n


def main() -> None:
    rows = []
    for dims in D_GRID:
        for n in N_GRID:
            errs = [run_bundling(dims, n, s) for s in SEEDS]
            eps_pred = float(np.exp(-dims / (2.0 * n)))  # Clarkson-style prediction
            row = {
                "dims": dims,
                "n": n,
                "mean_error": float(np.mean(errs)),
                "max_error": float(np.max(errs)),
                "predicted_eps": eps_pred,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    summary = {}
    for dims in D_GRID:
        r = [x for x in rows if x["dims"] == dims]
        clean = [x for x in r if x["mean_error"] <= 0.001]
        summary[f"D{dims}"] = {
            "max_n_error_free": max((x["n"] for x in clean), default=0),
            "n_at_1pct": max((x["n"] for x in r if x["mean_error"] <= 0.01), default=0),
        }
    result = {"probe": "4-capacity", "rows": rows, "summary": summary}
    (OUT / "probe4_capacity.json").write_text(json.dumps(result, indent=1))
    print("WROTE", OUT / "probe4_capacity.json")


if __name__ == "__main__":
    main()
