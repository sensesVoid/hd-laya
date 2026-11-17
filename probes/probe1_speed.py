"""Probe 1 - speed sweep: encode + scan throughput vs D, plus Clay counter ops.

Frozen bars (prereg): B1 truncated-k scan within 5% of plain-k;
full-D scan ~2x prefix-1024 within 10%; B4 encode at 2048 ~2x 1024 within 10%.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import BSCEncoder, build_labels, build_spans, scan_time  # noqa: E402

SPEED_DIMS = [512, 1024, 2048, 4096, 8192, 16384, 65536]
N_OPTIONS = 10_000
N_SCALE = 100_000
SCALE_DIMS = [512, 1024, 2048, 4096]
OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)


def time_encode(dims: int, texts: list[str]) -> float:
    """Median encode latency per text (seconds), warmed."""
    enc = BSCEncoder(dims=dims)
    for t in texts[:50]:
        enc.encode_bits(t)  # warm atom cache
    times = []
    for t in texts[:300]:
        t0 = time.perf_counter()
        enc.encode_bits(t)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    rng = np.random.default_rng(0)
    labels = build_labels(N_OPTIONS, seed=11)
    spans = build_spans(2_000)
    print(f"labels={len(labels)} spans={len(spans)}", flush=True)

    rows = []
    for dims in SPEED_DIMS:
        enc = BSCEncoder(dims=dims)
        t0 = time.perf_counter()
        mat = enc.encode_batch(labels)
        enc_s = time.perf_counter() - t0
        med, p95 = scan_time(mat, reps=20, seed=dims)
        row = {
            "dims": dims,
            "n": N_OPTIONS,
            "encode_ms_per_text": time_encode(dims, labels) * 1e3,
            "encode_texts_per_s": N_OPTIONS / enc_s,
            "scan_ms_10k": med * 1e3,
            "scan_p95_ms_10k": p95 * 1e3,
            "scan_opts_per_s": N_OPTIONS / med,
            "bytes_per_vec": dims // 8,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    scale_rows = []
    for dims in SCALE_DIMS:
        enc = BSCEncoder(dims=dims)
        idx = rng.integers(0, len(labels), size=N_SCALE)
        big = [labels[i] for i in idx]
        t0 = time.perf_counter()
        mat = enc.encode_batch(big)
        enc_s = time.perf_counter() - t0
        med, _ = scan_time(mat, reps=10, seed=dims)
        row = {
            "dims": dims,
            "n": N_SCALE,
            "encode_s_total": enc_s,
            "scan_ms_100k": med * 1e3,
            "scan_opts_per_s": N_SCALE / med,
            "matrix_mb": mat.nbytes / 1e6,
        }
        scale_rows.append(row)
        print(json.dumps(row), flush=True)

    # --- Clay-style int16 counter workspace ops (Herald clay.md, faithful)
    clay_rows = []
    for dims in [16384, 65536]:
        M = np.zeros(dims, dtype=np.int16)
        key = (np.random.default_rng(1).integers(0, 2, dims, dtype=np.int8) * 2 - 1)
        role = (np.random.default_rng(2).integers(0, 2, dims, dtype=np.int8) * 2 - 1)
        filler = (np.random.default_rng(3).integers(0, 2, dims, dtype=np.int8) * 2 - 1)
        shape = (role.astype(np.int32) * filler.astype(np.int32)).astype(np.int16)

        def t_write():
            t0 = time.perf_counter()
            np.add(M, (key.astype(np.int16) * shape), out=M)
            return time.perf_counter() - t0

        def t_activate():
            t0 = time.perf_counter()
            x = M.astype(np.int32) * key.astype(np.int32)
            return time.perf_counter() - t0, x

        codebook = np.stack(
            [
                (np.random.default_rng(10_000 + j).integers(0, 2, dims, dtype=np.int8) * 2 - 1)
                for j in range(1024)
            ]
        ).astype(np.float32)

        def t_decode():
            _, x = t_activate()
            t0 = time.perf_counter()
            _ = (x.astype(np.float32) @ codebook.T) / dims
            return time.perf_counter() - t0

        for _ in range(3):
            t_write()
        w = sorted(t_write() for _ in range(20))[10]
        for _ in range(3):
            t_activate()
        a = sorted(t_activate()[0] for _ in range(20))[10]
        for _ in range(3):
            t_decode()
        d = sorted(t_decode() for _ in range(20))[10]
        row = {
            "dims": dims,
            "write_ms": w * 1e3,
            "activate_ms": a * 1e3,
            "decode_8roles_ms": d * 1e3 * 8,
            "workspace_kb": dims * 2 / 1024,
            "codebook_mb": codebook.nbytes / 1e6,
        }
        clay_rows.append(row)
        print(json.dumps(row), flush=True)

    result = {"probe": "1-speed", "rows": rows, "scale_rows": scale_rows, "clay_rows": clay_rows}
    (OUT / "probe1_speed.json").write_text(json.dumps(result, indent=1))
    print("WROTE", OUT / "probe1_speed.json")


if __name__ == "__main__":
    main()
