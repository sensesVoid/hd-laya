"""Refine loop integration with a trained checkpoint (v1.0 T13 integration).

Loads a training checkpoint, rebuilds the head, wires it into RefineLoop, and
runs a gated step + latency measurement. Mirrors the realtime RLCD tier-3
path (23-63 ms/update measured on T4).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hd_laya.refine_loop import RefineLoop
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kaggle')); from train_hd_laya import build_tiny_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    head = build_tiny_head()
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    opt = torch.optim.Adam(head.parameters(), lr=1e-4)
    rl = RefineLoop(head, opt)

    x = torch.randn(8, 32)
    y = torch.randn(8, 4)
    # tier 1 + 2 (calib + memory) + tier 3 (head step)
    from hd_laya.zgate import StreamingCalib
    from hd_laya.workspace import Workspace
    ws = Workspace(dims=256)
    rl.memory_write(ws, "ctx", ["r0"], ["f0"])
    rl.calibrate(4.0)

    latencies = []
    for i in range(args.steps):
        t0 = time.perf_counter()
        res = rl.step(x, y)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        print(f"step {i+1}: loss {res.loss_before:.4f} -> {res.loss_after:.4f} stepped={res.stepped} ({latencies[-1]:.1f} ms)")

    print(f"latency: mean={sum(latencies)/len(latencies):.1f} ms  min={min(latencies):.1f}  max={max(latencies):.1f}")
    promoted = rl.promote()
    print(f"promote: {promoted}")
    print(f"replay buffer size: {len(rl._replay)}")
    print(f"calib: count={rl.calib.count()} mean={rl.calib.mean():.3f}")


if __name__ == "__main__":
    main()
