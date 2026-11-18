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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    from laya.agent import Agent
    from hd_laya.refine_loop import RefineLoop
    from hd_laya.zgate import StreamingCalib
    from hd_laya.workspace import Workspace

    agent = Agent("convaiinnovations/laya", device="cpu")
    model = agent.model
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"loaded checkpoint: {len(ckpt['state_dict'])} keys, missing={len(missing)}, unexpected={len(unexpected)}")

    # Use the real laya head (TransformerEncoder) + scorer + act_head + type_emb + temperature
    head = model.head
    head.eval()

    # For RefineLoop, we need a module that takes (batch, d_in) -> (batch, n_classes)
    # The laya head takes (batch, seq, 1024) + mask. Wrap it to flatten the input.
    # We'll use the first token's output as the "pooled" representation.
    class RefineHead(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head
            # Projection from head output (1024) to n_classes (4)
            self.proj = torch.nn.Linear(1024, 4)

        def forward(self, x):
            # x: (batch, 32) -> expand to (batch, 1, 1024) if needed, or use a different approach
            # Actually, the real head expects (batch, seq, 1024). 
            # We need to project x from 32 -> 1024 first, then add a seq dimension.
            pass

    # Simpler: use the scorer (which outputs per-marker scores) as the refinement target
    # The scorer: Sequential(LN, Linear(1024, 256), GELU, Linear(256, 1)) -> (batch, seq, 1)
    scorer = model.scorer
    scorer.eval()

    # RefineLoop expects a head that maps (batch, d_in) -> (batch, n_classes)
    # Let's create a compatible head: project 32 -> 1024, run through scorer on a single token
    class CompatHead(torch.nn.Module):
        def __init__(self, scorer, d_in=32, n_classes=4):
            super().__init__()
            self.proj = torch.nn.Linear(d_in, 1024)
            self.scorer = scorer
            self.n_classes = n_classes

        def forward(self, x):
            # x: (batch, d_in) -> (batch, 1024) -> (batch, 1, 1024) -> scorer -> (batch, 1) -> expand to n_classes
            h = self.proj(x).unsqueeze(1)  # (batch, 1, 1024)
            scores = self.scorer(h).squeeze(-1)  # (batch, 1)
            # Expand single score to n_classes (for demo; real would be per-option)
            return scores.repeat(1, self.n_classes)

    compat_head = CompatHead(scorer)
    compat_head.eval()

    opt = torch.optim.Adam(compat_head.parameters(), lr=1e-4)
    rl = RefineLoop(compat_head, opt)

    x = torch.randn(8, 32)
    y = torch.randn(8, 4)
    # tier 1 + 2 (calib + memory) + tier 3 (head step)
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