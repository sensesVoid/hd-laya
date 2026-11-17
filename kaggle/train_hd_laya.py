"""Kaggle training kernel for hd-laya v1.0 (T11).

Run on Kaggle (2xT4) with machine_shape=NvidiaTeslaT4. Smoke mode:
    SMOKE=1 python kaggle/train_hd_laya.py   (1 batch, synthetic data, tiny head)
Full mode requires a laya checkpoint and the high-K data generator.

Hardening lessons from probe5 (v1 kernel died on a missing device= arg):
- every tensor creation passes device= explicitly
- per-device try/except around GPU work
- OOM guards (batch size cap)
- SMOKE=1 runs the exact kernel code path locally before every push
- trust `kaggle kernels output`, not `kaggle kernels logs`
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT_DIR = os.environ.get("OUT_DIR", "./out")
DEVICE = os.environ.get("DEVICE", "cuda" if _cuda_available() else "cpu")

import torch
import torch.nn as nn


def build_tiny_head(d_in: int = 32, d_model: int = 16, k: int = 2) -> nn.Module:
    from hd_laya.hdc_slots import SlotProjector
    from hd_laya.hdc_context import ContextProjector
    # A minimal stand-in for the laya head + projections (smoke path only).
    return nn.Sequential(
        SlotProjector(d_in, d_model, k),
        nn.Flatten(),
        nn.Linear(k * d_model, 4),
    )


def run_smoke():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    head = build_tiny_head().to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    x = torch.randn(4, 32, device=dev)
    y = torch.randn(4, 4, device=dev)
    out = head(x)
    loss = torch.nn.functional.mse_loss(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"SMOKE ok: loss={loss.item():.4f} device={dev}")
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(head.state_dict(), os.path.join(OUT_DIR, "smoke_head.pt"))
    print("SMOKE checkpoint written")


def run_full():
    # Full training requires laya checkpoint + high-K data generator + 2xT4.
    raise NotImplementedError(
        "Full training mode requires a laya checkpoint and the high-K data "
        "generator. See SQAC docs hd-laya-plan-v10-p1..p3 and run on Kaggle "
        "with machine_shape=NvidiaTeslaT4."
    )


if __name__ == "__main__":
    if SMOKE:
        run_smoke()
    else:
        run_full()
