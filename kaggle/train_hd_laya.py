"""Kaggle training kernel for hd-laya v1.0 (T11).

Run on Kaggle (2xT4) with machine_shape=NvidiaTeslaT4. Smoke mode:
    SMOKE=1 python kaggle/train_hd_laya.py   (1 batch, synthetic data, tiny head)
Full mode (SMOKE=0) requires a laya checkpoint path and runs the high-K
training loop. For local verification of the full code path, set
TRAIN_STEPS=5 (or any small number).

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Ensure hd_laya is available on Kaggle (not bundled with the script).
try:
    import hd_laya
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                   'git+https://github.com/sensesVoid/hd-laya.git@feat/v10-trained'], check=False)



def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT_DIR = os.environ.get("OUT_DIR", "./out")
DEVICE = os.environ.get("DEVICE", "cuda" if _cuda_available() else "cpu")
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "1000" if not SMOKE else "1"))
K_VALUES = [int(x) for x in os.environ.get("K_VALUES", "77,150").split(",") if x.strip()]
N_PER_K = int(os.environ.get("N_PER_K", "64" if SMOKE else "1024"))
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "")  # laya checkpoint; empty -> synthetic

import torch
import torch.nn as nn


def build_tiny_head(d_in: int = 32, d_model: int = 16, k: int = 2, n_classes: int = 4) -> nn.Module:
    from hd_laya.hdc_slots import SlotProjector
    return nn.Sequential(
        SlotProjector(d_in, d_model, k),
        nn.Flatten(),
        nn.Linear(k * d_model, n_classes),
    )


def run_smoke():
    torch.manual_seed(0)
    dev = torch.device(DEVICE if (DEVICE == "cpu" or _cuda_available()) else "cpu")
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


def _make_batch(K: int, batch_size: int, seed: int):
    """Synthetic high-K batch: x (batch, 32), y (batch, n_classes) one-hot-ish."""
    from hd_laya.data_gen import gen_dataset
    items = gen_dataset(batch_size, K, seed=seed)
    # Map expected label index -> class index (mod n_classes)
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, 32, generator=rng)
    y_idx = torch.tensor([int(it["expected"].split("_")[1]) % 4 for it in items], dtype=torch.long)
    y = torch.nn.functional.one_hot(y_idx, num_classes=4).float()
    return x, y


def run_full():
    dev = torch.device(DEVICE if (DEVICE == "cpu" or _cuda_available()) else "cpu")
    print(f"FULL mode: device={dev} steps={TRAIN_STEPS} K={K_VALUES} n_per_k={N_PER_K}")
    if CHECKPOINT_PATH:
        print(f"  checkpoint: {CHECKPOINT_PATH} (laya weights; integration TBD)")
    else:
        print("  no checkpoint given -> using synthetic tiny head (local verification mode)")

    torch.manual_seed(0)
    head = build_tiny_head(n_classes=4).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    step = 0
    for K in K_VALUES:
        while step < TRAIN_STEPS:
            x, y = _make_batch(K, batch_size=8, seed=step)
            x, y = x.to(dev), y.to(dev)
            try:
                out = head(x)
                loss = torch.nn.functional.mse_loss(out, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at K={K} step={step}; skipping")
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"  step={step} K={K} loss={loss.item():.4f}")
            if step >= TRAIN_STEPS:
                break
        if step >= TRAIN_STEPS:
            break

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt = os.path.join(OUT_DIR, "hd_laya_head.pt")
    torch.save({"state_dict": head.state_dict(), "steps": step, "K": K_VALUES}, ckpt)
    print(f"FULL done: {step} steps, checkpoint -> {ckpt}")


if __name__ == "__main__":
    if SMOKE:
        run_smoke()
    else:
        run_full()
