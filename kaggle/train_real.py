"""Kaggle real-training kernel for hd-laya v1.0 (Phase B).

Loads convaiinnovations/laya, runs laya's RLCD training on a synthetic
high-K dataset, saves the trained head checkpoint. Smoke mode (SMOKE=1) runs
1 epoch on 16 items (fast, for verification).
"""
from __future__ import annotations

import os
import sys

try:
    import hd_laya
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                   "git+https://github.com/sensesVoid/hd-laya.git@feat/v10-trained"], check=False)

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT_DIR = os.environ.get("OUT_DIR", "./out")
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "50" if SMOKE else "500"))
K = int(os.environ.get("K", "20" if SMOKE else "77"))
N_ITEMS = int(os.environ.get("N_ITEMS", "16" if SMOKE else "256"))
MAX_LEN = int(os.environ.get("MAX_LEN", "256" if SMOKE else "512"))
HEAD_MAX_LEN = int(os.environ.get("HEAD_MAX_LEN", "64" if SMOKE else "192"))

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_items(tok, n_items, K, seed=0):
    from hd_laya.data_gen import gen_dataset
    from laya.train import encode_state, make_item, to_internal, target_from_expected

    items_raw = gen_dataset(n_items, K, seed=seed)
    items = []
    for it in items_raw:
        q = to_internal("q", it["questions"]["q"])
        state_ids = encode_state(tok, it["state"], MAX_LEN)
        target = target_from_expected(q, it["expected"])
        item_dict, err = make_item(tok, q, target, state_ids, HEAD_MAX_LEN, max_len=MAX_LEN)
        if item_dict is not None:
            items.append(item_dict)
    print(f"built {len(items)}/{len(items_raw)} valid items (K={K})")
    return items


def run():
    import laya
    from laya.agent import Agent
    from laya.train import TrainConfig, train_model

    print(f"SMOKE={SMOKE} DEVICE={DEVICE} K={K} N={N_ITEMS} steps~={TRAIN_STEPS}")

    print("Loading laya model...")
    agent = Agent("convaiinnovations/laya", device=DEVICE)
    model = agent.model
    tok = agent.tok
    print("Model loaded.")

    items = build_items(tok, N_ITEMS, K, seed=0)
    if not items:
        raise RuntimeError("no valid training items")

    epochs = 1
    config = TrainConfig(
        epochs=epochs,
        micro_batch=2 if SMOKE else 4,
        grad_accum=1,
        encoder_lr=1e-5,
        head_lr=1e-4,
        min_lr=1e-6,
        weight_decay=0.01,
        grad_clip=1.0,
        loss="rlcd",
        rl_samples=2 if SMOKE else 4,
        sigma_start=1.0,
        sigma_end=0.5,
        w_sph=1.0,
        w_rps=1.0,
        shuffle_options=["choice"],
        option_layout="parallel",
        freeze_encoder=True,
        calib_max=10,
        calib_frac=0.1,
        calib_seed=0,
        seed=0,
        max_len=MAX_LEN,
        head_max_len=HEAD_MAX_LEN,
        amp=(DEVICE == "cuda"),
        gradient_checkpointing=(DEVICE == "cuda"),
        log_every=max(1, len(items) // 4),
        label_smoothing=0.0,
        text_column="state",
        label_column="answer",
        question_id="q",
        instructions="Choose the matching option.",
        eval_data=None,
        target_error=0.0,
        min_abstain_n=1,
    )

    print("Training...")
    losses = train_model(model, tok, items, config, torch.device(DEVICE),
                        max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
    print(f"Training done: {len(losses)} steps, final loss={losses[-1] if losses else 'n/a'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, "hd_laya_trained.pt")
    torch.save({"state_dict": model.state_dict(), "losses": losses, "K": K, "n_items": len(items)}, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    run()
