"""Kaggle real-training kernel for hd-laya v1.0 (Phase B).

Loads convaiinnovations/laya, runs laya's RLCD training on the well-posed
high-K choice generator, and saves the trained head.

Defaults are the *gate* run: K=20 (the cardinality the model actually infers at
after HDC narrowing), 512 items, 2 epochs. It prints a GATE verdict comparing
the final loss with ln(K); a loss stuck at ln(K) is the signature of a task made
unlearnable by the option budget, and the earlier K=77 run sat there for four
epochs (3.89/4.42/4.05/3.96 against ln 77 = 4.34).

Scale up only after the gate passes:
    N_ITEMS=4096 EPOCHS=4 OUT_DIR=./out
"""
from __future__ import annotations

import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF = "8f2b86a"  # rewritten by kaggle/sync_bundles.py before a push
PKG_REF = os.environ.get("HD_LAYA_REF", DEFAULT_REF)
LAYA_PIN = os.environ.get("LAYA_PIN", "0.4.35")

if not os.path.isdir(os.path.join(os.path.dirname(HERE), "hd_laya")):
    import importlib.util

    if importlib.util.find_spec("hd_laya") is None:
        print(f"installing laya {LAYA_PIN} and hd_laya {PKG_REF}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"laya=={LAYA_PIN}"],
                       check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        f"git+https://github.com/sensesVoid/hd-laya.git@{PKG_REF}"], check=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT_DIR = os.environ.get("OUT_DIR", "./out")
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "50" if SMOKE else "500"))
K = int(os.environ.get("K", "20"))
N_ITEMS = int(os.environ.get("N_ITEMS", "16" if SMOKE else "512"))
EPOCHS = int(os.environ.get("EPOCHS", "1" if SMOKE else "2"))
MAX_LEN = int(os.environ.get("MAX_LEN", "256" if SMOKE else "512"))
HEAD_MAX_LEN = int(os.environ.get("HEAD_MAX_LEN", "64" if SMOKE else "192"))
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "2" if SMOKE else "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1"))
HEAD_LR = float(os.environ.get("HEAD_LR", "1e-4"))
LOSS = os.environ.get("LOSS", "rlcd")
RL_SAMPLES = int(os.environ.get("RL_SAMPLES", "2" if SMOKE else "4"))

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_items(tok, n_items, K, seed=0):
    from hd_laya.data_gen import gen_dataset
    from laya.train import encode_state, make_item, target_from_expected, to_internal

    items_raw = gen_dataset(n_items, K, seed=seed)
    items = []
    for it in items_raw:
        q = to_internal("q", it["questions"]["q"])
        state_ids = encode_state(tok, it["state"], MAX_LEN)
        target = target_from_expected(q, it["expected"])
        item_dict, err = make_item(tok, q, target, state_ids, HEAD_MAX_LEN, max_len=MAX_LEN)
        if item_dict is not None:
            items.append(item_dict)
    print(f"built {len(items)}/{len(items_raw)} valid items (K={K})", flush=True)
    return items


def run():
    from laya.agent import Agent
    from laya.train import TrainConfig, train_model

    print(f"SMOKE={SMOKE} DEVICE={DEVICE} K={K} N={N_ITEMS} EPOCHS={EPOCHS} "
          f"max_len={MAX_LEN} head_max_len={HEAD_MAX_LEN} loss={LOSS} "
          f"head_lr={HEAD_LR} micro_batch={MICRO_BATCH} grad_accum={GRAD_ACCUM}", flush=True)

    print("Loading laya model...", flush=True)
    agent = Agent("convaiinnovations/laya", device=DEVICE)
    model = agent.model
    tok = agent.tok
    print("Model loaded.", flush=True)

    items = build_items(tok, N_ITEMS, K, seed=0)
    if not items:
        raise RuntimeError("no valid training items")

    config = TrainConfig(
        epochs=EPOCHS,
        micro_batch=MICRO_BATCH,
        grad_accum=GRAD_ACCUM,
        encoder_lr=1e-5,
        head_lr=HEAD_LR,
        min_lr=1e-6,
        weight_decay=0.01,
        grad_clip=1.0,
        loss=LOSS,
        rl_samples=RL_SAMPLES,
        sigma_start=1.0,
        sigma_end=0.5,
        w_sph=1.0,
        w_rps=1.0,
        shuffle_options=["choice"],
        freeze_encoder=True,
        calib_max=10,
        calib_frac=0.1,
        calib_seed=0,
        seed=0,
        max_len=MAX_LEN,
        head_max_len=HEAD_MAX_LEN,
        amp=(DEVICE == "cuda"),
        gradient_checkpointing=(DEVICE == "cuda"),
        log_every=max(1, len(items) // 8),
        label_smoothing=0.0,
        text_column="state",
        label_column="answer",
        question_id="q",
        instructions="Choose the matching option.",
        eval_data=None,
        target_error=0.0,
        min_abstain_n=1,
    )

    print("Training...", flush=True)
    losses = train_model(model, tok, items, config, torch.device(DEVICE),
                         max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
    print(f"Training done: {len(losses)} steps, losses={losses}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    trainable_sd = {k: v for k, v in model.state_dict().items()
                    if not k.startswith("encoder.")}
    ckpt_path = os.path.join(OUT_DIR, "hd_laya_trained.pt")
    torch.save({"state_dict": trainable_sd, "losses": losses, "K": K,
                "n_items": len(items), "epochs": EPOCHS}, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}", flush=True)

    if losses:
        floor = math.log(K)
        final = losses[-1]
        verdict = "PASS" if final < floor else "FAIL"
        print(f"GATE {verdict}: final_loss={final:.4f} vs ln(K)={floor:.4f} "
              f"(margin={floor - final:+.4f})", flush=True)


if __name__ == "__main__":
    run()
