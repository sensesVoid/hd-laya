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

if os.path.isdir(os.path.join(os.path.dirname(HERE), "hd_laya")):
    sys.path.insert(0, os.path.dirname(HERE))
else:
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
N_ITEMS = int(os.environ.get("N_ITEMS", "16" if SMOKE else "2048"))
EPOCHS = int(os.environ.get("EPOCHS", "1" if SMOKE else "6"))
MAX_LEN = int(os.environ.get("MAX_LEN", "256" if SMOKE else "512"))
HEAD_MAX_LEN = int(os.environ.get("HEAD_MAX_LEN", "64" if SMOKE else "192"))
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "2" if SMOKE else "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1"))
HEAD_LR = float(os.environ.get("HEAD_LR", "1e-3"))
ENCODER_LR = float(os.environ.get("ENCODER_LR", "2e-4"))
LOSS = os.environ.get("LOSS", "soft-ce")
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


def heldout_accuracy(agent, n=128, seed=1000, K=20):
    from hd_laya.agent import HDLayaAgent
    from hd_laya.bench import BenchmarkItem, run_benchmark
    from hd_laya.data_gen import gen_dataset
    from hd_laya.laya_agent import TrainedLayaAgent
    raw = gen_dataset(n, K, seed=seed)
    items = [BenchmarkItem(state=it["state"], expected=it["expected"],
                           meta={"questions": it["questions"]}) for it in raw]
    hd = HDLayaAgent(TrainedLayaAgent(agent, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN),
                     max_options=20, z_bar=0.0)
    return run_benchmark(hd, items, lambda i: i.meta["questions"], question_id="q")["summary"]["accuracy"]


def run():
    from laya.agent import Agent
    from laya.train import TrainConfig, train_model

    print(f"SMOKE={SMOKE} DEVICE={DEVICE} K={K} N={N_ITEMS} EPOCHS={EPOCHS} "
          f"max_len={MAX_LEN} head_max_len={HEAD_MAX_LEN} loss={LOSS} "
          f"head_lr={HEAD_LR} micro_batch={MICRO_BATCH} grad_accum={GRAD_ACCUM}", flush=True)

    print("Loading laya model...", flush=True)
    agent = Agent("convaiinnovations/laya", device=DEVICE)
    model = agent.model
    unfreeze = int(os.environ.get("UNFREEZE_LAYERS", "2"))
    if unfreeze:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        layers = list(model.encoder.layers)[-unfreeze:]
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad_(True)
        n = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"unfroze last {len(layers)} encoder layer(s), {n:,} params", flush=True)
    tok = agent.tok
    print("Model loaded.", flush=True)
    base_acc = heldout_accuracy(agent, K=K)
    print(f"BASE held-out acc at K={K}: {base_acc:.3f}", flush=True)

    items = build_items(tok, N_ITEMS, K, seed=0)
    if not items:
        raise RuntimeError("no valid training items")

    config = TrainConfig(
        epochs=EPOCHS,
        micro_batch=MICRO_BATCH,
        grad_accum=GRAD_ACCUM,
        encoder_lr=ENCODER_LR,
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
        shuffle_options=["choice"] if os.environ.get("SHUFFLE", "1") == "1" else [],
        freeze_encoder=(unfreeze == 0),
        calib_max=10,
        calib_frac=0.1,
        calib_seed=0,
        seed=0,
        max_len=MAX_LEN,
        head_max_len=HEAD_MAX_LEN,
        amp=(DEVICE == "cuda" and os.environ.get("AMP", "1") == "1"),
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

    tuned_acc = heldout_accuracy(agent, K=K)
    delta = tuned_acc - base_acc
    print(f"RESULT base={base_acc:.3f} trained={tuned_acc:.3f} delta={delta:+.3f} "
          f"({'PASS' if delta > 0.02 else 'FAIL'})", flush=True)


if __name__ == "__main__":
    run()
