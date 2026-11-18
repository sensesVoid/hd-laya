"""Kaggle benchmark kernel for hd-laya v1.0 (high-K choice suite).

Runs K in {20,77,150,500,1000} through the full hd-laya path -- HDC two-stage
narrowing to max_options=20, laya forward on the survivors, z-gate abstention --
using the trained head checkpoint, and writes per-K JSON to the kernel output.

The checkpoint is located by globbing /kaggle/input (mount layouts differ), then
by an explicit path list, then from the Hugging Face repo. If none works the
kernel stops with the input tree in the log instead of quietly benchmarking an
untrained agent.

Smoke mode: SMOKE=1 -> K=20 only, 16 items.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF = "c2ec6f3"  # pinned by sync_bundles.py
PKG_REF = os.environ.get("HD_LAYA_REF", DEFAULT_REF)
LAYA_PIN = os.environ.get("LAYA_PIN", "0.4.35")


def _local_repo(start: str):
    """Nearest ancestor that contains the package (checkout layout), else None."""
    path = start
    for _ in range(4):
        if os.path.isdir(os.path.join(path, "hd_laya")):
            return path
        path = os.path.dirname(path)
    return None


repo = _local_repo(HERE)
if repo:
    # Prefer the checkout under test; keeps `python kaggle/bench_hd_laya.py` a
    # faithful smoke run of exactly what gets pushed.
    sys.path.insert(0, repo)
else:
    # Kaggle uploads only this file, so the package comes from pip there.
    import importlib.util

    if importlib.util.find_spec("hd_laya") is None:
        # Pin laya first: the head weights match the architecture it shipped with,
        # and `laya>=0.4.35` alone would float to whatever PyPI has today.
        print(f"installing laya {LAYA_PIN} and hd_laya {PKG_REF} from GitHub", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"laya=={LAYA_PIN}"],
                       check=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             f"git+https://github.com/sensesVoid/hd-laya.git@{PKG_REF}"],
            check=True,
        )

try:
    from hd_laya.laya_agent import TrainedLayaAgent  # noqa: E402
except ImportError as exc:  # pragma: no cover - stale install
    raise SystemExit(
        f"hd_laya has no laya_agent ({exc}); expected ref {PKG_REF}. "
        "Install it or unset the stale copy."
    )

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT_DIR = os.environ.get("OUT_DIR", "./out")
K_VALUES = [20, 77, 150, 500, 1000] if not SMOKE else [20]
N_PER_K = 16 if SMOKE else int(os.environ.get("N_PER_K", "256"))
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
HEAD_MAX_LEN = int(os.environ.get("HEAD_MAX_LEN", "192"))
MAX_OPTIONS = int(os.environ.get("MAX_OPTIONS", "20"))
Z_BAR = float(os.environ.get("Z_BAR", "5.0"))
CKPT_NAME = "hd_laya_trained.pt"
HF_REPO = os.environ.get("HF_REPO", "geraldaton20/hd-laya")
CKPT_PATH_ENV = os.environ.get("CHECKPOINT_PATH", "")

EXPLICIT_PATHS = (
    "./checkpoint/" + CKPT_NAME,
    "/kaggle/input/" + CKPT_NAME,
    "/kaggle/input/hd-laya-trained-checkpoint-v2/" + CKPT_NAME,
    "/kaggle/input/datasets/sensesvoid/hd-laya-trained-checkpoint-v2/" + CKPT_NAME,
)


def print_input_tree(root="/kaggle/input", depth=3):
    """Diagnostics: what the mount actually contains when discovery fails."""
    print("=== %s ===" % root, flush=True)
    if not os.path.isdir(root):
        print("  (missing)", flush=True)
        return
    base_depth = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.count("/") - base_depth
        if level >= depth:
            dirnames[:] = []
        pad = "  " * (level + 1)
        print("%s%s/" % (pad, os.path.basename(dirpath) or dirpath), flush=True)
        for name in sorted(filenames)[:12]:
            try:
                size = os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                size = -1
            print("%s  %s (%d bytes)" % (pad, name, size), flush=True)
    print("=== end tree ===", flush=True)


def find_checkpoint():
    """Return (path, source) or (None, None). Every route is logged."""
    if CKPT_PATH_ENV:
        if os.path.exists(CKPT_PATH_ENV):
            return CKPT_PATH_ENV, "env"
        print("CHECKPOINT_PATH set but missing: %s" % CKPT_PATH_ENV, flush=True)

    for path in glob.glob("/kaggle/input/**/%s" % CKPT_NAME, recursive=True):
        return path, "mount-glob"

    for path in EXPLICIT_PATHS:
        if os.path.exists(path):
            return path, "explicit"

    print("checkpoint not mounted; trying huggingface.co/%s ..." % HF_REPO, flush=True)
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(HF_REPO, CKPT_NAME, cache_dir="./hf_cache")
        return path, "hf"
    except Exception as exc:  # network, auth, missing file
        print("HF download failed: %s: %s" % (type(exc).__name__, exc), flush=True)

    return None, None


def expected_calibration_error(rows, bins=10):
    """Top-label ECE over the confidences the agent reported."""
    pairs = [(float(r["confidence"]), 1.0 if r["correct"] else 0.0)
             for r in rows if r.get("confidence") is not None]
    if not pairs:
        return None
    total = len(pairs)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, (conf, _) in enumerate(pairs)
               if (conf >= lo and conf < hi) or (b == bins - 1 and conf == 1.0)]
        if not idx:
            continue
        mean_conf = sum(pairs[i][0] for i in idx) / len(idx)
        acc = sum(pairs[i][1] for i in idx) / len(idx)
        ece += (len(idx) / total) * abs(acc - mean_conf)
    return round(ece, 4)


def main() -> int:
    import torch

    from hd_laya.agent import HDLayaAgent
    from hd_laya.bench import BenchmarkItem, dump_json, run_benchmark
    from hd_laya.data_gen import gen_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} smoke={SMOKE} K={K_VALUES} n={N_PER_K} "
          f"max_len={MAX_LEN} head_max_len={HEAD_MAX_LEN}", flush=True)

    ckpt_path, source = find_checkpoint()
    if ckpt_path is None:
        print("FATAL: no checkpoint found (source unavailable)", flush=True)
        print_input_tree()
        return 2
    print(f"checkpoint: {ckpt_path} (source={source}, "
          f"{os.path.getsize(ckpt_path) / 1e6:.1f} MB)", flush=True)

    print("loading laya ...", flush=True)
    from laya.agent import Agent

    laya_agent = Agent("convaiinnovations/laya", device=device)
    print("applying trained head ...", flush=True)
    trained = TrainedLayaAgent.from_checkpoint(
        ckpt_path,
        agent=laya_agent,
        max_len=MAX_LEN,
        head_max_len=HEAD_MAX_LEN,
    )
    n_head = sum(p.numel() for name, p in laya_agent.model.named_parameters()
                 if not name.startswith("encoder."))
    print(f"head params: {n_head:,}", flush=True)

    agent = HDLayaAgent(trained, max_options=MAX_OPTIONS, z_bar=Z_BAR)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

    def flush_summary():
        with open(os.path.join(OUT_DIR, "bench_summary.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    for K in K_VALUES:
        raw = gen_dataset(N_PER_K, K, seed=K)
        items = [BenchmarkItem(state=it["state"], expected=it["expected"],
                               meta={"K": K, "questions": it["questions"]}) for it in raw]

        t0 = time.perf_counter()
        done = 0
        rows = []
        # Run in one call but report progress by chunking, so a cancelled run
        # still shows how far it got.
        chunk = max(1, min(32, N_PER_K))
        for start in range(0, len(items), chunk):
            part = run_benchmark(agent, items[start:start + chunk],
                                 lambda item: item.meta["questions"], question_id="q")
            rows.extend(part["rows"])
            done += part["summary"]["n"]
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"  K={K} {done}/{len(items)} ({rate:.1f} items/s)",
                  flush=True)
        elapsed = time.perf_counter() - t0

        n = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        refused = sum(1 for r in rows if r["refused"])
        answered = n - refused
        narrowed = sum(1 for r in rows
                       if ((r.get("agent_meta") or {}).get("hd_index", {}).get("q") or {}).get("narrowed"))
        summary = {
            "K": K,
            "n": n,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "refusal_rate": round(refused / n, 4) if n else 0.0,
            "accuracy_on_answered": round(
                sum(1 for r in rows if r["correct"] and not r["refused"]) / answered, 4
            ) if answered else 0.0,
            "narrowed_items": narrowed,
            "ece": expected_calibration_error(rows),
            "elapsed_s": round(elapsed, 2),
            "items_per_s": round(n / elapsed, 2) if elapsed > 0 else 0.0,
        }
        all_results[f"K={K}"] = summary
        print(json.dumps(summary), flush=True)
        dump_json(os.path.join(OUT_DIR, f"bench_K{K}.json"), {"rows": rows, "summary": summary})
        flush_summary()

    print("ALL DONE", flush=True)
    print(json.dumps(all_results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
