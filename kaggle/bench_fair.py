"""Fair comparison: base laya vs trained, BOTH on the same narrowed 20 options.

Runs each K with a FIXED seed so items are comparable across K.
Reports: base-on-20 vs trained-on-20 (training contribution) plus base-on-K
(original, shows the ceiling problem).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_REF = "4621b92"
sys.path.insert(0, HERE)
repo = os.path.dirname(HERE)
if os.path.isdir(os.path.join(repo, "hd_laya")):
    sys.path.insert(0, repo)
else:
    import importlib.util
    if importlib.util.find_spec("hd_laya") is None:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.4.35"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "git+https://github.com/sensesVoid/hd-laya.git@%s" % PKG_REF], check=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT_DIR = os.environ.get("OUT_DIR", "./out")
K_VALUES = [20, 77, 150, 500, 1000] if not SMOKE else [20, 77]
N_PER_K = 16 if SMOKE else int(os.environ.get("N_PER_K", "64"))
MAX_LEN = 512
HEAD_MAX_LEN = 192
FIXED_SEED = 42

from hd_laya.agent import HDLayaAgent
from hd_laya.bench import BenchmarkItem, dump_json, run_benchmark
from hd_laya.data_gen import gen_dataset
from hd_laya.laya_agent import TrainedLayaAgent
from laya.agent import Agent

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device=%s SMOKE=%s K=%s n=%d seed=%d" % (device, SMOKE, K_VALUES, N_PER_K, FIXED_SEED), flush=True)

base_agent = Agent("convaiinnovations/laya", device=device)

ckpt_path = os.environ.get("CHECKPOINT_PATH", "")
if not ckpt_path or not os.path.exists(ckpt_path):
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download("geraldaton20/hd-laya", "hd_laya_trained_calibrated.pt",
                                cache_dir="./hf_cache")

trained = TrainedLayaAgent.from_checkpoint(ckpt_path, agent=Agent("convaiinnovations/laya", device=device),
                                           max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)

# Both arms narrow to 20 via HDC (fair comparison)
base_hd = HDLayaAgent(base_agent, max_options=20, z_bar=0.0)
trained_hd = HDLayaAgent(trained, max_options=20, z_bar=0.0)

os.makedirs(OUT_DIR, exist_ok=True)
all_results = {}

for K in K_VALUES:
    raw = gen_dataset(N_PER_K, K, seed=FIXED_SEED)
    items = [BenchmarkItem(state=it["state"], expected=it["expected"],
                           meta={"questions": it["questions"]}) for it in raw]

    def mq(item):
        return item.meta["questions"]

    t0 = time.perf_counter()
    res_base = run_benchmark(base_hd, items, mq, question_id="q")
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_trained = run_benchmark(trained_hd, items, mq, question_id="q")
    t_trained = time.perf_counter() - t0

    summary = {
        "K": K, "n": N_PER_K,
        "base_on_20_acc": round(res_base["summary"]["accuracy"], 4),
        "trained_on_20_acc": round(res_trained["summary"]["accuracy"], 4),
        "training_delta": round(res_trained["summary"]["accuracy"] - res_base["summary"]["accuracy"], 4),
        "base_items_per_s": round(N_PER_K / t_base, 1),
        "trained_items_per_s": round(N_PER_K / t_trained, 1),
    }
    all_results["K=%d" % K] = summary
    print(json.dumps(summary), flush=True)
    dump_json(os.path.join(OUT_DIR, "fair_K%d.json" % K), {"base": res_base, "trained": res_trained})

with open(os.path.join(OUT_DIR, "fair_summary.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print("ALL DONE")
print(json.dumps(all_results, indent=2))
PYEOF
echo "created"