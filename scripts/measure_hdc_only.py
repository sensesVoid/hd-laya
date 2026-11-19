"""Measure: base laya accuracy on HDC-narrowed sets (no trained head).

Uses HDLayaAgent with base laya. Measures for BANKING77 and CLINC150 and synthetic.
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pyarrow.parquet as pq
from laya.agent import Agent
from hd_laya.agent import HDLayaAgent
from hd_laya.bench import BenchmarkItem, run_benchmark


def get_label_names(local_path, label_col):
    """Extract label names from parquet metadata."""
    pf = pq.ParquetFile(local_path)
    meta = pf.metadata.metadata
    if meta and b"huggingface" in meta:
        features = json.loads(meta[b"huggingface"])["info"]["features"][label_col]
        if "names" in features:
            return features["names"]
    df = pd.read_parquet(local_path)
    return [str(i) for i in sorted(df[label_col].unique())]


def run_realworld(dataset_name, parquet_url, label_col, n=50, K=20, device="cuda"):
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"{dataset_name.lower()}_test.parquet")
    if not os.path.exists(local):
        urllib.request.urlretrieve(parquet_url, local)
    label_names = get_label_names(local, label_col)
    df = pd.read_parquet(local)
    print(f"{dataset_name}: {len(df)} rows, {len(label_names)} intents")

    items = []
    for _, row in df.head(n).iterrows():
        label_idx = int(row[label_col])
        expected = label_names[label_idx]
        criteria = {intent: intent.replace("_", " ") for intent in label_names}
        items.append({
            "state": row["text"],
            "questions": {"q": {
                "type": "choice",
                "instructions": f"Classify into one of {len(label_names)} intents.",
                "criteria": criteria,
            }},
            "expected": expected,
        })

    bench_items = [BenchmarkItem(state=it["state"], expected=it["expected"],
                                 meta={"questions": it["questions"]}) for it in items]

    agent = Agent("convaiinnovations/laya", device=device)
    hd = HDLayaAgent(agent, max_options=K, z_bar=0.0)
    res = run_benchmark(hd, bench_items, lambda i: i.meta["questions"], question_id="q")
    summary = res["summary"]
    print(f"  Base laya on HDC-narrowed({K}): accuracy={summary['accuracy']:.4f}")
    return {"dataset": dataset_name, "n": n, "K": K, "accuracy": summary["accuracy"]}


def run_synthetic(n=50, K=20, device="cuda"):
    from hd_laya.data_gen import gen_dataset
    items_raw = gen_dataset(n, 500, seed=100)
    bench_items = [BenchmarkItem(state=it["state"], expected=it["expected"],
                                 meta={"questions": it["questions"]}) for it in items_raw]
    agent = Agent("convaiinnovations/laya", device=device)
    hd = HDLayaAgent(agent, max_options=K, z_bar=0.0)
    res = run_benchmark(hd, bench_items, lambda i: i.meta["questions"], question_id="q")
    summary = res["summary"]
    print(f"Synthetic: base laya on HDC-narrowed({K}): accuracy={summary['accuracy']:.4f}")
    return {"dataset": "synthetic", "n": n, "K": K, "accuracy": summary["accuracy"]}


def main():
    results = []
    results.append(run_synthetic(device="cuda"))
    results.append(run_realworld(
        "BANKING77",
        "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data/test-00000-of-00001.parquet",
        "label", n=30, device="cuda"))
    results.append(run_realworld(
        "CLINC150",
        "https://huggingface.co/datasets/clinc/clinc_oos/resolve/main/plus/test-00000-of-00001.parquet",
        "intent", n=30, device="cuda"))

    print(f"\n{'='*60}")
    for r in results:
        print(f"  {r['dataset']}: {r['accuracy']:.4f}")
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hdc_only.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
