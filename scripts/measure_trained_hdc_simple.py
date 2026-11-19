"""Measure trained HDC recall + base laya accuracy on real-world data."""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import pandas as pd
from laya.agent import Agent
from hd_laya.agent import HDLayaAgent
from hd_laya.bench import BenchmarkItem, run_benchmark

from scripts.trained_index import TrainedOptionIndex
from scripts.train_hdc_encoder import HdcEncoder, text_to_features
from scripts.measure_hdc_only import get_label_names


def main():
    print("Loading trained HDC encoder...")
    encoder = HdcEncoder(dims=2048, hidden=2048)
    state = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hdc_encoder.pt"), map_location="cpu")["model"]
    encoder.load_state_dict(state)
    encoder.eval()

    # BANKING77
    print("\nBANKING77:")
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "banking77_test.parquet")
    if not os.path.exists(local):
        urllib.request.urlretrieve("https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data/test-00000-of-00001.parquet", local)
    label_names = get_label_names(local, "label")
    df = pd.read_parquet(local)

    n = 30
    # Measure recall
    idx = TrainedOptionIndex.from_file(label_names, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hdc_encoder.pt"))
    # Set options to label names for recall measurement
    idx.options = label_names

    hits = 0
    for _, row in df.head(n).iterrows():
        label_idx = int(row["label"])
        expected = label_names[label_idx]
        top_texts, _, _ = idx.rank(row["text"], M=20)
        if expected in top_texts:
            hits += 1
    print(f"  recall@20 = {hits}/{n} = {hits/n:.3f}")

    # Build items for base laya benchmark
    items = []
    for _, row in df.head(n).iterrows():
        label_idx = int(row["label"])
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

    bench_items = [BenchmarkItem(state=it["state"], expected=it["expected"], meta={"questions": it["questions"]}) for it in items]
    agent = Agent("convaiinnovations/laya", device="cuda")
    hd = HDLayaAgent(agent, max_options=20, z_bar=0.0)
    res = run_benchmark(hd, bench_items, lambda i: i.meta["questions"], question_id="q")
    print(f"  base laya accuracy = {res['summary']['accuracy']:.4f}")

    # CLINC150
    print("\nCLINC150:")
    local2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clinc150_test.parquet")
    if not os.path.exists(local2):
        urllib.request.urlretrieve("https://huggingface.co/datasets/clinc/clinc_oos/resolve/main/plus/test-00000-of-00001.parquet", local2)
    label_names2 = get_label_names(local2, "intent")
    df2 = pd.read_parquet(local2)

    idx2 = TrainedOptionIndex.from_file(label_names2, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hdc_encoder.pt"))
    idx2.options = label_names2

    hits2 = 0
    for _, row in df2.head(n).iterrows():
        label_idx = int(row["intent"])
        expected = label_names2[label_idx]
        top_texts, _, _ = idx2.rank(row["text"], M=20)
        if expected in top_texts:
            hits2 += 1
    print(f"  recall@20 = {hits2}/{n} = {hits2/n:.3f}")

    items2 = []
    for _, row in df2.head(n).iterrows():
        label_idx = int(row["intent"])
        expected = label_names2[label_idx]
        criteria = {intent: intent.replace("_", " ") for intent in label_names2}
        items2.append({
            "state": row["text"],
            "questions": {"q": {
                "type": "choice",
                "instructions": f"Classify into one of {len(label_names2)} intents.",
                "criteria": criteria,
            }},
            "expected": expected,
        })

    bench_items2 = [BenchmarkItem(state=it["state"], expected=it["expected"], meta={"questions": it["questions"]}) for it in items2]
    res2 = run_benchmark(hd, bench_items2, lambda i: i.meta["questions"], question_id="q")
    print(f"  base laya accuracy = {res2['summary']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
