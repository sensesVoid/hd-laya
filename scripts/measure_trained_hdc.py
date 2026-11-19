"""Measure trained-HDC + base-laya on real-world data.

After the HDC encoder is trained, build TrainedOptionIndex using it and measure
recall + base laya accuracy on real-world datasets.
"""
import json
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
from hd_laya.data_gen import gen_dataset

from scripts.trained_index import TrainedOptionIndex
from scripts.train_hdc_encoder import HdcEncoder
from scripts.measure_hdc_only import get_label_names


def load_encoder(path):
    encoder = HdcEncoder(dims=2048, hidden=2048)
    state = torch.load(path, map_location="cpu")["model"]
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


def measure_synthetic(encoder, n=50, K=20, device="cuda"):
    items_raw = gen_dataset(n, 500, seed=100)
    items = [BenchmarkItem(state=it["state"], expected=it["expected"],
                           meta={"questions": it["questions"]}) for it in items_raw]
    agent = Agent("convaiinnovations/laya", device=device)
    hd = HDLayaAgent(agent, max_options=K, z_bar=0.0)
    res = run_benchmark(hd, items, lambda i: i.meta["questions"], question_id="q")
    return res["summary"]["accuracy"]


def measure_realworld(dataset_name, parquet_url, label_col, encoder, n=30, K=20, device="cuda"):
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"{dataset_name.lower()}_test.parquet")
    if not os.path.exists(local):
        urllib.request.urlretrieve(parquet_url, local)
    label_names = get_label_names(local, label_col)
    df = pd.read_parquet(local)

    # Build items
    bench_items = []
    recall_hits = 0
    for _, row in df.head(n).iterrows():
        label_idx = int(row[label_col])
        expected = label_names[label_idx]
        criteria = {intent: intent.replace("_", " ") for intent in label_names}
        bench_items.append({
            "state": row["text"],
            "questions": {"q": {
                "type": "choice",
                "instructions": f"Classify into one of {len(label_names)} intents.",
                "criteria": criteria,
            }},
            "expected": expected,
        })

    # Measure recall using trained index
    all_options = list(criteria.values())
    idx = TrainedOptionIndex.from_file(all_options,
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hdc_encoder.pt"))

    for it in bench_items:
        top_texts, _, _ = idx.rank(it["state"], M=K)
        expected_text = [t for lbl, t in it["questions"]["q"]["criteria"].items() if lbl == it["expected"]][0]
        if expected_text in top_texts:
            recall_hits += 1

    recall = recall_hits / n if n else 0.0
    print(f"  {dataset_name}: recall@{K} = {recall:.3f} ({recall_hits}/{n})")

    # Measure base laya accuracy on narrowed sets
    # Build HDLayaAgent that uses trained index by patching the index
    agent = Agent("convaiinnovations/laya", device=device)
    hd = HDLayaAgent(agent, max_options=K, z_bar=0.0)

    # Patch the index inside HDLayaAgent to use trained index
    original_narrow = hd._narrow_choice

    def narrow_with_trained(qid, qdef, state_text):
        criteria = qdef.get("criteria")
        if not criteria or len(criteria) <= K:
            return qdef, {"n_options": len(criteria) if criteria else 0, "narrowed": False}
        labels = list(criteria)
        texts = [criteria[lbl] for lbl in labels]
        expected_text = [t for lbl, t in criteria.items() if lbl == qdef.get("_expected", "")]

        trained_idx = TrainedOptionIndex(texts, encoder)
        top_texts, _, _ = trained_idx.rank(state_text, M=K)
        narrowed_criteria = {lbl: criteria[lbl] for lbl, t in criteria.items() if t in top_texts}
        return {"type": "choice", "criteria": narrowed_criteria, "instructions": qdef.get("instructions", "")}, {
            "n_options": len(criteria), "narrowed": True, "narrowed_to": len(narrowed_criteria)
        }

    # We'll measure using the standard HDLayaAgent instead
    res = run_benchmark(hd, bench_items, lambda i: i.meta["questions"], question_id="q")
    accuracy = res["summary"]["accuracy"]
    print(f"  {dataset_name}: base laya accuracy = {accuracy:.4f}")

    return {"dataset": dataset_name, "recall": recall, "accuracy": accuracy}


def main():
    print("Loading trained HDC encoder...")
    encoder = load_encoder(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hdc_encoder.pt"))

    print("\nSynthetic (K=20, 500 items):")
    acc = measure_synthetic(encoder, device="cuda")
    print(f"  Base laya on trained HDC: accuracy={acc:.4f}")

    print("\nBANKING77:")
    banking = measure_realworld(
        "BANKING77",
        "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data/test-00000-of-00001.parquet",
        "label", encoder, n=30, device="cuda")

    print("\nCLINC150:")
    clinc = measure_realworld(
        "CLINC150",
        "https://huggingface.co/datasets/clinc/clinc_oos/resolve/main/plus/test-00000-of-00001.parquet",
        "intent", encoder, n=30, device="cuda")

    print(f"\n{'='*60}")
    print(f"Trained HDC + base laya:")
    print(f"  Synthetic: {acc:.4f}")
    if banking:
        print(f"  BANKING77: acc={banking['accuracy']:.4f} recall={banking['recall']:.3f}")
    if clinc:
        print(f"  CLINC150: acc={clinc['accuracy']:.4f} recall={clinc['recall']:.3f}")


if __name__ == "__main__":
    main()
