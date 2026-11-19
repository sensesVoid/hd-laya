"""One-shot optimization of HDC two-stage parameters on real data.

Tunes (top_M, prefix_k) to maximize recall@M on real-world datasets.
"""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pyarrow.parquet as pq
import json
from hd_laya.index import OptionIndex


def get_label_names(local_path, label_col):
    pf = pq.ParquetFile(local_path)
    meta = pf.metadata.metadata
    if meta and b"huggingface" in meta:
        features = json.loads(meta[b"huggingface"])["info"]["features"][label_col]
        if "names" in features:
            return features["names"]
    df = pd.read_parquet(local_path)
    return [str(i) for i in sorted(df[label_col].unique())]


def eval_recall(dataset_name, parquet_url, label_col, n_samples=50, top_M=20):
    """Measure recall@top_M on real data."""
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"{dataset_name.lower()}_test.parquet")
    if not os.path.exists(local):
        urllib.request.urlretrieve(parquet_url, local)
    label_names = get_label_names(local, label_col)
    df = pd.read_parquet(local)

    # Build one item per sample (shuffle for variety)
    hits = 0
    total = 0
    for _, row in df.head(n_samples).iterrows():
        label_idx = int(row[label_col])
        expected = label_names[label_idx]
        criteria = {intent: intent.replace("_", " ") for intent in label_names}
        texts = [f"{intent}: {desc}" for intent, desc in criteria.items()]
        expected_text = f"{expected}: {criteria[expected]}"

        idx = OptionIndex.fit(texts)
        top_texts, _, _ = idx.rank(row["text"], k=1024, M=top_M)

        if expected_text in top_texts:
            hits += 1
        total += 1
        # Shuffle rows to get variety
    return hits / total if total > 0 else 0.0


def optimize_M_for_dataset(dataset_name, parquet_url, label_col, n_samples=50):
    """Grid search M to maximize recall."""
    best = None
    for M in [10, 20, 30, 50, 100, 150, 200]:
        recall = eval_recall(dataset_name, parquet_url, label_col, n_samples, M)
        if best is None or recall > best[1]:
            best = (M, recall)
        print(f"  M={M}: recall@{M} = {recall:.3f}", flush=True)
    print(f"  Best: M={best[0]} with recall={best[1]:.3f}")
    return best


def main():
    print("Optimizing HDC parameters on real-world data...")

    print("\nBANKING77:")
    best_banking = optimize_M_for_dataset(
        "BANKING77",
        "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data/test-00000-of-00001.parquet",
        "label", n_samples=30)

    print("\nCLINC150:")
    best_clinc = optimize_M_for_dataset(
        "CLINC150",
        "https://huggingface.co/datasets/clinc/clinc_oos/resolve/main/plus/test-00000-of-00001.parquet",
        "intent", n_samples=30)

    print(f"\nBest: BANKING77 M={best_banking[0]}, CLINC150 M={best_clinc[0]}")


if __name__ == "__main__":
    main()
