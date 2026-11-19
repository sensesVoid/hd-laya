"""Run BANKING77 real-world benchmark: base laya vs hd-laya trained.

Downloads BANKING77 test split, converts to laya choice format (77 intents),
runs both models, and prints accuracy comparison.

Usage:
    python scripts/bench_banking77.py --n 200 --device cuda
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from laya.agent import Agent

    # Download test parquet
    url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data/test-00000-of-00001.parquet"
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "banking77_test.parquet")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if not os.path.exists(local):
        print("Downloading BANKING77 test split...")
        urllib.request.urlretrieve(url, local)

    # Get label names from metadata
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(local)
    meta = pf.metadata.metadata
    import json as _json
    features = _json.loads(meta[b"huggingface"])["info"]["features"]["label"]["names"]

    df = pd.read_parquet(local)
    print(f"BANKING77 test: {len(df)} rows, {len(features)} intents")

    # Build laya-format items (first N)
    items_raw = []
    for _, row in df.head(args.n).iterrows():
        text = row["text"]
        label_idx = int(row["label"])
        expected = features[label_idx]
        criteria = {intent: intent.replace("_", " ") for intent in features}
        items_raw.append({
            "state": text,
            "questions": {"q": {
                "type": "choice",
                "instructions": "Classify the banking intent of this customer query.",
                "criteria": criteria,
            }},
            "expected": expected,
        })

    from hd_laya.bench import BenchmarkItem, run_benchmark
    items = [BenchmarkItem(state=it["state"], expected=it["expected"],
                           meta={"questions": it["questions"]}) for it in items_raw]

    def mq(item):
        return item.meta["questions"]

    # Base laya
    print("\nLoading base laya...")
    base = Agent("convaiinnovations/laya", device=args.device)
    res_base = run_benchmark(base, items, mq, question_id="q")
    print(f"Base laya: accuracy={res_base['summary']['accuracy']:.4f} "
          f"refusal={res_base['summary']['refusal_rate']:.4f}")

    # hd-laya trained
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        # Download from HF
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download("geraldaton20/hd-laya", "hd_laya_trained_calibrated.pt",
                                    cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hf_cache"))

    print("Loading hd-laya trained...")
    from hd_laya.agent import HDLayaAgent
    from hd_laya.laya_agent import TrainedLayaAgent
    trained = TrainedLayaAgent.from_checkpoint(ckpt_path, agent=Agent("convaiinnovations/laya", device=args.device),
                                               max_len=512, head_max_len=192)
    hd = HDLayaAgent(trained, max_options=20, z_bar=0.0)  # z_bar=0 = no refusals
    res_hd = run_benchmark(hd, items, mq, question_id="q")
    print(f"hd-laya trained: accuracy={res_hd['summary']['accuracy']:.4f} "
          f"refusal={res_hd['summary']['refusal_rate']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"BANKING77 ({args.n} items, {len(features)} intents)")
    print(f"  Base laya:      {res_base['summary']['accuracy']:.4f}")
    print(f"  hd-laya trained: {res_hd['summary']['accuracy']:.4f}")
    print(f"  Delta: {res_hd['summary']['accuracy'] - res_base['summary']['accuracy']:+.4f}")
    print(f"  (base narrowed to 20 by HDC, so base sees all {len(features)} but "
          f"hd-laya sees 20)")

    # Save results
    import json
    out = {
        "dataset": "BANKING77",
        "n_items": args.n,
        "n_intents": len(features),
        "base_accuracy": res_base["summary"]["accuracy"],
        "base_refusal": res_base["summary"]["refusal_rate"],
        "hd_accuracy": res_hd["summary"]["accuracy"],
        "hd_refusal": res_hd["summary"]["refusal_rate"],
        "delta": res_hd["summary"]["accuracy"] - res_base["summary"]["accuracy"],
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "banking77.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
