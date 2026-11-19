"""Run a real-world intent classification benchmark: base laya vs hd-laya trained.

Works with any HuggingFace dataset in parquet format with text + label columns.

Usage:
    python scripts/bench_realworld.py --dataset CLINC150 --url <parquet_url> --label-col intent --n 100 --device cuda
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset name for labeling")
    ap.add_argument("--url", required=True, help="URL to test parquet file")
    ap.add_argument("--label-col", default="label", help="Label column name")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--head-max-len", type=int, default=192)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from laya.agent import Agent

    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", f"{args.dataset.lower()}_test.parquet")
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if not os.path.exists(local):
        print(f"Downloading {args.dataset} test split...")
        urllib.request.urlretrieve(args.url, local)

    pf = pq.ParquetFile(local)
    meta = pf.metadata.metadata
    features = None
    if meta and b"huggingface" in meta:
        import json as _j
        features = _j.loads(meta[b"huggingface"])["info"]["features"][args.label_col]["names"]

    df = pd.read_parquet(local)
    print(f"{args.dataset} test: {len(df)} rows, {df[args.label_col].nunique()} intents")

    if features is None:
        features = [str(i) for i in sorted(df[args.label_col].unique())]

    # Build laya-format items (first N)
    items_raw = []
    for _, row in df.head(args.n).iterrows():
        text = row["text"]
        label_idx = int(row[args.label_col])
        expected = features[label_idx]
        criteria = {intent: intent.replace("_", " ") for intent in features}
        items_raw.append({
            "state": text,
            "questions": {"q": {
                "type": "choice",
                "instructions": f"Classify the intent of this user query into one of {len(features)} categories.",
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
    base_agent = Agent("convaiinnovations/laya", device=args.device)
    # Wrap to pass max_len/head_max_len
    class MaxLenAgent:
        def __init__(self, agent, max_len, head_max_len):
            self._agent = agent
            self._max_len = max_len
            self._head_max_len = head_max_len
        def predict(self, state, questions):
            return self._agent.predict(state, questions, max_len=self._max_len, head_max_len=self._head_max_len)
    base = MaxLenAgent(base_agent, args.max_len, args.head_max_len)
    res_base = run_benchmark(base, items, mq, question_id="q")
    print(f"Base laya: accuracy={res_base['summary']['accuracy']:.4f} "
          f"refusal={res_base['summary']['refusal_rate']:.4f}")

    # hd-laya trained
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download("geraldaton20/hd-laya", "hd_laya_trained_calibrated.pt",
                                    cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hf_cache"))

    print("Loading hd-laya trained...")
    from hd_laya.agent import HDLayaAgent
    from hd_laya.laya_agent import TrainedLayaAgent
    trained = TrainedLayaAgent.from_checkpoint(ckpt_path, agent=Agent("convaiinnovations/laya", device=args.device),
                                               max_len=args.max_len, head_max_len=args.head_max_len)
    hd = HDLayaAgent(trained, max_options=20, z_bar=0.0)  # z_bar=0 = no refusals
    res_hd = run_benchmark(hd, items, mq, question_id="q")
    print(f"hd-laya trained: accuracy={res_hd['summary']['accuracy']:.4f} "
          f"refusal={res_hd['summary']['refusal_rate']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"{args.dataset} ({args.n} items, {len(features)} intents)")
    print(f"  Base laya:       {res_base['summary']['accuracy']:.4f}")
    print(f"  hd-laya trained: {res_hd['summary']['accuracy']:.4f}")
    print(f"  Delta: {res_hd['summary']['accuracy'] - res_base['summary']['accuracy']:+.4f}")

    # Save results
    out = {
        "dataset": args.dataset,
        "n_items": args.n,
        "n_intents": len(features),
        "base_accuracy": res_base["summary"]["accuracy"],
        "base_refusal": res_base["summary"]["refusal_rate"],
        "hd_accuracy": res_hd["summary"]["accuracy"],
        "hd_refusal": res_hd["summary"]["refusal_rate"],
        "delta": res_hd["summary"]["accuracy"] - res_base["summary"]["accuracy"],
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"{args.dataset.lower()}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
