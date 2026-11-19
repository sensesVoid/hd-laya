"""Measure HDC index recall on CLINC150 real-world dataset."""
import os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import pyarrow.parquet as pq
import json
from hd_laya.index import OptionIndex

url = "https://huggingface.co/datasets/clinc/clinc_oos/resolve/main/plus/test-00000-of-00001.parquet"
local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clinc_test.parquet")
if not os.path.exists(local):
    urllib.request.urlretrieve(url, local)

pf = pq.ParquetFile(local)
features = json.loads(pf.metadata.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
df = pd.read_parquet(local)

# Build one item per intent (first occurrence)
seen = set()
items = []
for _, row in df.iterrows():
    intent = features[int(row["intent"])]
    if intent in seen:
        continue
    seen.add(intent)
    text = row["text"]
    criteria = {i: i.replace("_", " ") for i in features}
    items.append((text, criteria, intent))
    if len(items) >= 20:
        break

print(f"Testing {len(items)} intents from CLINC150")
hits = 0
for text, criteria, expected in items:
    labels = list(criteria)
    texts = [f"{lbl}: {criteria[lbl]}" for lbl in labels]
    expected_text = f"{expected}: {criteria[expected]}"
    idx = OptionIndex.fit(texts)
    top_texts, _, _ = idx.rank(text, k=1024, M=20)
    hit = expected_text in top_texts
    hits += int(hit)
    if not hit:
        print(f"  MISS: {text[:60]} -> expected {expected}")

print(f"\nRecall@20: {hits}/{len(items)} = {hits/len(items):.3f}")