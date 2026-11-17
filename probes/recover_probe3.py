"""One-off: reconstruct probe3_semantic.json from the stdout rows of the run
that crashed in summary aggregation (same int-vs-str key bug as probe2)."""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path(r"C:\Users\geral\.local\share\opencode\shell\9780453cd0b7fb6a1cdd55676002b075bd231a70\sh_1e76ed3c3001rkUbW2pNUgLwzH.out")
OUT = Path(__file__).resolve().parent.parent / "results" / "probe3_semantic.json"

rows = []
embed_seconds = None
for line in SRC.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = line.strip()
    if line.startswith('{"k"'):
        rows.append(json.loads(line))
    elif "embedded options in" in line:
        embed_seconds = float(line.split("embedded options in")[1].split("s")[0])

assert rows, "no rows found"
print(f"parsed {len(rows)} rows; embed_seconds={embed_seconds}")

summary = {}
for k in sorted({r["k"] for r in rows}):
    rs = [r for r in rows if r["k"] == k]
    n = len(rs)
    summary[f"k{k}"] = {
        "seeds": n,
        "mean_bit_top1_vs_float_agree": sum(r["bit_top1_vs_float_top1_agree"] for r in rs) / n,
        "mean_bit_gold_acc": sum(r["bit_gold_acc"] for r in rs) / n,
        "mean_recall_at_64_vs_float": sum(r["recall_at_M_vs_float"]["64"] for r in rs) / n,
        "mean_two_stage_disagree": sum(r["two_stage_final_disagree_vs_float_top1"] for r in rs) / n,
    }

result = {
    "probe": "3-semantic-simhash",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "embed_seconds": embed_seconds,
    "rows": rows,
    "summary": summary,
    "recovered_from_stdout": True,
}
OUT.write_text(json.dumps(result, indent=1))
print("WROTE", OUT)
for k, v in sorted(summary.items()):
    print(f"{k:>5} agree={v['mean_bit_top1_vs_float_agree']:.4f} gold={v['mean_bit_gold_acc']:.4f} "
          f"R@64={v['mean_recall_at_64_vs_float']:.4f} 2stage={v['mean_two_stage_disagree']:.5f}")
