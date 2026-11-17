"""One-off: reconstruct probe2_matryoshka.json from the stdout rows of the
crashed run (all 45 rows printed; only the summary aggregation crashed on an
int-vs-str key). Parses JSON lines starting with '{"dfull"' from the shell
output file and recomputes the summary.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

SRC = Path(r"C:\Users\geral\.local\share\opencode\shell\9780453cd0b7fb6a1cdd55676002b075bd231a70\sh_1e75c5248001DAuwBkpAJHt1OV.out")
OUT = Path(__file__).resolve().parent.parent / "results" / "probe2_matryoshka.json"

rows = []
bit_identity = None
for line in SRC.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = line.strip()
    if line.startswith('{"dfull"'):
        rows.append(json.loads(line))
    elif line.startswith("bit_identity:"):
        bit_identity = ast.literal_eval(line[len("bit_identity: "):])

assert rows, "no rows found"
print(f"parsed {len(rows)} rows; bit_identity={bit_identity}")

summary = {}
for r in rows:
    key = f"{r['k']}of{r['dfull']}"
    s = summary.setdefault(key, {"n": 0})
    s["n"] += 1
    for f in ["full_acc", "k_acc", "acc_delta_pp", "flip_rate", "two_stage_final_disagree"]:
        s.setdefault(f"sum_{f}", 0.0)
        s[f"sum_{f}"] += r[f]
    s.setdefault("sum_recall64", 0.0)
    s["sum_recall64"] += r["recall_at_M"]["64"]

summary_out = {}
for key, s in summary.items():
    n = s["n"]
    summary_out[key] = {
        "seeds": n,
        "mean_full_acc": s["sum_full_acc"] / n,
        "mean_k_acc": s["sum_k_acc"] / n,
        "mean_acc_delta_pp": s["sum_acc_delta_pp"] / n,
        "mean_flip_rate": s["sum_flip_rate"] / n,
        "mean_recall_at_64": s["sum_recall64"] / n,
        "mean_two_stage_disagree": s["sum_two_stage_final_disagree"] / n,
    }

result = {
    "probe": "2-matryoshka-lexical",
    "bit_identity": bit_identity,
    "rows": rows,
    "summary": summary_out,
    "recovered_from_stdout": True,
}
OUT.write_text(json.dumps(result, indent=1))
print("WROTE", OUT)
for k, v in sorted(summary_out.items()):
    print(f"{k:>10} full={v['mean_full_acc']:.4f} k={v['mean_k_acc']:.4f} "
          f"delta={v['mean_acc_delta_pp']:+.3f}pp flips={v['mean_flip_rate']:.4f} "
          f"R@64={v['mean_recall_at_64']:.4f} 2stage_dis={v['mean_two_stage_disagree']:.5f}")
