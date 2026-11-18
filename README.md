# hd-laya

**hd-laya** is a neuro-symbolic decision model built on top of the pure,
un-distilled upstream [laya](https://github.com/NandhaKishorM/laya) System-1
engine. It removes laya's ~20-option choice ceiling and lets a
hyperdimensional (HDC/VSA) workspace manage context, with measured (speed,
accuracy) sweet spots instead of assumed ones.

- **v0.1** (this release): a no-train HDC harness wrapping unmodified laya
  checkpoints — two-stage coarse-to-fine option index, int16 counter
  workspace with exact undo, z-gate abstention, and an engine-agnostic
  benchmark runner.
- **v1.0** (planned): trained HDC option slots + HDC-context projection +
  realtime RLCD refinement (see SQAC docs `hd-laya-design-spec-p1..p3`).

License: Apache-2.0 (derived from laya; attribution to NandhaKishorM/laya).

## Install

```bash
pip install hd-laya
# optional semantic tier (MiniLM):
pip install "hd-laya[sem]"
```

Requires Python >= 3.10. The laya engine is pulled in automatically.

## Quickstart

```python
from hd_laya.index import OptionIndex
from hd_laya.workspace import Workspace
from hd_laya.zgate import decide, z_from_scores

# Two-stage option index: handles hundreds of options in one pass.
idx = OptionIndex.fit(["refund order", "cancel subscription", "track package", "..."])
labels, scores, meta = idx.rank("where is my package")
print(labels[:5], meta["margin"], meta["near_tie"])

# HDC context workspace: write, erase (exact undo), decode with z-gate.
ws = Workspace(dims=65536)
ws.write("ctx", ["role:order"], ["filler:42"])
print(ws.decode("ctx", ["role:order"], ["filler:0", "filler:42"]))

# Z-gate abstention.
z = z_from_scores([0.9, 0.05, 0.05])
print(decide(z, bar=5.0))   # "answer" or "REFUSE"
```

## Benchmarks

Reuses laya's fixed-question harness for apples-to-apples comparison
(BANKING77, CLINC150, MASSIVE, ...) plus a novel high-K stress suite
(500 / 1000 / 2000 options). Run:

```bash
python -m pytest tests/ -q
```

Real checkpoint runs are documented as manual steps (need a laya checkpoint
and `laya` installed from PyPI).

## Layout

- `hd_laya/` — package: `hv.py`, `index.py`, `workspace.py`, `zgate.py`,
  `agent.py`, `bench.py`
- `probes/` — measurement code (pre-registered bars)
- `results/` — probe outputs (JSON)
- `tests/` — test suite (offline, no hub downloads)

## Documentation policy

All durable/agent-facing documentation lives in **SQAC** (docs cartridge).
`README.md` is the only `.md` file in this repo.
