# hd-laya

**hd-laya** is a neuro-symbolic decision model built on top of the pure,
un-distilled upstream [laya](https://github.com/NandhaKishorM/laya) System-1
engine. It removes laya's ~20-option choice ceiling and lets a
hyperdimensional (HDC/VSA) workspace manage context.

- **v0.1**: no-train HDC harness — two-stage option index, int16 counter
  workspace with exact undo, z-gate abstention, engine-agnostic bench runner.
- **v1.0**: the harness plus a well-posed high-K suite, a strict
  trained-checkpoint loader, an ONNX export path, and a refine loop — **and a
  negative result on head fine-tuning**, documented below rather than rounded up.

License: Apache-2.0 (derived from laya; attribution to NandhaKishorM/laya).

## Install

```bash
pip install hd-laya
pip install "hd-laya[sem]"   # optional semantic tier (MiniLM)
```

Requires Python >= 3.10; the laya engine is pulled in automatically.

## Quickstart

```python
from hd_laya.index import OptionIndex
from hd_laya.workspace import Workspace
from hd_laya.zgate import decide, z_from_scores

idx = OptionIndex.fit(["refund order", "cancel subscription", "track package"])
labels, scores, meta = idx.rank("where is my package")
print(labels[:3], meta["margin"])

ws = Workspace(dims=65536)
ws.write("ctx", ["role:order"], ["filler:42"])
print(ws.decode("ctx", ["role:order"], ["filler:0", "filler:42"]))

print(decide(z_from_scores([0.9, 0.05, 0.05]), bar=5.0))   # answer / REFUSE
```

### Beyond laya's option ceiling

```python
from laya.agent import Agent
from hd_laya.agent import HDLayaAgent

base = Agent("convaiinnovations/laya", device="cuda")
agent = HDLayaAgent(base, max_options=20, z_bar=5.0)
result = agent.predict("customer wants a refund", {"q": {"type": "choice"}})
```

Any number of criteria: the index narrows to 20 before the forward pass and the
answer maps back to the original labels. `TrainedLayaAgent` (in
`hd_laya.laya_agent`) loads a head-only checkpoint over that base model and
refuses to continue if any non-encoder weight fails to land.

## Measured

The high-K task is well-posed: every option carries a unique order key and the
state names exactly one of them, so the attainable accuracy is 1.0 at any K. An
earlier generator drew all options from the same 12-topic vocabulary, which made
~K/12 options equally consistent with the state; accuracy could never exceed
12/K and the high-K rows measured that ceiling instead of retrieval.

| measurement | K=20 | K=77 | K=150 | K=500 | K=1000 | K=2000 |
|---|---|---|---|---|---|---|
| index recall@20 | – | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Base laya plus narrowing, 48 items: **75.0% at K=200** (83.7% on answered),
against 52–56% at K=20 direct.

`scripts/measure_index_recall.py` reproduces the recall; it needs numpy only.

### Negative result: head fine-tuning

Three configurations, all measured (`scripts/verify_trained_inference.py
--compare` prints the delta and prediction agreement):

| config | steps | outcome |
|---|---|---|
| RLCD, N=128, lr 1e-4 | 128 | weights moved 0.0035; predictions identical to base on its own training items (30/64 both) |
| soft-CE, N=512, lr 3e-4 | 1024 | loss flat at 1.00; no accuracy gain at matched coverage (base 62.5% vs 60.9%) |
| overfit test, N=32, lr 1e-3 | 320 | loss plateaus ~0.85; train top-1 43.8% — it cannot memorize 32 items |

A model that cannot memorize 32 items is not starved of data or steps. With the
encoder frozen and a 2-layer head, exact key matching across 20 options plateaus
far below memorisation, so the fine-tune is published as a pipeline artifact and
not as an improvement. The gate for any future training claim is a held-out
accuracy delta at matched coverage — never training loss, which is stochastic
under RLCD and swung 5.38 → 1.10 → 2.42 while barely moving the weights.

## Real-world benchmarks

| dataset | intents | base laya | hd-laya trained | delta | index recall@20 |
|---|---|---|---|---|---|
| BANKING77 | 77 | 4.0% | 24.0% | +20.0% | ~1.0 |
| CLINC150 | 151 | 20.0% | 3.3% | −16.7% | 0.75 |

**BANKING77** (50 items): base laya is at chance (4% for 77-way). hd-laya
narrows to 20 via HDC and achieves 24% — a 6× improvement. The HDC index
achieves ~100% recall@20 on BANKING77.

**CLINC150** (30 items): base laya gets 20% (well above chance for 151-way).
hd-laya gets 3.3% because the HDC index only achieves 75% recall@20 on
CLINC150's more semantically diverse intents. When the correct intent isn't
in the narrowed set, the model can't recover.

## Layout

- `hd_laya/` — `hv`, `index`, `workspace`, `zgate`, `agent`, `bench`,
  `laya_agent`, `data_gen`, `hdc_slots`, `hdc_context`, `refine_loop`,
  `export_onnx`
- `kaggle/` — training and benchmark kernels, pushed with `kaggle/sync_bundles.py`
- `probes/`, `results/` — pre-registered measurements and their JSON
- `scripts/` — verification harnesses (`verify_trained_inference.py`,
  `measure_index_recall.py`, `export_checkpoint.py`)
- `tests/` — offline suite, no hub downloads

## Documentation policy

All durable/agent-facing documentation lives in **SQAC** (docs cartridge).
`README.md` is the only `.md` file in this repo.

