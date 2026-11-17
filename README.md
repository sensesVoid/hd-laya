# hd-laya

HDC/VSA integration for **laya** — the pure, un-distilled upstream System-1
decision engine (`NandhaKishorM/laya`). Goal: remove the ~20-option choice
ceiling and let a hypervector workspace manage laya's context, with measured
(speed, accuracy) sweet spots instead of assumed ones.

## Layout

- `probes/` — measurement code (pre-registered bars; see SQAC doc `hd-laya-probe-1-prereg`)
  - `probe1_speed.py` — encode/scan throughput vs dimension D + Clay counter ops
  - `probe2_matryoshka.py` — prefix-truncation ("matryoshka") accuracy + two-stage coarse-to-fine
  - `probe3_semantic.py` — MiniLM SimHash tier vs float-cosine ground truth
  - `probe4_capacity.py` — bundling capacity vs (D, n)
- `results/` — probe outputs (JSON)

## Documentation policy

All durable documentation lives in **SQAC** (`sqac-mcp`, docs cartridge —
`mem_search` / agent-facing). **`README.md` is the only `.md` file in this
repo.**

## Environment

Probe runner: `C:\Users\geral\AppData\Local\Programs\Python\Python312\python.exe`
(numpy 2.4.6, torch 2.6.0+cu124, transformers 5.23; MiniLM cached offline).

```powershell
& "C:\Users\geral\AppData\Local\Programs\Python\Python312\python.exe" probes\probe1_speed.py
```

Run probes one at a time — timings are the measurement, don't overlap runs.
