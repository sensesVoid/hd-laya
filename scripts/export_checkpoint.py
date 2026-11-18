"""Load a trained checkpoint and export the head to ONNX (v1.0 T12).

Usage:
    python scripts/export_checkpoint.py --checkpoint path/to/hd_laya_trained.pt \
        --out-dir path/to/export --batch 4
"""
from __future__ import annotations

import argparse
import os
import sys

# Fix Windows emoji encoding for torch.onnx logging (must be BEFORE any torch import)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hd_laya.export_onnx import export_onnx, verify_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    from laya.agent import Agent
    agent = Agent("convaiinnovations/laya", device="cpu")
    model = agent.model
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"loaded checkpoint: {len(ckpt['state_dict'])} keys, missing={len(missing)}, unexpected={len(unexpected)}")

    h = model.head
    h.eval()

    # Wrapper: takes src (batch, seq, dim) + pad (batch, seq) -> output
    class HeadWrapper(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, src, src_key_padding_mask):
            return self.head(src, src_key_padding_mask=src_key_padding_mask)

    wrapper = HeadWrapper(h)
    wrapper.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    dummy_src = torch.randn(args.batch, 8, 1024)
    dummy_pad = torch.zeros(args.batch, 8, dtype=torch.bool)
    out_path = os.path.join(args.out_dir, "head.onnx")
    
    # Export via torch.export -> torch.onnx.export(ExportedProgram) for TransformerEncoder compatibility
    ep = torch.export.export(wrapper, (dummy_src, dummy_pad), strict=False)
    torch.onnx.export(ep, (dummy_src, dummy_pad), out_path,
                      input_names=["src", "src_key_padding_mask"],
                      output_names=["output"],
                      opset_version=18)
    verify_onnx(out_path)
    print(f"exported: {out_path}")

    # Runtime verification
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(args.batch, 8, 1024).astype("float32")
    pad = np.zeros((args.batch, 8), dtype=bool)
    y = sess.run(None, {"src": x, "src_key_padding_mask": pad})[0]
    print(f"ORT ok: input {x.shape} -> output {y.shape}")
    print(f"K={ckpt.get('K')} n_items={ckpt.get('n_items')} losses={ckpt.get('losses')}")


if __name__ == "__main__":
    main()