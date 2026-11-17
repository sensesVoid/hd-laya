"""Load a training checkpoint and export the head to ONNX (v1.0 T12 integration).

Usage:
    python scripts/export_checkpoint.py --checkpoint path/to/hd_laya_head.pt \
        --out-dir path/to/export --batch 4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hd_laya.export_onnx import export_onnx, verify_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'kaggle'))
    from train_hd_laya import build_tiny_head
    head = build_tiny_head()
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    dummy = torch.randn(args.batch, 32)
    out_path = os.path.join(args.out_dir, "head.onnx")
    export_onnx(head, dummy, out_path, input_names=["input"], output_names=["output"])
    print(f"exported: {out_path}")

    # Runtime verification via ONNX Runtime.
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    import numpy as np
    x = np.random.randn(args.batch, 32).astype("float32")
    y = sess.run(None, {"input": x})[0]
    print(f"ORT ok: input {x.shape} -> output {y.shape}")
    print(f"steps={ckpt.get('steps')} K={ckpt.get('K')}")


if __name__ == "__main__":
    main()
