"""Tests for hd_laya.export_onnx: ONNX export + validation (T12)."""
import os
import tempfile

import pytest

try:
    import torch
    import torch.nn as nn
except Exception:
    pytest.skip("torch not available", allow_module_level=True)

import onnx

from hd_laya.export_onnx import export_onnx, verify_onnx


def test_export_onnx_produces_valid_model():
    model = nn.Linear(4, 2)
    dummy = torch.randn(3, 4)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.onnx")
        out = export_onnx(model, dummy, p)
        assert out.exists()
        verify_onnx(out)
        m = onnx.load(str(out))
        assert m.graph.input[0].name == "input"
        assert m.graph.output[0].name == "output"


def test_export_onnx_dynamic_axes():
    model = nn.Linear(4, 2)
    dummy = torch.randn(2, 4)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.onnx")
        export_onnx(model, dummy, p, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        verify_onnx(p)
