"""ONNX export helpers for hd-laya v1.0 (T12).

Exports a (frozen) torch module to ONNX and validates the resulting file with
the ``onnx`` checker. Runtime verification uses ONNX Runtime when available
(optional dependency); the export itself only needs ``torch`` + ``onnx``.

The v1.0 split: frozen encoder graph + promoted head.onnx + PyTorch head
sidecar for live refinement. This module covers the export half; the promote/
swap logic lives in the refine loop's promote().
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import torch
    _TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH = False

try:
    import onnx
    _ONNX = True
except Exception:  # pragma: no cover
    onnx = None  # type: ignore
    _ONNX = False

__all__ = ["export_onnx", "verify_onnx"]


def export_onnx(
    module: Any,
    dummy_input: Any,
    path: "str | Path",
    *,
    input_names: Sequence[str] = ("input",),
    output_names: Sequence[str] = ("output",),
    dynamic_axes: Optional[dict] = None,
    opset_version: int = 17,
) -> Path:
    if not _TORCH:
        raise RuntimeError("export_onnx requires torch")
    if not _ONNX:
        raise RuntimeError("export_onnx requires onnx")
    module = module.eval() if hasattr(module, "eval") else module
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy_input,
            str(p),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            dynamo=False,
        )
    verify_onnx(p)
    return p


def verify_onnx(path: "str | Path") -> None:
    if not _ONNX:
        raise RuntimeError("verify_onnx requires onnx")
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    # optional runtime check
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _ = sess.get_inputs()
    except ImportError:
        pass
