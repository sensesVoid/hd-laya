"""Tests for hd_laya.hdc_slots: HDC option-slot projection (T9)."""
import numpy as np
import pytest

from hd_laya.hdc_slots import project_slots


def test_project_slots_shape_and_dtype():
    vec = np.arange(64, dtype=np.float32)
    out = project_slots(vec, k=2, d_model=16, seed="s")
    assert out.shape == (2, 16)
    assert out.dtype == np.float32
    # batched input
    batch = np.zeros((3, 64), dtype=np.float32)
    outb = project_slots(batch, k=1, d_model=8, seed="s")
    assert outb.shape == (3, 1, 8)


def test_project_slots_deterministic():
    vec = np.random.default_rng(0).normal(size=32).astype(np.float32)
    a = project_slots(vec, k=3, d_model=8, seed="hd-laya-slots-v1")
    b = project_slots(vec, k=3, d_model=8, seed="hd-laya-slots-v1")
    assert np.array_equal(a, b)


def test_project_slots_seed_changes_output():
    vec = np.ones(16, dtype=np.float32)
    a = project_slots(vec, k=1, d_model=8, seed="a")
    b = project_slots(vec, k=1, d_model=8, seed="b")
    assert not np.array_equal(a, b)


def test_project_slots_guards():
    with pytest.raises(ValueError):
        project_slots(np.ones(4), k=0, d_model=8)
    with pytest.raises(ValueError):
        project_slots(np.ones(4), k=1, d_model=0)
    with pytest.raises(ValueError):
        project_slots(np.array(1.0), k=1, d_model=8)


def test_project_slots_torch_parity_if_available():
    try:
        import torch
    except Exception:
        pytest.skip("torch not available")
    from hd_laya.hdc_slots import SlotProjector
    sp = SlotProjector(d_in=16, d_model=8, k=2, seed="t")
    x = torch.zeros(16)
    y = sp(x)
    assert tuple(y.shape) == (2, 8)
    # determinism: same seed -> identical weights
    sp2 = SlotProjector(d_in=16, d_model=8, k=2, seed="t")
    assert torch.allclose(sp.weight, sp2.weight)
