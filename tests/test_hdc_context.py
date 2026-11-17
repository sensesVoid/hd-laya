"""Tests for hd_laya.hdc_context: workspace superposition -> hypertokens (T10)."""
import numpy as np

from hd_laya.hdc_context import project_context


def test_project_context_shape_dtype():
    vec = np.zeros(65536, dtype=np.float32)
    out = project_context(vec, k=4, d_model=16)
    assert out.shape == (4, 16)
    assert out.dtype == np.float32


def test_project_context_deterministic():
    rng = np.random.default_rng(1)
    vec = rng.normal(size=1024).astype(np.float32)
    a = project_context(vec, k=2, d_model=8)
    b = project_context(vec, k=2, d_model=8)
    assert np.array_equal(a, b)


def test_project_context_differs_from_slots_seed():
    # context seed differs from slot seed -> different projection for same input
    from hd_laya.hdc_slots import project_slots
    vec = np.ones(32, dtype=np.float32)
    c = project_context(vec, k=1, d_model=8, d_in=32)
    s = project_slots(vec, k=1, d_model=8, seed="hd-laya-slots-v1", d_in=32)
    assert not np.array_equal(c, s)


def test_project_context_batched():
    batch = np.zeros((2, 1024), dtype=np.float32)
    out = project_context(batch, k=3, d_model=4, d_in=1024)
    assert out.shape == (2, 3, 4)
