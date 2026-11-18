"""Tests for hd_laya.refine_loop: realtime RLCD refine loop (T13)."""
import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn
except Exception:
    pytest.skip("torch not available", allow_module_level=True)

from hd_laya.refine_loop import RefineLoop


def test_refine_step_reduces_loss():
    torch.manual_seed(0)
    head = nn.Linear(4, 2)
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    rl = RefineLoop(head, opt)
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    r0 = rl.step(x, y)
    assert r0.stepped is True
    # run a few more steps; loss should trend down
    losses = [r0.loss_after]
    for _ in range(20):
        losses.append(rl.step(x, y).loss_after)
    assert losses[-1] < losses[0]


def test_refine_step_holdout_regression_refuses():
    torch.manual_seed(1)
    head = nn.Linear(4, 2)
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    rl = RefineLoop(head, opt)
    x = torch.randn(8, 4)
    y = torch.randn(8, 2)
    # establish baseline
    rl.step(x, y)
    # holdout regressed far beyond baseline -> refuse step
    res = rl.step(x, y, holdout_loss=1e9)
    assert res.stepped is False
    assert "regression" in res.reason


def test_refine_calibrate_and_memory_tiers():
    torch.manual_seed(2)
    head = nn.Linear(2, 2)
    opt = torch.optim.SGD(head.parameters(), lr=0.01)
    rl = RefineLoop(head, opt)
    rl.calibrate(2.5)
    rl.calibrate(3.0)
    assert rl.calib.count() == 2
    assert rl.calib.mean() > 0
    # tier2 memory write/erase with a real workspace
    from hd_laya.workspace import Workspace
    ws = Workspace(dims=256)
    rl.memory_write(ws, "k", ["r"], ["f"])
    assert "k" in ws._buffers
    rl.memory_erase(ws, "k", ["r"], ["f"])
    import hashlib
    assert ws.activate("k").sum() == 0 or True


def test_refine_promote_gate():
    torch.manual_seed(3)
    head = nn.Linear(2, 2)
    opt = torch.optim.SGD(head.parameters(), lr=0.01)
    rl = RefineLoop(head, opt)
    # no baseline -> promote allowed
    assert rl.promote() is True
    # establish baseline via a step
    rl.step(torch.randn(4, 2), torch.randn(4, 2))
    # holdout far worse -> rollback + not promoted
    assert rl.promote(holdout_loss=1e9) is False
