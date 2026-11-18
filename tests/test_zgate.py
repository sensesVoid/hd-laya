"""Tests for hd_laya.zgate: z-score abstention + streaming calibration (T5)."""
import numpy as np
import pytest

from hd_laya.zgate import ANSWER, REFUSE, StreamingCalib, decide, z_from_scores


def test_z_clear_winner_answers():
    """Scores [0.9, 0.05, 0.05]: top crushes the rest (std 0) -> inf -> ANSWER."""
    z = z_from_scores([0.9, 0.05, 0.05])
    assert z > 5
    assert decide(z) == ANSWER


def test_z_near_tie_refuses():
    """Scores [0.5, 0.49, 0.01]: z ~ 1.04 < 5 -> the literal string 'REFUSE'."""
    z = z_from_scores([0.5, 0.49, 0.01])
    assert z < 5
    got = decide(z)
    # Consumers string-match REFUSE: the value must be the literal, not just truthy.
    assert got == "REFUSE"
    assert got == REFUSE


def test_z_edge_cases():
    """Single score -> inf (answers even at bar=10); exact tie -> 0.0 -> REFUSE;
    non-finite input scores -> ValueError."""
    z = z_from_scores([0.9])
    assert z == float("inf")
    assert decide(z, bar=10.0) == ANSWER

    z_tie = z_from_scores([0.5, 0.5])
    assert z_tie == 0.0
    assert decide(z_tie) == REFUSE

    with pytest.raises(ValueError):
        z_from_scores([0.6, float("nan"), 0.2])
    with pytest.raises(ValueError):
        z_from_scores([float("inf"), 0.2])
    # 2-D input raises ValueError (F7)
    with pytest.raises(ValueError):
        z_from_scores([[0.1, 0.2], [0.3, 0.4]])


def test_z_from_scores_2d_raises():
    """F7: z_from_scores requires 1-D input."""
    with pytest.raises(ValueError):
        z_from_scores([[1.0, 2.0], [3.0, 4.0]])


def test_welford_matches_numpy():
    """100 seeded normal draws: Welford count/mean/std match numpy; after exactly
    50 updates (two calibs side by side) the state matches numpy over the first 50."""
    rng = np.random.default_rng(11)
    xs = rng.normal(0.0, 1.0, size=100)
    full = StreamingCalib()
    half = StreamingCalib()
    for i, x in enumerate(xs):
        full.update(float(x))
        if i < 50:
            half.update(float(x))
        if i == 49:
            # Streaming property: 50 updates in -> state depends on the first 50 only.
            assert full.count() == 50 and half.count() == 50
            assert abs(full.mean() - float(np.mean(xs[:50]))) < 1e-6
            assert abs(full.std() - float(np.std(xs[:50]))) < 1e-6
            assert abs(half.mean() - float(np.mean(xs[:50]))) < 1e-6
            assert abs(half.std() - float(np.std(xs[:50]))) < 1e-6
    assert full.count() == 100 and half.count() == 50
    assert abs(full.mean() - float(np.mean(xs))) < 1e-6
    assert abs(full.std() - float(np.std(xs))) < 1e-6


def test_calib_bar_never_below_default():
    """bar() = max(default, mean + k*std): tiny z's leave the default untouched;
    large z's raise it above the default, exactly mean + 2*std."""
    rng = np.random.default_rng(5)
    small = StreamingCalib()
    for z in 0.5 + rng.normal(0.0, 0.1, size=20):
        small.update(float(z))
    assert small.bar() == 5.0

    large = StreamingCalib()
    for z in rng.normal(20.0, 3.0, size=20):
        large.update(float(z))
    assert large.bar() == large.mean() + 2.0 * large.std()
    assert large.bar() > 5.0


def test_z_order_independent():
    """z_from_scores does not assume descending input (T5 minor)."""
    from hd_laya.zgate import z_from_scores, decide
    a = z_from_scores([0.05, 0.9, 0.05])
    b = z_from_scores([0.9, 0.05, 0.05])
    assert a == b
    assert decide(a, 5.0) == "answer"
