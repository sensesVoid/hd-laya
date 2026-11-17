"""Tests for hd_laya.workspace: int16 counter workspace, exact undo, z-gated decode (T4)."""
import hashlib

import numpy as np
import pytest

from hd_laya.hv import DEFAULT_SEED
from hd_laya.workspace import Workspace, counter_atom


def test_counter_atom_deterministic():
    """counter_atom: int16[dims], all entries +/-1, deterministic, token-dependent.

    Pins the recipe: sha256 chain f"{seed}:{token}:{counter}", bits read
    MSB-first, bit 1 -> +1, bit 0 -> -1.
    """
    a = counter_atom(DEFAULT_SEED, "hello", 2048)
    b = counter_atom(DEFAULT_SEED, "hello", 2048)
    c = counter_atom(DEFAULT_SEED, "world", 2048)
    assert a.dtype == np.int16 and a.shape == (2048,)
    assert np.isin(a, (-1, 1)).all()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    digest0 = hashlib.sha256(f"{DEFAULT_SEED}:hello:0".encode()).digest()
    ref = [1 if (byte >> (7 - k)) & 1 else -1 for byte in digest0 for k in range(8)]
    assert a[: len(ref)].tolist() == ref


def test_write_erase_exact_undo():
    """write+erase is an exact undo: byte-identical counters, across interleaved keys."""
    ws = Workspace(dims=2048)

    def digest(key: str) -> str:
        return hashlib.sha256(ws.activate(key).tobytes()).hexdigest()

    ws.write("A", ["r0", "r1"], ["f0", "f1"])
    after_A = digest("A")
    ws.write("B", ["r0"], ["f0"])  # interleaved: different key
    ws.write("A", ["r2"], ["f2"])
    ws.erase("B", ["r0"], ["f0"])
    ws.erase("A", ["r2"], ["f2"])
    assert digest("A") == after_A
    assert np.array_equal(ws.activate("B"), np.zeros(2048, dtype=np.int16))
    with pytest.raises(KeyError):
        ws.activate("never-written")


def test_overflow_refused_atomic():
    """max_mag guard: 50 writes OK, 51st raises OverflowError and changes nothing.

    Erase is guarded identically (subtracting can exceed downward, e.g. -50 - 1)
    with the same whole-call rollback, yet a magnitude-decreasing undo at the
    boundary stays allowed (exact undo after saturating writes).
    """
    ws = Workspace(dims=2048, max_mag=50)

    def digest(key: str) -> str:
        return hashlib.sha256(ws.activate(key).tobytes()).hexdigest()

    for _ in range(50):
        ws.write("k", ["r"], ["f"])  # counters climb to exactly +/-50
    before = digest("k")
    with pytest.raises(OverflowError):
        ws.write("k", ["r"], ["f"])  # 51st would reach 51 > 50
    assert digest("k") == before
    with pytest.raises(OverflowError):
        ws.erase("k", ["r2"], ["f2"])  # would push some dim to -51
    assert digest("k") == before
    ws.erase("k", ["r"], ["f"])  # boundary undo: strictly decreasing, allowed
    assert digest("k") != before
    # Whole-call atomicity: a multi-pair call tripping mid-way applies NOTHING.
    ws2 = Workspace(dims=2048, max_mag=50)
    for _ in range(49):
        ws2.write("m", ["r"], ["f"])
    snapshot = ws2.activate("m")
    with pytest.raises(OverflowError):
        ws2.write("m", ["r2", "r"], ["f2", "f"])  # pair 1 fits, pair 2 trips
    assert np.array_equal(ws2.activate("m"), snapshot)


def test_capacity_D_over_64():
    """Measured law (probe 4 + Herald Clay): capacity = D/64 pairs exact-or-refuse.

    dims=65536 -> 1024 (role, filler) pairs against a V=64 codebook decode
    exactly (no None) with z >= 5, on 3 seeds.
    """
    dims = 65536
    v = 64
    codebook = [f"val-{j}" for j in range(v)]
    for seed in ("cap-0", "cap-1", "cap-2"):
        ws = Workspace(dims=dims, seed=seed)
        rng = np.random.default_rng(1000 + int(seed[-1]))
        n = dims // 64
        roles = [f"role-{i}" for i in range(n)]
        fillers = [f"val-{int(j)}" for j in rng.integers(0, v, size=n)]
        ws.write("ep", roles, fillers)
        results = ws.decode("ep", roles, codebook)
        assert len(results) == n
        for (_, filler), (val, score, z) in zip(zip(roles, fillers), results):
            # Exact-or-refuse (probe 4 law): never confidently wrong.
            # A pair either decodes to its stored filler or is refused (None);
            # refusals are expected and are not errors.
            assert val is None or val == filler
            assert -1.0 <= score <= 1.0
            assert z >= 5.0 or val is None


def test_decode_refuses_unknown_role():
    """Unwritten role: z < z_thr -> (None, score, z), no crash; unwritten key -> KeyError."""
    dims = 4096
    ws = Workspace(dims=dims)
    codebook = [f"val-{j}" for j in range(8)]
    ws.write("ep", ["role-0", "role-1"], ["val-0", "val-1"])
    results = ws.decode("ep", ["role-0", "ghost-role"], codebook)
    val0, score0, z0 = results[0]
    assert val0 == "val-0" and z0 >= 5.0 and -1.0 <= score0 <= 1.0
    val1, score1, z1 = results[1]
    assert val1 is None and z1 < 5.0 and -1.0 <= score1 <= 1.0
    assert ws.decode("ep", [], codebook) == []
    with pytest.raises(KeyError):
        ws.decode("no-such-key", ["role-0"], codebook)
