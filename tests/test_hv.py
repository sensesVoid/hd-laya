"""Tests for hd_laya.hv: seeded atoms, bit packing, packed Hamming scans (T1)."""
import numpy as np

from hd_laya.hv import atom, hamming_counts, pack_bits


def test_atom_bit_identity():
    """P1 law: atom(dims=1024) is bit-identical to the 128-byte prefix of atom(dims=2048)."""
    a1024 = atom("hd-laya-v1", "blue credit card", 1024)
    a2048 = atom("hd-laya-v1", "blue credit card", 2048)
    assert a1024.dtype == np.uint8 and a1024.shape == (128,)
    assert a2048.dtype == np.uint8 and a2048.shape == (256,)
    assert np.array_equal(a1024, a2048[:128])


def test_atom_deterministic():
    """Same inputs -> identical arrays; different tokens -> different arrays."""
    a = atom("hd-laya-v1", "hello", 2048)
    b = atom("hd-laya-v1", "hello", 2048)
    c = atom("hd-laya-v1", "world", 2048)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.dtype == np.uint8 and a.shape == (256,)


def test_pack_roundtrip():
    """pack_bits is MSB-first and unpackbits round-trips a random bool array."""
    rng = np.random.default_rng(0)
    bits = rng.random((5, 2048)) < 0.5
    packed = pack_bits(bits)
    assert packed.dtype == np.uint8 and packed.shape == (5, 256)
    # unpackbits needs axis=-1: without it the array is treated as a flat buffer.
    assert np.array_equal(np.unpackbits(packed, axis=-1, bitorder="big"), bits)


def test_hamming_scan_matches_reference():
    """LUT scan equals a plain-Python (x ^ y).bit_count() reference, exactly."""
    rng = np.random.default_rng(7)
    matrix = rng.integers(0, 256, size=(50, 256), dtype=np.uint8)
    queries = rng.integers(0, 256, size=(3, 256), dtype=np.uint8)
    got = hamming_counts(queries, matrix)
    assert got.dtype == np.uint16 and got.shape == (3, 50)
    for qi in range(queries.shape[0]):
        x = int.from_bytes(queries[qi].tobytes(), "big")
        for ri in range(matrix.shape[0]):
            y = int.from_bytes(matrix[ri].tobytes(), "big")
            assert got[qi, ri] == (x ^ y).bit_count()
    # Single packed query row: uint8[B] x uint8[n, B] -> uint16[n].
    single = hamming_counts(queries[0], matrix)
    assert single.dtype == np.uint16 and single.shape == (50,)
    assert np.array_equal(single, got[0])
