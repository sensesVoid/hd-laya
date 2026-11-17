"""Hyperdimensional-vector primitives for the hd-laya harness.

Bit-exact port of SQAC's seeded BSC atoms plus the packed Hamming scan
from the dimension probes. Atoms chain sha256(f"{seed}:{token}:{counter}")
digests until dims//8 bytes are filled, so a shorter atom is always the
byte-prefix of a longer one with the same seed (matryoshka property).
"""
from __future__ import annotations

import hashlib

import numpy as np

D_FULL = 2048
D_PREFIX = 1024
DEFAULT_SEED = "hd-laya-v1"

# Popcount LUT: _LUT[x] = number of set bits in byte x.
_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def atom(seed: str, token: str, dims: int) -> np.ndarray:
    """Seeded atom as packed uint8 bits, shape (dims//8,).

    Bit-exact port of SQAC's atom: chain sha256(f"{seed}:{token}:{counter}")
    digests (counter starts 0, increments per digest) until dims//8 bytes
    are filled.
    """
    if dims % 8 != 0:
        raise ValueError("dims must be a multiple of 8")
    need = dims // 8
    out = bytearray()
    counter = 0
    while len(out) < need:
        out.extend(hashlib.sha256(f"{seed}:{token}:{counter}".encode()).digest())
        counter += 1
    return np.frombuffer(bytes(out[:need]), dtype=np.uint8).copy()


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Pack a bool array (..., D) into uint8 (..., D//8), MSB-first."""
    bits = np.asarray(bits, dtype=bool)
    if bits.shape[-1] % 8 != 0:
        raise ValueError("last axis length must be a multiple of 8")
    return np.packbits(bits, axis=-1, bitorder="big")


def hamming_counts(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Packed Hamming counts via XOR + 256-entry popcount LUT.

    uint8[B]    x uint8[n, B] -> uint16[n]      (one query)
    uint8[q, B] x uint8[n, B] -> uint16[q, n]  (batched queries)
    """
    q = np.asarray(q, dtype=np.uint8)
    matrix = np.asarray(matrix, dtype=np.uint8)
    single = q.ndim == 1
    if single:
        q = q[np.newaxis, :]
    xor = np.bitwise_xor(q[:, np.newaxis, :], matrix[np.newaxis, :, :])
    counts = _LUT[xor].sum(axis=-1, dtype=np.uint16)
    return counts[0] if single else counts
