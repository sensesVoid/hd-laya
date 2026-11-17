"""Tests for hd_laya.index: two-stage coarse-to-fine option index (T2).

The text->HV reference below re-implements the brief's recipe independently
from the hd_laya.hv primitives, so OptionIndex ranking is checked against a
second encoding path (itself pinned bit-exact to SQAC's chain by T1).
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest

from hd_laya.hv import DEFAULT_SEED, atom, hamming_counts, pack_bits
from hd_laya.index import OptionIndex

_WORDS = (
    "card credit blue top up failed refund issued twice march pending order "
    "cancel ship late item wrong size color return label email code verify "
    "account login password reset store pickup ready today tomorrow delivery "
    "address change billing invoice paid charge dispute open closed ticket "
    "support agent chat phone call back week month year plan price total tax "
    "shipping free express standard track number package lost found damaged "
    "broken swap exchange gift wrap note urgent low stock waitlist notify "
    "available discount coupon apply promo balance points reward member gold "
    "silver status update profile settings report archive restore draft"
).split()

Fixture = namedtuple("Fixture", "options idx query src matrix")


def _make_options(n: int, seed: int) -> list[str]:
    """n unique option-like strings: 2-4 word combos from the word pool."""
    rng = np.random.default_rng(seed)
    out: set[str] = set()
    guard = 0
    while len(out) < n and guard < n * 50:
        k = int(rng.integers(2, 5))
        words = [_WORDS[int(i)] for i in rng.integers(0, len(_WORDS), size=k)]
        out.add(" ".join(words))
        guard += 1
    if len(out) < n:
        raise AssertionError("word pool too small for unique options")
    return sorted(out)


def _encode_ref(text: str, seed: str = DEFAULT_SEED, ngram: int = 3,
                dims: int = 2048) -> np.ndarray:
    """Independent reference for the text->HV recipe (the bit-exact target).

    normalize: f"  {text.strip().lower()}  "; char n-grams over the padded
    string; bundle = per-bit majority of atom(seed, gram, dims); empty -> zeros.
    """
    padded = f"  {text.strip().lower()}  "
    if not padded.strip():
        return np.zeros(dims // 8, dtype=np.uint8)
    grams = [padded[i:i + ngram] for i in range(0, max(len(padded) - ngram + 1, 1))]
    vecs = [atom(seed, g, dims) for g in grams]
    if len(vecs) == 1:
        return vecs[0]
    bits = np.unpackbits(np.stack(vecs), axis=1, bitorder="big")
    return pack_bits(bits.sum(axis=0) > len(vecs) / 2)


def _swap_two_chars(text: str, rng: np.random.Generator) -> str:
    """Swap one adjacent distinct char pair; guaranteed to change the string."""
    for _ in range(32):
        i = int(rng.integers(0, len(text) - 1))
        if text[i] != text[i + 1]:
            new = text[:i] + text[i + 1] + text[i] + text[i + 2:]
            if new != text:
                return new
    raise AssertionError("no adjacent distinct pair to swap")


@pytest.fixture(scope="module")
def fx() -> Fixture:
    """200 seeded options + a 2-char-swapped query with an unambiguous winner.

    Selection criteria (frozen seeds, deterministic): the query retrieves its
    own source at full D with margin >= 0.02*2048 bits, the prefix-top-64
    candidate set has pairwise-distinct full-D distances, and the full-D
    top-16 lies inside that candidate set — i.e. no near-ties anywhere the
    tests below compare.
    """
    options = _make_options(200, seed=0)
    matrix = np.stack([_encode_ref(o) for o in options])
    for src in range(len(options)):
        query = _swap_two_chars(options[src], np.random.default_rng(100 + src))
        q = _encode_ref(query)
        dists = hamming_counts(q, matrix)
        order = np.argsort(dists, kind="stable")
        prefix64 = np.argsort(hamming_counts(q[:128], matrix[:, :128]),
                              kind="stable")[:64]
        if (int(order[0]) == src
                and int(dists[order[1]]) - int(dists[order[0]]) >= int(0.02 * 2048)
                and {int(i) for i in order[:16]} <= {int(i) for i in prefix64}):
            return Fixture(options, OptionIndex.fit(options), query, src, matrix)
    raise AssertionError("no source option with an unambiguous perturbed query")


def test_two_stage_matches_fulldim(fx):
    """Two-stage (prefix-k scan -> top-M full-D re-rank) == brute-force full-D rank."""
    labels, scores, meta = fx.idx.rank(fx.query)  # defaults k=1024, M=64
    dists = hamming_counts(_encode_ref(fx.query), fx.matrix)
    full_labels = [fx.options[int(i)] for i in np.argsort(dists, kind="stable")]
    assert labels[0] == full_labels[0]        # same top-1
    assert labels[:16] == full_labels[:16]    # top-16 order identical: 0 flips
    assert meta["final_top1"] == full_labels[0]


def test_prefix_recall_at_M(fx):
    """Full-D top-1 lies inside the stage-1 prefix top-64: recall@64 = 1.0."""
    q = _encode_ref(fx.query)
    full_top1 = int(np.argmin(hamming_counts(q, fx.matrix)))
    prefix_order = np.argsort(hamming_counts(q[:128], fx.matrix[:, :128]),
                              kind="stable")
    top64 = {fx.options[int(i)] for i in prefix_order[:64]}
    assert fx.options[full_top1] in top64
    labels, scores, meta = fx.idx.rank(fx.query)
    assert set(labels) == top64  # rank returns exactly the stage-1 top-M set


def test_scores_descending_in_unit_range(fx):
    """scores = 1 - hamming_full/2048 as float32, strictly descending, in [0,1]."""
    labels, scores, meta = fx.idx.rank(fx.query)
    assert scores.dtype == np.float32
    assert scores.shape == (len(labels),)
    assert np.all(np.diff(scores) <= 0)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    # Score recipe pinned exactly: (2048-c)/2048 is exact in float32.
    dists = hamming_counts(_encode_ref(fx.query), fx.matrix)
    assert float(scores[0]) == 1.0 - int(dists.min()) / 2048


def test_degenerate_inputs(fx):
    """Empty query, single-option index, and clamped k/M all behave."""
    # Empty text -> zero HV (recipe rule); rank does not crash.
    assert np.array_equal(fx.idx.encode(""), np.zeros(256, dtype=np.uint8))
    labels, scores, meta = fx.idx.rank("", k=9999, M=9999)
    assert len(labels) == len(fx.options) == 200  # clamped M=n -> all options
    assert meta["k"] == 2048 and meta["M"] == 200
    assert set(labels) == set(fx.options)
    labels2, scores2, meta2 = fx.idx.rank("")     # default k/M: safe
    assert len(labels2) == 64 and len(scores2) == 64
    # Single-option index: returned with margin 0.0 and exact score 1.0.
    solo = OptionIndex.fit(["refund issued twice march"])
    labs, sc, m = solo.rank("refund issued twice march")
    assert labs == ["refund issued twice march"]
    assert sc.shape == (1,) and float(sc[0]) == 1.0
    assert m["margin"] == 0.0 and m["near_tie"] is True
    # k > dims_full and M > n clamp without error on a normal query too.
    labs3, sc3, m3 = fx.idx.rank(fx.query, k=9999, M=9999)
    assert m3["k"] == 2048 and m3["M"] == 200 and len(labs3) == 200
