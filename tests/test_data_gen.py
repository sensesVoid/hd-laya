"""Tests for hd_laya.data_gen: high-K choice data generator (v1.0)."""
import numpy as np

from hd_laya.data_gen import K_VALUES, gen_choice_item, gen_dataset


def test_gen_choice_item_shape():
    rng = np.random.default_rng(0)
    item = gen_choice_item(rng, 50)
    assert isinstance(item["state"], str)
    assert "q" in item["questions"]
    crit = item["questions"]["q"]["criteria"]
    assert len(crit) == 50
    assert item["expected"] in crit
    assert item["questions"]["q"]["type"] == "choice"


def test_gen_dataset_count_and_K():
    items = gen_dataset(10, 77, seed=1)
    assert len(items) == 10
    assert all(len(it["questions"]["q"]["criteria"]) == 77 for it in items)


def test_gen_dataset_deterministic():
    a = gen_dataset(5, 20, seed=42)
    b = gen_dataset(5, 20, seed=42)
    assert a == b


def test_gen_dataset_guards():
    try:
        gen_dataset(0, 20)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n=0")
    try:
        gen_dataset(5, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for K=1")


def test_k_values_include_ceiling_breakers():
    assert 20 in K_VALUES
    assert 1000 in K_VALUES
    assert 2000 in K_VALUES
