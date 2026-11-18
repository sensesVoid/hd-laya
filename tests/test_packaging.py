"""Tests for packaging + golden fixture (T8)."""
import json
import subprocess
import tomllib
from pathlib import Path

import numpy as np

from hd_laya.index import OptionIndex

ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_metadata():
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    proj = data["project"]
    assert proj["name"] == "hd-laya"
    assert proj["version"] == "0.1.0"
    assert proj["license"]["text"] == "Apache-2.0"
    assert proj["requires-python"] == ">=3.10"
    deps = proj["dependencies"]
    assert any(d.startswith("numpy") for d in deps)
    assert any(d.startswith("laya") for d in deps)
    assert "sem" in proj["optional-dependencies"]
    assert any(d.startswith("sentence-transformers") for d in proj["optional-dependencies"]["sem"])
    assert data["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_readme_only_md():
    out = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    files = [line for line in out.stdout.splitlines() if line.strip()]
    assert files == ["README.md"], files


def test_golden_fixture_reproduces():
    fixture = json.loads((ROOT / "tests" / "golden" / "two_stage_fixture.json").read_text(encoding="utf-8"))
    idx = OptionIndex.fit(fixture["options"], seed=fixture["seed"])
    for case, q in zip(fixture["cases"], fixture["queries"]):
        labels, scores, meta = idx.rank(q)
        assert labels[0] == case["top1"], (labels[0], case["top1"])
        assert [round(float(s), 6) for s in scores[:2]] == case["top2_scores"]
        assert round(float(meta["margin"]), 6) == case["margin"]
