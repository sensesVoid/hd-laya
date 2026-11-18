"""Publish helper for hd-laya v1.0 (T14): HF + Kaggle.

Skeleton: documents the exact commands; real runs require trained artifacts.
HF:    hf upload geraldaton20/hd-laya <dir> .  (needs `hf auth login`)
Kaggle: kaggle datasets create / kaggle models create (needs ~/.kaggle/kaggle.json)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def publish_hf(artifact_dir: str, repo: str = "geraldaton20/hd-laya") -> None:
    print(f"[HF] uploading {artifact_dir} -> {repo}")
    subprocess.run(["hf", "upload", repo, artifact_dir, "."], check=True)


def publish_kaggle(artifact_dir: str, slug: str = "sensesvoid/hd-laya") -> None:
    print(f"[Kaggle] publishing {artifact_dir} -> {slug}")
    # Real flow: kaggle datasets create -p <dir> ; or kaggle models create ...
    subprocess.run(["kaggle", "datasets", "create", "-p", artifact_dir], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--target", choices=["hf", "kaggle", "both"], default="both")
    args = ap.parse_args()
    if args.target in ("hf", "both"):
        publish_hf(args.artifacts)
    if args.target in ("kaggle", "both"):
        publish_kaggle(args.artifacts)


if __name__ == "__main__":
    main()
