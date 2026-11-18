"""Sync the Kaggle bench bundle from the package sources.

The pushed bundle carries its own copy of `hd_laya/` and of the kernel code, so
editing the package without re-syncing silently benchmarks an older version --
which is exactly how a run once scored a placeholder agent for 20 minutes.

Run before every `kaggle kernels push -p kaggle/bench_bundle`.
"""
from __future__ import annotations

import glob
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "kaggle", "bench_bundle")
PKG = os.path.join(ROOT, "hd_laya")


def main() -> int:
    os.makedirs(os.path.join(BUNDLE, "hd_laya"), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "kaggle", "bench_hd_laya.py"),
                 os.path.join(BUNDLE, "bench_hd_laya.py"))
    for path in glob.glob(os.path.join(PKG, "*.py")):
        shutil.copy2(path, os.path.join(BUNDLE, "hd_laya", os.path.basename(path)))
    for cache in glob.glob(os.path.join(BUNDLE, "**", "__pycache__"), recursive=True):
        shutil.rmtree(cache, ignore_errors=True)
    for junk in ("out", "hf_cache", "checkpoint"):
        shutil.rmtree(os.path.join(BUNDLE, junk), ignore_errors=True)

    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(BUNDLE, "hd_laya", "*.py")))
    print("bundle synced: %d package files, kernel %d bytes"
          % (len(files), os.path.getsize(os.path.join(BUNDLE, "bench_hd_laya.py"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
