"""Sync the Kaggle bench bundle before a push.

Kaggle uploads only the kernel's code file, so the bundle holds the kernel and
its metadata and nothing else -- the package arrives by pip at the ref baked
into the kernel (``HD_LAYA_REF``). Keeping a second copy of the package in the
bundle is how a pushed kernel once ran a stale version for 20 minutes.

Run before every `kaggle kernels push -p kaggle/bench_bundle`.
"""
from __future__ import annotations

import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "kaggle", "bench_bundle")


def main() -> int:
    os.makedirs(BUNDLE, exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "kaggle", "bench_hd_laya.py"),
                 os.path.join(BUNDLE, "bench_hd_laya.py"))
    shutil.rmtree(os.path.join(BUNDLE, "hd_laya"), ignore_errors=True)
    for junk in ("out", "hf_cache", "checkpoint"):
        shutil.rmtree(os.path.join(BUNDLE, junk), ignore_errors=True)

    kernel = os.path.join(BUNDLE, "bench_hd_laya.py")
    text = open(kernel, encoding="utf-8").read()
    ref = text.split('HD_LAYA_REF", "')[1].split('"')[0]
    print("bundle synced: kernel %d bytes, package ref %s" % (os.path.getsize(kernel), ref))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
