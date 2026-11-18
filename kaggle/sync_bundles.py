"""Sync the Kaggle bundles before a push.

Kaggle uploads only the kernel's code file, so each bundle holds its kernel and
metadata and nothing else -- the package arrives by pip at the ref baked into
the kernel. This script writes the current commit into that ref, so a pushed
kernel always installs the code it was built from. Keeping a copy of the package
in the bundle is how one pushed kernel ran a stale version for 20 minutes.

Run before `kaggle kernels push -p kaggle/<bundle>`.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLES = (("bench_bundle", "bench_hd_laya.py"), ("train_bundle", "train_real.py"))
REF_LINE = re.compile(r'^DEFAULT_REF = "[0-9a-f]{7,40}".*$', re.MULTILINE)


def head_ref() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def main() -> int:
    ref = head_ref()
    for bundle, kernel in BUNDLES:
        src = os.path.join(ROOT, "kaggle", kernel)
        dst_dir = os.path.join(ROOT, "kaggle", bundle)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, kernel)

        text = open(src, encoding="utf-8").read()
        if not REF_LINE.search(text):
            raise SystemExit(f"{kernel}: no DEFAULT_REF line to pin")
        text = REF_LINE.sub(f'DEFAULT_REF = "{ref}"  # pinned by sync_bundles.py', text)
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

        shutil.rmtree(os.path.join(dst_dir, "hd_laya"), ignore_errors=True)
        for junk in ("out", "hf_cache", "checkpoint", "__pycache__"):
            shutil.rmtree(os.path.join(dst_dir, junk), ignore_errors=True)
        print(f"{bundle}: {kernel} pinned to {ref} ({len(text)} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
