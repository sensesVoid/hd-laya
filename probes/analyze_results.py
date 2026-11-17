"""Aggregate probe results into the sweet-spot decision tables.

Run after probes 1-4 have written their JSONs:
    python probes/analyze_results.py
Prints: (a) speed table, (b) matryoshka verdict vs frozen bars,
(c) two-stage operating points, (d) capacity table, (e) verdict lines.
"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=3):
    return "?" if x is None else f"{x:.{nd}f}"


def main() -> None:
    p1 = load("probe1_speed.json")
    p2 = load("probe2_matryoshka.json")
    p3 = load("probe3_semantic.json")
    p4 = load("probe4_capacity.json")

    if p1:
        print("== Probe 1: speed ==")
        print(f"{'D':>6} {'enc ms/text':>12} {'scan ms/10k':>12} {'scan kopts/s':>13} {'vec bytes':>10}")
        for r in p1["rows"]:
            print(f"{r['dims']:>6} {r['encode_ms_per_text']:>12.3f} {r['scan_ms_10k']:>12.2f} "
                  f"{r['scan_opts_per_s']/1e3:>13.0f} {r['bytes_per_vec']:>10}")
        print("scale (100k options):")
        for r in p1["scale_rows"]:
            print(f"  D={r['dims']:>6} scan_ms={r['scan_ms_100k']:>8.1f} kopts/s={r['scan_opts_per_s']/1e3:>8.0f} matrix={r['matrix_mb']:.0f}MB")
        print("clay int16 counter workspace:")
        for r in p1["clay_rows"]:
            print(f"  D={r['dims']:>6} write={r['write_ms']*1000:.0f}us activate={r['activate_ms']*1000:.0f}us "
                  f"decode8={r['decode_8roles_ms']:.1f}ms ws={r['workspace_kb']:.0f}KB")

    if p2:
        print("\n== Probe 2: matryoshka (lexical BSC) ==")
        print(f"{'config':>12} {'full acc':>9} {'k acc':>9} {'delta pp':>9} {'flip rate':>10} {'R@64':>7} {'2stage dis':>10}")
        for key, s in p2["summary"].items():
            print(f"{key:>12} {s['mean_full_acc']:>9.4f} {s['mean_k_acc']:>9.4f} "
                  f"{s['mean_acc_delta_pp']:>9.3f} {s['mean_flip_rate']:>10.4f} "
                  f"{s['mean_recall_at_64']:>7.4f} {s['mean_two_stage_disagree']:>10.5f}")
        print("bit-identity (P1):", p2["bit_identity"])
        # near-tie conditioning for the headline config
        rows = [r for r in p2["rows"] if r["dfull"] == 2048 and r["k"] == 1024]
        if rows:
            print("flips by full-D margin (1024of2048):")
            for b, v in rows[0]["flips_by_margin"].items():
                fr = v["flip_rate"]
                print(f"  margin {b}: n={v['n']} flip_rate={'?' if fr is None else f'{fr:.4f}'}")

    if p3:
        print("\n== Probe 3: semantic SimHash vs float truth ==")
        print(f"{'config':>8} {'agree(f)':>10} {'R@64':>7} {'2stage dis':>10} {'flips clear':>11}")
        for key, s in p3["summary"].items():
            print(f"{key:>8} {s['mean_bit_top1_vs_float_agree']:>10.4f} {s['mean_recall_at_64_vs_float']:>7.4f} "
                  f"{s['mean_two_stage_disagree']:>10.5f} {s['mean_flips_clear_margin']:>11.5f}")

    if p4:
        print("\n== Probe 4: bundling capacity ==")
        print(f"{'D':>6} {'max_n_err_free':>14} {'n@1pct':>7}")
        for key, s in p4["summary"].items():
            print(f"{key:>6} {s['max_n_error_free']:>14} {s['n_at_1pct']:>7}")
        print("error vs n (per D):")
        for dims in [1024, 4096, 16384]:
            r = [x for x in p4["rows"] if x["dims"] == dims]
            line = " ".join(f"n={x['n']}:{x['mean_error']:.3f}" for x in r)
            print(f"  D={dims:>6}: {line}")

    print("\n== Verdict hooks (fill after reading) ==")
    print(" - B1 (scan speed): see probe1 rows")
    print(" - B2 (flip <=1% clear margin): see probe2/probe3 flip tables")
    print(" - B3 (two-stage == full-D): see two_stage columns")
    print(" - B4 (encode 2x): compare probe1 encode at 1024 vs 2048")


if __name__ == "__main__":
    main()
