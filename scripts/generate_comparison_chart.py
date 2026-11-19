"""Generate a comparison chart for the HF model card.

Usage:
    python scripts/generate_comparison_chart.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_PATH = os.path.join(ROOT, "hf_chart.png")


def load_results():
    # High-K stress suite (from Kaggle benchmark)
    high_k = {
        "K": [20, 77, 150, 500, 1000],
        "base": [0.473, 0.145, 0.051, 0.016, 0.008],
        "trained": [0.465, 0.266, 0.445, 0.805, 0.797],
    }
    # BANKING77 (from local run)
    banking77 = None
    bk_path = os.path.join(RESULTS_DIR, "banking77.json")
    if os.path.exists(bk_path):
        with open(bk_path) as f:
            banking77 = json.load(f)
    return high_k, banking77


def main():
    high_k, banking77 = load_results()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: High-K stress suite
    K = high_k["K"]
    x = np.arange(len(K))
    w = 0.35
    ax1.bar(x - w/2, high_k["base"], w, label="base laya", color="#999999")
    ax1.bar(x + w/2, high_k["trained"], w, label="hd-laya trained", color="#2196F3")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(k) for k in K])
    ax1.set_xlabel("K (number of options)")
    ax1.set_ylabel("accuracy")
    ax1.set_title("High-K stress suite (K options, 256 items each)")
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="chance (1/20)")

    # Plot 2: BANKING77 if available
    if banking77:
        models = ["base laya", "hd-laya"]
        accs = [banking77["base_accuracy"], banking77["hd_accuracy"]]
        ax2.bar(models, accs, color=["#999999", "#2196F3"])
        ax2.set_ylabel("accuracy")
        ax2.set_title(f"BANKING77 ({banking77['n_items']} items, {banking77['n_intents']} intents)")
        ax2.set_ylim(0, 1.0)
        ax2.text(0.5, max(accs) + 0.05, f"delta: {banking77['delta']:+.4f}",
                 ha="center", fontsize=11)
    else:
        ax2.text(0.5, 0.5, "BANKING77 results pending", ha="center", va="center")
        ax2.set_title("BANKING77 (pending)")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
