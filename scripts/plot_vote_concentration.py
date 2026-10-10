#!/usr/bin/env python3
"""Descriptive scatter of mean winning-vote share vs Oracle-Selected gap for the nine final primary
pools (E4 off, E2b off), from artifacts/final_primary/concentration/summary.csv. No fitted line or
correlation coefficient. Writes vote_concentration_gap.{png,pdf} next to the summary."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "final_primary" / "concentration" / "summary.csv"
MODELS = {"mini": ("GPT-4o-mini", "o"), "qwen_b1": ("Qwen Run 1", "s"), "qwen_b2": ("Qwen Run 2", "D")}
METHODS = {"BoN": ("BoN", "tab:blue"), "CoT": ("CoT-tree", "tab:orange"), "Five": ("MO-Tree", "tab:green")}


def main():
    rows = list(csv.DictReader(SRC.open()))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for r in rows:
        model, marker = MODELS[r["bundle"]]
        method, color = METHODS[r["pool"]]
        ax.scatter(float(r["win_share_mean"]), int(r["oracle"]) - int(r["selected"]), marker=marker, color=color,
                   s=45, edgecolor="black", linewidth=0.5)
    for model, marker in MODELS.values():
        ax.scatter([], [], marker=marker, color="white", edgecolor="black", label=model)
    for method, color in METHODS.values():
        ax.scatter([], [], marker="o", color=color, label=method)
    ax.set_xlabel("mean winning-vote share")
    ax.set_ylabel("Oracle $-$ Selected (tasks)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(SRC.parent / f"vote_concentration_gap.{ext}", dpi=150)
    print("wrote", SRC.parent / "vote_concentration_gap.png")


if __name__ == "__main__":
    main()
