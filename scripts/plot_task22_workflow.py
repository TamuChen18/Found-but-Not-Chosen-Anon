#!/usr/bin/env python3
"""
Task 22 schematic: reference workflow vs the modal GPT-4o-mini BoN error (final no-seed pool).
Both panels share the node layout; only the Step-2 input edge differs. Vote counts come from
artifacts/selection_rebuild/task22.json.

Output: artifacts/final_primary/figures/task22_workflow.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "final_primary" / "figures"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

NODES = {
    0: ((0.50, 0.74), "Step 0", "Temporal_High_Resolution_\nImage_Generation"),
    1: ((0.23, 0.25), "Step 1", "Geospatial_Object_\nSegmentation"),
    2: ((0.77, 0.25), "Step 2", "Landslide_Segmentation"),
}
W, H = 0.40, 0.25
GOOD, BAD, GREY = "#2b6cb0", "#c53030", "#a0aec0"


def node(ax, k, highlight=False):
    (x, y), step, name = NODES[k]
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc="#fff5f5" if highlight else "#f7fafc",
                                ec=BAD if highlight else "#4a5568", lw=1.1))
    ax.text(x, y + 0.065, step, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(x, y - 0.035, name, ha="center", va="center", fontsize=6.2, family="DejaVu Sans Mono", linespacing=1.1)


def edge(ax, a, b, color, label, *, style="-", lw=1.4, label_xy=None, rot=0, cross=False):
    (xa, ya), (xb, yb) = NODES[a][0], NODES[b][0]
    if a == 0:
        start = (xa + (xb - xa) * 0.30, ya - H / 2)
        end = (xb + (xa - xb) * 0.12, yb + H / 2)
    else:
        start, end = (xa + W / 2, ya), (xb - W / 2, yb)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=color,
                                 lw=lw, linestyle=style, shrinkA=2, shrinkB=2))
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=5.8, family="DejaVu Sans Mono", color=color,
                rotation=rot, bbox=dict(fc="white", ec="none", pad=0.6))
    if cross:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my, "\u2715", ha="center", va="center", fontsize=9, color=BAD, fontweight="bold")


def main():
    t22 = json.loads((ROOT / "artifacts" / "selection_rebuild" / "task22.json").read_text())["mini"]["BoN"]
    wrong, total = t22["landslide_miswired_votes"], t22["n_valid"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.06)
    titles = [("Reference workflow (correct)", "#1a202c"),
              (f"Modal BoN error: structurally valid, incorrect ({wrong}/{total} votes)", BAD)]
    for ax, (title, col) in zip(axes, titles):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=8.5, color=col, pad=3)
        ax.add_patch(FancyBboxPatch((0.005, 0.005), 0.99, 0.99, boxstyle="round,pad=0,rounding_size=0.02",
                                    fc="none", ec="#cbd5e0", lw=0.8, transform=ax.transAxes))

    ax = axes[0]
    for k in NODES:
        node(ax, k)
    edge(ax, 0, 1, GOOD, "high_res_image_T_prime_path", label_xy=(0.27, 0.50))
    edge(ax, 0, 2, GOOD, "high_res_image_T_prime_path", label_xy=(0.73, 0.50))

    ax = axes[1]
    for k in NODES:
        node(ax, k, highlight=(k == 2))
    edge(ax, 0, 1, GOOD, "high_res_image_T_prime_path", label_xy=(0.27, 0.50))
    edge(ax, 0, 2, GREY, None, style=(0, (3, 2)), lw=1.0, cross=True)
    edge(ax, 1, 2, BAD, None, lw=1.6)
    ax.text(0.5, 0.065, "segmentation_map_image_path", ha="center", va="center", fontsize=5.8,
            family="DejaVu Sans Mono", color=BAD)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"task22_workflow.{ext}", dpi=300)
    print("wrote", OUT / "task22_workflow.pdf")


if __name__ == "__main__":
    main()
