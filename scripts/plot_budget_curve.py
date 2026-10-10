#!/usr/bin/env python3
"""Plot Oracle / Selected vs actual tokens per task.

  --mode prefix  search-budget scaling: BoN / CoT-tree truncated to their first N calls
                 (artifacts/budget_curve/search_budget_prefix.csv), plus complete MO-Tree runs
                 (GPT-4o-mini locked run; Qwen batch 20260725 narrow / medium / original width)
  --mode random  candidate-budget / subsampling sensitivity (candidate_subsampling.csv)
Qwen panel uses batch 20260725 so that it matches the breadth runs.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "budget_curve"
PANELS = {"GPT-4o-mini": "mini", "Qwen3.5-9B (Run 1)": "qwen_b1"}
MO_RUNS = {
    "mini": [("locked", ROOT / "artifacts/phase2_fulltest/FINAL_LOCKED/review_bundle")],
    "qwen_b1": [("CoT 8 / DC 5 (locked)", ROOT / "artifacts/qwen_nothink_main183/eval/b20260725"),
                ("CoT 12 / DC 8", ROOT / "artifacts/qwen_nothink_breadth/medium/eval"),
                ("CoT 16 / DC 10", ROOT / "artifacts/qwen_nothink_breadth/orig/eval")],
}


FINAL_OUT = ROOT / "artifacts" / "final_primary" / "budget_curve"
FINAL_MO_RUNS = {
    "mini": [("no-seed", ROOT / "artifacts/mini_noseed_trees_full/eval_e2b_off")],
    "qwen_b1": [("Narrow", ROOT / "artifacts/e2b_off_primary/qwen_b1"),
                ("Medium", ROOT / "artifacts/final_primary/breadth/medium"),
                ("Default", ROOT / "artifacts/final_primary/breadth/orig")],
}


def mo_points(run, mo_runs=MO_RUNS):
    pts = []
    for label, bundle in mo_runs[run]:
        rs = [x for x in csv.DictReader((bundle / "per_task.csv").open()) if x["pool"] == "Five"]
        pts.append((sum(float(x["total_tokens"]) for x in rs) / len(rs) / 1000,
                    sum(x["oracle"] == "True" for x in rs),
                    sum(x["filter_majority_correct"] == "True" for x in rs), label))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prefix", "random"], default="prefix")
    ap.add_argument("--final", action="store_true", help="final primary pools (no-seed GPT-4o-mini, E2b off)")
    ap.add_argument("--panel", choices=["both", "gpt"], default="both", help="gpt: GPT-4o-mini panel only (*_gpt)")
    args = ap.parse_args()
    out, mo_runs = (FINAL_OUT, FINAL_MO_RUNS) if args.final else (OUT, MO_RUNS)
    src = out / ("search_budget_prefix.csv" if args.mode == "prefix" else "candidate_subsampling.csv")
    rows = list(csv.DictReader(src.open()))
    panels = PANELS if args.panel == "both" else dict(list(PANELS.items())[:1])
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.4) if len(panels) == 2 else (5.8, 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    colors = {"BoN": "tab:blue", "CoT": "tab:orange", "PMBoN": "tab:purple"}
    names = {"BoN": "BoN", "CoT": "CoT-tree", "PMBoN": "Prompt-matched BoN"}
    for ax, (title, run) in zip(axes, panels.items()):
        for pool in ("BoN", "CoT", "PMBoN"):
            pts = sorted((float(r["tokens_per_task"]) / 1000, float(r["oracle"]), float(r["selected"]))
                         for r in rows if r["run"] == run and r["pool"] == pool)
            if not pts:
                continue
            name = names[pool]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=colors[pool], label=f"{name} Oracle")
            ax.plot([p[0] for p in pts], [p[2] for p in pts], "--s", color=colors[pool], label=f"{name} Selected")
        mo = mo_points(run, mo_runs)
        ax.plot([p[0] for p in mo], [p[1] for p in mo], "-^", color="tab:green", ms=8, label="MO-Tree Oracle")
        ax.plot([p[0] for p in mo], [p[2] for p in mo], "--v", color="tab:green", ms=8, mfc="none",
                label="MO-Tree Selected")
        if len(mo) > 1:
            for i, (x, o, _, label) in enumerate(mo):
                ax.annotate(label, (x, o), textcoords="offset points", xytext=(0, 7 if i % 2 == 0 else -13),
                            ha="center", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("realized tokens per task (thousands)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("tasks correct (of 183)")
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = (len(labels) + 1) // 2 if len(panels) == 2 else 2
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=ncol, frameon=False)
    fig.tight_layout(rect=(0, 0.11 if len(panels) == 2 else 0.16, 1, 1))
    name = "search_budget_prefix.png" if args.mode == "prefix" else "candidate_subsampling.png"
    if args.panel == "gpt":
        name = name.replace(".png", "_gpt.png")
    fig.savefig(out / name, dpi=150)
    if args.final:
        fig.savefig(out / name.replace(".png", ".pdf"))
    print("wrote", out / name)


if __name__ == "__main__":
    main()
