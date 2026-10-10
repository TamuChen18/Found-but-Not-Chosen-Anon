#!/usr/bin/env python3
"""
Figure 1 result panels, offline from the 11 final pools (9 primary + GPT/Qwen Prompt-matched BoN).
No LLM calls.

Groups = exact-JSON workflows among valid (hard0) candidates in selector order (votes desc, smallest
candidate_id first), as in analyze_vote_rank.py. Strict vote losses come from
analyze_modal_errors.load_cases.

  fig1_attribution.csv       Oracle of CoT-tree vs Prompt-matched BoN (GPT; Qwen Run 1 / Run 2)
  fig1_validation.csv        Oracle-positive pool-tasks before (any normalized candidate) / after (hard0)
                             structural validation
  fig1_loss_localization.csv filter / vote / tie over the Oracle - Selected gap
  fig1_topk.csv              Top-k oracle (correct group among the first k groups), k = 1..K_MAX and Oracle
  fig1_topk_by_pool.csv      the same per pool
  fig1_diversity.csv         per-pool median / mean distinct valid groups, all tasks vs strict vote-loss tasks
                             (11 pools: 9 primary + GPT/Qwen Prompt-matched BoN)
  fig1_diversity_tasks.csv   per (pool, task) raw group counts, same 11 pools
  fig1_topk_by_model.csv     the same pooled within GPT-4o-mini (4 pools) and Qwen3.5-9B (7 pools)
  fig1_1_attribution, fig1_2_validation, fig1_3_loss_localization, fig1_4_topk_recovery, fig1_5_diversity
                             .{pdf,png}: equal size, no titles, rounded frame

Output: artifacts/final_primary/fig1/
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_modal_errors as amc  # noqa: E402
from analyze_vote_rank import BUNDLES, key  # noqa: E402

OUT = ROOT / "artifacts" / "final_primary" / "fig1"
K_MAX = 10
MODEL = {"mini": "GPT-4o-mini", "qwen_b1": "Qwen3.5-9B", "qwen_b2": "Qwen3.5-9B", "qwen": "Qwen3.5-9B"}
RUN = {"mini": "primary", "qwen_b1": "Run 1", "qwen_b2": "Run 2", "qwen": "control"}
SHORT = {"BoN": "BoN", "CoT-tree": "CoT", "MO-Tree": "MO", "Prompt-matched BoN": "Prompt"}

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})


def wcsv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def pool_label(run, method):
    m = "GPT" if run == "mini" else "Qwen"
    r = {"qwen_b1": " R1", "qwen_b2": " R2"}.get(run, "")
    return f"{m}{r} {SHORT[method]}"


def collect():
    pools, tasks = [], []
    for run, bundle, names in BUNDLES:
        b = ROOT / bundle
        amc.METHODS = names
        cases, _ = amc.load_cases(b, run)
        cands = defaultdict(list)
        for line in (b / "per_candidate.jsonl").open():
            c = json.loads(line)
            if c["pool"] in names:
                cands[(c["pool"], int(c["task_id"]))].append(c)
        pt = {(r["pool"], int(r["task_id"])): r for r in csv.DictReader((b / "per_task.csv").open()) if r["pool"] in names}
        stage = {r["method"]: r for r in csv.DictReader(
            (ROOT / ("artifacts/selection_rebuild/stage_loss.csv" if names != {"PMBoN": "Prompt-matched BoN"} else
                     f"{bundle}/stage_loss.csv")).open())
            if r["run"] == (run if run != "qwen" else "qwen_b1")}
        for pool, name in names.items():
            tids = sorted(t for (p, t) in pt if p == pool)
            assert len(tids) == 183
            topk = Counter()
            for tid in tids:
                r = pt[(pool, tid)]
                selected = r["filter_majority_correct"] == "True"
                valid = [c for c in cands[(pool, tid)] if c["hard0"]]
                votes, first, correct = Counter(), {}, {}
                for c in valid:
                    kk = key(c["plan"])
                    votes[kk] += 1
                    first[kk] = min(first.get(kk, 10 ** 9), int(c["candidate_id"]))
                    correct[kk] = c["correct"]
                order = sorted(votes, key=lambda kk: (-votes[kk], first[kk]))
                crank = next((i + 1 for i, kk in enumerate(order) if correct[kk]), None)
                if order:
                    assert bool(correct[order[0]]) == selected, (run, pool, tid)
                assert (crank is not None) == (r["oracle"] == "True"), (run, pool, tid)
                for k in range(1, K_MAX + 1):
                    topk[k] += (crank is not None and crank <= k) if order else selected
                tasks.append({"model": MODEL[run], "run": RUN[run], "procedure": name, "pool": pool_label(run, name),
                              "task_id": tid, "is_vote_loss": int((pool, tid) in cases),
                              "num_distinct_valid_workflow_groups": len(order), "best_correct_rank": crank,
                              "selected_correct": int(selected),
                              "oracle_prevalidation": int(any(c["correct"] for c in cands[(pool, tid)])),
                              "oracle": int(r["oracle"] == "True")})
            st = stage[name]
            sel = sum(t["selected_correct"] for t in tasks[-183:])
            orc = sum(t["oracle"] for t in tasks[-183:])
            assert (sel, orc) == (int(st["selected"]), int(st["oracle"])), (run, name)
            assert topk[1] == sel
            pools.append({"run": run, "procedure": name, "pool": pool_label(run, name), "selected": sel, "oracle": orc,
                          "oracle_prevalidation": sum(t["oracle_prevalidation"] for t in tasks[-183:]),
                          "filter": int(st["filter_loss"]), "vote": int(st["vote_loss"]), "tie": int(st["tie_loss"]),
                          **{f"top{k}": topk[k] for k in range(1, K_MAX + 1)}})
    assert len(pools) == 11 and len(tasks) == 11 * 183
    return pools, tasks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pools, tasks = collect()
    P = {(p["run"], p["procedure"]): p for p in pools}

    attr = []
    for run, proc in (("mini", "CoT-tree"), ("mini", "Prompt-matched BoN"), ("qwen_b1", "CoT-tree"),
                      ("qwen_b2", "CoT-tree"), ("qwen", "Prompt-matched BoN")):
        o = P[(run, proc)]["oracle"]
        attr.append({"model": MODEL[run], "procedure": "Prompt-matched" if proc != "CoT-tree" else proc,
                     "run": "control" if proc != "CoT-tree" else RUN[run], "oracle_count": o, "n": 183, "oracle_pct": round(100 * o / 183, 1)})
    wcsv("fig1_attribution.csv", attr)

    n = 11 * 183
    before, after = sum(p["oracle_prevalidation"] for p in pools), sum(p["oracle"] for p in pools)
    wcsv("fig1_validation.csv", [
        {"stage": "Before validation", "correct_workflow_available": before, "no_correct_workflow": n - before},
        {"stage": "After validation", "correct_workflow_available": after, "no_correct_workflow": n - after}])

    gap = sum(p["oracle"] - p["selected"] for p in pools)
    loss = {k: sum(p[k] for p in pools) for k in ("filter", "vote", "tie")}
    assert sum(loss.values()) == gap
    wcsv("fig1_loss_localization.csv",
         [{"loss_type": f"{k.capitalize()} loss", "count": v, "pct": round(100 * v / gap, 1)} for k, v in loss.items()]
         + [{"loss_type": "Total", "count": gap, "pct": 100.0}])

    sel, orc = sum(p["selected"] for p in pools), sum(p["oracle"] for p in pools)
    topk = [{"k": k, "correct_count": sum(p[f"top{k}"] for p in pools),
             "recovered_loss": sum(p[f"top{k}"] for p in pools) - sel} for k in range(1, K_MAX + 1)]
    topk.append({"k": "Oracle", "correct_count": orc, "recovered_loss": gap})
    for t in topk:
        t["recovery_pct"] = round(100 * t["recovered_loss"] / gap, 1)
    wcsv("fig1_topk.csv", topk)
    wcsv("fig1_topk_by_pool.csv", [{k: v for k, v in p.items() if k not in ("filter", "vote", "tie", "oracle_prevalidation")}
                                   for p in pools])
    by_model = []
    for model, runs in (("GPT-4o-mini", ("mini",)), ("Qwen3.5-9B", ("qwen_b1", "qwen_b2", "qwen"))):
        ps = [p for p in pools if p["run"] in runs]
        s_, g_ = sum(p["selected"] for p in ps), sum(p["oracle"] - p["selected"] for p in ps)
        for k in list(range(1, K_MAX + 1)) + ["Oracle"]:
            c = sum(p["oracle"] if k == "Oracle" else p[f"top{k}"] for p in ps)
            by_model.append({"model": model, "n_pools": len(ps), "gap": g_, "k": k, "correct_count": c,
                             "recovered_loss": c - s_, "recovery_pct": round(100 * (c - s_) / g_, 1)})
    wcsv("fig1_topk_by_model.csv", by_model)
    beyond = [t for t in tasks if t["best_correct_rank"] and t["best_correct_rank"] > K_MAX]
    print(f"pool-tasks whose best correct group ranks > {K_MAX}: {len(beyond)}")

    div_pools, div_tasks = pools, tasks
    wcsv("fig1_diversity_tasks.csv", [{"model_run": t["model"] + ("" if t["run"] in ("primary", "control") else
                                                                   " " + t["run"]),
                                       "procedure": t["procedure"], "pool": t["pool"], "task_id": t["task_id"],
                                       "is_vote_loss": t["is_vote_loss"],
                                       "num_distinct_valid_workflow_groups": t["num_distinct_valid_workflow_groups"]}
                                      for t in div_tasks])
    div = []
    for p in div_pools:
        ts = [t for t in div_tasks if t["pool"] == p["pool"]]
        a = [t["num_distinct_valid_workflow_groups"] for t in ts]
        v = [t["num_distinct_valid_workflow_groups"] for t in ts if t["is_vote_loss"]]
        assert len(v) == p["vote"]
        div.append({"pool": p["pool"].replace(" ", "_"), "all_task_median": statistics.median(a),
                    "vote_loss_median": statistics.median(v), "n_tasks": len(a), "n_vote_loss": len(v),
                    "all_task_mean": round(statistics.fmean(a), 3), "vote_loss_mean": round(statistics.fmean(v), 3)})
    wcsv("fig1_diversity.csv", div)

    for name, rows in (("attribution", attr), ("topk", topk), ("diversity", div)):
        print(f"\n{name}")
        for r in rows:
            print("  ", r)
    print(f"\nvalidation: before {before} after {after} (of {n}); loss {loss} gap {gap}")

    plot(attr, (before, after, n), loss, topk, by_model, div)


COT, PM, GPT_C, QWEN_C = "#2b6cb0", "#dd6b20", "#2b6cb0", "#c05621"
VOTE_C, TIE_C = "#c53030", "#a0aec0"
SIZE = (3.6, 2.8)
NAMES = ("fig1_1_attribution", "fig1_2_validation", "fig1_3_loss_localization", "fig1_4_topk_recovery",
         "fig1_5_diversity")


def new_fig():
    fig, ax = plt.subplots(figsize=SIZE, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.10, h_pad=0.10)
    return fig, ax


def save(fig, name):
    fig.patch.set_alpha(0)
    fig.patches.append(FancyBboxPatch((0.004, 0.005), 0.992, 0.99, boxstyle="round,pad=0,rounding_size=0.05",
                                      mutation_aspect=SIZE[0] / SIZE[1], transform=fig.transFigure,
                                      fc="white", ec="#cbd5e0", lw=0.8, zorder=-10))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None, facecolor="none")
    plt.close(fig)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def plot(attr, validation, loss, topk, by_model, div):
    for old in list(OUT.glob("fig1[a-d]_*.p*")) + list(OUT.glob("fig1_panels.*")):
        old.unlink()

    fig, ax = new_fig()
    xs = [0, 1, 2.6, 3.6, 4.6]
    order = [attr[0], attr[1], attr[2], attr[4], attr[3]]
    labels = ["CoT-tree", "Prompt-\nmatched", "CoT-tree\nRun 1", "Prompt-\nmatched", "CoT-tree\nRun 2"]
    for x, r in zip(xs, order):
        cot = r["procedure"] == "CoT-tree"
        ax.scatter(x, r["oracle_pct"], s=42, color=COT if cot else PM, marker="o" if cot else "D", zorder=3)
        ax.text(x, r["oracle_pct"] + 0.45, f"{r['oracle_count']}\n({r['oracle_pct']:.1f})", ha="center",
                va="bottom", fontsize=6.6, linespacing=1.0)
    ax.axvline(1.8, color="#e2e8f0", lw=0.8)
    ax.set_xticks(xs, labels, fontsize=6.6)
    ax.set_xlim(-0.6, 5.2)
    ax.set_ylim(80, 92)
    ax.set_ylabel("Oracle EM (%)")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.set_xlabel("GPT-4o-mini" + " " * 30 + "Qwen3.5-9B", fontsize=7.5, fontweight="bold")
    clean(ax)
    save(fig, NAMES[0])

    before, after, n = validation
    fig, ax = new_fig()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [(0.84, f"Oracle-positive before validation\n{before}", "#ebf8ff", COT),
             (0.50, "Structural validation\n(hard-constraint filter)", "#f7fafc", "#4a5568"),
             (0.16, f"Oracle-positive after validation\n{after}", "#ebf8ff", COT)]
    for y, text, fc, ec in boxes:
        ax.add_patch(FancyBboxPatch((0.17, y - 0.11), 0.66, 0.22, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=fc, ec=ec, lw=1.0))
        first, second = text.split("\n")
        big = second.isdigit()
        ax.text(0.5, y + (0.045 if big else 0.04), first, ha="center", va="center", fontsize=7.2)
        ax.text(0.5, y - (0.045 if big else 0.04), second, ha="center", va="center",
                fontsize=11 if big else 6.6, fontweight="bold" if big else "normal", color=ec if big else "#4a5568")
    for y0, y1 in ((0.73, 0.61), (0.39, 0.27)):
        ax.annotate("", (0.5, y1), (0.5, y0), arrowprops={"arrowstyle": "-|>", "color": "#4a5568", "lw": 1.1})
    ax.text(0.87, 0.50, f"Filter loss\n= {before - after}", ha="left", va="center", fontsize=8, fontweight="bold",
            color=VOTE_C)
    ax.text(0.5, -0.02, f"{n} pool-tasks (11 pools × 183)", ha="center", va="top", fontsize=6.4, color="#718096")
    save(fig, NAMES[1])

    fig, ax = new_fig()
    vals, cols = [loss["vote"], loss["tie"]], [VOTE_C, TIE_C]
    tot = sum(vals)
    wedges, _ = ax.pie(vals, colors=cols, startangle=90, counterclock=False, radius=1.0,
                       wedgeprops={"width": 0.40, "edgecolor": "white"})
    for w, v, inside in zip(wedges, vals, (True, False)):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        r = 0.80 if inside else 1.24
        ax.text(r * np.cos(ang), r * np.sin(ang), f"{v}\n{100 * v / tot:.1f}%", ha="center", va="center",
                fontsize=8 if inside else 7, fontweight="bold", color="white" if inside else "#4a5568",
                linespacing=1.0)
    ax.text(0, 0.07, f"{tot}", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0, -0.17, "lost cases", ha="center", va="center", fontsize=7)
    ax.legend(wedges, ["Vote loss", "Tie loss"], loc="center left", bbox_to_anchor=(1.02, 0.6), fontsize=7.5,
              frameon=False)
    ax.text(1.0, 0.30, f"Filter loss = {loss['filter']}", transform=ax.transAxes, ha="left", fontsize=7.5,
            fontweight="bold")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    save(fig, NAMES[2])

    fig, ax = new_fig()
    ks = [t["k"] for t in topk]
    xs = list(range(len(ks)))
    for model, color, marker, off in (("GPT-4o-mini", GPT_C, "o", (5, -11)), ("Qwen3.5-9B", QWEN_C, "s", (-20, 5))):
        rs = [r for r in by_model if r["model"] == model]
        ys = [r["recovery_pct"] for r in rs]
        ax.plot(xs[:-1], ys[:-1], "-", marker=marker, color=color, ms=3.2, lw=1.3,
                label=f"{model} ({rs[0]['n_pools']} pools, gap {rs[0]['gap']})")
        ax.plot([xs[-2], xs[-1]], [ys[-2], ys[-1]], ":", color=color, lw=1.1)
        for i in (1, 2):
            ax.annotate(f"{ys[i]:.1f}", (xs[i], ys[i]), textcoords="offset points", xytext=off,
                        fontsize=6.5, color=color)
    ax.plot(xs[-1], 100, "D", color="#2d3748", ms=3.5)
    ax.set_xticks(xs, [str(k) for k in ks], fontsize=6.6)
    ax.set_xlabel("Top-k groups (selector order)")
    ax.set_ylabel("Recovered gap (%)")
    ax.set_ylim(-3, 105)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    clean(ax)
    save(fig, NAMES[3])

    fig, ax = new_fig()
    ys = list(range(len(div)))[::-1]
    for y, d in zip(ys, div):
        ax.plot([d["all_task_median"], d["vote_loss_median"]], [y, y], color="#cbd5e0", lw=1.6, zorder=1)
        ax.scatter(d["all_task_median"], y, color="#718096", s=16, zorder=2)
        ax.scatter(d["vote_loss_median"], y, color=VOTE_C, s=16, zorder=2)
    ax.set_yticks(ys, [d["pool"].replace("_", " ") for d in div], fontsize=6.4)
    ax.set_xlabel("Median distinct valid workflows / task")
    ax.scatter([], [], color="#718096", s=16, label="All tasks")
    ax.scatter([], [], color=VOTE_C, s=16, label="Vote-loss tasks")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    ax.set_xlim(0, max(d["vote_loss_median"] for d in div) + 2)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    clean(ax)
    save(fig, NAMES[4])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
