#!/usr/bin/env python3
"""
Margin-gated Top-3 override, offline from the deployable Top-3 selector (no new LLM calls).

For each of the 1,647 (pool, task) pairs of the 9 final primary pools, with valid exact-JSON groups in
selector order (votes desc, smallest candidate_id first) and v1 >= v2 the top two vote counts:
  absolute margin    d = v1 - v2
  normalized margin  m = (v1 - v2) / sum_j v_j     (sum over all valid groups of the pool-task)
Gated selector: use the cached GPT-4o Top-3 choice if the pool-task has >= 2 groups and margin <= t,
otherwise keep the majority pick. t = inf reproduces the verifier-only contested selector.

Also reports leave-one-run-out threshold selection (choose t maximizing net on the other two runs,
ties -> smallest t), so the gated result is not tuned on the evaluated run.

Output: artifacts/final_primary/margin_gated/{per_task.csv,sweep_normalized.csv,sweep_absolute.csv,
sweep_by_method.csv,lorocv.csv}
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_strict_matched import paired_stats  # noqa: E402
from run_top3_selector import load_items  # noqa: E402

SEL = ROOT / "artifacts" / "final_primary" / "top3_selector" / "per_task.csv"
OUT = ROOT / "artifacts" / "final_primary" / "margin_gated"
T_ABS = [0, 1, 2, 3, 4, 5, 8, 10, math.inf]
T_NORM = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, math.inf]
RUNS = ("mini", "qwen_b1", "qwen_b2")


def load():
    sel = {(r["run"], r["method"], int(r["task_id"])): r for r in csv.DictReader(SEL.open())}
    rows = []
    for it in load_items():
        s = sel[(it["run"], it["method"], it["task_id"])]
        assert (s["majority_ok"] == "True") == it["majority_ok"]
        top = it["top"]
        contested = s["mode"] == "verifier"
        assert contested == (it["n_groups"] >= 2)
        v1 = top[0]["votes"] if top else 0
        v2 = top[1]["votes"] if len(top) > 1 else 0
        rows.append({
            "run": it["run"], "method": it["method"], "task_id": it["task_id"],
            "oracle": it["oracle"], "majority_ok": it["majority_ok"], "n_groups": it["n_groups"],
            "contested": contested, "v1": v1, "v2": v2, "valid_votes": it["valid_votes"],
            "abs_margin": v1 - v2 if contested else None,
            "norm_margin": (v1 - v2) / it["valid_votes"] if contested else None,
            "verifier_ok": s["verifier_ok"] == "True",
            "chosen_rank": int(s["chosen_rank"]) if contested else 1,
            "top3_has_correct": it["top3_has_correct"],
        })
    assert len(rows) == 1647 and sum(r["contested"] for r in rows) == 1105
    return rows


def gated_ok(r, kind, t):
    if r["contested"] and r[kind] <= t:
        return r["verifier_ok"], True
    return r["majority_ok"], False


def net(rows, kind, t):
    return sum(int(gated_ok(r, kind, t)[0]) - int(r["majority_ok"]) for r in rows)


def sweep_row(rows, kind, t, extra=None):
    pick, gate = zip(*(gated_ok(r, kind, t) for r in rows))
    maj = [r["majority_ok"] for r in rows]
    st = paired_stats(list(pick), maj)
    gated = [r for r, g in zip(rows, gate) if g]
    row = dict(extra or {})
    row.update({
        "margin": kind, "t": "inf" if t == math.inf else t, "n_tasks": len(rows),
        "gated": len(gated), "overrides": sum(r["chosen_rank"] != 1 for r in gated),
        "gated_majority_ok": sum(r["majority_ok"] for r in gated),
        "gated_top3_has_correct": sum(r["top3_has_correct"] for r in gated),
        "fixed": st["a_only"], "broken": st["b_only"], "net": st["diff_tasks"],
        "majority_selected": sum(maj), "final_selected": sum(pick), "oracle": sum(r["oracle"] for r in rows),
        "mcnemar_p": st["mcnemar_exact_p"], "boot_lo_pp": st["boot_lo_pp"], "boot_hi_pp": st["boot_hi_pp"],
    })
    return row


def write(name, data):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader()
        w.writerows(data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    write("per_task.csv", rows)
    inf_row = sweep_row(rows, "norm_margin", math.inf)
    ref = {r["run"] + r["method"]: r for r in csv.DictReader((SEL.parent / "summary.csv").open())}["allall"]
    assert (inf_row["fixed"], inf_row["broken"], inf_row["final_selected"]) == \
        (int(ref["fixed"]), int(ref["broken"]), int(ref["verifier_selected"])), "t=inf must equal selector"

    for kind, grid, name in (("norm_margin", T_NORM, "sweep_normalized.csv"), ("abs_margin", T_ABS, "sweep_absolute.csv")):
        data = [sweep_row(rows, kind, t) for t in grid]
        write(name, data)
        print(f"\n{kind}")
        for d in data:
            print(f"  t={d['t']!s:>5} gated={d['gated']:4d} overrides={d['overrides']:4d} fixed={d['fixed']:3d} "
                  f"broken={d['broken']:3d} net={d['net']:+4d} selected={d['final_selected']} "
                  f"(maj {d['majority_selected']}, oracle {d['oracle']}) p={d['mcnemar_p']:.3g} "
                  f"boot[{d['boot_lo_pp']:+.1f},{d['boot_hi_pp']:+.1f}]")

    by = []
    for kind, grid in (("norm_margin", T_NORM), ("abs_margin", T_ABS)):
        for m in ("BoN", "CoT-tree", "MO-Tree"):
            sub = [r for r in rows if r["method"] == m]
            by += [sweep_row(sub, kind, t, {"method": m}) for t in grid]
    write("sweep_by_method.csv", by)

    cv = []
    for kind, grid in (("norm_margin", T_NORM), ("abs_margin", T_ABS)):
        for held in RUNS:
            train = [r for r in rows if r["run"] != held]
            test = [r for r in rows if r["run"] == held]
            best = max(grid, key=lambda t: (net(train, kind, t), -grid.index(t)))
            cv.append(sweep_row(test, kind, best, {"held_out_run": held, "train_net": net(train, kind, best)}))
        tot = {"held_out_run": "all", "margin": kind, "t": "per-fold", "n_tasks": sum(c["n_tasks"] for c in cv[-3:])}
        for k in ("gated", "overrides", "fixed", "broken", "net", "majority_selected", "final_selected", "oracle"):
            tot[k] = sum(c[k] for c in cv[-3:])
        cv.append(tot)
    write("lorocv.csv", cv)
    print("\nleave-one-run-out")
    for c in cv:
        print(f"  {c['held_out_run']:8s} {c['margin']:11s} t={c['t']!s:>8} gated={c['gated']} fixed={c['fixed']} "
              f"broken={c['broken']} net={c['net']:+d} selected={c['final_selected']}/{c['majority_selected']}")


if __name__ == "__main__":
    main()
