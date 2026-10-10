#!/usr/bin/env python3
"""
§5 diagnostics on the locked review bundle (no LLM calls):
  1) Filter→Maj ties: how many tied tasks contain a correct top-voted plan,
     how many the smallest-candidate_id tie-break gets right, oracle tie-break ceiling.
  2) Oracle / Selected by gold workflow topology derived from test.jsonl.

Topology is derived from the gold `dependence` graph (not a benchmark-provided label):
  Node      = 1 step
  Chain     = >=2 steps forming a single linear path
  Parallel  = >=2 steps, no inter-step dependence
  Branching = any other DAG (fan-out / fan-in / partial chains)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts" / "phase2_fulltest" / "FINAL_LOCKED" / "review_bundle"
POOLS = ("BoN", "CoT", "Five")
TOPOS = ("Node", "Chain", "Parallel", "Branching")


def topology(plan):
    n = len(plan)
    if n == 1:
        return "Node"
    parents = {i: [d for d in (s.get("dependence") or [-1]) if d != -1] for i, s in enumerate(plan)}
    children = Counter(p for ps in parents.values() for p in ps)
    edges = sum(len(v) for v in parents.values())
    if edges == 0:
        return "Parallel"
    if all(len(v) <= 1 for v in parents.values()) and all(c <= 1 for c in children.values()) and edges == n - 1:
        return "Chain"
    return "Branching"


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(BUNDLE))
    ap.add_argument("--out", default=None, help="defaults to the bundle directory")
    args = ap.parse_args()
    bundle = Path(args.bundle)
    out = Path(args.out) if args.out else bundle
    out.mkdir(parents=True, exist_ok=True)
    topo = {int(o["task_id"]): topology(o["structured_plan"])
            for o in map(json.loads, (ROOT / "data" / "test.jsonl").open()) if o}

    cands = defaultdict(list)
    for line in (bundle / "per_candidate.jsonl").open():
        c = json.loads(line)
        cands[(c["pool"], c["task_id"])].append(c)
    per_task = {(r["pool"], int(r["task_id"])): r for r in csv.DictReader((bundle / "per_task.csv").open())}
    ids = sorted(topo)

    # ---- 1) ties ----
    tie_rows, tie_summary = [], []
    for pool in POOLS:
        s = Counter()
        for tid in ids:
            hard0 = [c for c in cands[(pool, tid)] if c["hard0"]]
            if not hard0:
                continue
            groups = defaultdict(list)
            for c in hard0:
                groups[json.dumps(c["plan"], sort_keys=True)].append(c)
            top = max(len(g) for g in groups.values())
            tied = [g for g in groups.values() if len(g) == top]
            if len(tied) < 2:
                continue
            any_correct = any(g[0]["correct"] for g in tied)
            chosen_ok = per_task[(pool, tid)]["filter_majority_correct"] == "True"
            s["tied"] += 1
            s["tied_with_correct"] += any_correct
            s["tiebreak_correct"] += chosen_ok
            s["oracle_tiebreak_gain"] += any_correct and not chosen_ok
            tie_rows.append({
                "pool": pool, "task_id": tid, "topology": topo[tid],
                "top_votes": top, "n_tied_groups": len(tied),
                "n_tied_groups_correct": sum(g[0]["correct"] for g in tied),
                "tiebreak_correct": chosen_ok,
                "task_oracle": per_task[(pool, tid)]["oracle"] == "True",
            })
        tie_summary.append({
            "pool": pool,
            "tied_tasks": s["tied"],
            "tied_with_a_correct_top_group": s["tied_with_correct"],
            "min_candidate_id_tiebreak_correct": s["tiebreak_correct"],
            "oracle_tiebreak_extra_tasks_ceiling": s["oracle_tiebreak_gain"],
            "filter_majority": sum(per_task[(pool, t)]["filter_majority_correct"] == "True" for t in ids),
        })
    write_csv(out / "tie_analysis_tasks.csv", tie_rows)
    write_csv(out / "tie_analysis_summary.csv", tie_summary)

    # ---- 2) topology gap ----
    topo_rows = []
    groupings = {
        "4way": {t: t for t in TOPOS},
        "3way_parallel_as_branching": {"Node": "Node", "Chain": "Chain", "Parallel": "Branching", "Branching": "Branching"},
    }
    for scheme, mapping in groupings.items():
        for label in dict.fromkeys(mapping.values()):
            tids = [t for t in ids if mapping[topo[t]] == label]
            row = {"scheme": scheme, "topology": label, "n": len(tids)}
            for pool in POOLS:
                o = sum(per_task[(pool, t)]["oracle"] == "True" for t in tids)
                f = sum(per_task[(pool, t)]["filter_majority_correct"] == "True" for t in tids)
                row[f"{pool}_oracle"] = o
                row[f"{pool}_selected"] = f
                row[f"{pool}_gap"] = o - f
            for pool in ("CoT", "Five"):
                row[f"{pool}_oracle_minus_BoN"] = row[f"{pool}_oracle"] - row["BoN_oracle"]
                row[f"{pool}_selected_minus_BoN"] = row[f"{pool}_selected"] - row["BoN_selected"]
            topo_rows.append(row)
    write_csv(out / "topology_gap.csv", topo_rows)
    (out / "test_topology.json").write_text(json.dumps({str(k): v for k, v in topo.items()}, indent=0))

    for r in tie_summary:
        print(r)
    print()
    for r in topo_rows:
        print(r)


if __name__ == "__main__":
    main()
