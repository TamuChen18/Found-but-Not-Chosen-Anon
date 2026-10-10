#!/usr/bin/env python3
"""
Checks on the DisasterBench partial-credit metrics used in the vote-loss partial-credit analysis.

A. Every Exact-Match-correct candidate in the 11 final pools passes Tool / Parameter / Dependency
   (asserted), i.e. Parameter = 0% on selected wrong workflows is not an unattainable metric.
B. For the 265 strict vote-loss winners whose tool sequence is correct, why Parameter fails:
   per aligned step, inputs differ in keys (different input names) or only in values, and value
   differences split into literal values vs GENERATED references.

Writes artifacts/final_primary/partial_credit_definition/{summary.csv, tool_correct_param_fail.csv}.
No LLM calls.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_vote_rank import BUNDLES  # noqa: E402
from evaluators.evaluators import Evaluator  # noqa: E402

TEST = ROOT / "data" / "test.jsonl"
CASES = ROOT / "artifacts" / "final_primary" / "voteloss_partial_credit" / "per_case.csv"
OUT = ROOT / "artifacts" / "final_primary" / "partial_credit_definition"


def wcsv(p, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def is_ref(v):
    return isinstance(v, str) and v.startswith("<GENERATED>")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = Evaluator()
    golds = {int(o["task_id"]): json.dumps(o["structured_plan"]) for o in map(json.loads, TEST.open()) if o}

    n_correct = 0
    selected = {}
    for run, bundle, names in BUNDLES:
        for line in (ROOT / bundle / "per_candidate.jsonl").open():
            c = json.loads(line)
            if c["pool"] in names and c["correct"]:
                plan, gold = json.dumps(c["plan"]), golds[int(c["task_id"])]
                assert ev.check_tools_correctness(plan, gold), (run, c["pool"], c["task_id"])
                assert ev.check_parameters_correctness(plan, gold), (run, c["pool"], c["task_id"])
                assert ev.check_dependencies_correctness(plan, gold), (run, c["pool"], c["task_id"])
                n_correct += 1
        for r in csv.DictReader((ROOT / bundle / "per_task.csv").open()):
            if r["pool"] in names:
                selected[(run, names[r["pool"]], int(r["task_id"]))] = r["selected_workflow"]
    assert n_correct == 26236, n_correct

    rows = []
    for c in csv.DictReader(CASES.open()):
        if c["tool_accuracy"] != "1":
            continue
        tid = int(c["task_id"])
        plan = Evaluator._normalize_plan_steps(json.loads(selected[(c["run"], c["procedure"], tid)]))
        gold = Evaluator._normalize_plan_steps(json.loads(golds[tid]))
        kinds = Counter()
        for p, g in zip(plan, gold):
            pi, gi = p.get("inputs", {}) or {}, g.get("inputs", {}) or {}
            if pi == gi:
                continue
            if set(pi) != set(gi):
                kinds["input_names_differ"] += 1
                continue
            for k in gi:
                if pi[k] != gi[k]:
                    kinds["generated_ref_differs" if is_ref(pi[k]) or is_ref(gi[k]) else "literal_value_differs"] += 1
        assert kinds, (c["pool"], tid)
        rows.append({"pool": c["pool"], "task_id": tid, "dependency_accuracy": c["dependency_accuracy"],
                     **{k: kinds.get(k, 0) for k in ("input_names_differ", "literal_value_differs",
                                                     "generated_ref_differs")}})
    n = len(rows)
    summary = [{
        "correct_candidates_checked": n_correct, "correct_candidates_failing_any_metric": 0,
        "voteloss_winners": sum(1 for _ in csv.DictReader(CASES.open())),
        "tool_correct_winners": n,
        "with_input_name_diff": sum(r["input_names_differ"] > 0 for r in rows),
        "with_literal_value_diff": sum(r["literal_value_differs"] > 0 for r in rows),
        "with_generated_ref_diff": sum(r["generated_ref_differs"] > 0 for r in rows),
        "literal_value_diff_only": sum(r["literal_value_differs"] > 0 and not r["input_names_differ"]
                                       and not r["generated_ref_differs"] for r in rows),
        "dependency_correct_among_tool_correct": sum(r["dependency_accuracy"] == "1" for r in rows),
    }]
    wcsv(OUT / "summary.csv", summary)
    wcsv(OUT / "tool_correct_param_fail.csv", rows)
    print(summary[0])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
