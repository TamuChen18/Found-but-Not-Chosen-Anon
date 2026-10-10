#!/usr/bin/env python3
"""
Exact-Match flips caused by strict normalization (E1-E3, E4 off, E2b off) on the 11 final pools.

Per candidate: raw_correct = EM of the parsed raw answer, correct = EM of the normalized plan
(both stored in per_candidate.jsonl). Counts incorrect->correct and correct->incorrect flips,
overall and among valid (hard0) candidates, plus Oracle-level flips per task.
Asserts there is no correct->incorrect flip.

Writes artifacts/final_primary/normalization_em_flips/{summary.csv, flip_cases.csv}. No LLM calls.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_vote_rank import BUNDLES  # noqa: E402

OUT = ROOT / "artifacts" / "final_primary" / "normalization_em_flips"


def wcsv(p, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, flips = [], []
    for run, bundle, names in BUNDLES:
        per = defaultdict(lambda: defaultdict(list))
        for line in (ROOT / bundle / "per_candidate.jsonl").open():
            c = json.loads(line)
            if c["pool"] in names:
                per[c["pool"]][int(c["task_id"])].append(c)
        for pool, tasks in per.items():
            cs = [c for v in tasks.values() for c in v]
            up = [c for c in cs if not c["raw_correct"] and c["correct"]]
            down = [c for c in cs if c["raw_correct"] and not c["correct"]]
            assert not down, (run, pool, len(down))
            summary.append({
                "run": run, "method": names[pool], "tasks": len(tasks), "candidates": len(cs),
                "raw_correct": sum(c["raw_correct"] for c in cs), "normalized_correct": sum(c["correct"] for c in cs),
                "incorrect_to_correct": len(up), "correct_to_incorrect": len(down),
                "incorrect_to_correct_hard0": sum(c["hard0"] for c in up),
                "tasks_with_flip": len({int(c["task_id"]) for c in up}),
                "oracle_raw": sum(any(c["raw_correct"] for c in v) for v in tasks.values()),
                "oracle_normalized": sum(any(c["correct"] for c in v) for v in tasks.values()),
            })
            flips += [{"run": run, "method": names[pool], "task_id": c["task_id"], "candidate_id": c["candidate_id"],
                       "hard0": c["hard0"]} for c in up]
    tot = {"run": "all", "method": f"{len(summary)} pools"}
    for k in summary[0]:
        if k not in tot:
            tot[k] = sum(r[k] for r in summary)
    summary.append(tot)
    wcsv(OUT / "summary.csv", summary)
    wcsv(OUT / "flip_cases.csv", flips)
    for r in summary:
        print(r)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
