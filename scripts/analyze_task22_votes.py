#!/usr/bin/env python3
"""
Task 22 vote breakdown on the 11 final pools (9 primary + GPT/Qwen prompt-matched BoN).

Valid (hard0) candidates grouped by exact normalized JSON:
  correct_votes        votes for groups that are Exact-Match correct
  miswired_votes       votes for the modal BoN error (Landslide_Segmentation fed
                       <GENERATED>-1-<segmentation_map_image_path>), same rule as
                       analyze_selection_rebuild.py
  other_wrong_votes    remaining valid votes
Asserts the primary-pool numbers equal artifacts/selection_rebuild/task22.json.

Writes artifacts/final_primary/task22_votes.csv. No LLM calls.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_vote_rank import BUNDLES, key  # noqa: E402

TID = 22
OUT = ROOT / "artifacts" / "final_primary" / "task22_votes.csv"
REF = ROOT / "artifacts" / "selection_rebuild" / "task22.json"


def miswired(k):
    return '"<GENERATED>-1-<segmentation_map_image_path>"' in k and "Landslide_Segmentation" in k


def main():
    ref = json.loads(REF.read_text())
    rows = []
    for run, bundle, names in BUNDLES:
        pt = {r["pool"]: r for r in csv.DictReader((ROOT / bundle / "per_task.csv").open()) if int(r["task_id"]) == TID}
        valid = {p: [] for p in names}
        for line in (ROOT / bundle / "per_candidate.jsonl").open():
            c = json.loads(line)
            if c["pool"] in names and int(c["task_id"]) == TID and c["hard0"]:
                valid[c["pool"]].append(c)
        for pool, cs in valid.items():
            groups = Counter(key(c["plan"]) for c in cs)
            correct = {key(c["plan"]) for c in cs if c["correct"]}
            cv = sum(groups[k] for k in correct)
            mv = sum(v for k, v in groups.items() if miswired(k))
            r = {"run": run, "method": names[pool], "valid_votes": len(cs), "groups": len(groups),
                 "correct_votes": cv, "miswired_votes": mv, "other_wrong_votes": len(cs) - cv - mv,
                 "selected_correct": pt[pool]["filter_majority_correct"] == "True"}
            rows.append(r)
            if run in ref and names[pool] in ref[run]:
                e = ref[run][names[pool]]
                assert (e["n_valid"], e["n_groups"], e["landslide_miswired_votes"], e["top_is_correct"]) == \
                    (r["valid_votes"], r["groups"], mv, r["selected_correct"]), (run, pool, e, r)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
