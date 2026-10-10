#!/usr/bin/env python3
"""
Appendix numbers on the earlier fixed-seed GPT-4o-mini pools (request seed 20260808 sent).

normalization.csv   Original vs common post-generation normalization, per procedure:
                    candidate invalidity (% failing the hard0 structural check) and unfiltered
                    exact-JSON majority (correct tasks / 183). Original = each procedure's stored
                    answers (BoN applies its online postprocess, trees store raw extracts);
                    Common = strict postprocess with E4 and E2b off.
pooled_accounting.csv
                    Earlier exploratory pooled candidate set: first eight CoT-tree candidates plus
                    all narrow MO-Tree candidates. Realized cost = full CoT-tree run + full narrow
                    MO-Tree run (mean tokens per task from the answer sheets); the 8/32-scaled
                    CoT-tree cost is the deprecated approximation.

Inputs: artifacts/e2b_off_primary/mini (freeze_strict_matched.py --keep_multi_pred on the fixed-seed
manifest) and the answer sheets of runs/gpt-4o-mini/fixed_seed/{cot_tree,mo_tree_narrow}.
Expected values are asserted. Writes artifacts/final_primary/historical/. No LLM calls.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from freeze_strict_matched import majority_pick  # noqa: E402

BUNDLE = ROOT / "artifacts" / "e2b_off_primary" / "mini"
RUNS = ROOT / "runs" / "gpt-4o-mini" / "fixed_seed"
OUT = ROOT / "artifacts" / "final_primary" / "historical"
NAMES = {"BoN": "BoN", "CoT": "CoT-tree", "Five": "MO-Tree"}
EXPECTED = {
    "BoN": (2.0, 0.9, 117, 117),
    "CoT": (27.4, 2.9, 104, 118),
    "Five": (25.1, 6.9, 100, 115),
}


def wcsv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def majority_correct(cands, plan_field, correct_field):
    pick = majority_pick([{"plan": c[plan_field], "correct": c[correct_field], "candidate_id": c["candidate_id"]}
                          for c in cands])
    return bool(pick and pick["correct"])


def normalization():
    by_task = defaultdict(list)
    for line in open(BUNDLE / "per_candidate.jsonl"):
        c = json.loads(line)
        by_task[(c["pool"], c["task_id"])].append(c)
    rows = []
    for pool, name in NAMES.items():
        tasks = [cs for (p, _), cs in by_task.items() if p == pool]
        cands = [c for cs in tasks for c in cs]
        row = {
            "procedure": name,
            "candidates": len(cands),
            "invalid_pct_original": round(100 * sum(not c["raw_hard0"] for c in cands) / len(cands), 1),
            "invalid_pct_common": round(100 * sum(not c["hard0"] for c in cands) / len(cands), 1),
            "unfiltered_majority_original": sum(majority_correct(cs, "raw_plan", "raw_correct") for cs in tasks),
            "unfiltered_majority_common": sum(majority_correct(cs, "plan", "correct") for cs in tasks),
        }
        got = (row["invalid_pct_original"], row["invalid_pct_common"],
               row["unfiltered_majority_original"], row["unfiltered_majority_common"])
        assert got == EXPECTED[pool], (pool, got, EXPECTED[pool])
        rows.append(row)
    return rows


def mean_tokens(run):
    sheets = sorted((RUNS / run / "answer_sheets").glob("Task * - Answer.json"))
    assert len(sheets) == 183, (run, len(sheets))
    return statistics.mean(json.loads(p.read_text())["usage"]["total_tokens"] for p in sheets)


def pooled_accounting():
    cot, narrow = mean_tokens("cot_tree"), mean_tokens("mo_tree_narrow")
    rows = [
        {"quantity": "cot_tree_full_run", "mean_tokens_per_task": round(cot)},
        {"quantity": "mo_tree_narrow_full_run", "mean_tokens_per_task": round(narrow)},
        {"quantity": "pooled_realized", "mean_tokens_per_task": round(cot + narrow)},
        {"quantity": "pooled_deprecated_8_of_32_approx", "mean_tokens_per_task": round(cot * 8 / 32 + narrow)},
    ]
    assert [r["mean_tokens_per_task"] for r in rows] == [242475, 171241, 413716, 231860], rows
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("normalization.csv", normalization()), ("pooled_accounting.csv", pooled_accounting())):
        wcsv(OUT / name, rows)
        for r in rows:
            print(r)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
