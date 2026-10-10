#!/usr/bin/env python3
"""
Backend sampling facts cited in the paper (setup and MO-Tree n>1 paragraphs).

  gpt_mini_mo_n_gt1      GPT-4o-mini no-seed MO-Tree: requests with n>1 and how many
                         returned a single choice (OpenRouter ignores n).
  qwen_mo_n_gt1          Qwen3.5-9B MO-Tree Run 1 / Run 2: requests with n>1 and how many
                         returned n completions (vLLM honors n).
  mo_tiebreak_calls      MO-Tree local-aggregation confidence tie-break calls
                         (max_tokens=10) per run.
  qwen_thinking_no_plan  Earlier thinking-enabled Qwen3.5-9B CoT-tree run (OpenRouter,
                         40 tasks x 34 samples): samples whose text contains no JSON
                         array opener '[{' (whitespace removed), i.e. no workflow.

Expected values are asserted. Writes artifacts/final_primary/backend_sampling/summary.csv.
No LLM calls.
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT = ROOT / "artifacts" / "final_primary" / "backend_sampling" / "summary.csv"

GPT_MO = LOGS / "calllogs_mini_noseed_trees_full" / "five.jsonl"
QWEN_MO = {
    "qwen_b1": LOGS / "calllogs_main183" / "qwen35nothink_test_five_r1_da4_cot8_dc5_d5_seed20260725_main183.jsonl",
    "qwen_b2": LOGS / "calllogs_main183" / "qwen35nothink_test_five_r1_da4_cot8_dc5_d5_seed20260726_main183.jsonl",
}
THINK_COT = "runs/qwen3.5-9b-thinking/cot_tree"

EXPECTED = {
    ("gpt_mini_mo_n_gt1", "mini"): (5082, 5082),
    ("qwen_mo_n_gt1", "qwen_b1"): (4607, 4607),
    ("qwen_mo_n_gt1", "qwen_b2"): (4598, 4598),
    ("mo_tiebreak_calls", "mini"): (0, None),
    ("mo_tiebreak_calls", "qwen_b1"): (216, None),
    ("mo_tiebreak_calls", "qwen_b2"): (187, None),
    ("qwen_thinking_no_plan", "qwen_thinking_cot_seed20260725"): (1360, 672),
}


def ok_calls(path):
    with open(path) as f:
        return [r for r in map(json.loads, f) if r.get("status") == "ok"]


def main():
    rows = []

    calls = ok_calls(GPT_MO)
    multi = [r for r in calls if r["n"] > 1]
    rows.append(("gpt_mini_mo_n_gt1", "mini", len(multi), sum(r["n_choices"] == 1 for r in multi),
                 "n>1 requests / returned exactly one choice"))
    rows.append(("mo_tiebreak_calls", "mini", sum(r["max_tokens"] == 10 for r in calls), None,
                 "confidence tie-break calls (max_tokens=10)"))

    for label, path in QWEN_MO.items():
        calls = ok_calls(path)
        multi = [r for r in calls if r["n"] > 1]
        rows.append(("qwen_mo_n_gt1", label, len(multi),
                     sum(len(r["finish_reasons"]) == r["n"] for r in multi),
                     "n>1 requests / returned n completions"))
        rows.append(("mo_tiebreak_calls", label, sum(r["max_tokens"] == 10 for r in calls), None,
                     "confidence tie-break calls (max_tokens=10)"))

    run_dir = glob.glob(str(ROOT / THINK_COT))
    assert len(run_dir) == 1, run_dir
    total = no_plan = 0
    for f in glob.glob(run_dir[0] + "/answer_sheets/Task * - Final Solutions.json"):
        for sol in json.load(open(f)):
            text = sol["trace"]["0"]["chain_of_thought"]
            text = text.get("text") if isinstance(text, dict) else text
            total += 1
            no_plan += "[{" not in text.replace(" ", "").replace("\n", "")
    rows.append(("qwen_thinking_no_plan", "qwen_thinking_cot_seed20260725", total, no_plan,
                 "samples / samples without a JSON workflow"))

    for metric, run, a, b, _ in rows:
        exp = EXPECTED[(metric, run)]
        assert (a, b) == exp, (metric, run, a, b, exp)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "run", "count", "subcount", "definition"])
        w.writerows(rows)
    for r in rows:
        print(*r, sep="\t")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
