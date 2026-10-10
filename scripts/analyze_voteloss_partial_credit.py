#!/usr/bin/env python3
"""
Partial credit of the selected (winning) wrong workflow on strict vote-loss cases. No LLM calls.

Cases: the 265 strict vote losses of the 11 pools (9 primary + GPT/Qwen Prompt-matched BoN), from
artifacts/final_primary/vote_rank/cases.csv. The selected workflow is the locked final-protocol pick
(per_task.csv selected_workflow). It is scored against the gold plan with the DisasterBench evaluator:
  Tool Acc        Evaluator.check_tools_correctness        (step, agent) sequence matches exactly
  Parameter Acc   Evaluator.check_parameters_correctness   step/agent and inputs match at every step
  Dependency Acc  Evaluator.check_dependencies_correctness step/agent, dependence and dependence_content match
Each metric is binary per workflow, and Parameter / Dependency both require the per-step tools to match,
so neither can exceed Tool Acc.

Descriptive step-level columns (not DisasterBench metrics): same_n_steps, and tool_step_match =
positionally matched (step, agent) pairs / max(len(selected), len(gold)).

Normalized vote margin = (winner votes - best correct votes) / valid votes; low / high split at the median.

Output: artifacts/final_primary/voteloss_partial_credit/{per_case.csv,summary_by_rank.csv,summary_by_margin.csv}
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_vote_rank import BUNDLES  # noqa: E402
from evaluators.evaluators import Evaluator  # noqa: E402

CASES = ROOT / "artifacts" / "final_primary" / "vote_rank" / "cases.csv"
TEST = ROOT / "data" / "test.jsonl"
OUT = ROOT / "artifacts" / "final_primary" / "voteloss_partial_credit"
SHORT = {"BoN": "BoN", "CoT-tree": "CoT", "MO-Tree": "MO", "Prompt-matched BoN": "Prompt"}


def pool_label(run, method):
    r = {"mini": "GPT", "qwen_b1": "Qwen_R1", "qwen_b2": "Qwen_R2", "qwen": "Qwen"}[run]
    return f"{r}_{SHORT[method]}"


def tool_step_match(plan, gold):
    p = Evaluator._normalize_plan_steps(json.loads(plan))
    g = Evaluator._normalize_plan_steps(json.loads(gold))
    ps = [(x.get("step"), x.get("agent")) for x in p if isinstance(x, dict)]
    gs = [(x.get("step"), x.get("agent")) for x in g if isinstance(x, dict)]
    return sum(a == b for a, b in zip(ps, gs)) / max(len(ps), len(gs), 1)


def rank_group(r):
    return "Rank 2" if r == 2 else "Rank 3" if r == 3 else "Rank >3"


def wcsv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def summarize(rows, key, order):
    out = []
    for g in order + ["All"]:
        rs = rows if g == "All" else [r for r in rows if r[key] == g]
        n = len(rs)
        out.append({key: g, "n": n,
                    "tool_acc": round(100 * sum(r["tool_accuracy"] for r in rs) / n, 1),
                    "parameter_acc": round(100 * sum(r["parameter_accuracy"] for r in rs) / n, 1),
                    "dependency_acc": round(100 * sum(r["dependency_accuracy"] for r in rs) / n, 1),
                    "tool_only_ok_pct": round(100 * sum(r["tool_accuracy"] and not r["parameter_accuracy"]
                                                        and not r["dependency_accuracy"] for r in rs) / n, 1),
                    "param_wrong_only_pct": round(100 * sum(r["tool_accuracy"] and not r["parameter_accuracy"]
                                                            and r["dependency_accuracy"] for r in rs) / n, 1),
                    "dep_wrong_only_pct": round(100 * sum(r["tool_accuracy"] and r["parameter_accuracy"]
                                                          and not r["dependency_accuracy"] for r in rs) / n, 1),
                    "tool_wrong_pct": round(100 * sum(not r["tool_accuracy"] for r in rs) / n, 1),
                    "same_n_steps_pct": round(100 * sum(r["same_n_steps"] for r in rs) / n, 1),
                    "tool_step_match_mean": round(statistics.fmean(r["tool_step_match"] for r in rs), 3),
                    "tool_wrong_one_step_off_pct": round(100 * sum(
                        not r["tool_accuracy"] and r["same_n_steps"] and
                        round(r["tool_step_match"] * r["gold_n_steps"]) == r["gold_n_steps"] - 1 for r in rs) / n, 1),
                    "norm_margin_median": round(statistics.median(r["norm_vote_margin"] for r in rs), 3)})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = Evaluator()
    golds = {int(o["task_id"]): json.dumps(o["structured_plan"]) for o in map(json.loads, TEST.open()) if o}
    selected = {}
    for run, bundle, names in BUNDLES:
        inv = {v: k for k, v in names.items()}
        for r in csv.DictReader((ROOT / bundle / "per_task.csv").open()):
            if r["pool"] in inv.values():
                method = next(m for m, p in inv.items() if p == r["pool"])
                selected[(run, method, int(r["task_id"]))] = r
    rows = []
    for c in csv.DictReader(CASES.open()):
        run, method, tid = c["run"], c["method"], int(c["task_id"])
        pt = selected[(run, method, tid)]
        assert pt["filter_majority_correct"] == "False" and pt["oracle"] == "True", (run, method, tid)
        plan = pt["selected_workflow"]
        gold = golds[tid]
        assert not ev.check_answers_equivalence(plan, gold), (run, method, tid)
        rank = int(c["correct_rank"])
        rows.append({
            "pool": pool_label(run, method), "run": run, "procedure": method, "task_id": tid,
            "best_correct_rank": rank, "rank_group": rank_group(rank),
            "winner_votes": int(c["winner_votes"]), "best_correct_votes": int(c["correct_votes"]),
            "valid_votes": int(c["n_valid"]), "vote_margin": int(c["margin"]),
            "norm_vote_margin": round(int(c["margin"]) / int(c["n_valid"]), 4),
            "tool_accuracy": int(bool(ev.check_tools_correctness(plan, gold))),
            "parameter_accuracy": int(bool(ev.check_parameters_correctness(plan, gold))),
            "dependency_accuracy": int(bool(ev.check_dependencies_correctness(plan, gold))),
            "selected_n_steps": len(json.loads(plan)), "gold_n_steps": len(json.loads(gold)),
            "same_n_steps": int(len(json.loads(plan)) == len(json.loads(gold))),
            "tool_step_match": round(tool_step_match(plan, gold), 4),
        })
    assert len(rows) == 265
    for r in rows:
        assert r["parameter_accuracy"] <= r["tool_accuracy"] and r["dependency_accuracy"] <= r["tool_accuracy"]
    med = statistics.median(r["norm_vote_margin"] for r in rows)
    for r in rows:
        r["margin_group"] = "low-margin" if r["norm_vote_margin"] <= med else "high-margin"
    wcsv("per_case.csv", rows)
    by_rank = summarize(rows, "rank_group", ["Rank 2", "Rank 3", "Rank >3"])
    by_margin = summarize(rows, "margin_group", ["low-margin", "high-margin"])
    wcsv("summary_by_rank.csv", by_rank)
    wcsv("summary_by_margin.csv", by_margin)
    print(f"median normalized margin = {med:.3f}")
    for s in by_rank + by_margin:
        print(s)


if __name__ == "__main__":
    main()
