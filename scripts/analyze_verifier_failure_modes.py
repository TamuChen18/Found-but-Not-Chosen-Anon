#!/usr/bin/env python3
"""
Failure modes of the deployable Top-3 verifier (offline, no new LLM calls).

Cases: the 1,105 contested pool-tasks (>= 2 valid groups) of the 9 primary pools, with the cached GPT-4o
choice. broken = majority correct, verifier wrong (175); fixed = majority wrong, verifier correct (90).
Group order, votes and plans come from run_top3_selector.load_items; the verifier's choice and rank from
artifacts/final_primary/margin_gated/per_task.csv.

Per case: absolute margin v1 - v2, normalized margin (v1 - v2) / valid votes, winner (rank-1) vote share,
number of valid groups, chosen rank, and DisasterBench Tool / Parameter / Dependency checks of the
verifier-chosen workflow against gold (binary per workflow; Parameter and Dependency require per-step tools
to match), and step counts of the chosen, majority (rank-1) and gold workflows.

Output: artifacts/final_primary/verifier_failure_modes/{per_case.csv,summary.csv,margin_bins.csv}
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

from evaluators.evaluators import Evaluator  # noqa: E402
from run_top3_selector import load_items  # noqa: E402

GATED = ROOT / "artifacts" / "final_primary" / "margin_gated" / "per_task.csv"
OUT = ROOT / "artifacts" / "final_primary" / "verifier_failure_modes"
POOL = {"mini": "GPT", "qwen_b1": "Qwen_R1", "qwen_b2": "Qwen_R2"}
SHORT = {"BoN": "BoN", "CoT-tree": "CoT", "MO-Tree": "MO"}
BINS = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01)]


def wcsv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def med(xs):
    return round(statistics.median(xs), 3) if xs else None


def pct(k, n):
    return round(100 * k / n, 1) if n else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = Evaluator()
    golds = {int(o["task_id"]): json.dumps(o["structured_plan"])
             for o in map(json.loads, (ROOT / "data" / "test.jsonl").open()) if o}
    items = {(it["run"], it["method"], it["task_id"]): it for it in load_items()}
    rows = []
    for g in csv.DictReader(GATED.open()):
        if g["contested"] != "True":
            continue
        it = items[(g["run"], g["method"], int(g["task_id"]))]
        rank = int(g["chosen_rank"])
        chosen = it["top"][rank - 1]
        assert chosen["correct"] == (g["verifier_ok"] == "True")
        plan, gold = json.dumps(chosen["plan"]), golds[it["task_id"]]
        maj_ok, ver_ok = g["majority_ok"] == "True", g["verifier_ok"] == "True"
        rows.append({
            "pool": f"{POOL[g['run']]}_{SHORT[g['method']]}", "run": g["run"], "procedure": g["method"],
            "task_id": it["task_id"],
            "outcome": "broken" if maj_ok and not ver_ok else "fixed" if ver_ok and not maj_ok else
                       "both_correct" if maj_ok else "both_wrong",
            "vote_margin": int(g["abs_margin"]), "norm_vote_margin": round(float(g["norm_margin"]), 4),
            "winner_share": round(int(g["v1"]) / int(g["valid_votes"]), 4),
            "num_valid_groups": it["n_groups"], "valid_votes": int(g["valid_votes"]),
            "majority_correct": int(maj_ok), "verifier_correct": int(ver_ok), "verifier_chosen_rank": rank,
            "n_options": min(3, it["n_groups"]),
            "tool_acc": int(bool(ev.check_tools_correctness(plan, gold))),
            "param_acc": int(bool(ev.check_parameters_correctness(plan, gold))),
            "dep_acc": int(bool(ev.check_dependencies_correctness(plan, gold))),
            "chosen_n_steps": len(chosen["plan"]), "majority_n_steps": len(it["top"][0]["plan"]),
            "gold_n_steps": len(json.loads(gold)),
        })
    by = {o: [r for r in rows if r["outcome"] == o] for o in ("broken", "fixed", "both_correct", "both_wrong")}
    assert len(rows) == 1105 and len(by["broken"]) == 175 and len(by["fixed"]) == 90
    assert all(r["verifier_chosen_rank"] > 1 for r in by["broken"] + by["fixed"])
    wcsv("per_case.csv", rows)

    maj_ok_contested = by["broken"] + by["both_correct"]
    summary = []
    for name, rs in (("broken", by["broken"]), ("fixed", by["fixed"]),
                     ("majority-correct contested (638)", maj_ok_contested)):
        wrong = [r for r in rs if not r["verifier_correct"]]
        summary.append({
            "group": name, "n": len(rs),
            "median_abs_margin": med([r["vote_margin"] for r in rs]),
            "median_norm_margin": med([r["norm_vote_margin"] for r in rs]),
            "norm_margin_gt_0.3_pct": pct(sum(r["norm_vote_margin"] > 0.3 for r in rs), len(rs)),
            "median_winner_share": med([r["winner_share"] for r in rs]),
            "median_num_valid_groups": med([r["num_valid_groups"] for r in rs]),
            "mean_num_valid_groups": round(statistics.fmean(r["num_valid_groups"] for r in rs), 2),
            "chose_rank1_pct": pct(sum(r["verifier_chosen_rank"] == 1 for r in rs), len(rs)),
            "chose_rank2_pct": pct(sum(r["verifier_chosen_rank"] == 2 for r in rs), len(rs)),
            "chose_rank3_pct": pct(sum(r["verifier_chosen_rank"] == 3 for r in rs), len(rs)),
            "two_option_pct": pct(sum(r["n_options"] == 2 for r in rs), len(rs)),
            "chosen_more_steps_than_majority_pct": pct(sum(r["chosen_n_steps"] > r["majority_n_steps"]
                                                           for r in rs if r["verifier_chosen_rank"] > 1),
                                                       sum(r["verifier_chosen_rank"] > 1 for r in rs)),
            "chosen_fewer_steps_than_majority_pct": pct(sum(r["chosen_n_steps"] < r["majority_n_steps"]
                                                            for r in rs if r["verifier_chosen_rank"] > 1),
                                                        sum(r["verifier_chosen_rank"] > 1 for r in rs)),
            "n_chosen_wrong": len(wrong),
            "chosen_wrong_more_steps_than_gold_pct": pct(sum(r["chosen_n_steps"] > r["gold_n_steps"] for r in wrong),
                                                         len(wrong)),
            "chosen_wrong_tool_acc": pct(sum(r["tool_acc"] for r in wrong), len(wrong)),
            "chosen_wrong_param_acc": pct(sum(r["param_acc"] for r in wrong), len(wrong)),
            "chosen_wrong_dep_acc": pct(sum(r["dep_acc"] for r in wrong), len(wrong)),
        })
    wcsv("summary.csv", summary)

    bins = []
    for lo, hi in BINS:
        base = [r for r in maj_ok_contested if lo <= r["norm_vote_margin"] < hi]
        br = [r for r in base if not r["verifier_correct"]]
        fx = [r for r in by["fixed"] if lo <= r["norm_vote_margin"] < hi]
        bins.append({"norm_margin_bin": f"[{lo:.1f}, {min(hi, 1.0):.1f}{']' if hi > 1 else ')'}",
                     "majority_correct_contested": len(base), "broken": len(br),
                     "break_rate_pct": pct(len(br), len(base)), "share_of_all_broken_pct": pct(len(br), 175),
                     "fixed": len(fx), "net": len(fx) - len(br)})
    wcsv("margin_bins.csv", bins)

    for s in summary:
        print(s)
    print()
    for b in bins:
        print(b)


if __name__ == "__main__":
    main()
