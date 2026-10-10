#!/usr/bin/env python3
"""
Candidate-pool / voting diagnostics under the locked protocol, applied uniformly to any
review bundle (per_candidate.jsonl + per_task.csv). No LLM calls.

Per (pool, task), over valid retained candidates (hard0 = hard_violation_count == 0 after strict
postprocess); plans are the strict-normalized workflows, grouped by exact JSON:
  n_candidates        retained candidates
  n_valid             valid retained candidates
  n_unique_valid      unique normalized workflows among valid candidates
  win_votes           size of the largest group
  win_share           win_votes / n_valid
  tied                >= 2 groups share the largest size
  correct_votes       size of the largest correct group (0 if none)
  correct_share       correct_votes / n_valid
  stage               Oracle-positive tasks only: selected / lost_at_filter / lost_at_vote /
                      lost_at_tie / no_valid_fallback_{selected,lost}
Tasks with n_valid == 0 fall back to raw majority in the locked protocol; they are excluded from
the vote statistics and counted separately.

Usage:
  python3 scripts/analyze_pool_concentration.py \
      --bundle mini=artifacts/phase2_fulltest/FINAL_LOCKED/review_bundle \
      --bundle qwen_b1=artifacts/qwen_nothink_main183/eval/b20260725 \
      --out artifacts/pool_concentration
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOLS = ("BoN", "CoT", "Five")
STAGES = ("selected", "lost_at_filter", "lost_at_vote", "lost_at_tie",
          "no_valid_fallback_selected", "no_valid_fallback_lost")


def key(plan):
    return json.dumps(plan, sort_keys=True, ensure_ascii=False)


def task_rows(bundle: Path, label: str):
    cands = defaultdict(list)
    for line in (bundle / "per_candidate.jsonl").open():
        c = json.loads(line)
        cands[(c["pool"], int(c["task_id"]))].append(c)
    per_task = {(r["pool"], int(r["task_id"])): r for r in csv.DictReader((bundle / "per_task.csv").open())}
    rows = []
    for (pool, tid), pt in sorted(per_task.items()):
        if pool not in POOLS:
            continue
        cs = cands.get((pool, tid), [])
        valid = [c for c in cs if c["hard0"]]
        oracle = pt["oracle"] == "True"
        selected = pt["filter_majority_correct"] == "True"
        r = {"bundle": label, "pool": pool, "task_id": tid, "n_candidates": len(cs),
             "n_valid": len(valid), "oracle": oracle, "selected": selected}
        if valid:
            groups = Counter(key(c["plan"]) for c in valid)
            correct_keys = {key(c["plan"]) for c in valid if c["correct"]}
            top = max(groups.values())
            r.update(n_unique_valid=len(groups), win_votes=top, win_share=top / len(valid),
                     tied=sum(v == top for v in groups.values()) >= 2,
                     correct_votes=max((groups[k] for k in correct_keys), default=0))
            r["correct_share"] = r["correct_votes"] / len(valid)
        if oracle:
            if not valid:
                r["stage"] = "no_valid_fallback_selected" if selected else "no_valid_fallback_lost"
            elif selected:
                r["stage"] = "selected"
            elif not r["correct_votes"]:
                r["stage"] = "lost_at_filter"
            else:
                r["stage"] = "lost_at_tie" if r["correct_votes"] == r["win_votes"] else "lost_at_vote"
        rows.append(r)
    return rows


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def summarize(rows):
    out = []
    for (label, pool) in dict.fromkeys((r["bundle"], r["pool"]) for r in rows):
        rs = [r for r in rows if r["bundle"] == label and r["pool"] == pool]
        v = [r for r in rs if r["n_valid"]]
        op = [r for r in rs if r["oracle"]]
        opv = [r for r in op if r["n_valid"]]
        st = Counter(r["stage"] for r in op)
        out.append({
            "bundle": label, "pool": pool, "n_tasks": len(rs),
            "oracle": len(op), "selected": sum(r["selected"] for r in rs),
            "retained_cands_mean": mean([r["n_candidates"] for r in rs]),
            "valid_cands_mean": mean([r["n_valid"] for r in rs]),
            "tasks_no_valid": len(rs) - len(v),
            "tasks_single_valid": sum(r["n_valid"] == 1 for r in v),
            "unique_valid_wf_mean": mean([r["n_unique_valid"] for r in v]),
            "unique_valid_wf_median": statistics.median([r["n_unique_valid"] for r in v]) if v else None,
            "win_votes_mean": mean([r["win_votes"] for r in v]),
            "win_share_mean": mean([r["win_share"] for r in v]),
            "win_share_median": statistics.median([r["win_share"] for r in v]) if v else None,
            "tasks_unanimous": sum(r["n_unique_valid"] == 1 for r in v),
            "tied_tasks": sum(r["tied"] for r in v),
            "oraclepos_correct_share_mean": mean([r["correct_share"] for r in opv]),
            "oraclepos_win_share_mean": mean([r["win_share"] for r in opv]),
            **{s: st.get(s, 0) for s in STAGES},
            "lost_total": len(op) - st.get("selected", 0) - st.get("no_valid_fallback_selected", 0),
        })
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="append", required=True, help="label=path")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "pool_concentration"))
    args = ap.parse_args()
    rows = []
    for spec in args.bundle:
        label, path = spec.split("=", 1)
        rows += task_rows(ROOT / path, label)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_task.csv", rows)
    summary = summarize(rows)
    write_csv(out / "summary.csv", summary)
    for s in summary:
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items()})


if __name__ == "__main__":
    main()
