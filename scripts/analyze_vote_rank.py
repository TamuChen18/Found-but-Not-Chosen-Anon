#!/usr/bin/env python3
"""
Vote-rank diagnostics on the final primary pools (and the prompt-matched BoN control).

Groups = exact-JSON normalized workflows among valid (hard0) candidates, ordered by
(votes desc, smallest candidate_id asc), i.e. the selector's own order, so rank 1 is the
selected workflow and Top-1 reproduces Selected exactly.

A. correct rank: for strict vote-loss cases (load_cases: Oracle-positive, not selected, best valid
   correct group strictly out-voted), rank of the best correct group (2 / 3 / >3).
B. top-k oracle: a task counts if a correct group is among the first k groups; tasks without valid
   candidates keep their fallback outcome. Reported for k = 1, 2, 3 next to Oracle.
C. vote-loss vote shares: correct share mean / median, winner-minus-correct margin (all cases and
   rank-2 cases), cases with margin <= 1 / <= 2 votes.

Output: artifacts/final_primary/vote_rank/{summary.csv,cases.csv}
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_modal_errors as amc  # noqa: E402

OUT = ROOT / "artifacts" / "final_primary" / "vote_rank"
BUNDLES = [
    ("mini", "artifacts/mini_noseed_trees_full/eval_e2b_off", {"BoN": "BoN", "CoT": "CoT-tree", "Five": "MO-Tree"}),
    ("mini", "artifacts/prompt_matched_bon/eval", {"PMBoN": "Prompt-matched BoN"}),
    ("qwen_b1", "artifacts/e2b_off_primary/qwen_b1", {"BoN": "BoN", "CoT": "CoT-tree", "Five": "MO-Tree"}),
    ("qwen_b2", "artifacts/e2b_off_primary/qwen_b2", {"BoN": "BoN", "CoT": "CoT-tree", "Five": "MO-Tree"}),
    ("qwen", "artifacts/prompt_matched_bon_qwen/eval_vs_b1", {"PMBoN": "Prompt-matched BoN"}),
]


def key(plan):
    return json.dumps(plan, sort_keys=True, ensure_ascii=False)


def wcsv(p, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def med(xs):
    return statistics.median(xs) if xs else None


def mean(xs):
    return statistics.fmean(xs) if xs else None


def main():
    summary, case_rows = [], []
    for run, bundle, pools in BUNDLES:
        b = ROOT / bundle
        amc.METHODS = pools
        cases, _ = amc.load_cases(b, run)
        cands = defaultdict(list)
        for line in (b / "per_candidate.jsonl").open():
            c = json.loads(line)
            if c["pool"] in pools:
                cands[(c["pool"], int(c["task_id"]))].append(c)
        pt = {(r["pool"], int(r["task_id"])): r for r in csv.DictReader((b / "per_task.csv").open()) if r["pool"] in pools}

        for pool, name in pools.items():
            tids = sorted(t for (p, t) in pt if p == pool)
            topk = Counter()
            ranks, shares, margins, margins_r2 = [], [], [], []
            groups_vl, groups_all = [], []
            for tid in tids:
                r = pt[(pool, tid)]
                selected = r["filter_majority_correct"] == "True"
                valid = [c for c in cands[(pool, tid)] if c["hard0"]]
                if not valid:
                    for k in (1, 2, 3):
                        topk[k] += selected
                    continue
                votes, first, correct = Counter(), {}, {}
                for c in valid:
                    kk = key(c["plan"])
                    votes[kk] += 1
                    first[kk] = min(first.get(kk, 10 ** 9), int(c["candidate_id"]))
                    correct[kk] = c["correct"]
                order = sorted(votes, key=lambda kk: (-votes[kk], first[kk]))
                groups_all.append(len(order))
                assert correct[order[0]] == selected, (run, pool, tid)
                crank = next((i + 1 for i, kk in enumerate(order) if correct[kk]), None)
                for k in (1, 2, 3):
                    topk[k] += crank is not None and crank <= k
                if (pool, tid) in cases:
                    cs = cases[(pool, tid)]
                    m = cs["vote_margin"]
                    ranks.append(crank)
                    groups_vl.append(len(order))
                    shares.append(cs["correct_vote_share"])
                    margins.append(m)
                    if crank == 2:
                        margins_r2.append(m)
                    case_rows.append({"run": run, "method": name, "task_id": tid, "n_valid": len(valid),
                                      "n_groups": len(order), "correct_rank": crank,
                                      "winner_votes": cs["winning_wrong_votes"], "correct_votes": cs["best_correct_votes"],
                                      "margin": m, "correct_share": cs["correct_vote_share"],
                                      "winner_share": cs["winner_vote_share"]})
            sel = sum(pt[(pool, t)]["filter_majority_correct"] == "True" for t in tids)
            orc = sum(pt[(pool, t)]["oracle"] == "True" for t in tids)
            assert topk[1] == sel, (run, pool, topk[1], sel)
            n = len(ranks)
            summary.append({
                "run": run, "method": name, "n_tasks": len(tids),
                "selected": sel, "top2": topk[2], "top3": topk[3], "oracle": orc,
                "strict_vote_loss": n,
                "rank2": sum(x == 2 for x in ranks), "rank3": sum(x == 3 for x in ranks),
                "rank_gt3": sum(x > 3 for x in ranks),
                "rank2_pct": 100 * sum(x == 2 for x in ranks) / n if n else None,
                "rank3_pct": 100 * sum(x == 3 for x in ranks) / n if n else None,
                "rank_gt3_pct": 100 * sum(x > 3 for x in ranks) / n if n else None,
                "correct_share_mean": mean(shares), "correct_share_median": med(shares),
                "margin_mean": mean(margins), "margin_median": med(margins),
                "margin_rank2_mean": mean(margins_r2), "margin_rank2_median": med(margins_r2),
                "margin_le1": sum(m <= 1 for m in margins), "margin_le2": sum(m <= 2 for m in margins),
                "groups_voteloss_mean": mean(groups_vl), "groups_voteloss_median": med(groups_vl),
                "groups_alltasks_mean": mean(groups_all), "groups_alltasks_median": med(groups_all),
            })

    agg = []
    for label, rows in (("9 primary pools", [s for s in summary if s["method"] != "Prompt-matched BoN"]),
                        ("11 pools (+ GPT/Qwen Prompt-matched BoN)", summary)):
        tot = {k: sum(s[k] for s in rows) for k in
               ("selected", "top2", "top3", "oracle", "strict_vote_loss", "rank2", "rank3", "rank_gt3")}
        gap = tot["oracle"] - tot["selected"]
        names = {(s["run"], s["method"]) for s in rows}
        cs = [c for c in case_rows if (c["run"], c["method"]) in names]
        assert len(cs) == tot["strict_vote_loss"]
        ms = [c["margin"] for c in cs]
        nms = [c["margin"] / c["n_valid"] for c in cs]
        r2 = [c["margin"] for c in cs if c["correct_rank"] == 2]
        agg.append({"pools": label, "n_pools": len(rows), **tot, "gap": gap,
                    "top2_recovered": tot["top2"] - tot["selected"], "top3_recovered": tot["top3"] - tot["selected"],
                    "top2_recovery_pct": 100 * (tot["top2"] - tot["selected"]) / gap,
                    "top3_recovery_pct": 100 * (tot["top3"] - tot["selected"]) / gap,
                    "correct_share_mean": mean([c["correct_share"] for c in cs]),
                    "correct_share_median": med([c["correct_share"] for c in cs]),
                    "winner_share_median": med([c["winner_share"] for c in cs]),
                    "margin_mean": mean(ms), "margin_median": med(ms),
                    "norm_margin_mean": mean(nms), "norm_margin_median": med(nms),
                    "margin_rank2_mean": mean(r2), "margin_rank2_median": med(r2),
                    "margin_le1": sum(m <= 1 for m in ms), "margin_le2": sum(m <= 2 for m in ms)})

    OUT.mkdir(parents=True, exist_ok=True)
    wcsv(OUT / "summary.csv", summary)
    wcsv(OUT / "cases.csv", case_rows)
    wcsv(OUT / "aggregate.csv", agg)
    for a in agg:
        print(f"{a['pools']}: gap={a['gap']} VL={a['strict_vote_loss']} r2/r3/>3={a['rank2']}/{a['rank3']}/"
              f"{a['rank_gt3']} top2 +{a['top2_recovered']} ({a['top2_recovery_pct']:.1f}%) "
              f"top3 +{a['top3_recovered']} ({a['top3_recovery_pct']:.1f}%)")
    for s in summary:
        print(f"{s['run']:8s} {s['method']:19s} S={s['selected']} T2={s['top2']} T3={s['top3']} O={s['oracle']} | "
              f"VL={s['strict_vote_loss']} r2/r3/>3={s['rank2']}/{s['rank3']}/{s['rank_gt3']} | "
              f"cshare {s['correct_share_mean']:.3f}/{s['correct_share_median']:.3f} | "
              f"margin {s['margin_mean']:.2f}/{s['margin_median']} r2 {s['margin_rank2_mean'] or 0:.2f}/{s['margin_rank2_median']} | "
              f"<=1:{s['margin_le1']} <=2:{s['margin_le2']} | groups VL {s['groups_voteloss_mean']:.2f}/"
              f"{s['groups_voteloss_median']} all {s['groups_alltasks_mean']:.3f}/{s['groups_alltasks_median']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
