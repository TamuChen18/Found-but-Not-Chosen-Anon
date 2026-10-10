#!/usr/bin/env python3
"""
Consistency audit of the reported selection-loss and verifier numbers. No LLM calls.

  1. Prompt-Qwen vs Qwen CoT-tree Run1/Run2 Oracle paired CIs, recomputed from the per-task
     Oracle vectors with freeze_strict_matched.paired_stats and checked against paired_ci.csv.
  2. Prompt-Qwen pool statistics (valid/task, ties, ...) from summary.csv.
  3. Stage loss over the 11 pools (9 primary + GPT and Qwen Prompt-matched BoN):
     filter + vote + tie == oracle - selected per pool, and the pooled totals.
  4. Top-3 verifier diagnostic vs deployable selector: case partition and task-level agreement.

Writes artifacts/final_primary/consistency_audit.log; exits non-zero on any failed assert.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_strict_matched import paired_stats  # noqa: E402

A = ROOT / "artifacts"
LOG = A / "final_primary" / "consistency_audit.log"
QWEN_PM = A / "prompt_matched_bon_qwen"
EXPECT_POOLED = {"pools": 11, "gap": 289, "filter": 0, "vote": 265, "tie": 24}
EXPECT_VERIFIER = {"eligible": 181, "fixed": 90, "majority_ok_called": 638, "broken": 175,
                   "called": 1105, "top3_no_gold": 286}

lines: list[str] = []


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


def check(cond: bool, msg: str) -> None:
    log(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        LOG.write_text("\n".join(lines) + "\n")
        sys.exit(1)


def rows(p: Path) -> list[dict]:
    with open(p) as f:
        return list(csv.DictReader(f))


def oracle_vec(per_task: list[dict], pool: str) -> dict[str, bool]:
    return {r["task_id"]: r["oracle"] == "True" for r in per_task if r["pool"] == pool}


def main() -> None:
    log("## 1. Prompt-Qwen vs Qwen CoT-tree: Oracle paired CI (recomputed)")
    for b in ("b1", "b2"):
        ev = QWEN_PM / f"eval_vs_{b}"
        pt = rows(ev / "per_task.csv")
        pm, cot = oracle_vec(pt, "PMBoN"), oracle_vec(pt, "CoT")
        check(sorted(pm) == sorted(cot) and len(pm) == 183, f"{b}: 183 aligned tasks")
        ids = sorted(pm, key=int)
        s = paired_stats([pm[t] for t in ids], [cot[t] for t in ids])
        ref = next(r for r in rows(ev / "paired_ci.csv")
                   if r["a"] == "PMBoN" and r["b"] == "CoT" and r["metric"] == "oracle")
        same = (int(ref["diff_tasks"]) == s["diff_tasks"]
                and abs(float(ref["boot_lo_pp"]) - s["boot_lo_pp"]) < 1e-9
                and abs(float(ref["boot_hi_pp"]) - s["boot_hi_pp"]) < 1e-9)
        log(f"  {b}: PM {sum(pm.values())} vs CoT {sum(cot.values())}  diff {s['diff_tasks']:+d} "
            f"({s['a_only']}/{s['b_only']})  {s['diff_pp']:+.1f}pp  "
            f"boot95 [{s['boot_lo_pp']:+.1f}, {s['boot_hi_pp']:+.1f}]  McNemar p={s['mcnemar_exact_p']:.3f}")
        check(same, f"{b}: recomputed CI equals paired_ci.csv")

    log("\n## 2. Prompt-Qwen pool statistics")
    for pool in ("PMBoN", "PMBoN_dedup"):
        r = next(x for x in rows(QWEN_PM / "eval_vs_b1" / "summary.csv") if x["pool"] == pool)
        r2 = next(x for x in rows(QWEN_PM / "eval_vs_b2" / "summary.csv") if x["pool"] == pool)
        check(r == r2, f"{pool}: summary identical in eval_vs_b1 and eval_vs_b2")
        ret, val = float(r["retained_per_task"]), float(r["valid_per_task"])
        log(f"  {pool}: oracle {r['oracle']} selected {r['selected']} retained/task {ret:.3f} "
            f"valid/task {val:.3f} invalid {100 * (ret - val) / ret:.2f}% unique_valid/task "
            f"{float(r['unique_valid_wf_per_task']):.3f} winner_share {float(r['winner_share']):.3f} "
            f"ties {r['ties']} tokens/task {float(r['tokens_per_task']):.0f}")

    log("\n## 3. Stage loss over 11 pools")
    pools = [r for r in rows(A / "selection_rebuild" / "stage_loss.csv")
             if r["run"] in ("mini", "qwen_b1", "qwen_b2")]
    pools += [r for r in rows(A / "prompt_matched_bon" / "eval" / "stage_loss.csv")
              if r["method"] == "Prompt-matched BoN"]
    pools += [r for r in rows(QWEN_PM / "eval_vs_b1" / "stage_loss.csv")
              if r["method"] == "Prompt-matched BoN"]
    tot = {"gap": 0, "filter": 0, "vote": 0, "tie": 0}
    for r in pools:
        gap = int(r["oracle"]) - int(r["selected"])
        f, v, t = int(r["filter_loss"]), int(r["vote_loss"]), int(r["tie_loss"])
        log(f"  {r['run']:8s} {r['method']:20s} O={r['oracle']} S={r['selected']} gap={gap} "
            f"filter={f} vote={v} tie={t}")
        check(f + v + t == gap, f"{r['run']}:{r['method']} filter+vote+tie == gap")
        for k, x in zip(("gap", "filter", "vote", "tie"), (gap, f, v, t)):
            tot[k] += x
    qb2 = next(r for r in rows(QWEN_PM / "eval_vs_b2" / "stage_loss.csv")
               if r["method"] == "Prompt-matched BoN")
    q = pools[-1]
    check(all(qb2[k] == q[k] for k in ("oracle", "selected", "filter_loss", "vote_loss", "tie_loss")),
          "Qwen Prompt-matched BoN stage loss identical in eval_vs_b1 and eval_vs_b2")
    log(f"  totals: pools={len(pools)} {tot}")
    check(len(pools) == EXPECT_POOLED["pools"], "11 pools")
    for k in ("gap", "filter", "vote", "tie"):
        check(tot[k] == EXPECT_POOLED[k], f"sum({k}) == {EXPECT_POOLED[k]}")
    check(tot["filter"] + tot["vote"] + tot["tie"] == tot["gap"], "pooled filter+vote+tie == gap")
    vr = {(r["run"].replace("qwen", "qwen_b1") if r["run"] == "qwen" else r["run"], r["method"]): r
          for r in rows(A / "final_primary" / "vote_rank" / "summary.csv")}
    for r in pools:
        v = vr[(r["run"], r["method"])]
        check((v["selected"], v["oracle"], v["strict_vote_loss"]) == (r["selected"], r["oracle"], r["vote_loss"]),
              f"vote_rank {r['run']}:{r['method']} selected/oracle/vote-loss match stage loss")
    agg = {r["n_pools"]: r for r in rows(A / "final_primary" / "vote_rank" / "aggregate.csv")}["11"]
    check((int(agg["gap"]), int(agg["strict_vote_loss"])) == (tot["gap"], tot["vote"]),
          "vote_rank 11-pool aggregate gap/vote-loss == stage-loss totals")
    check(int(agg["rank2"]) + int(agg["rank3"]) + int(agg["rank_gt3"]) == tot["vote"], "rank2+rank3+>3 == vote loss")
    log(f"  11-pool Top-k: gap {agg['gap']} top2 +{agg['top2_recovered']} ({float(agg['top2_recovery_pct']):.1f}%) "
        f"top3 +{agg['top3_recovered']} ({float(agg['top3_recovery_pct']):.1f}%)")

    log("\n## 4. Top-3 verifier diagnostic vs deployable selector")
    sel = rows(A / "final_primary" / "top3_selector" / "per_task.csv")
    diag = rows(A / "final_primary" / "top3_verifier" / "per_case.csv")
    T = lambda x: x == "True"  # noqa: E731
    called = [r for r in sel if r["mode"] == "verifier"]
    eligible = [r for r in called if not T(r["majority_ok"]) and T(r["top3_has_correct"])]
    maj_ok = [r for r in called if T(r["majority_ok"])]
    no_gold = [r for r in called if not T(r["majority_ok"]) and not T(r["top3_has_correct"])]
    fixed = [r for r in called if not T(r["majority_ok"]) and T(r["verifier_ok"])]
    broken = [r for r in called if T(r["majority_ok"]) and not T(r["verifier_ok"])]
    got = {"eligible": len(eligible), "fixed": len(fixed), "majority_ok_called": len(maj_ok),
           "broken": len(broken), "called": len(called), "top3_no_gold": len(no_gold)}
    log(f"  selector: {got}")
    for k, v in EXPECT_VERIFIER.items():
        check(got[k] == v, f"{k} == {v}")
    check(len(eligible) + len(maj_ok) + len(no_gold) == len(called), "181 + 638 + 286 == 1105")
    check(len(sel) == 9 * 183 and all((r["mode"] == "verifier") == (int(r["n_groups"]) >= 2) for r in sel),
          "1647 pool-tasks; verifier called iff >= 2 groups")
    check(not any(T(r["verifier_ok"]) for r in no_gold), "no fix possible without gold in Top-3")
    check(len(diag) == len(eligible), "diagnostic cases == eligible (181)")
    k = lambda r: (r["run"], r["method"], r["task_id"])  # noqa: E731
    dmap = {k(r): T(r["recovered"]) for r in diag}
    smap = {k(r): T(r["verifier_ok"]) for r in eligible}
    check(set(dmap) == set(smap), "diagnostic and eligible cover the same (run, method, task)")
    mism = [x for x in dmap if dmap[x] != smap[x]]
    check(not mism, f"task-level agreement diagnostic.recovered == selector.verifier_ok ({len(mism)} mismatches)")
    check(sum(dmap.values()) == len(fixed) == 90, "diagnostic correct == deployable fixed == 90")

    log("\nALL CHECKS PASSED")
    LOG.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
