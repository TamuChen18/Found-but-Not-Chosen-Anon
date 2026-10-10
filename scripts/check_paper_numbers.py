#!/usr/bin/env python3
"""
Check the numbers reported in the paper against the regenerated result files.

Every table of the main text and appendix that reports experimental numbers, the Figure 1 panels and the key
statistics quoted in the text are listed below with the value printed in the paper; each is recomputed from
artifacts/ (rounded as in the paper) and compared. Writes artifacts/final_primary/paper_check.csv and exits
non-zero if anything differs.

  python3 scripts/check_paper_numbers.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
FP = A / "final_primary"
OUT = FP / "paper_check.csv"

RESULTS = []


def check(item, paper, got):
    RESULTS.append({"item": item, "paper": paper, "reproduced": got, "ok": paper == got})


def rows(path, **match):
    with open(A / path, newline="") as f:
        out = [r for r in csv.DictReader(f) if all(r.get(k) == v for k, v in match.items())]
    return out


def row(path, **match):
    out = rows(path, **match)
    assert len(out) == 1, (path, match, len(out))
    return out[0]


def i(x):
    return int(float(x))


def f1(x):
    return f"{float(x):.1f}"


def f2(x):
    return f"{float(x):.2f}"


def s3(x):
    """.769-style share."""
    return f"{float(x):.3f}"[1:] if float(x) < 1 else f"{float(x):.3f}"


def k(x):
    return f"{round(float(x) / 1000)}k"


def p3(x):
    x = float(x)
    return "<.001" if x < 0.001 else (f"{x:.3f}"[1:] if x < 1 else "1.000")


RUN = {"GPT-4o-mini": "mini", "Qwen Run 1": "qwen_b1", "Qwen Run 2": "qwen_b2"}
POOL = {"BoN": "BoN", "CoT-tree": "CoT", "MO-Tree": "Five"}
BUNDLE = {"mini": "mini_noseed_trees_full/eval_e2b_off", "qwen_b1": "e2b_off_primary/qwen_b1",
          "qwen_b2": "e2b_off_primary/qwen_b2"}
PM = {"GPT-4o-mini": ("prompt_matched_bon/eval/summary.csv", "PMBoN"),
      "Qwen Control": ("prompt_matched_bon_qwen/eval_vs_b1/summary.csv", "PMBoN")}


# ---------------------------------------------------------------- Table: main results
MAIN = {  # Oracle, Selected, Gap, Filter, Vote, Tie, Tokens/task
    ("GPT-4o-mini", "BoN"): (139, 121, 18, 0, 18, 0, "238k"),
    ("GPT-4o-mini", "CoT-tree"): (153, 131, 22, 0, 22, 0, "242k"),
    ("GPT-4o-mini", "MO-Tree"): (146, 123, 23, 0, 21, 2, "239k"),
    ("GPT-4o-mini", "Prompt-matched BoN"): (154, 129, 25, 0, 24, 1, "242k"),
    ("Qwen Control", "Prompt-matched BoN"): (160, 127, 33, 0, 32, 1, "253k"),
    ("Qwen Run 1", "BoN"): (159, 129, 30, 0, 28, 2, "263k"),
    ("Qwen Run 1", "CoT-tree"): (162, 132, 30, 0, 28, 2, "253k"),
    ("Qwen Run 1", "MO-Tree"): (147, 122, 25, 0, 19, 6, "285k"),
    ("Qwen Run 2", "BoN"): (162, 128, 34, 0, 33, 1, "263k"),
    ("Qwen Run 2", "CoT-tree"): (157, 133, 24, 0, 22, 2, "252k"),
    ("Qwen Run 2", "MO-Tree"): (147, 122, 25, 0, 18, 7, "284k"),
}


def main_results():
    for (run, proc), paper in MAIN.items():
        if proc == "Prompt-matched BoN":
            r = row(PM[run][0], pool=PM[run][1])
            got = (i(r["oracle"]), i(r["selected"]), i(r["oracle"]) - i(r["selected"]), i(r["lost_at_filter"]),
                   i(r["lost_at_vote"]), i(r["lost_at_tie"]), k(r["tokens_per_task"]))
        else:
            c = row("final_primary/concentration/summary.csv", bundle=RUN[run], pool=POOL[proc])
            b = row(f"{BUNDLE[RUN[run]]}/summary.csv", pool=POOL[proc])
            assert (b["oracle"], b["filter_majority"]) == (c["oracle"], c["selected"])
            got = (i(c["oracle"]), i(c["selected"]), i(c["oracle"]) - i(c["selected"]), i(c["lost_at_filter"]),
                   i(c["lost_at_vote"]), i(c["lost_at_tie"]), k(b["mean_total_tokens"]))
        check(f"tab:main-results {run} {proc}", paper, got)


# ---------------------------------------------------------------- Table: pool statistics
POOL_STATS = {  # Retained, Valid, Unique valid WF, Winning share, Ties
    # Qwen control unique valid WF = 742/183 = 4.055 -> 4.05 (the submitted draft printed 4.06 here, in the run-level
    # table and twice in the text: double rounding of 4.055; see artifacts/final_primary/PROVENANCE.md)
    ("GPT-4o-mini", "BoN"): ("32.0", "31.7", "2.32", ".881", 1),
    ("GPT-4o-mini", "Prompt-matched BoN"): ("32.0", "31.1", "4.09", ".770", 3),
    ("GPT-4o-mini", "CoT-tree"): ("31.5", "30.6", "4.22", ".769", 5),
    ("GPT-4o-mini", "MO-Tree"): ("12.7", "11.9", "4.75", ".583", 17),
    ("Qwen Control", "Prompt-matched BoN"): ("34.0", "33.5", "4.05", ".797", 5),
    ("Qwen Run 1", "BoN"): ("34.0", "28.7", "4.29", ".776", 6),
    ("Qwen Run 1", "CoT-tree"): ("11.2", "11.0", "3.91", ".733", 14),
    ("Qwen Run 1", "MO-Tree"): ("7.3", "6.2", "2.67", ".689", 20),
    ("Qwen Run 2", "BoN"): ("34.0", "28.3", "4.31", ".773", 4),
    ("Qwen Run 2", "CoT-tree"): ("11.1", "10.8", "4.01", ".740", 8),
    ("Qwen Run 2", "MO-Tree"): ("7.5", "6.2", "2.48", ".729", 17),
}


def pool_stats():
    for (run, proc), paper in POOL_STATS.items():
        if proc == "Prompt-matched BoN":
            r = row(PM[run][0], pool=PM[run][1])
            got = (f1(r["retained_per_task"]), f1(r["valid_per_task"]), f2(r["unique_valid_wf_per_task"]),
                   s3(r["winner_share"]), i(r["ties"]))
        else:
            c = row("final_primary/concentration/summary.csv", bundle=RUN[run], pool=POOL[proc])
            got = (f1(c["retained_cands_mean"]), f1(c["valid_cands_mean"]), f2(c["unique_valid_wf_mean"]),
                   s3(c["win_share_mean"]), i(c["tied_tasks"]))
        check(f"tab:app-pool-stats {run} {proc}", paper, got)


# ---------------------------------------------------------------- Table: prompt-matched run-level sensitivity
def run_sensitivity():
    pm_g, pm_q1, pm_q2 = "prompt_matched_bon/eval", "prompt_matched_bon_qwen/eval_vs_b1", "prompt_matched_bon_qwen/eval_vs_b2"
    spec = {  # Oracle, Selected, Tokens/task, Unique valid, Winner share
        "GPT-4o-mini CoT-tree": ((153, 131, "242,476", "4.22", ".769"), pm_g, "CoT", "CoT"),
        "GPT-4o-mini Prompt-matched": ((154, 129, "242,193", "4.09", ".770"), pm_g, "PMBoN", "PMBoN"),
        "Qwen CoT-tree R1": ((162, 132, "252,678", "3.91", ".733"), pm_q1, "CoT", "CoT"),
        "Qwen CoT-tree R2": ((157, 133, "252,445", "4.01", ".740"), pm_q2, "CoT", "CoT"),
        # winner share of the Qwen control under CoT-tree's retention convention (caption)
        "Qwen Prompt-matched": ((160, 127, "252,666", "4.05", ".734"), pm_q1, "PMBoN", "PMBoN_dedup"),
    }
    for name, (paper, d, pool, share_pool) in spec.items():
        r, s = row(f"{d}/summary.csv", pool=pool), row(f"{d}/summary.csv", pool=share_pool)
        got = (i(r["oracle"]), i(r["selected"]), f"{round(float(r['tokens_per_task'])):,}",
               f2(r["unique_valid_wf_per_task"]), s3(s["winner_share"]))
        check(f"tab:app-run-sensitivity {name}", paper, got)
    r = row(f"{pm_q1}/summary.csv", pool="PMBoN_dedup")
    check("text: Qwen control under CoT-tree retention (Selected, retained, valid, ties)",
          (130, "11.24", "11.02", 11),
          (i(r["selected"]), f2(r["retained_per_task"]), f2(r["valid_per_task"]), i(r["ties"])))


# ---------------------------------------------------------------- Table: seed sensitivity (GPT-4o-mini)
SEED = {  # Oracle, Selected, Retained/task, Unique valid/task, Winner share
    # Winner share of the tree rows: .806 / .769 / .555 / .583 (the submitted draft printed .810 / .770 /
    # .550 / .580; transcription correction, see artifacts/final_primary/PROVENANCE.md)
    ("BoN", "fixed seed"): (133, 121, "32.0", "1.60", ".925"),
    ("BoN", "no seed"): (139, 121, "32.0", "2.32", ".881"),
    ("CoT-tree", "fixed seed"): (144, 118, "23.0", "3.22", ".806"),
    ("CoT-tree", "no seed"): (153, 131, "31.5", "4.22", ".769"),
    ("MO-Tree", "fixed seed"): (150, 117, "12.8", "4.82", ".555"),
    ("MO-Tree", "no seed"): (146, 123, "12.7", "4.75", ".583"),
}


def seed_sensitivity():
    for (proc, sampling), paper in SEED.items():
        if proc == "BoN":
            r = row("mini_noseed_bon_full/no_seed_bon_summary.csv", protocol="e2b_off",
                    run="seeded_rerun" if sampling == "fixed seed" else "noseed")
            got = (i(r["oracle"]), i(r["selected"]), "32.0", f2(r["unique_valid_workflows_per_task"]),
                   s3(r["win_share_mean"]))
        else:
            r = row("mini_noseed_trees_full/noseed_trees_summary.csv", protocol="e2b_off", method=proc,
                    seed="sent (20260808)" if sampling == "fixed seed" else "omitted")
            got = (i(r["oracle"]), i(r["selected"]), f1(r["retained_per_task"]),
                   f2(r["distinct_valid_workflows_per_task"]), s3(r["win_share_mean"]))
        check(f"tab:seed-sensitivity {proc} {sampling}", paper, got)


# ---------------------------------------------------------------- Tables: E2b audit and normalization (fixed-seed pools)
def e2b_and_normalization():
    paper = {"BoN": (132, 132, 118, 117), "CoT-tree": (144, 144, 120, 118), "MO-Tree": (150, 150, 117, 117)}
    for proc, p in paper.items():
        on = row("mini_noseed_trees_full/noseed_trees_summary.csv", protocol="e2b_on", method=proc, seed="sent (20260808)")
        off = row("mini_noseed_trees_full/noseed_trees_summary.csv", protocol="e2b_off", method=proc, seed="sent (20260808)")
        check(f"tab:e2b-ablation {proc}", p,
              (i(on["oracle"]), i(off["oracle"]), i(on["selected"]), i(off["selected"])))
    paper = {"BoN": ("2.0", "0.9", 117, 117), "CoT-tree": ("27.4", "2.9", 104, 118), "MO-Tree": ("25.1", "6.9", 100, 115)}
    for proc, p in paper.items():
        r = row("final_primary/historical/normalization.csv", procedure=proc)
        check(f"tab:app-normalization {proc}", p,
              (f1(r["invalid_pct_original"]), f1(r["invalid_pct_common"]),
               i(r["unfiltered_majority_original"]), i(r["unfiltered_majority_common"])))
    acc = {r["quantity"]: i(r["mean_tokens_per_task"]) for r in rows("final_primary/historical/pooled_accounting.csv")}
    check("text: pooled fixed-seed CoT-tree + narrow MO-Tree accounting (realized, deprecated)",
          (413716, 231860), (acc["pooled_realized"], acc["pooled_deprecated_8_of_32_approx"]))


# ---------------------------------------------------------------- Table: paired tests
PAIRED = {  # (run, comparison, metric): (delta tasks, delta pp, CI low, CI high, McNemar p) as printed
    ("GPT-4o-mini", "PMBoN-BoN", "Oracle"): ("+15", "+8.2", "+3.8", "+13.1", ".0015"),
    ("GPT-4o-mini", "PMBoN-CoT", "Oracle"): ("+1", "+0.5", "-3.3", "+4.4", "1.000"),
    ("GPT-4o-mini", "CoT-BoN", "Oracle"): ("+14", "+7.7", "+3.3", "+12.6", ".0026"),
    ("GPT-4o-mini", "Five-BoN", "Oracle"): ("+7", "+3.8", "-1.1", "+8.7", ".210"),
    ("GPT-4o-mini", "PMBoN-BoN", "Selected"): ("+8", "+4.4", "-1.1", "+9.8", ".169"),
    ("GPT-4o-mini", "PMBoN-CoT", "Selected"): ("-2", "-1.1", "-4.4", "+1.6", ".727"),
    ("GPT-4o-mini", "CoT-BoN", "Selected"): ("+10", "+5.5", "0.0", "+11.5", ".087"),
    ("GPT-4o-mini", "Five-BoN", "Selected"): ("+2", "+1.1", "-4.9", "+7.1", ".860"),
    ("Qwen Control R1", "PMBoN-CoT", "Oracle"): ("-2", "-1.1", "-3.3", "+1.1", ".630"),
    ("Qwen Control R2", "PMBoN-CoT", "Oracle"): ("+3", "+1.6", "-1.6", "+4.9", ".508"),
    ("Qwen Control R1", "PMBoN-CoT", "Selected"): ("-5", "-2.7", "-6.6", "+0.5", ".230"),
    ("Qwen Control R2", "PMBoN-CoT", "Selected"): ("-6", "-3.3", "-7.1", "+0.5", ".150"),
    ("Qwen Run 1", "CoT-BoN", "Oracle"): ("+3", "+1.6", "-1.6", "+5.5", ".549"),
    ("Qwen Run 1", "Five-BoN", "Oracle"): ("-12", "-6.6", "-10.9", "-2.2", ".008"),
    ("Qwen Run 1", "CoT-BoN", "Selected"): ("+3", "+1.6", "-2.2", "+5.5", ".581"),
    ("Qwen Run 1", "Five-BoN", "Selected"): ("-7", "-3.8", "-8.7", "+1.1", ".210"),
    ("Qwen Run 2", "CoT-BoN", "Oracle"): ("-5", "-2.7", "-7.1", "+1.1", ".302"),
    ("Qwen Run 2", "Five-BoN", "Oracle"): ("-15", "-8.2", "-13.1", "-3.8", ".001"),
    ("Qwen Run 2", "CoT-BoN", "Selected"): ("+5", "+2.7", "-0.5", "+6.6", ".227"),
    ("Qwen Run 2", "Five-BoN", "Selected"): ("-6", "-3.3", "-7.7", "+1.1", ".238"),
}
PAIRED_SRC = {"GPT-4o-mini": "prompt_matched_bon/eval", "Qwen Control R1": "prompt_matched_bon_qwen/eval_vs_b1",
              "Qwen Control R2": "prompt_matched_bon_qwen/eval_vs_b2", "Qwen Run 1": "e2b_off_primary/qwen_b1",
              "Qwen Run 2": "e2b_off_primary/qwen_b2"}


def pp(x):
    s = f"{float(x):+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s


def p_printed(paper, x):
    """Printed p-values mix three decimals and two significant digits; accept either rounding of x."""
    x = float(x)
    if paper.startswith("<"):
        return paper if x < float(paper[1:]) else f"{x:.4g}"
    decimals = len(paper.split(".")[1])
    two_sig = Decimal(repr(x)).quantize(Decimal(1).scaleb(Decimal(repr(x)).adjusted() - 1), rounding=ROUND_HALF_UP)
    if round(x, decimals) == float(paper) or float(two_sig) == float(paper):
        return paper
    return f"{x:.4g}"


def paired_tests():
    for (run, comp, metric), paper in PAIRED.items():
        a, b = comp.split("-")
        src = PAIRED_SRC[run]
        if run == "GPT-4o-mini" and a == "Five":
            src = BUNDLE["mini"]
        r = row(f"{src}/paired_ci.csv", a=a, b=b,
                metric="oracle" if metric == "Oracle" else "filter_majority_correct")
        got = (f"{i(r['diff_tasks']):+d}", pp(r["diff_pp"]), pp(r["boot_lo_pp"]), pp(r["boot_hi_pp"]),
               p_printed(paper[4], r["mcnemar_exact_p"]))
        check(f"tab:app-paired-tests {run} {comp} {metric}", paper, got)


# ---------------------------------------------------------------- Table: calibration-task exclusion
def calibration_exclusion():
    paper = {"BoN": (135, 117, None, None), "CoT-tree": (148, 127, "+13 (.004)", "+10 (.087)"),
             "MO-Tree": (142, 119, "+7 (.210)", "+2 (.860)")}
    for proc, p in paper.items():
        s = row("final_primary/excl_calib5_178/summary.csv", pool=POOL[proc])
        d = [None, None]
        if proc != "BoN":
            for j, metric in enumerate(("oracle", "filter_majority_correct")):
                c = row("final_primary/excl_calib5_178/paired_ci.csv", a=POOL[proc], b="BoN", metric=metric)
                d[j] = f"{i(c['diff_tasks']):+d} ({p3(c['mcnemar_exact_p'])})"
        check(f"tab:app-calibration-exclusion {proc}", p, (i(s["oracle"]), i(s["filter_majority"]), *d))


# ---------------------------------------------------------------- Table: Qwen MO-Tree width
def qwen_width():
    paper = {"Narrow": ("285k", "33.6", "7.3", "6.2", "2.67", 147, 122),
             "Medium": ("376k", "42.7", "10.1", "8.6", "2.90", 148, 122),
             "Original": ("438k", "48.6", "11.7", "9.8", "2.95", 150, 121)}
    bundles = {"Narrow": "e2b_off_primary/qwen_b1", "Medium": "final_primary/breadth/medium",
               "Original": "final_primary/breadth/orig"}
    for name, p in paper.items():
        b = row(f"{bundles[name]}/summary.csv", pool="Five")
        tasks = defaultdict(list)
        for line in open(A / bundles[name] / "per_candidate.jsonl"):
            c = json.loads(line)
            if c["pool"] == "Five":
                tasks[c["task_id"]].append(c)
        valid = [[c for c in cs if c["hard0"]] for cs in tasks.values()]
        n = len(tasks)
        got = (k(b["mean_total_tokens"]), f1(b["mean_api_calls"]), f1(b["mean_candidates"]),
               f1(sum(map(len, valid)) / n),
               f2(sum(len({json.dumps(c["plan"], sort_keys=True) for c in v}) for v in valid) / n),
               i(b["oracle"]), i(b["filter_majority"]))
        check(f"tab:app-qwen-width {name}", p, got)


# ---------------------------------------------------------------- Tables: Top-k and vote margins
TOPK = {  # Vote loss, Rank 2, Rank 3, >3, Selected, Top-2, Top-3, Oracle
    ("mini", "BoN"): (18, 11, 5, 2, 121, 132, 137, 139),
    ("mini", "CoT-tree"): (22, 10, 2, 10, 131, 141, 143, 153),
    ("mini", "MO-Tree"): (21, 13, 3, 5, 123, 138, 141, 146),
    ("mini", "Prompt-matched BoN"): (24, 13, 6, 5, 129, 143, 149, 154),
    ("qwen", "Prompt-matched BoN"): (32, 17, 7, 8, 127, 145, 152, 160),
    ("qwen_b1", "BoN"): (28, 17, 7, 4, 129, 148, 155, 159),
    ("qwen_b1", "CoT-tree"): (28, 14, 5, 9, 132, 148, 153, 162),
    ("qwen_b1", "MO-Tree"): (19, 15, 4, 0, 122, 141, 146, 147),
    ("qwen_b2", "BoN"): (33, 20, 8, 5, 128, 149, 157, 162),
    ("qwen_b2", "CoT-tree"): (22, 12, 3, 7, 133, 147, 150, 157),
    ("qwen_b2", "MO-Tree"): (18, 9, 5, 4, 122, 134, 140, 147),
}
MARGINS = {  # Correct share mean, Vote deficit mean, Rank-2 deficit median, <=1 vote, <=2 votes
    ("mini", "BoN"): (".154", "16.3", "18", 0, 1),
    ("mini", "CoT-tree"): (".147", "8.1", "5", 2, 4),
    ("mini", "MO-Tree"): (".219", "2.2", "2", 7, 15),
    ("mini", "Prompt-matched BoN"): (".162", "10.0", "7", 1, 4),
    ("qwen", "Prompt-matched BoN"): (".138", "13.3", "16", 1, 5),
    ("qwen_b1", "BoN"): (".169", "11.0", "10", 1, 1),
    ("qwen_b1", "CoT-tree"): (".144", "3.6", "3.5", 4, 9),
    ("qwen_b1", "MO-Tree"): (".229", "2.3", "2", 9, 14),
    ("qwen_b2", "BoN"): (".176", "10.1", "8.5", 2, 5),
    ("qwen_b2", "CoT-tree"): (".166", "3.6", "3", 6, 10),
    ("qwen_b2", "MO-Tree"): (".212", "2.6", "3", 6, 9),
}


def num(x):
    x = float(x)
    return str(int(x)) if x == int(x) else str(x)


def vote_rank():
    for (run, method), p in TOPK.items():
        r = row("final_primary/vote_rank/summary.csv", run=run, method=method)
        check(f"tab:app-topk-selection {run} {method}", p,
              tuple(i(r[c]) for c in ("strict_vote_loss", "rank2", "rank3", "rank_gt3", "selected", "top2", "top3", "oracle")))
    for (run, method), p in MARGINS.items():
        r = row("final_primary/vote_rank/summary.csv", run=run, method=method)
        check(f"tab:app-vote-margins {run} {method}", p,
              (s3(r["correct_share_mean"]), f1(r["margin_mean"]), num(r["margin_rank2_median"]),
               i(r["margin_le1"]), i(r["margin_le2"])))


# ---------------------------------------------------------------- Figure 1 panels and loss localization
def fig1():
    att = {(r["model"], r["procedure"], r["run"]): i(r["oracle_count"]) for r in rows("final_primary/fig1/fig1_attribution.csv")}
    check("fig1(1) Oracle CoT-tree vs prompt-matched (GPT 153/154; Qwen R1 162, R2 157, control 160)",
          (153, 154, 162, 157, 160),
          (att[("GPT-4o-mini", "CoT-tree", "primary")], att[("GPT-4o-mini", "Prompt-matched", "control")],
           att[("Qwen3.5-9B", "CoT-tree", "Run 1")], att[("Qwen3.5-9B", "CoT-tree", "Run 2")],
           att[("Qwen3.5-9B", "Prompt-matched", "control")]))
    v = {r["stage"]: (i(r["correct_workflow_available"]), i(r["no_correct_workflow"]))
         for r in rows("final_primary/fig1/fig1_validation.csv")}
    check("fig1(2) Oracle-positive pool-tasks before / after structural validation", (1686, 1686),
          (v["Before validation"][0], v["After validation"][0]))
    loss = {r["loss_type"]: (i(r["count"]), r["pct"]) for r in rows("final_primary/fig1/fig1_loss_localization.csv")}
    check("fig1(3) gap = filter + vote + tie (11 pools)", (289, 0, 265, "91.7", 24, "8.3"),
          (loss["Total"][0], loss["Filter loss"][0], loss["Vote loss"][0], loss["Vote loss"][1],
           loss["Tie loss"][0], loss["Tie loss"][1]))
    topk = {r["k"]: r["recovery_pct"] for r in rows("final_primary/fig1/fig1_topk.csv")}
    check("fig1(4) pooled Top-2 / Top-3 recovery of the gap (%)", ("58.5", "78.2"), (topk["2"], topk["3"]))
    div = rows("final_primary/fig1/fig1_diversity.csv")
    check("fig1(5) pools with vote-loss median > all-task median; pooled medians", (11, "5", "2"),
          (sum(float(r["vote_loss_median"]) > float(r["all_task_median"]) for r in div),
           num(statistics.median([float(r["vote_loss_median"]) for r in div])),
           num(statistics.median([float(r["all_task_median"]) for r in div]))))


# ---------------------------------------------------------------- Tables: selector sensitivity and judge
SENS = {  # Primary, Unfiltered, Occurrence, Unique text, First valid, Tie upper bound
    ("mini", "BoN"): (121, 121, 121, 119, 119, 121), ("mini", "CoT-tree"): (131, 129, 130, 131, 112, 131),
    ("mini", "MO-Tree"): (123, 122, 123, 123, 104, 125), ("qwen_b1", "BoN"): (129, 129, 129, 128, 121, 131),
    ("qwen_b1", "CoT-tree"): (132, 132, 127, 132, 122, 134), ("qwen_b1", "MO-Tree"): (122, 122, 127, 122, 114, 128),
    ("qwen_b2", "BoN"): (128, 128, 128, 130, 119, 129), ("qwen_b2", "CoT-tree"): (133, 133, 128, 133, 118, 135),
    ("qwen_b2", "MO-Tree"): (122, 121, 122, 122, 110, 129),
}
JUDGE = {  # Primary, GPT-4o judge, Tie upper bound, Oracle
    ("mini", "BoN"): (121, 123, 121, 139), ("mini", "CoT-tree"): (131, 112, 131, 153),
    ("mini", "MO-Tree"): (123, 112, 125, 146), ("qwen_b1", "BoN"): (129, 110, 131, 159),
    ("qwen_b1", "CoT-tree"): (132, 114, 134, 162), ("qwen_b1", "MO-Tree"): (122, 128, 128, 147),
    ("qwen_b2", "BoN"): (128, 106, 129, 162), ("qwen_b2", "CoT-tree"): (133, 115, 135, 157),
    ("qwen_b2", "MO-Tree"): (122, 121, 129, 147),
}


def selectors():
    for (run, method), p in SENS.items():
        r = row("final_primary/selector_sensitivity.csv", run=run, method=method)
        check(f"tab:app-voting-sensitivity {run} {method}", p,
              tuple(i(r[c]) for c in ("primary", "unfiltered_majority", "occurrence", "unique_text", "first_valid",
                                      "oracle_tiebreak")))
    for (run, method), p in JUDGE.items():
        r = row("final_primary/selector_sensitivity.csv", run=run, method=method)
        check(f"tab:app-judge {run} {method}", p,
              tuple(i(r[c]) for c in ("primary", "gpt4o_judge", "oracle_tiebreak", "oracle")))


# ---------------------------------------------------------------- Tables: Top-3 verifier
VERIFIER = {  # Majority, Verifier, Fixed, Broken, Net, McNemar p
    ("mini", "BoN"): (121, 123, 11, 9, 2, ".820"), ("mini", "CoT-tree"): (131, 113, 3, 21, -18, "<.001"),
    ("mini", "MO-Tree"): (123, 108, 12, 27, -15, ".024"), ("qwen_b1", "BoN"): (129, 116, 9, 22, -13, ".029"),
    ("qwen_b1", "CoT-tree"): (132, 117, 11, 26, -15, ".020"), ("qwen_b1", "MO-Tree"): (122, 121, 12, 13, -1, "1.000"),
    ("qwen_b2", "BoN"): (128, 118, 13, 23, -10, ".130"), ("qwen_b2", "CoT-tree"): (133, 122, 8, 19, -11, ".052"),
    ("qwen_b2", "MO-Tree"): (122, 118, 11, 15, -4, ".560"),
}
GATED = {  # Routed, Changed, Fixed, Broken, Net, Selected, McNemar p
    "0": (92, 49, 10, 9, 1, 1142, "1.000"), "0.05": (114, 65, 13, 13, 0, 1141, "1.000"),
    "0.1": (190, 118, 22, 24, -2, 1139, ".880"), "0.15": (243, 147, 31, 27, 4, 1145, ".690"),
    "0.2": (321, 178, 37, 34, 3, 1144, ".810"), "0.3": (437, 231, 53, 51, 2, 1143, ".920"),
    "0.4": (542, 271, 66, 63, 3, 1144, ".860"), "0.5": (652, 321, 76, 88, -12, 1129, ".390"),
    "inf": (1105, 455, 90, 175, -85, 1056, "<.001"),
}


def p_two(x):
    """McNemar p as printed in the verifier tables (two significant digits, three decimals)."""
    x = float(x)
    if x < 0.001:
        return "<.001"
    if x >= 0.9995:
        return "1.000"
    return f"{float(f'{x:.2g}'):.3f}"[1:]


def verifier():
    for (run, method), p in VERIFIER.items():
        r = row("final_primary/top3_selector/summary.csv", run=run, method=method)
        check(f"tab:app-targeted-verifier {run} {method}", p,
              (i(r["majority_selected"]), i(r["verifier_selected"]), i(r["fixed"]), i(r["broken"]), i(r["net"]),
               p_two(r["mcnemar_p"])))
    a = row("final_primary/top3_selector/summary.csv", run="all", method="all")
    check("text: deployable Top-3 over 9 pools (majority, verifier, fixed, broken, contested calls)",
          (1141, 1056, 90, 175, 1105),
          (i(a["majority_selected"]), i(a["verifier_selected"]), i(a["fixed"]), i(a["broken"]), i(a["verifier_calls"])))
    d = {r["group"]: r for r in rows("final_primary/verifier_failure_modes/summary.csv")}
    fx, br, mc = d["fixed"], d["broken"], d["majority-correct contested (638)"]
    check("tab:app-verifier-changed-decisions fixes (90)", (90, "0.258", "0.556", "3", "84.4", "15.6", "27.8"),
          (i(fx["n"]), f"{float(fx['median_norm_margin']):.3f}", f"{float(fx['median_winner_share']):.3f}",
           num(fx["median_num_valid_groups"]), f1(fx["chose_rank2_pct"]), f1(fx["chose_rank3_pct"]),
           f1(fx["chosen_more_steps_than_majority_pct"])))
    check("tab:app-verifier-changed-decisions false overrides (175)", (175, "0.500", "0.700", "4", "64.6", "35.4", "52.6"),
          (i(br["n"]), f"{float(br['median_norm_margin']):.3f}", f"{float(br['median_winner_share']):.3f}",
           num(br["median_num_valid_groups"]), f1(br["chose_rank2_pct"]), f1(br["chose_rank3_pct"]),
           f1(br["chosen_more_steps_than_majority_pct"])))
    check("text: false overrides preserving the reference tool sequence (%)", "20.6", f1(br["chosen_wrong_tool_acc"]))
    check("text: majority-correct contested cases with normalized margin > 0.3 (%)", "76",
          f"{round(float(mc['norm_margin_gt_0.3_pct']))}")
    bins = {"[0.0, 0.1)": (51, 22, "43.1", 21, -1), "[0.1, 0.2)": (39, 9, "23.1", 16, 7),
            "[0.2, 0.3)": (60, 18, "30.0", 15, -3), "[0.3, 0.5)": (94, 28, "29.8", 20, -8),
            "[0.5, 1.0]": (394, 98, "24.9", 18, -80)}
    for b, p in bins.items():
        r = row("final_primary/verifier_failure_modes/margin_bins.csv", norm_margin_bin=b)
        check(f"tab:app-verifier-margin-bins {b}", p,
              (i(r["majority_correct_contested"]), i(r["broken"]), f1(r["break_rate_pct"]), i(r["fixed"]), i(r["net"])))
    for t, p in GATED.items():
        r = row("final_primary/margin_gated/sweep_normalized.csv", margin="norm_margin", t=t)
        check(f"tab:app-margin-gated-verifier t={t}", p,
              (i(r["gated"]), i(r["overrides"]), i(r["fixed"]), i(r["broken"]), i(r["net"]), i(r["final_selected"]),
               p_two(r["mcnemar_p"])))


# ---------------------------------------------------------------- vote-loss audit, partial credit, modal agreement, Task 22
def audits():
    for etype, p in {"Tool": (17, 18, 35), "Dependency": (13, 9, 22), "SemanticEquivalent": (0, 3, 3)}.items():
        r = row("vote_loss_audit_final/tally_primary_by_model.csv", primary_error_type=etype)
        check(f"tab:app-error-audit {etype}", p,
              (i(r["GPT-4o-mini"]), i(r["Qwen3.5-9B non-thinking (batch 20260725)"]), i(r["Total"])))
    for g, p in {"Rank 2": (151, "29.8", "0.0", "2.6"), "Rank 3": (55, "25.5", "0.0", "1.8"),
                 "Rank >3": (59, "23.7", "0.0", "1.7"), "All": (265, "27.5", "0.0", "2.3")}.items():
        r = row("final_primary/voteloss_partial_credit/summary_by_rank.csv", rank_group=g)
        check(f"tab:app-vote-loss-partial-rank {g}", p,
              (i(r["n"]), f1(r["tool_acc"]), f1(r["parameter_acc"]), f1(r["dependency_acc"])))
    for g, p in {"low-margin": (133, "18.8", "0.0", "3.0"), "high-margin": (132, "36.4", "0.0", "1.5")}.items():
        r = row("final_primary/voteloss_partial_credit/summary_by_margin.csv", margin_group=g)
        check(f"tab:app-vote-loss-partial-margin {g}", p,
              (i(r["n"]), f1(r["tool_acc"]), f1(r["parameter_acc"]), f1(r["dependency_acc"])))
    r = row("final_primary/voteloss_partial_credit/summary_by_rank.csv", rank_group="All")
    check("text: vote-loss winners: wrong tools / tools only ok / params only wrong (%)", ("72.5", "25.3", "2.3"),
          (f1(r["tool_wrong_pct"]), f1(r["tool_only_ok_pct"]), f1(r["param_wrong_only_pct"])))
    modal = {"mini": ("3/4", "4/6", "3/6"), "qwen_b1": ("14/15", "5/5", "6/7"), "qwen_b2": ("13/15", "4/6", "3/5")}
    for run, p in modal.items():
        got = []
        for m in ("BoN & CoT-tree", "BoN & MO-Tree", "CoT-tree & MO-Tree"):
            r = row("selection_rebuild/primary/modal_error_agreement.csv", run=run, methods=m)
            got.append(f"{r['exact_same_winner']}/{r['overlap_tasks']}")
        check(f"tab:app-modal-agreement {run}", p, tuple(got))
    t22 = {"BoN": (7, 25, "False"), "Prompt-matched BoN": (29, 1, "True"), "CoT-tree": (28, 2, "True"),
           "MO-Tree": (4, 3, "True")}
    for m, p in t22.items():
        r = row("final_primary/task22_votes.csv", run="mini", method=m)
        check(f"tab:app-task22 {m}", p, (i(r["correct_votes"]), i(r["miswired_votes"]), r["selected_correct"]))


# ---------------------------------------------------------------- normalization flips and backend audit (text)
def text_audits():
    r = row("final_primary/normalization_em_flips/summary.csv", run="all")
    check("text: normalization EM flips over 11 pools (candidates, incorrect->correct, correct->incorrect, Oracle)",
          (45246, 1612, 0, 1686),
          (i(r["candidates"]), i(r["incorrect_to_correct"]), i(r["correct_to_incorrect"]), i(r["oracle_normalized"])))
    b = {(r["metric"], r["run"]): r for r in rows("final_primary/backend_sampling/summary.csv")}
    check("text: backend audit (GPT n>1 single-choice; Qwen n>1 full; tie-break calls; thinking no-JSON)",
          ((5082, 5082), (4607, 4607), (4598, 4598), (0, 216, 187), (672, 1360)),
          ((i(b["gpt_mini_mo_n_gt1", "mini"]["count"]), i(b["gpt_mini_mo_n_gt1", "mini"]["subcount"])),
           (i(b["qwen_mo_n_gt1", "qwen_b1"]["count"]), i(b["qwen_mo_n_gt1", "qwen_b1"]["subcount"])),
           (i(b["qwen_mo_n_gt1", "qwen_b2"]["count"]), i(b["qwen_mo_n_gt1", "qwen_b2"]["subcount"])),
           tuple(i(b["mo_tiebreak_calls", r]["count"]) for r in ("mini", "qwen_b1", "qwen_b2")),
           (i(b["qwen_thinking_no_plan", "qwen_thinking_cot_seed20260725"]["subcount"]),
            i(b["qwen_thinking_no_plan", "qwen_thinking_cot_seed20260725"]["count"]))))


def main():
    for fn in (main_results, pool_stats, run_sensitivity, seed_sensitivity, e2b_and_normalization, paired_tests,
               calibration_exclusion, qwen_width, vote_rank, fig1, selectors, verifier, audits, text_audits):
        fn()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "paper", "reproduced", "ok"])
        w.writeheader()
        w.writerows(RESULTS)
    bad = [r for r in RESULTS if not r["ok"]]
    for r in bad:
        print(f"MISMATCH {r['item']}: paper {r['paper']} reproduced {r['reproduced']}")
    print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} paper checks match; wrote {OUT.relative_to(ROOT)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
