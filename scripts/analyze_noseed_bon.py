#!/usr/bin/env python3
"""
Seeded vs no-seed GPT-4o-mini BoN-32 (183 tasks), offline, under both evaluation protocols
(E2b on = locked, E2b off = new primary). No LLM calls.

Per candidate: task_id, candidate_id, raw_output (raw_completion; only stored by the no-seed run),
model_answer (extracted answer before normalization), normalized_workflow, valid (hard0), exact_match.
Per task: oracle, selected, n_valid, invalid, unique answer strings (pre-normalization; comparable across
both runs), unique raw completions (no-seed only), unique valid normalized workflows, winning share,
tied, tokens. Selection = locked rule (hard0 filter, exact-JSON occurrence majority, tie -> smallest
candidate_id). The seeded reconstruction is checked against the locked bundles.

Paired: Oracle / Selected (exact McNemar, paired bootstrap, Wald) and per-task diversity differences
(paired bootstrap of the mean). No-seed BoN Oracle is also paired against CoT-tree and MO-Tree.
"""
import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_budget_curve as abc  # noqa: E402
from freeze_strict_matched import majority_pick, paired_stats  # noqa: E402

BUNDLES = {"e2b_on": ROOT / "artifacts/phase2_fulltest/FINAL_LOCKED/review_bundle",
           "e2b_off": ROOT / "artifacts/e2b_off_primary/mini"}


def key(plan):
    return json.dumps(plan, sort_keys=True, ensure_ascii=False)


def evaluate_run(run_dir, collapse):
    abc.COLLAPSE_MULTI = collapse
    cand_rows, tasks = [], {}
    for tid, a in sorted(abc.load_answers(run_dir).items()):
        cs = sorted(a.get("all_candidates") or [], key=lambda x: int(x.get("candidate_id", 0)))
        items = []
        for c in cs:
            ma = c.get("model_answer")
            s = abc.score(ma, tid)
            it = {**s, "candidate_id": int(c["candidate_id"])}
            items.append(it)
            cand_rows.append({"task_id": tid, "candidate_id": it["candidate_id"],
                              "raw_output": c.get("raw_completion"),
                              "model_answer": ma if isinstance(ma, str) else json.dumps(ma, ensure_ascii=False),
                              "normalized_workflow": key(s["plan"]) if s["plan"] is not None else None,
                              "valid": s["hard0"], "exact_match": s["correct"]})
        valid = [x for x in items if x["hard0"]]
        pick = majority_pick(valid) if valid else majority_pick(items)
        g = Counter(key(x["plan"]) for x in valid)
        top = max(g.values()) if g else 0
        answers = [c.get("model_answer") if isinstance(c.get("model_answer"), str)
                   else json.dumps(c.get("model_answer"), sort_keys=True) for c in cs]
        raws = [c.get("raw_completion") for c in cs]
        u = a.get("usage") or {}
        tasks[tid] = {"task_id": tid, "oracle": any(x["correct"] for x in items),
                      "selected": bool(pick and pick["correct"]),
                      "n_candidates": len(items), "n_valid": len(valid),
                      "unique_answer_strings": len(set(answers)),
                      "unique_raw_completions": len({r.strip() for r in raws}) if all(raws) else None,
                      "unique_valid_workflows": len(g),
                      "win_share": top / len(valid) if valid else None,
                      "tied": sum(v == top for v in g.values()) >= 2 if g else False,
                      "total_tokens": u.get("total_tokens"), "completion_tokens": u.get("completion_tokens")}
    return cand_rows, tasks


def summarize(label, proto, T):
    rs = list(T.values())
    n_c = sum(r["n_candidates"] for r in rs)
    v = [r for r in rs if r["win_share"] is not None]
    raw = [r["unique_raw_completions"] for r in rs if r["unique_raw_completions"] is not None]
    return {"run": label, "protocol": proto, "n_tasks": len(rs),
            "oracle": sum(r["oracle"] for r in rs), "selected": sum(r["selected"] for r in rs),
            "invalid_pct": 100 * (1 - sum(r["n_valid"] for r in rs) / n_c),
            "unique_answer_strings_per_task": statistics.fmean(r["unique_answer_strings"] for r in rs),
            "unique_raw_completions_per_task": statistics.fmean(raw) if raw else None,
            "unique_valid_workflows_per_task": statistics.fmean(r["unique_valid_workflows"] for r in rs),
            "tasks_single_workflow": sum(r["unique_valid_workflows"] == 1 for r in rs),
            "win_share_mean": statistics.fmean(r["win_share"] for r in v),
            "win_share_median": statistics.median(r["win_share"] for r in v),
            "tied_tasks": sum(r["tied"] for r in rs),
            "tokens_per_task": statistics.fmean(r["total_tokens"] for r in rs),
            "completion_tokens_per_task": statistics.fmean(r["completion_tokens"] for r in rs)}


def boot_mean_diff(a, b, n_boot=10000, seed=0):
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(d)
    ms = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return statistics.fmean(d), ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot) - 1]


def wcsv(p, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeded_dir", default=json.load(open(ROOT / "artifacts/phase2_fulltest/manifest_mini.json"))["runs"][0]["dir"])
    ap.add_argument("--noseed_dir", default=json.load(open(ROOT / "artifacts/mini_noseed_bon_full/run.json"))["dir"])
    ap.add_argument("--rerun_dir", default=None,
                    help="optional seeded rerun made at the same time as the no-seed run (separates seed from drift)")
    ap.add_argument("--out", default=str(ROOT / "artifacts/mini_noseed_bon_full"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = {"seeded": args.seeded_dir, "noseed": args.noseed_dir}
    if args.rerun_dir:
        run_dirs["seeded_rerun"] = args.rerun_dir
    pairs = [("noseed", "seeded")]
    if args.rerun_dir:
        pairs += [("noseed", "seeded_rerun"), ("seeded_rerun", "seeded")]
    summary, paired, task_rows, checks = [], [], [], {}
    for proto, collapse in (("e2b_off", False), ("e2b_on", True)):
        ev_runs = {lab: evaluate_run(ROOT / d, collapse) for lab, d in run_dirs.items()}
        R = {lab: T for lab, (_, T) in ev_runs.items()}
        locked = {(r["pool"], int(r["task_id"])): r for r in csv.DictReader(open(BUNDLES[proto] / "per_task.csv"))}
        ids = sorted(set.intersection(*(set(T) for T in R.values())) & {t for p, t in locked if p == "BoN"})
        S = R["seeded"]
        checks[proto] = {"n_common_tasks": len(ids), "seeded_reconstruction_mismatches": sum(
            (S[t]["oracle"] != (locked[("BoN", t)]["oracle"] == "True")) or
            (S[t]["selected"] != (locked[("BoN", t)]["filter_majority_correct"] == "True")) for t in ids)}
        R = {lab: {t: T[t] for t in ids} for lab, T in R.items()}
        summary += [summarize(lab, proto, T) for lab, T in R.items()]
        for a_lab, b_lab in pairs:
            A, B = R[a_lab], R[b_lab]
            for m in ("oracle", "selected"):
                paired.append({"protocol": proto, "comparison": f"{a_lab} BoN - {b_lab} BoN", "metric": m,
                               **paired_stats([A[t][m] for t in ids], [B[t][m] for t in ids])})
            for m in ("unique_answer_strings", "unique_valid_workflows", "win_share", "n_valid"):
                a = [A[t][m] if A[t][m] is not None else 0 for t in ids]
                b = [B[t][m] if B[t][m] is not None else 0 for t in ids]
                d, lo, hi = boot_mean_diff(a, b)
                paired.append({"protocol": proto, "comparison": f"{a_lab} BoN - {b_lab} BoN", "metric": f"{m}_mean",
                               "n": len(ids), "a_mean": statistics.fmean(a), "b_mean": statistics.fmean(b),
                               "diff_mean": d, "boot_lo": lo, "boot_hi": hi,
                               "tasks_a_higher": sum(x > y for x, y in zip(a, b)),
                               "tasks_b_higher": sum(y > x for x, y in zip(a, b))})
        for lab in R:
            for pool, name in (("CoT", "CoT-tree"), ("Five", "MO-Tree")):
                for m, col in (("oracle", "oracle"), ("selected", "filter_majority_correct")):
                    paired.append({"protocol": proto, "comparison": f"{name} - {lab} BoN", "metric": m,
                                   **paired_stats([locked[(pool, t)][col] == "True" for t in ids],
                                                  [R[lab][t][m] for t in ids])})
        for t in ids:
            row = {"protocol": proto, "task_id": t}
            for lab, T in R.items():
                row.update({f"{lab}_{k}": v for k, v in T[t].items() if k != "task_id"})
            task_rows.append(row)
        if proto == "e2b_off":
            for lab, (cs, _) in ev_runs.items():
                wcsv(out / f"{lab}_per_candidate.csv", [r for r in cs if r["task_id"] in set(ids)])
    wcsv(out / "no_seed_bon_summary.csv", summary)
    wcsv(out / "no_seed_bon_paired.csv", paired)
    wcsv(out / "no_seed_bon_per_task.csv", task_rows)
    (out / "no_seed_bon_summary.json").write_text(json.dumps(
        {"checks": checks, "summary": summary, "paired": paired}, indent=1, default=str))
    print(json.dumps(checks))
    for r in summary:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    for r in paired:
        if "mcnemar_exact_p" in r:
            print(f"[{r['protocol']}] {r['comparison']:26s} {r['metric']:9s} {r['diff_tasks']:+d} ({r['a_only']}/{r['b_only']}) "
                  f"boot[{r['boot_lo_pp']:+.1f},{r['boot_hi_pp']:+.1f}]pp p={r['mcnemar_exact_p']:.3g}")
        else:
            print(f"[{r['protocol']}] {r['comparison']:34s} {r['metric']:28s} {r['a_mean']:.3f} vs {r['b_mean']:.3f} "
                  f"diff {r['diff_mean']:+.3f} [{r['boot_lo']:+.3f},{r['boot_hi']:+.3f}] "
                  f"tasks +{r['tasks_a_higher']}/-{r['tasks_b_higher']}")


if __name__ == "__main__":
    main()
