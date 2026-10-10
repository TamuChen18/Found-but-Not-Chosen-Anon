#!/usr/bin/env python3
"""
No-seed GPT-4o-mini CoT-tree / MO-Tree reruns (183 tasks) against the seeded August runs and the no-seed
BoN-32, under the primary protocol (E4 off, E2b off, hard0 filter, exact-JSON majority; BoN occurrence
voting, trees unique-text voting; tie -> smallest candidate_id). No LLM calls.

Inputs:
  artifacts/mini_noseed_trees_full/eval_e2b_off  freeze_strict_matched.py --keep_multi_pred on the all-no-seed manifest
  artifacts/mini_noseed_trees_full/eval_e2b_on   same, E2b on (sensitivity)
  artifacts/e2b_off_primary/mini, FINAL_LOCKED/review_bundle   seeded August bundles (E2b off / on)
  answer sheets of both run sets (raw solution texts), logs/calllogs_mini_noseed_trees_full (request provenance)

Per task: oracle, selected, api_calls, raw completions (sum of n over requests, from logs; seeded from the
locked config), retained candidates, unique retained solution texts, valid, distinct valid workflows,
winning share, tied, tokens.

Outputs (artifacts/mini_noseed_trees_full/): noseed_trees_summary.csv, noseed_trees_paired.csv,
noseed_trees_per_task.csv, provenance_summary.json, noseed_trees_results.json
"""
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_noseed_bon import boot_mean_diff, wcsv  # noqa: E402
from freeze_strict_matched import load_tasks, paired_stats  # noqa: E402

O = ROOT / "artifacts/mini_noseed_trees_full"
BUNDLES = {
    "e2b_off": {"noseed": O / "eval_e2b_off", "seeded": ROOT / "artifacts/e2b_off_primary/mini"},
    "e2b_on": {"noseed": O / "eval_e2b_on", "seeded": ROOT / "artifacts/phase2_fulltest/FINAL_LOCKED/review_bundle"},
}
MANIFESTS = {"noseed": O / "manifest_allnoseed.json", "seeded": ROOT / "artifacts/phase2_fulltest/manifest_mini.json"}
RUN_KEYS = {"BoN": "BoN-CoT-M", "CoT": "CoT-only-M", "Five": "Five-M"}
NAME = {"BoN": "BoN", "CoT": "CoT-tree", "Five": "MO-Tree"}
LOGS = {"CoT": ROOT / "logs/calllogs_mini_noseed_trees_full/cot.jsonl",
        "Five": ROOT / "logs/calllogs_mini_noseed_trees_full/five.jsonl",
        "BoN": ROOT / "logs/calllogs_mini_noseed_full/bon32.jsonl"}
PAIRS = [(("noseed", "CoT"), ("seeded", "CoT")), (("noseed", "CoT"), ("noseed", "BoN")),
         (("noseed", "Five"), ("seeded", "Five")), (("noseed", "Five"), ("noseed", "BoN")),
         (("noseed", "CoT"), ("noseed", "Five")), (("noseed", "BoN"), ("seeded", "BoN"))]


def key(plan):
    return json.dumps(plan, sort_keys=True, ensure_ascii=False)


def run_dirs(label):
    m = json.load(open(MANIFESTS[label]))
    by = {r["key"]: ROOT / r["dir"] for r in m["runs"] if int(r["seed"]) == 20260808}
    return {p: by[k] for p, k in RUN_KEYS.items()}


def log_stats():
    out, per_task = {}, {}
    for pool, path in LOGS.items():
        R = [json.loads(l) for l in open(path)]
        ok = [r for r in R if r["status"] == "ok"]
        by_task = defaultdict(list)
        for r in ok:
            by_task[r.get("task_id")].append(r)
        per_task[pool] = {t: {"logged_calls": len(rs), "raw_completions": sum(r.get("n_choices") or r["n"] for r in rs),
                              "max_same_prompt_calls": max(Counter(r.get("prompt_sha256") for r in rs).values())
                              if rs[0].get("prompt_sha256") else None}
                          for t, rs in by_task.items()}
        out[NAME[pool]] = {
            "log": str(path.relative_to(ROOT)), "records": len(R), "ok": len(ok),
            "status": dict(Counter(r["status"] for r in R)),
            "tasks": len(by_task),
            "seed_sent": dict(Counter(str(r.get("seed_sent")) for r in ok)),
            "seed_field": dict(Counter(str(r.get("seed")) for r in ok)) if "seed" in ok[0] else "not logged",
            "temperature": dict(Counter(str(r.get("temperature")) for r in ok)),
            "top_p": dict(Counter(str(r.get("top_p")) for r in ok)) if "top_p" in ok[0] else "not logged",
            "max_tokens": dict(Counter(str(r.get("max_tokens")) for r in ok)),
            "n": dict(Counter(str(r["n"]) for r in ok)),
            "operator": dict(Counter(str(r.get("operator")) for r in ok)),
            "response_id_present": sum(bool(r.get("response_id")) for r in ok),
            "unique_call_ids": len({r.get("call_id") for r in ok}) if "call_id" in ok[0] else "not logged",
            "calls_with_task_id": sum(r.get("task_id") is not None for r in ok),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in ok),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in ok),
        }
    return out, per_task


def task_metrics(bundle, label, dirs, logs):
    cands = defaultdict(list)
    for l in open(bundle / "per_candidate.jsonl"):
        c = json.loads(l)
        cands[(c["pool"], int(c["task_id"]))].append(c)
    pt = {(r["pool"], int(r["task_id"])): r for r in csv.DictReader(open(bundle / "per_task.csv"))}
    sheets = {p: load_tasks(d) for p, d in dirs.items()}
    T = {}
    for (pool, tid), r in pt.items():
        cs = cands[(pool, tid)]
        valid = [c for c in cs if c["hard0"]]
        g = Counter(key(c["plan"]) for c in valid)
        top = max(g.values()) if g else 0
        sol = [c.get("solution") or c.get("model_answer") for c in sheets[pool][tid].get("all_candidates") or []]
        lg = logs.get(pool, {}).get(tid) if label == "noseed" else None
        api = int(r["api_calls"])
        raw = lg["raw_completions"] if lg else (api if pool in ("BoN", "CoT") else None)
        T[(pool, tid)] = {
            "oracle": r["oracle"] == "True", "selected": r["filter_majority_correct"] == "True",
            "api_calls": api, "raw_completions": raw, "retained": len(cs),
            "unique_retained_texts": len({str(s).strip() for s in sol}),
            "duplicates_removed": (raw - len(cs)) if raw is not None and pool != "Five" else None,
            "valid": len(valid), "distinct_valid_workflows": len(g),
            "win_share": top / len(valid) if valid else None, "win_votes": top,
            "tied": sum(v == top for v in g.values()) >= 2 if g else False,
            "total_tokens": int(r["total_tokens"]), "completion_tokens": int(r["completion_tokens"]),
            "logged_calls": lg["logged_calls"] if lg else None,
            "max_same_prompt_calls": lg["max_same_prompt_calls"] if lg else None,
        }
    return T


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def main():
    prov, logs = log_stats()
    dirs = {lab: run_dirs(lab) for lab in MANIFESTS}
    summary, paired, task_rows, checks = [], [], [], {}
    for proto, bs in BUNDLES.items():
        M = {lab: task_metrics(b, lab, dirs[lab], logs) for lab, b in bs.items()}
        ids = sorted({t for lab in M for (_, t) in M[lab]})
        assert all(len({t for (p, t) in M[lab] if p == pool}) == 183 for lab in M for pool in NAME)
        for lab in M:
            for pool in NAME:
                rs = [M[lab][(pool, t)] for t in ids]
                summary.append({
                    "protocol": proto, "seed": "omitted" if lab == "noseed" else "sent (20260808)",
                    "method": NAME[pool], "n_tasks": len(rs),
                    "oracle": sum(r["oracle"] for r in rs), "selected": sum(r["selected"] for r in rs),
                    "api_calls_per_task": mean(r["api_calls"] for r in rs),
                    "raw_completions_per_task": mean(r["raw_completions"] for r in rs),
                    "retained_per_task": mean(r["retained"] for r in rs),
                    "unique_retained_texts_per_task": mean(r["unique_retained_texts"] for r in rs),
                    "duplicates_removed_per_task": mean(r["duplicates_removed"] for r in rs),
                    "tasks_with_any_duplicate": sum((r["duplicates_removed"] or 0) > 0 for r in rs)
                    if rs[0]["duplicates_removed"] is not None else None,
                    "valid_per_task": mean(r["valid"] for r in rs),
                    "invalid_pct": 100 * (1 - sum(r["valid"] for r in rs) / sum(r["retained"] for r in rs)),
                    "distinct_valid_workflows_per_task": mean(r["distinct_valid_workflows"] for r in rs),
                    "tasks_single_workflow": sum(r["distinct_valid_workflows"] == 1 for r in rs),
                    "win_share_mean": mean(r["win_share"] for r in rs),
                    "win_share_median": statistics.median(r["win_share"] for r in rs if r["win_share"] is not None),
                    "win_votes_mean": mean(r["win_votes"] for r in rs),
                    "tied_tasks": sum(r["tied"] for r in rs),
                    "tokens_per_task": mean(r["total_tokens"] for r in rs),
                    "completion_tokens_per_task": mean(r["completion_tokens"] for r in rs),
                })
        if proto == "e2b_off":
            checks["usage_api_calls_vs_logged_ok_calls_mismatch"] = {
                NAME[p]: sum(M["noseed"][(p, t)]["api_calls"] != M["noseed"][(p, t)]["logged_calls"] for t in ids)
                for p in ("CoT", "Five")}
            checks["n_gt1_requests_returning_one_choice"] = {
                NAME[p]: sum(r["n"] > 1 and r.get("n_choices") == 1 for r in map(json.loads, open(LOGS[p])))
                for p in ("CoT", "Five")}
        for (la, pa), (lb, pb) in PAIRS:
            A, B = M[la], M[lb]
            comp = f"{la} {NAME[pa]} - {lb} {NAME[pb]}"
            for m in ("oracle", "selected"):
                paired.append({"protocol": proto, "comparison": comp, "metric": m,
                               **paired_stats([A[(pa, t)][m] for t in ids], [B[(pb, t)][m] for t in ids])})
            for m in ("retained", "valid", "distinct_valid_workflows", "win_share", "total_tokens"):
                a = [A[(pa, t)][m] or 0 for t in ids]
                b = [B[(pb, t)][m] or 0 for t in ids]
                d, lo, hi = boot_mean_diff(a, b)
                paired.append({"protocol": proto, "comparison": comp, "metric": f"{m}_mean", "n": len(ids),
                               "a_mean": statistics.fmean(a), "b_mean": statistics.fmean(b),
                               "diff_mean": d, "boot_lo": lo, "boot_hi": hi,
                               "tasks_a_higher": sum(x > y for x, y in zip(a, b)),
                               "tasks_b_higher": sum(y > x for x, y in zip(a, b))})
        for t in ids:
            row = {"protocol": proto, "task_id": t}
            for lab in M:
                for p in NAME:
                    row.update({f"{lab}_{p}_{k}": v for k, v in M[lab][(p, t)].items()})
            task_rows.append(row)

    wcsv(O / "noseed_trees_summary.csv", summary)
    wcsv(O / "noseed_trees_paired.csv", paired)
    wcsv(O / "noseed_trees_per_task.csv", task_rows)
    (O / "provenance_summary.json").write_text(json.dumps(prov, indent=1))
    (O / "noseed_trees_results.json").write_text(json.dumps(
        {"checks": checks, "summary": summary, "paired": paired, "provenance": prov}, indent=1, default=str))
    print(json.dumps(checks))
    for r in summary:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    for r in paired:
        if "mcnemar_exact_p" in r:
            print(f"[{r['protocol']}] {r['comparison']:30s} {r['metric']:9s} {r['diff_tasks']:+d} ({r['a_only']}/{r['b_only']}) "
                  f"boot[{r['boot_lo_pp']:+.1f},{r['boot_hi_pp']:+.1f}]pp p={r['mcnemar_exact_p']:.3g}")
        elif r["protocol"] == "e2b_off":
            print(f"[{r['protocol']}] {r['comparison']:30s} {r['metric']:30s} {r['a_mean']:.3f} vs {r['b_mean']:.3f} "
                  f"diff {r['diff_mean']:+.3f} [{r['boot_lo']:+.3f},{r['boot_hi']:+.3f}]")
    for k, v in prov.items():
        print(k, {x: v[x] for x in ("records", "ok", "tasks", "seed_sent", "seed_field", "temperature", "max_tokens", "n")})


if __name__ == "__main__":
    main()
