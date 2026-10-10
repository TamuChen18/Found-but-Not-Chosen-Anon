#!/usr/bin/env python3
"""
Selector sensitivity and GPT-4o judge on the final primary pools (E4 off, E2b off, hard0 filter).
GPT-4o-mini = all-no-seed pools (artifacts/mini_noseed_trees_full/eval_e2b_off); Qwen pools are unchanged
from the E2b-off primary bundles, so their selector / judge results are reused from there.

Inputs:
  analyze_vote_multiplicity.py --keep_multi_pred on the no-seed manifest  -> artifacts/final_primary/vote_multiplicity
  run_locked_judge.py on the no-seed bundle (fresh cache)                  -> artifacts/final_primary/judge
  artifacts/e2b_off_primary/{vote_multiplicity,judge}                      (Qwen)
  artifacts/selection_rebuild/stage_loss.csv                               (tie losses)

Selectors (all after the same hard0 filter unless stated):
  primary          exact-JSON majority (BoN occurrence, trees unique-text), tie -> smallest candidate_id
  unfiltered       same majority without the hard0 filter
  occurrence       one vote per generation unit
  unique_text      one vote per unique solution text
  first_valid      first valid candidate in generation order
  oracle_tiebreak  diagnostic upper bound of tie-breaking: primary + tie losses
  gpt4o_judge      blind GPT-4o pick among <=8 unique valid workflows

Outputs (artifacts/final_primary/): selector_sensitivity.csv, judge_paired.csv, final_primary_selection.json
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_strict_matched import paired_stats  # noqa: E402

O = ROOT / "artifacts/final_primary"
BUNDLES = {"mini": ROOT / "artifacts/mini_noseed_trees_full/eval_e2b_off",
           "qwen_b1": ROOT / "artifacts/e2b_off_primary/qwen_b1",
           "qwen_b2": ROOT / "artifacts/e2b_off_primary/qwen_b2"}
VM = {"mini": O / "vote_multiplicity", "qwen_b1": ROOT / "artifacts/e2b_off_primary/vote_multiplicity",
      "qwen_b2": ROOT / "artifacts/e2b_off_primary/vote_multiplicity"}
JD = {"mini": O / "judge", "qwen_b1": ROOT / "artifacts/e2b_off_primary/judge",
      "qwen_b2": ROOT / "artifacts/e2b_off_primary/judge"}
POOLS = ["BoN", "CoT", "Five"]
NAME = {"BoN": "BoN", "CoT": "CoT-tree", "Five": "MO-Tree"}


def rd(p):
    return list(csv.DictReader(open(p)))


def wcsv(p, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main():
    stage = {(r["run"], r["method"]): r for r in rd(ROOT / "artifacts/selection_rebuild/stage_loss.csv")}
    sel, paired, checks = [], [], {}
    for b, bundle in BUNDLES.items():
        summ = {r["pool"]: r for r in rd(bundle / "summary.csv")}
        pt = {(r["pool"], int(r["task_id"])): r for r in rd(bundle / "per_task.csv")}
        vm = {r["pool"]: r for r in rd(VM[b] / "summary.csv") if r["run"] == b}
        vmt = {(r["pool"], int(r["task_id"])): r for r in rd(VM[b] / "per_task.csv") if r["run"] == b}
        js = {r["pool"]: r for r in rd(JD[b] / "summary.csv") if r["bundle"] == b}
        jt = {(r["pool"], int(r["task_id"])): r for r in rd(JD[b] / "per_task.csv") if r["bundle"] == b}
        ids = sorted({t for p, t in pt if p == "BoN"})
        checks[b] = {
            "judge_majority_matches_bundle": {p: sum((jt[(p, t)]["filter_majority_ok"] == "True") !=
                                                     (pt[(p, t)]["filter_majority_correct"] == "True") for t in ids) == 0
                                              for p in POOLS},
            "judge_oracle_matches_bundle": {p: sum((jt[(p, t)]["oracle"] == "True") != (pt[(p, t)]["oracle"] == "True")
                                                   for t in ids) == 0 for p in POOLS},
            "selector_reproduction_mismatches": {p: int(vm[p]["locked_reproduction_mismatches"]) for p in POOLS},
        }
        for p in POOLS:
            s, v, j, st = summ[p], vm[p], js[p], stage[(b, NAME[p])]
            sel.append({
                "run": b, "method": NAME[p], "oracle": int(s["oracle"]),
                "primary": int(s["filter_majority"]),
                "primary_rule": "occurrence" if p == "BoN" else "unique-text",
                "unfiltered_majority": int(s["raw_majority"]),
                "occurrence": int(v["multiplicity"]), "unique_text": int(v["dedup"]),
                "first_valid": int(v["first_valid"]),
                "oracle_tiebreak": int(s["filter_majority"]) + int(st["tie_loss"]),
                "gpt4o_judge": int(j["judge"]), "judge_fixes": int(j["judge_fixes"]),
                "judge_breaks": int(j["judge_breaks"]), "judge_calls": int(j["judge_calls"]),
                "judge_subsampled_tasks": int(j["subsampled_tasks"]),
                "vote_loss": int(st["vote_loss"]), "tie_loss": int(st["tie_loss"]),
            })
            prim = [pt[(p, t)]["filter_majority_correct"] == "True" for t in ids]
            for name, xs in (("gpt4o_judge", [jt[(p, t)]["judge_ok"] == "True" for t in ids]),
                             ("first_valid", [vmt[(p, t)]["first_valid"] == "True" for t in ids]),
                             ("alternative_rule", [vmt[(p, t)]["dedup" if p == "BoN" else "multiplicity"] == "True"
                                                   for t in ids])):
                paired.append({"run": b, "comparison": f"{NAME[p]}: {name} - primary", "metric": "selected",
                               **paired_stats(xs, prim)})
        for a, c in (("CoT", "BoN"), ("Five", "BoN"), ("CoT", "Five")):
            for name, f in (("gpt4o_judge", lambda p, t: jt[(p, t)]["judge_ok"] == "True"),
                            ("primary", lambda p, t: pt[(p, t)]["filter_majority_correct"] == "True")):
                paired.append({"run": b, "comparison": f"{NAME[a]} - {NAME[c]} under {name}", "metric": "selected",
                               **paired_stats([f(a, t) for t in ids], [f(c, t) for t in ids])})

    wcsv(O / "selector_sensitivity.csv", sel)
    wcsv(O / "judge_paired.csv", paired)
    (O / "final_primary_selection.json").write_text(json.dumps(
        {"pools": {k: str(v.relative_to(ROOT)) for k, v in BUNDLES.items()}, "checks": checks,
         "selector_sensitivity": sel, "paired": paired}, indent=1))
    print(json.dumps(checks))
    for r in sel:
        print(f"{r['run']:8s} {r['method']:9s} O={r['oracle']} primary={r['primary']} unfilt={r['unfiltered_majority']} "
              f"occ={r['occurrence']} uniq={r['unique_text']} first={r['first_valid']} tieUB={r['oracle_tiebreak']} "
              f"judge={r['gpt4o_judge']} (+{r['judge_fixes']}/-{r['judge_breaks']}, calls {r['judge_calls']})")
    for r in paired:
        print(f"{r['run']:8s} {r['comparison']:40s} {r['diff_tasks']:+d} ({r['a_only']}/{r['b_only']}) "
              f"boot[{r['boot_lo_pp']:+.1f},{r['boot_hi_pp']:+.1f}]pp p={r['mcnemar_exact_p']:.3g}")


if __name__ == "__main__":
    main()
