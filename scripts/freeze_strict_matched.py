#!/usr/bin/env python3
"""
Reproduce the locked Phase-2 main table (BoN / CoT-tree / Five) under the strict
matched postprocess, export per-task and per-candidate results, and compute
paired confidence intervals between methods.

Pipeline per candidate (identical for all pools):
  stored model_answer (already extracted via Evaluator.extract_answer_from_model_completion,
  which applies json.loads -> json_repair fallback)
  -> strict_postprocess (E1 GENERATED canonicalization, E2/E2b dependence rebuild,
     E3 outputs fill; E4 single-output key remap DISABLED)
  -> parse_plan
  -> Evaluator.check_answers_equivalence(plan, gold)   # Oracle / correctness
  -> hard_violation_count(plan) == 0                     # structural filter

No LLM calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluators.evaluators import Evaluator  # noqa: E402
from utils.answer_selector import hard_violation_count, load_tools_manifest, parse_plan  # noqa: E402

MANIFEST = ROOT / "artifacts" / "phase2_fulltest" / "manifest_mini.json"
TOOLS = ROOT / "src" / "interfaces" / "tools" / "tools_manifest.json"
TEST = ROOT / "data" / "test.jsonl"
SEED = 20260808
COT_K = 8
POOLS = ("BoN", "CoT", "Five")
RUN_KEYS = {"BoN": "BoN-CoT-M", "CoT": "CoT-only-M", "Five": "Five-M", "H24": "Five-H24"}

GEN_PAT = re.compile(r"^<GENERATED>-(\d+)-<?([^<>]+)>?$")


def load_runs(manifest=MANIFEST, seed=SEED):
    m = json.loads(Path(manifest).read_text())
    return {r["key"]: ROOT / r["dir"] for r in m["runs"] if int(r["seed"]) == int(seed)}


def load_tasks(run_dir: Path):
    out = {}
    for p in (run_dir / "answer_sheets").glob("Task * - Answer.json"):
        a = json.loads(p.read_text())
        out[int(a["task_id"])] = a
    return out


def strict_postprocess(answer_text, agent_outputs, *, enable_e4=False, collapse_multi=True):
    if not isinstance(answer_text, str):
        return answer_text
    try:
        plan = json.loads(answer_text)
    except json.JSONDecodeError:
        return answer_text
    if not isinstance(plan, list):
        return answer_text

    step_to_outputs = {}
    for item in plan:
        if not isinstance(item, dict) or not isinstance(item.get("step"), int):
            continue
        outputs = item.get("outputs", [])
        if not isinstance(outputs, list) or len(outputs) == 0:
            agent = item.get("agent")
            outputs = agent_outputs.get(agent, []) if isinstance(agent, str) else []
            if outputs:
                item["outputs"] = outputs
        cur = item.get("outputs", [])
        step_to_outputs[item["step"]] = cur if isinstance(cur, list) else []

    for item in plan:
        if not isinstance(item, dict):
            continue
        inputs = item.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
            item["inputs"] = inputs
        preds, consumed = [], defaultdict(list)
        for name, value in list(inputs.items()):
            if not isinstance(value, str):
                continue
            m = GEN_PAT.match(value.strip())
            if not m:
                continue
            ref_step, ref_key = int(m.group(1)), m.group(2).strip()
            valid = step_to_outputs.get(ref_step, [])
            if enable_e4 and valid and ref_key not in valid and len(valid) == 1:
                ref_key = valid[0]
            inputs[name] = f"<GENERATED>-{ref_step}-<{ref_key}>"
            if ref_step not in preds:
                preds.append(ref_step)
            if ref_key not in consumed[ref_step]:
                consumed[ref_step].append(ref_key)
        if not preds:
            item["dependence"] = [-1]
            item["dependence_content"] = None
        elif collapse_multi or len(preds) == 1:
            item["dependence"] = [preds[0]]
            item["dependence_content"] = {str(preds[0]): consumed[preds[0]]}
        else:
            item["dependence"] = list(preds)
            item["dependence_content"] = {str(p): consumed[p] for p in preds}
    return json.dumps(plan, ensure_ascii=False)


def plan_key(plan):
    if plan is None:
        return None
    return json.dumps(plan, sort_keys=True, ensure_ascii=False)


def majority_pick(items):
    """items: list of dicts with plan, correct, candidate_id.
    Exact-JSON grouping (sort_keys); tie -> group containing the smallest candidate_id;
    representative = smallest candidate_id in the winning group."""
    groups = defaultdict(list)
    for it in items:
        k = plan_key(it["plan"])
        if k is not None:
            groups[k].append(it)
    if not groups:
        return None
    best = max(groups, key=lambda k: (len(groups[k]), -min(x["candidate_id"] for x in groups[k])))
    return min(groups[best], key=lambda x: x["candidate_id"])


def paired_stats(a, b, n_boot=10000, seed=0):
    """a, b: aligned lists of bools. Returns diff (a-b) in tasks and pp, McNemar exact p,
    paired Wald CI and paired bootstrap percentile CI (pp)."""
    n = len(a)
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    c10 = sum(1 for x, y in zip(a, b) if y and not x)
    d = (b01 - c10) / n
    se = math.sqrt(max(b01 + c10 - (b01 - c10) ** 2 / n, 0.0)) / n
    k, m = min(b01, c10), b01 + c10
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(int(a[i]) - int(b[i]) for i in idx) / n)
    diffs.sort()
    return {
        "n": n,
        "a_only": b01,
        "b_only": c10,
        "diff_tasks": b01 - c10,
        "diff_pp": 100 * d,
        "wald_lo_pp": 100 * (d - 1.96 * se),
        "wald_hi_pp": 100 * (d + 1.96 * se),
        "boot_lo_pp": 100 * diffs[int(0.025 * n_boot)],
        "boot_hi_pp": 100 * diffs[int(0.975 * n_boot) - 1],
        "mcnemar_exact_p": p,
    }


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "phase2_fulltest" / "FINAL_LOCKED" / "review_bundle"))
    ap.add_argument("--enable_e4", action="store_true")
    ap.add_argument("--exclude_ids_file", default=None, help="test task_ids to drop, e.g. data/judgment_calib5_ids.txt")
    ap.add_argument("--keep_multi_pred", action="store_true",
                    help="disable E2b: keep every GENERATED source step in dependence / dependence_content")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.manifest, args.seed)
    stores = {k: load_tasks(runs[v]) for k, v in RUN_KEYS.items() if v in runs}
    ids = sorted(set.intersection(*(set(s) for s in stores.values())))
    if args.exclude_ids_file:
        excluded = {int(x) for x in Path(args.exclude_ids_file).read_text().split() if x.strip().isdigit()}
        ids = [t for t in ids if t not in excluded]
    golds = {int(o["task_id"]): o["structured_plan"] for o in map(json.loads, TEST.open()) if o}
    ctx = load_tools_manifest(TOOLS)
    agent_outputs = {a: list(s.get("output", {}).keys()) for a, s in json.loads(TOOLS.read_text()).items()}
    ev = Evaluator()

    def evaluate(ans, gold):
        text = ans if isinstance(ans, str) or ans is None else json.dumps(ans, ensure_ascii=False)
        raw = parse_plan(text)
        post = parse_plan(strict_postprocess(
            text, agent_outputs, enable_e4=args.enable_e4, collapse_multi=not args.keep_multi_pred)) if text else None

        def corr(plan):
            return plan is not None and bool(ev.check_answers_equivalence(json.dumps(plan), json.dumps(gold)))

        return {
            "raw_plan": raw,
            "plan": post,
            "raw_correct": corr(raw),
            "correct": corr(post),
            "raw_hard0": raw is not None and hard_violation_count(raw, ctx) == 0,
            "hard0": post is not None and hard_violation_count(post, ctx) == 0,
        }

    cand_path = out / "per_candidate.jsonl"
    per_task = []
    with cand_path.open("w") as fc:
        for pool in POOLS:
            for tid in ids:
                task = stores[pool][tid]
                items = []
                for c in sorted(task.get("all_candidates") or [], key=lambda x: int(x.get("candidate_id", 0))):
                    r = evaluate(c.get("model_answer"), golds[tid])
                    it = {"candidate_id": int(c.get("candidate_id", 0)), **r}
                    items.append(it)
                    fc.write(json.dumps({
                        "pool": pool, "task_id": tid, "candidate_id": it["candidate_id"],
                        "node_type": c.get("node_type"), "depth": c.get("depth"),
                        "stored_correct_raw": bool(c.get("correct")),
                        "raw_correct": r["raw_correct"], "correct": r["correct"],
                        "raw_hard0": r["raw_hard0"], "hard0": r["hard0"],
                        "raw_plan": r["raw_plan"], "plan": r["plan"],
                    }, ensure_ascii=False) + "\n")

                maj = majority_pick(items)
                hard0 = [x for x in items if x["hard0"]]
                filt = majority_pick(hard0) if hard0 else maj
                u = task.get("usage") or {}
                per_task.append({
                    "pool": pool,
                    "task_id": tid,
                    "n_candidates": len(items),
                    "n_hard0": len(hard0),
                    "n_unique_hard0": len({plan_key(x["plan"]) for x in hard0}),
                    "oracle_raw": any(x["raw_correct"] for x in items),
                    "oracle": any(x["correct"] for x in items),
                    "raw_majority_correct": bool(maj and maj["correct"]),
                    "filter_majority_correct": bool(filt and filt["correct"]),
                    "filter_fell_back_to_raw_majority": not hard0,
                    "selected_candidate_id": filt["candidate_id"] if filt else None,
                    "selected_workflow": json.dumps(filt["plan"], ensure_ascii=False) if filt else None,
                    "api_calls": u.get("api_calls"),
                    "prompt_tokens": u.get("prompt_tokens"),
                    "completion_tokens": u.get("completion_tokens"),
                    "total_tokens": u.get("total_tokens"),
                })
    write_csv(out / "per_task.csv", per_task)

    by = {(r["pool"], r["task_id"]): r for r in per_task}
    summary = []
    for pool in POOLS:
        rs = [by[(pool, t)] for t in ids]
        n_c = sum(r["n_candidates"] for r in rs)
        summary.append({
            "pool": pool,
            "n_tasks": len(ids),
            "oracle": sum(r["oracle"] for r in rs),
            "raw_majority": sum(r["raw_majority_correct"] for r in rs),
            "filter_majority": sum(r["filter_majority_correct"] for r in rs),
            "oracle_raw_unnormalized": sum(r["oracle_raw"] for r in rs),
            "invalid_pct_matched": 100 * (1 - sum(r["n_hard0"] for r in rs) / n_c),
            "mean_candidates": n_c / len(ids),
            "mean_api_calls": sum(r["api_calls"] for r in rs) / len(ids),
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rs) / len(ids),
            "mean_completion_tokens": sum(r["completion_tokens"] for r in rs) / len(ids),
            "mean_total_tokens": sum(r["total_tokens"] for r in rs) / len(ids),
            "sum_total_tokens": sum(r["total_tokens"] for r in rs),
        })
    write_csv(out / "summary.csv", summary)

    paired = []
    for a, b in [("CoT", "BoN"), ("Five", "BoN"), ("CoT", "Five")]:
        for metric in ["oracle", "raw_majority_correct", "filter_majority_correct"]:
            s = paired_stats([by[(a, t)][metric] for t in ids], [by[(b, t)][metric] for t in ids])
            paired.append({"a": a, "b": b, "metric": metric, **s})
    write_csv(out / "paired_ci.csv", paired)

    for r in summary:
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
    print()
    for r in paired:
        print(f"{r['a']}-{r['b']} {r['metric']}: {r['diff_tasks']:+d} ({r['a_only']}/{r['b_only']}) "
              f"{r['diff_pp']:+.1f}pp boot[{r['boot_lo_pp']:+.1f},{r['boot_hi_pp']:+.1f}] "
              f"wald[{r['wald_lo_pp']:+.1f},{r['wald_hi_pp']:+.1f}] p={r['mcnemar_exact_p']:.3g}")
    print("wrote", out)


if __name__ == "__main__":
    main()
