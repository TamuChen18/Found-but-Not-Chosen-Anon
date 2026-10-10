#!/usr/bin/env python3
"""
Evaluate Prompt-matched BoN under the final-primary protocol (E4 off, E2b off, hard0 filter,
exact-JSON majority, tie -> smallest candidate_id) and compare it with the locked no-seed
GPT-4o-mini BoN and CoT-tree pools (artifacts/mini_noseed_trees_full/eval_e2b_off).

Pools written to the bundle:
  BoN, CoT        copied unchanged from the locked bundle
  PMBoN           all N samples, one vote per sample (BoN retention / voting convention)
  PMBoN_dedup     samples deduplicated by stripped text, JSON-array texts only, one vote per unique
                  text (CoT-tree retention / voting convention)
  BoN_dedup       original BoN raw completions under the CoT-tree convention (convention control)

No LLM calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_modal_errors as amc  # noqa: E402
import analyze_pool_concentration as apc  # noqa: E402
from analyze_selection_rebuild import stage_row  # noqa: E402
from evaluators.evaluators import Evaluator  # noqa: E402
from freeze_strict_matched import (TEST, TOOLS, load_tasks, majority_pick, paired_stats,  # noqa: E402
                                   plan_key, strict_postprocess, write_csv)
from utils.agent_utils import _JSON_ARRAY_RE  # noqa: E402
from utils.answer_selector import hard_violation_count, load_tools_manifest, parse_plan  # noqa: E402

LOCKED = ROOT / "artifacts" / "mini_noseed_trees_full" / "eval_e2b_off"
BON_RUN = ROOT / "runs/gpt-4o-mini/no_seed/bon"
NEW_POOLS = ("PMBoN", "PMBoN_dedup", "BoN_dedup")
ALL_POOLS = ("BoN", "CoT") + NEW_POOLS
STAGE_NAMES = {"BoN": "BoN", "CoT": "CoT-tree", "PMBoN": "Prompt-matched BoN",
               "PMBoN_dedup": "Prompt-matched BoN (unique-text)", "BoN_dedup": "BoN (unique-text)"}


def dedup(cands):
    seen, out = set(), []
    for c in sorted(cands, key=lambda x: int(x["candidate_id"])):
        t = (c.get("raw_completion") or "").strip()
        if not t or t in seen or not _JSON_ARRAY_RE.findall(t):
            continue
        seen.add(t)
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "artifacts" / "prompt_matched_bon" / "run"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "prompt_matched_bon" / "eval"))
    ap.add_argument("--partial", action="store_true", help="evaluate only the tasks already generated")
    ap.add_argument("--locked", default=str(LOCKED), help="final-protocol bundle with the BoN and CoT pools")
    ap.add_argument("--bon_run", default=str(BON_RUN), help="raw BoN run (answer sheets) behind the locked BoN pool")
    ap.add_argument("--label", default="mini")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    locked = Path(args.locked) if Path(args.locked).is_absolute() else ROOT / args.locked
    locked_pt = [r for r in csv.DictReader((locked / "per_task.csv").open()) if r["pool"] in ("BoN", "CoT")]
    ids = sorted({int(r["task_id"]) for r in locked_pt})
    pm = load_tasks(Path(args.run))
    bon = load_tasks(Path(args.bon_run) if Path(args.bon_run).is_absolute() else ROOT / args.bon_run)
    missing = [t for t in ids if t not in pm]
    if missing and args.partial:
        ids = [t for t in ids if t in pm]
        locked_pt = [r for r in locked_pt if int(r["task_id"]) in set(ids)]
    elif missing:
        sys.exit(f"PM-BoN missing {len(missing)} tasks, e.g. {missing[:10]}")

    golds = {int(o["task_id"]): o["structured_plan"] for o in map(json.loads, TEST.open()) if o}
    ctx = load_tools_manifest(TOOLS)
    agent_outputs = {a: list(s.get("output", {}).keys()) for a, s in json.loads(TOOLS.read_text()).items()}
    ev = Evaluator()

    def evaluate(ans, gold):
        text = ans if isinstance(ans, str) or ans is None else json.dumps(ans, ensure_ascii=False)
        raw = parse_plan(text)
        post = parse_plan(strict_postprocess(text, agent_outputs, enable_e4=False, collapse_multi=False)) if text else None

        def corr(plan):
            return plan is not None and bool(ev.check_answers_equivalence(json.dumps(plan), json.dumps(gold)))

        return {"raw_plan": raw, "plan": post, "raw_correct": corr(raw), "correct": corr(post),
                "raw_hard0": raw is not None and hard_violation_count(raw, ctx) == 0,
                "hard0": post is not None and hard_violation_count(post, ctx) == 0}

    per_task = list(locked_pt)
    with (out / "per_candidate.jsonl").open("w") as fc:
        for line in (locked / "per_candidate.jsonl").open():
            if json.loads(line)["pool"] in ("BoN", "CoT"):
                fc.write(line)
        for pool in NEW_POOLS:
            for tid in ids:
                sheet = (bon if pool == "BoN_dedup" else pm)[tid]
                cands = sheet["all_candidates"]
                if pool != "PMBoN":
                    cands = dedup(cands)
                items = []
                for c in cands:
                    r = evaluate(c.get("model_answer"), golds[tid])
                    it = {"candidate_id": int(c["candidate_id"]), **r}
                    items.append(it)
                    fc.write(json.dumps({"pool": pool, "task_id": tid, "candidate_id": it["candidate_id"],
                                         "raw_correct": r["raw_correct"], "correct": r["correct"],
                                         "raw_hard0": r["raw_hard0"], "hard0": r["hard0"],
                                         "raw_plan": r["raw_plan"], "plan": r["plan"]}, ensure_ascii=False) + "\n")
                maj = majority_pick(items)
                hard0 = [x for x in items if x["hard0"]]
                filt = majority_pick(hard0) if hard0 else maj
                u = sheet.get("usage") or {}
                per_task.append({
                    "pool": pool, "task_id": tid, "n_candidates": len(items), "n_hard0": len(hard0),
                    "n_unique_hard0": len({plan_key(x["plan"]) for x in hard0}),
                    "oracle_raw": any(x["raw_correct"] for x in items),
                    "oracle": any(x["correct"] for x in items),
                    "raw_majority_correct": bool(maj and maj["correct"]),
                    "filter_majority_correct": bool(filt and filt["correct"]),
                    "filter_fell_back_to_raw_majority": not hard0,
                    "selected_candidate_id": filt["candidate_id"] if filt else None,
                    "selected_workflow": json.dumps(filt["plan"], ensure_ascii=False) if filt else None,
                    "api_calls": u.get("api_calls"), "prompt_tokens": u.get("prompt_tokens"),
                    "completion_tokens": u.get("completion_tokens"), "total_tokens": u.get("total_tokens"),
                })
    write_csv(out / "per_task.csv", per_task)

    apc.POOLS = ALL_POOLS
    conc = apc.summarize(apc.task_rows(out, args.label))
    tok = {}
    for pool in ALL_POOLS:
        rs = [r for r in per_task if r["pool"] == pool]
        tok[pool] = {k: sum(float(r[k]) for r in rs) / len(rs) for k in ("total_tokens", "prompt_tokens", "completion_tokens", "api_calls")}
    summary = []
    for c in conc:
        p = c["pool"]
        summary.append({"pool": p, "n_tasks": c["n_tasks"], "oracle": c["oracle"], "selected": c["selected"],
                        "raw_majority": sum(str(r["raw_majority_correct"]) == "True" for r in per_task if r["pool"] == p),
                        "tokens_per_task": tok[p]["total_tokens"], "prompt_tokens_per_task": tok[p]["prompt_tokens"],
                        "completion_tokens_per_task": tok[p]["completion_tokens"], "api_calls_per_task": tok[p]["api_calls"],
                        "retained_per_task": c["retained_cands_mean"], "valid_per_task": c["valid_cands_mean"],
                        "unique_valid_wf_per_task": c["unique_valid_wf_mean"], "winner_share": c["win_share_mean"],
                        "ties": c["tied_tasks"], "tasks_no_valid": c["tasks_no_valid"],
                        "lost_at_filter": c["lost_at_filter"], "lost_at_vote": c["lost_at_vote"], "lost_at_tie": c["lost_at_tie"]})
    write_csv(out / "summary.csv", summary)

    by = {(r["pool"], int(r["task_id"])): r for r in per_task}
    b = lambda pool, t, m: str(by[(pool, t)][m]) == "True"  # noqa: E731
    paired = []
    for a, c in [("PMBoN", "BoN"), ("PMBoN", "CoT"), ("CoT", "BoN"),
                 ("PMBoN_dedup", "BoN"), ("PMBoN_dedup", "CoT"), ("PMBoN_dedup", "PMBoN"), ("BoN_dedup", "BoN")]:
        for metric in ("oracle", "filter_majority_correct", "raw_majority_correct"):
            s = paired_stats([b(a, t, metric) for t in ids], [b(c, t, metric) for t in ids])
            paired.append({"a": a, "b": c, "metric": metric, **s})
    write_csv(out / "paired_ci.csv", paired)

    amc.METHODS = {p: STAGE_NAMES[p] for p in ALL_POOLS}
    cases, _ = amc.load_cases(out, args.label)
    cands = {}
    for line in (out / "per_candidate.jsonl").open():
        c = json.loads(line)
        cands.setdefault((c["pool"], int(c["task_id"])), []).append(c)
    pt_str = list(csv.DictReader((out / "per_task.csv").open()))
    stage = [stage_row(args.label, p, STAGE_NAMES[p], pt_str, cands, cases) for p in ALL_POOLS]
    for s in stage:
        assert s["filter_loss"] + s["vote_loss"] + s["tie_loss"] == s["oracle"] - s["selected"], s
    write_csv(out / "stage_loss.csv", stage)
    for s in stage:
        print(f"stage {s['method']:22s} O={s['oracle']} S={s['selected']} filter={s['filter_loss']} "
              f"vote={s['vote_loss']} tie={s['tie_loss']} (sum={s['filter_loss'] + s['vote_loss'] + s['tie_loss']})")
    print()

    for r in summary:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print()
    for r in paired:
        print(f"{r['a']}-{r['b']} {r['metric']}: {r['diff_tasks']:+d} ({r['a_only']}/{r['b_only']}) "
              f"{r['diff_pp']:+.1f}pp boot[{r['boot_lo_pp']:+.1f},{r['boot_hi_pp']:+.1f}] p={r['mcnemar_exact_p']:.3g}")
    print("wrote", out)


if __name__ == "__main__":
    main()
