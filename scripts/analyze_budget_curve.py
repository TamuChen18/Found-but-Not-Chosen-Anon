#!/usr/bin/env python3
"""
Offline budget curve for BoN and CoT-tree by subsampling their independent n=1 calls.
No new generation.

BoN: each call is one candidate (all_candidates in candidate_id order).
PMBoN (prompt-matched BoN control): CoT-tree root prompt, independent n=1 calls, scored and voted
  exactly like BoN (one vote per call).
CoT-tree: each call is one leaf (Final Solutions traces in generation order); as in
  mcts_search.py, leaves whose text contains a JSON array become candidates, deduplicated by
  stripped full text, ordered by first appearance.
Candidates are scored under the locked protocol (strict postprocess E4 off / E2b on,
hard0 filter, exact-JSON majority, tie -> smallest candidate_id). The full-N reconstruction is
checked task-by-task against the locked per_task.csv before any subsampling.

--mode prefix (search-budget scaling): the first N calls in true generation order. BoN issues its
  calls in a sequential loop and candidate_id is the call index; CoT-tree (rollouts=1) creates its
  leaves in a sequential loop and Final Solutions lists them in creation order (calls that returned
  empty text leave no trace). Both methods are non-adaptive, so the prefix equals a run stopped
  after N calls.
--mode random (candidate-budget / subsampling sensitivity): R random subsets of calls per task
  (without replacement, original order kept); Oracle / Selected are expected counts over subsets.
Tokens = locked per-task total tokens x N / N_full (calls are homogeneous in prompt length).

CoT "freq" variant: same leaves but votes weighted by duplicate count (as BoN does); diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluators.evaluators import Evaluator  # noqa: E402
from utils.answer_selector import hard_violation_count, load_tools_manifest, parse_plan  # noqa: E402
from utils.agent_utils import _JSON_ARRAY_RE  # noqa: E402
from freeze_strict_matched import TOOLS, majority_pick, strict_postprocess  # noqa: E402

golds = {int(o["task_id"]): o["structured_plan"] for o in map(json.loads, (ROOT / "data" / "test.jsonl").open()) if o}
ctx = load_tools_manifest(TOOLS)
agent_outputs = {a: list(s.get("output", {}).keys()) for a, s in json.loads(Path(TOOLS).read_text()).items()}
ev = Evaluator()
COLLAPSE_MULTI = True  # E2b; importers may set False
BON_LIKE = ("BoN", "PMBoN")


def score(ans, tid):
    text = ans if isinstance(ans, str) or ans is None else json.dumps(ans, ensure_ascii=False)
    plan = parse_plan(strict_postprocess(text, agent_outputs, collapse_multi=COLLAPSE_MULTI)) if text else None
    return {"plan": plan,
            "correct": plan is not None and bool(ev.check_answers_equivalence(json.dumps(plan), json.dumps(golds[tid]))),
            "hard0": plan is not None and hard_violation_count(plan, ctx) == 0}


def select(items):
    hard0 = [x for x in items if x["hard0"]]
    pick = majority_pick(hard0) if hard0 else majority_pick(items)
    return bool(pick and pick["correct"])


def load_answers(run_dir):
    out = {}
    for p in (run_dir / "answer_sheets").glob("Task * - Answer.json"):
        a = json.loads(p.read_text())
        out[int(a["task_id"])] = a
    return out


def bon_calls(run_dir):
    """task -> list of per-call scored candidates (index = call order)."""
    out = {}
    for tid, a in load_answers(run_dir).items():
        cs = sorted(a.get("all_candidates") or [], key=lambda x: int(x.get("candidate_id", 0)))
        out[tid] = [score(c.get("model_answer"), tid) for c in cs]
    return out


def cot_calls(run_dir):
    """task -> list of per-call (text_key or None, scored candidate or None)."""
    out = {}
    for tid in load_answers(run_dir):
        p = run_dir / "answer_sheets" / f"Task {tid} - Final Solutions.json"
        calls, memo = [], {}
        for r in json.load(p.open()):
            cot = r["trace"]["0"].get("chain_of_thought")
            text = cot.get("text") if isinstance(cot, dict) else cot
            if not text or not _JSON_ARRAY_RE.findall(text):
                calls.append((None, None))
                continue
            k = text.strip()
            if k not in memo:
                memo[k] = score(ev.extract_answer_from_model_completion(k), tid)
            calls.append((k, memo[k]))
        out[tid] = calls
    return out


def bon_eval(calls, idx):
    items = [{**calls[i], "candidate_id": j} for j, i in enumerate(idx)]
    return any(x["correct"] for x in items), select(items)


def cot_eval(calls, idx, freq=False):
    order, count = [], defaultdict(int)
    for i in idx:
        k, c = calls[i]
        if k is None:
            continue
        if k not in count:
            order.append((k, c))
        count[k] += 1
    items = []
    for j, (k, c) in enumerate(order):
        reps = count[k] if freq else 1
        items += [{**c, "candidate_id": j}] * reps
    return any(x["correct"] for x in items), select(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="label=pool:run_dir:bundle_dir, pool in {BoN, PMBoN, CoT}")
    ap.add_argument("--ns", default="4,8,16,full")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--mode", choices=["prefix", "random"], default="prefix")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "budget_curve"))
    ap.add_argument("--keep_multi_pred", action="store_true", help="E2b off (bundles must be E2b-off bundles)")
    args = ap.parse_args()
    global COLLAPSE_MULTI
    COLLAPSE_MULTI = not args.keep_multi_pred
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.run:
        label, rest = spec.split("=", 1)
        pool, run_dir, bundle = rest.split(":", 2)
        run_dir, bundle = ROOT / run_dir, ROOT / bundle
        locked = {int(r["task_id"]): r for r in csv.DictReader((bundle / "per_task.csv").open()) if r["pool"] == pool}
        calls = bon_calls(run_dir) if pool in BON_LIKE else cot_calls(run_dir)
        tids = sorted(locked)
        n_full = max(len(calls[t]) for t in tids)

        mism = 0
        for t in tids:
            full = list(range(len(calls[t])))
            o, s = bon_eval(calls[t], full) if pool in BON_LIKE else cot_eval(calls[t], full)
            mism += (o != (locked[t]["oracle"] == "True")) + (s != (locked[t]["filter_majority_correct"] == "True"))
        print(f"{label} {pool}: full-N reconstruction mismatches vs locked = {mism}", flush=True)

        tok_full = sum(float(locked[t]["total_tokens"]) for t in tids) / len(tids)
        for n_s in args.ns.split(","):
            n = n_full if n_s == "full" else int(n_s)
            rng = random.Random(0)
            reps = 1 if (n == n_full or args.mode == "prefix") else args.reps
            o_sum = s_sum = f_sum = 0.0
            for t in tids:
                m = len(calls[t])
                for _ in range(reps):
                    if args.mode == "prefix":
                        idx = list(range(min(n, m)))
                    else:
                        idx = sorted(rng.sample(range(m), min(n, m)))
                    if pool in BON_LIKE:
                        o, s = bon_eval(calls[t], idx)
                        f = s
                    else:
                        o, s = cot_eval(calls[t], idx)
                        f = cot_eval(calls[t], idx, freq=True)[1]
                    o_sum += o / reps
                    s_sum += s / reps
                    f_sum += f / reps
            rows.append({"mode": args.mode, "run": label, "pool": pool, "N": n, "n_tasks": len(tids),
                         "tokens_per_task": tok_full * n / n_full,
                         "oracle": round(o_sum, 2), "selected": round(s_sum, 2),
                         "selected_freq_vote": round(f_sum, 2) if pool == "CoT" else None,
                         "reconstruction_mismatches": mism})
            print(rows[-1], flush=True)

    name = "search_budget_prefix.csv" if args.mode == "prefix" else "candidate_subsampling.csv"
    with (out / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out / name)


if __name__ == "__main__":
    main()
