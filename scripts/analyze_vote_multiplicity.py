#!/usr/bin/env python3
"""
Unified multiplicity-vs-dedup voting sensitivity for BoN / CoT-tree / MO-Tree (offline).

Every variant uses the locked pipeline (strict postprocess E4 off / E2b on, hard0 filter, exact-JSON
majority over normalized workflows, tie -> smallest candidate_id); only the vote weights differ:
  multiplicity  one vote per generation unit: BoN / CoT-tree = one per API call,
                MO-Tree = one per solution node (CoT/DC nodes already collapse their n samples by
                intra-node majority, so sample-level multiplicity is not recoverable for MO-Tree)
  dedup         one vote per unique solution text (the generation-time rule of the tree runs)
  first_valid   no votes: the first valid candidate in generation order (frequency ignored)
Locked protocol = multiplicity for BoN, dedup for CoT-tree / MO-Tree.

Unit reconstruction:
  BoN       all_candidates (one per call). Dedup key = raw_completion text; runs that did not store
            raw text fall back to the extracted model_answer string (coarser; flagged dedup_key=answer).
  CoT-tree  Final Solutions traces (one per call, generation order); JSON-array texts are candidates.
  MO-Tree   Final Solutions nodes, deduplicated by node path (rollouts re-list earlier nodes). A node
            whose path equals a candidate's model_path is that candidate; any other node whose trace
            contains a candidate's exact solution text (excluding ancestor candidates) is a duplicate
            of it; nodes matching nothing carry no plan. Ambiguous matches are counted.
Each reconstruction is checked against the locked per_task.csv (dedup / multiplicity as locked).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from utils.agent_utils import _JSON_ARRAY_RE  # noqa: E402
import analyze_budget_curve as abc  # noqa: E402
from analyze_budget_curve import ev, load_answers, score, select  # noqa: E402

KEYS = {"BoN": "BoN-CoT-M", "CoT": "CoT-only-M", "Five": "Five-M"}


def leaves(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from leaves(v)


def units_bon(a, tid):
    cs = sorted(a.get("all_candidates") or [], key=lambda x: int(x.get("candidate_id", 0)))
    raw = all(c.get("raw_completion") for c in cs)
    units = []
    for c in cs:
        ma = c.get("model_answer")
        k = c["raw_completion"].strip() if raw else (ma if isinstance(ma, str) else json.dumps(ma, sort_keys=True))
        units.append((k, score(ma, tid)))
    return units, ("raw_text" if raw else "answer"), 0


def units_cot(run_dir, tid):
    memo, units = {}, []
    p = run_dir / "answer_sheets" / f"Task {tid} - Final Solutions.json"
    for r in json.load(p.open()):
        cot = r["trace"]["0"].get("chain_of_thought")
        text = cot.get("text") if isinstance(cot, dict) else cot
        if not text or not _JSON_ARRAY_RE.findall(text):
            continue
        k = text.strip()
        if k not in memo:
            memo[k] = score(ev.extract_answer_from_model_completion(k), tid)
        units.append((k, memo[k]))
    return units, "raw_text", 0


def units_five(run_dir, a, tid):
    """Returns units in generation order plus (#ambiguous node matches, #candidates with no node)."""
    cands = sorted(a.get("all_candidates") or [], key=lambda x: int(x.get("candidate_id", 0)))
    texts = {c["solution"].strip(): c for c in cands if isinstance(c.get("solution"), str)}
    scored = {int(c["candidate_id"]): {**score(c.get("model_answer"), tid), "cid": int(c["candidate_id"])}
              for c in cands}
    p = run_dir / "answer_sheets" / f"Task {tid} - Final Solutions.json"
    seen, units, ambiguous, used = set(), [], 0, set()
    for r in json.load(p.open()):
        trace = r["trace"]
        nid = json.dumps(trace, sort_keys=True)
        if nid in seen:
            continue
        seen.add(nid)
        path = trace.get("0", {}).get("path")
        last = trace[max(trace, key=int)]
        own = [texts[s.strip()] for s in leaves(last) if s.strip() in texts]
        pool = own or [texts[s.strip()] for s in leaves(trace) if s.strip() in texts]
        hits = list({int(c["candidate_id"]): c for c in pool}.values())
        if len(hits) > 1:
            exact = [c for c in hits if c.get("model_path") == path]
            ambiguous += len(exact) != 1
            hits = exact or hits
        if not hits:
            continue
        c = hits[-1]
        used.add(int(c["candidate_id"]))
        units.append((c["solution"].strip(), scored[int(c["candidate_id"])]))
    unmapped = [c for c in cands if int(c["candidate_id"]) not in used]
    units += [(c["solution"].strip(), scored[int(c["candidate_id"])]) for c in unmapped]
    return units, "raw_text", (ambiguous, len(unmapped))


def vote(units, mode):
    order, count = [], Counter()
    for k, c in units:
        if k not in count:
            order.append((k, c))
        count[k] += 1
    # candidate order (tie-break) = first appearance, or the stored candidate_id when known
    order = [kc for _, kc in sorted(enumerate(order), key=lambda x: (x[1][1].get("cid", x[0]), x[0]))]
    if mode == "first_valid":
        for j, (k, c) in enumerate(order):
            if c["hard0"]:
                return bool(c["correct"])
        return select([{**c, "candidate_id": j} for j, (k, c) in enumerate(order)])
    items = []
    for j, (k, c) in enumerate(order):
        items += [{**c, "candidate_id": j}] * (count[k] if mode == "multiplicity" else 1)
    return select(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True, help="label=manifest.json:seed:bundle_dir")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "vote_multiplicity"))
    ap.add_argument("--keep_multi_pred", action="store_true", help="E2b off (bundle_dir must be an E2b-off bundle)")
    args = ap.parse_args()
    abc.COLLAPSE_MULTI = not args.keep_multi_pred
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, task_rows = [], []
    for spec in args.manifest:
        label, rest = spec.split("=", 1)
        mpath, seed, bundle = rest.split(":", 2)
        runs = {r["key"]: ROOT / r["dir"] for r in json.load(open(ROOT / mpath))["runs"] if str(r["seed"]) == seed}
        locked = {(r["pool"], int(r["task_id"])): r for r in csv.DictReader((ROOT / bundle / "per_task.csv").open())}
        for pool, key in KEYS.items():
            if key not in runs:
                continue
            A = load_answers(runs[key])
            tids = sorted(t for p, t in locked if p == pool)
            acc = Counter()
            keys_used, amb, unmapped, n_units, n_unique, mism = set(), 0, 0, 0, 0, 0
            for t in tids:
                if pool == "BoN":
                    units, kk, a = units_bon(A[t], t)
                elif pool == "CoT":
                    units, kk, a = units_cot(runs[key], t)
                else:
                    units, kk, a = units_five(runs[key], A[t], t)
                    a, u = a
                    unmapped += u
                keys_used.add(kk)
                amb += a
                n_units += len(units)
                n_unique += len({k for k, _ in units})
                res = {m: vote(units, m) for m in ("multiplicity", "dedup", "first_valid")}
                locked_mode = "multiplicity" if pool == "BoN" else "dedup"
                mism += res[locked_mode] != (locked[(pool, t)]["filter_majority_correct"] == "True")
                acc.update({m: int(v) for m, v in res.items()})
                task_rows.append({"run": label, "pool": pool, "task_id": t, "n_units": len(units),
                                  "n_unique_texts": len({k for k, _ in units}), **res})
            rows.append({"run": label, "pool": pool, "n_tasks": len(tids),
                         "oracle": sum(locked[(pool, t)]["oracle"] == "True" for t in tids),
                         "locked_selected": sum(locked[(pool, t)]["filter_majority_correct"] == "True" for t in tids),
                         "multiplicity": acc["multiplicity"], "dedup": acc["dedup"], "first_valid": acc["first_valid"],
                         "locked_mode": "multiplicity" if pool == "BoN" else "dedup",
                         "units_per_task": n_units / len(tids), "unique_texts_per_task": n_unique / len(tids),
                         "dedup_key": "/".join(sorted(keys_used)), "ambiguous_node_matches": amb,
                         "candidates_without_node": unmapped,
                         "locked_reproduction_mismatches": mism})
            print(rows[-1], flush=True)
    for name, data in (("summary.csv", rows), ("per_task.csv", task_rows)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    print("wrote", out)


if __name__ == "__main__":
    main()
