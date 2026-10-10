#!/usr/bin/env python3
"""
Deployable Top-3 selector on all 183 tasks of the 9 final primary pools.

Per (pool, task), over valid (hard0) exact-JSON workflow groups in selector order
(votes desc, smallest candidate_id first):
  0 groups  -> keep the locked fallback (raw majority)
  1 group   -> select it, no call
  >=2 groups-> GPT-4o picks among the top 3, shown in shuffled A/B/C order with the task and the full
               specs of the tools they use, no vote counts, temperature 0. Prompt, shuffle and cache
               key are those of run_top3_verifier.py, so its 181 diagnostic answers are reused.

Output: artifacts/final_primary/top3_selector/{per_task.csv,summary.csv,usage.json};
cache shared with artifacts/final_primary/top3_verifier/cache.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_strict_matched import paired_stats  # noqa: E402
from interfaces.OpenRouter_client import OpenRouterClient  # noqa: E402
from interfaces.call_context import call_context  # noqa: E402
from run_top3_verifier import (BUNDLES, LABELS, METHODS, POOL_CODE, RUN_CODE, build_prompt, key,  # noqa: E402
                               parse_choice)

OUT = ROOT / "artifacts" / "final_primary" / "top3_selector"
CACHE = ROOT / "artifacts" / "final_primary" / "top3_verifier" / "cache.json"


def load_items():
    descs = {int(o["task_id"]): o["task_desc"] for o in map(json.loads, open(ROOT / "data" / "test.jsonl")) if o}
    items = []
    for run, rel in BUNDLES.items():
        b = ROOT / rel
        cands = defaultdict(list)
        for line in (b / "per_candidate.jsonl").open():
            c = json.loads(line)
            if c["pool"] in METHODS:
                cands[(c["pool"], int(c["task_id"]))].append(c)
        for r in csv.DictReader((b / "per_task.csv").open()):
            pool, tid = r["pool"], int(r["task_id"])
            if pool not in METHODS:
                continue
            votes, first, plans, correct = Counter(), {}, {}, {}
            for c in cands[(pool, tid)]:
                if c["hard0"]:
                    k = key(c["plan"])
                    votes[k] += 1
                    first[k] = min(first.get(k, 10 ** 9), int(c["candidate_id"]))
                    plans[k], correct[k] = c["plan"], c["correct"]
            order = sorted(votes, key=lambda k: (-votes[k], first[k]))
            majority_ok = r["filter_majority_correct"] == "True"
            if order:
                assert bool(correct[order[0]]) == majority_ok, (run, pool, tid)
            items.append({"run": run, "pool": pool, "method": METHODS[pool], "task_id": tid, "task": descs[tid],
                          "oracle": r["oracle"] == "True", "majority_ok": majority_ok, "n_groups": len(order),
                          "valid_votes": sum(votes.values()),
                          "top": [{"rank": i + 1, "plan": plans[k], "correct": bool(correct[k]), "votes": votes[k]}
                                  for i, k in enumerate(order[:3])],
                          "top3_has_correct": any(correct[k] for k in order[:3])})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--rng_seed", type=int, default=20261004)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    n_cached0 = len(cache)
    items = load_items()
    need = [it for it in items if it["n_groups"] >= 2]
    print(f"{len(items)} pool-tasks; verifier calls needed for {len(need)} "
          f"({sum(f'{i['run']}:{i['pool']}:{i['task_id']}' in cache for i in need)} cached)", flush=True)
    client = OpenRouterClient(args.model)

    def run_one(it):
        row = {k: it[k] for k in ("run", "method", "task_id", "oracle", "majority_ok", "n_groups", "top3_has_correct")}
        if it["n_groups"] == 0:
            row.update(mode="no_valid_fallback", verifier_ok=it["majority_ok"], chosen_rank=None, choice=None)
            return row
        if it["n_groups"] == 1:
            row.update(mode="single_group", verifier_ok=it["top"][0]["correct"], chosen_rank=1, choice=None)
            return row
        rng = random.Random(args.rng_seed + RUN_CODE[it["run"]] * 100003 + POOL_CODE[it["pool"]] * 1009 + it["task_id"])
        perm = list(range(len(it["top"])))
        rng.shuffle(perm)
        labeled = [(LABELS[i], it["top"][j]) for i, j in enumerate(perm)]
        ck = f"{it['run']}:{it['pool']}:{it['task_id']}"
        order = [t["rank"] for _, t in labeled]
        if ck not in cache or cache[ck].get("choice") is None:
            prompt = build_prompt(it["task"], [(lab, t["plan"]) for lab, t in labeled])
            with call_context(task_id=it["task_id"], operator="TOP3_SELECTOR", run=it["run"], pool=it["pool"]):
                raw = client.generate(prompt, temperature=0.0, max_tokens=300)
            cache[ck] = {"raw": raw, "choice": parse_choice(raw, LABELS[: len(labeled)]), "order": order}
        assert cache[ck]["order"] == order, ck
        ch = cache[ck]["choice"]
        pick = next((t for lab, t in labeled if lab == ch), None)
        if pick is None:
            row.update(mode="parse_fail_majority", verifier_ok=it["majority_ok"], chosen_rank=1, choice=ch)
        else:
            row.update(mode="verifier", verifier_ok=pick["correct"], chosen_rank=pick["rank"], choice=ch)
        row["n_options"] = len(labeled)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(run_one, it) for it in items]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 100 == 0 or i == len(futs):
                CACHE.write_text(json.dumps(cache, ensure_ascii=False))
                print(f"[{i}/{len(futs)}]", flush=True)
    rows.sort(key=lambda r: (r["run"], r["method"], r["task_id"]))

    summary = []
    for run in BUNDLES:
        for m in METHODS.values():
            rs = [r for r in rows if r["run"] == run and r["method"] == m]
            s = paired_stats([r["verifier_ok"] for r in rs], [r["majority_ok"] for r in rs])
            called = [r for r in rs if r["mode"] in ("verifier", "parse_fail_majority")]
            summary.append({
                "run": run, "method": m, "n_tasks": len(rs),
                "majority_selected": sum(r["majority_ok"] for r in rs),
                "verifier_selected": sum(r["verifier_ok"] for r in rs),
                "fixed": s["a_only"], "broken": s["b_only"], "net": s["diff_tasks"],
                "mcnemar_p": s["mcnemar_exact_p"], "boot_lo_pp": s["boot_lo_pp"], "boot_hi_pp": s["boot_hi_pp"],
                "oracle": sum(r["oracle"] for r in rs), "top3_upper": sum(r["top3_has_correct"] for r in rs),
                "verifier_calls": len(called),
                "called_majority_ok": sum(r["majority_ok"] for r in called),
                "called_verifier_ok": sum(r["verifier_ok"] for r in called),
                "called_top3_has_correct": sum(r["top3_has_correct"] for r in called),
                "chose_rank1": sum(r["chosen_rank"] == 1 for r in called),
                "parse_fail": sum(r["mode"] == "parse_fail_majority" for r in rs),
            })
    tot = {"run": "all", "method": "all"}
    for k in summary[0]:
        if k not in ("run", "method", "mcnemar_p", "boot_lo_pp", "boot_hi_pp"):
            tot[k] = sum(s[k] for s in summary)
    st = paired_stats([r["verifier_ok"] for r in rows], [r["majority_ok"] for r in rows])
    tot.update(mcnemar_p=st["mcnemar_exact_p"], boot_lo_pp=st["boot_lo_pp"], boot_hi_pp=st["boot_hi_pp"])
    summary.append(tot)

    for name, data in (("per_task.csv", rows), ("summary.csv", summary)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            w.writeheader()
            w.writerows(data)
    (out / "usage.json").write_text(json.dumps({"usage": client.get_usage(), "cache_entries_before": n_cached0,
                                                "cache_entries_after": len(cache), "args": {**vars(args), "out": os.path.relpath(args.out, ROOT)}}, indent=2))
    for s in summary:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()
