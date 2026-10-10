#!/usr/bin/env python3
"""
Controlled semantic judge (same prompt / k / label shuffling / judge model as
run_controlled_semantic_judge.py) applied to locked review bundles.

Candidates, validity (hard0) and correctness come from the bundle's per_candidate.jsonl,
so the judge sees exactly the strict-normalized valid pool that Filter->Maj votes over:
  valid pool -> exact-plan dedup -> if >k unique: random subsample k -> blind shuffled judge
  0 unique: locked Filter->Maj pick (raw-majority fallback); 1 unique: that plan.

Usage:
  python3 scripts/run_locked_judge.py \
      --bundle mini=artifacts/phase2_fulltest/FINAL_LOCKED/review_bundle \
      --bundle qwen_b1=artifacts/qwen_nothink_main183/eval/b20260725 \
      --out artifacts/locked_judge
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_semantic_judge as base  # noqa: E402

POOL_CODE = {"CoT": 1, "Five": 2, "BoN": 4}


def key(plan):
    return json.dumps(plan, sort_keys=True, ensure_ascii=False)


def load_bundle(path: Path):
    cands = defaultdict(list)
    for line in (path / "per_candidate.jsonl").open():
        c = json.loads(line)
        cands[(c["pool"], int(c["task_id"]))].append(c)
    per_task = {(r["pool"], int(r["task_id"])): r for r in csv.DictReader((path / "per_task.csv").open())}
    return cands, per_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="append", required=True, help="label=path")
    ap.add_argument("--api", default="openrouter")
    ap.add_argument("--judge_model", default="openai/gpt-4o")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rng_seed", type=int, default=20260914)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max_tasks", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "locked_judge"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    descs = base.load_task_descs()
    ctx = base.load_tools_manifest(base.TOOLS)
    tools_card = base.compact_tools_card(ctx)
    client = base.make_client(args.api, args.judge_model)
    cache_path = out / f"cache_k{args.k}_{args.judge_model.replace('/', '_')}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    work = []
    for spec in args.bundle:
        label, path = spec.split("=", 1)
        cands, per_task = load_bundle(ROOT / path)
        items = sorted(k for k in per_task if k[0] in POOL_CODE)
        if args.max_tasks:
            keep = sorted({t for _, t in items})[: args.max_tasks]
            items = [k for k in items if k[1] in keep]
        for pool, tid in items:
            work.append((label, pool, tid, cands[(pool, tid)], per_task[(pool, tid)]))

    def run_one(item):
        label, pool, tid, cs, pt = item
        groups = {}
        for c in sorted(cs, key=lambda x: int(x["candidate_id"])):
            if c["hard0"]:
                groups.setdefault(key(c["plan"]), {"model_answer": json.dumps(c["plan"], ensure_ascii=False),
                                                   "correct": c["correct"]})
        uniq_all = list(groups.values())
        rng = random.Random(args.rng_seed + tid * 17 + POOL_CODE[pool] * 97 + args.k * 13)
        uniq = rng.sample(uniq_all, args.k) if len(uniq_all) > args.k else uniq_all
        fm_ok = pt["filter_majority_correct"] == "True"
        row = {"bundle": label, "pool": pool, "task_id": tid, "oracle": pt["oracle"] == "True",
               "filter_majority_ok": fm_ok, "n_unique_valid": len(uniq_all), "n_judged": len(uniq),
               "subsampled": len(uniq_all) > args.k, "judge_label": None}
        if not uniq:
            row.update(mode="empty_valid_fallback", judge_ok=fm_ok)
        elif len(uniq) == 1:
            row.update(mode="singleton_valid", judge_ok=bool(uniq[0]["correct"]))
        else:
            ck = f"{label}:{pool}:{tid}:k{args.k}"
            if ck not in cache:
                pick, meta = base.judge_unique(client, descs.get(tid, ""), tools_card, uniq, rng)
                cache[ck] = {"pick_key": pick["model_answer"] if pick else None,
                             "chosen_label": meta.get("chosen_label"), "raw": meta.get("raw"),
                             "label_order": meta.get("label_order")}
            pk = cache[ck]["pick_key"]
            pick = next((u for u in uniq if u["model_answer"] == pk), None)
            if pick is None:
                row.update(mode="judge_parse_fail_fallback_filter_maj", judge_ok=fm_ok)
            else:
                row.update(mode="judge", judge_ok=bool(pick["correct"]), judge_label=cache[ck]["chosen_label"])
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(run_one, it) for it in work]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 50 == 0 or i == len(work):
                print(f"[{i}/{len(work)}]", flush=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False))
    cache_path.write_text(json.dumps(cache, ensure_ascii=False))
    rows.sort(key=lambda r: (r["bundle"], r["pool"], r["task_id"]))

    summary = []
    for (label, pool) in dict.fromkeys((r["bundle"], r["pool"]) for r in rows):
        rs = [r for r in rows if r["bundle"] == label and r["pool"] == pool]
        summary.append({
            "bundle": label, "pool": pool, "n": len(rs),
            "oracle": sum(r["oracle"] for r in rs),
            "filter_majority": sum(r["filter_majority_ok"] for r in rs),
            "judge": sum(r["judge_ok"] for r in rs),
            "judge_fixes": sum(r["judge_ok"] and not r["filter_majority_ok"] for r in rs),
            "judge_breaks": sum(r["filter_majority_ok"] and not r["judge_ok"] for r in rs),
            "judge_calls": sum(r["mode"] != "singleton_valid" and r["mode"] != "empty_valid_fallback" for r in rs),
            "subsampled_tasks": sum(r["subsampled"] for r in rs),
            "modes": json.dumps(dict(Counter(r["mode"] for r in rs))),
        })
    for name, data in (("per_task.csv", rows), ("summary.csv", summary)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    (out / "usage.json").write_text(json.dumps({"usage": client.get_usage(), "args": vars(args)}, indent=2))
    for s in summary:
        print(s)
    print("usage", client.get_usage())


if __name__ == "__main__":
    main()
