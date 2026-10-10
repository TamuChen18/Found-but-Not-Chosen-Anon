#!/usr/bin/env python3
"""
Targeted Top-3 verifier on the final primary pools.

Cases: Oracle-positive, Selected-wrong tasks whose best correct workflow group is among the top 3
groups in selector order (valid hard0 candidates, exact-JSON groups, votes desc, smallest
candidate_id first). The verifier (GPT-4o, temperature 0) sees the task, the top-3 normalized
workflows in a shuffled A/B/C order, and the full specs of every tool they use; vote counts are not
shown. Recovery = the chosen workflow is correct.

Output: artifacts/final_primary/top3_verifier/{per_case.csv,summary.csv,cache.json,usage.json}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interfaces.OpenRouter_client import OpenRouterClient  # noqa: E402
from interfaces.call_context import call_context  # noqa: E402

OUT = ROOT / "artifacts" / "final_primary" / "top3_verifier"
TOOLS = json.loads((ROOT / "src" / "interfaces" / "tools" / "tools_manifest.json").read_text())
METHODS = {"BoN": "BoN", "CoT": "CoT-tree", "Five": "MO-Tree"}
POOL_CODE = {"CoT": 1, "Five": 2, "BoN": 4}
RUN_CODE = {"mini": 1, "qwen_b1": 2, "qwen_b2": 3}
BUNDLES = {"mini": "artifacts/mini_noseed_trees_full/eval_e2b_off",
           "qwen_b1": "artifacts/e2b_off_primary/qwen_b1",
           "qwen_b2": "artifacts/e2b_off_primary/qwen_b2"}
LABELS = "ABC"

SYSTEM = """You are verifying candidate tool-use workflows for a disaster-response planning task.
Exactly one of the candidate workflows below is the intended plan. All candidates are structurally
well formed, so judge them on planning semantics:
- Tool choice: does each step use the tool whose function matches what the task asks for, and are
  all requested analyses (and no unrequested ones) performed?
- Dependency semantics: does each input come from the right source (a file supplied in the task, or
  the output of the right earlier step), so that the data flow matches the task's intent?
In a workflow, "<GENERATED>-k-<key>" means the input is output <key> of step k, and "dependence"
lists the steps an input is taken from (-1 = none).
The candidate labels and their order are arbitrary.
Respond with ONLY a JSON object: {"choice": "<letter>", "reason": "<one short sentence>"}"""


def key(plan):
    return json.dumps(plan, sort_keys=True, ensure_ascii=False)


def tool_specs(plans):
    names = sorted({s.get("agent") for p in plans for s in p if isinstance(s, dict) and s.get("agent") in TOOLS})
    return "\n\n".join(f"### {n}\n{json.dumps(TOOLS[n], ensure_ascii=False, indent=1)}" for n in names)


def build_prompt(task, labeled):
    blocks = "\n\n".join(f"### Candidate {lab}\n```json\n{json.dumps(p, ensure_ascii=False, indent=1)}\n```"
                         for lab, p in labeled)
    return (f"{SYSTEM}\n\n## Task\n{task}\n\n## Tool specifications\n{tool_specs([p for _, p in labeled])}\n\n"
            f"## Candidates\n{blocks}\n\nReturn JSON only.")


def parse_choice(text, labels):
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            c = str(json.loads(m.group(0)).get("choice", "")).strip().upper()
            if c[:1] in labels:
                return c[:1]
        except json.JSONDecodeError:
            pass
    m = re.search(r'"choice"\s*:\s*"([A-C])"', text) or re.search(r"\b([A-C])\b", text)
    return m.group(1) if m and m.group(1) in labels else None


def load_cases():
    descs = {int(o["task_id"]): o["task_desc"] for o in map(json.loads, open(ROOT / "data" / "test.jsonl")) if o}
    cases = []
    for run, rel in BUNDLES.items():
        b = ROOT / rel
        cands = defaultdict(list)
        for line in (b / "per_candidate.jsonl").open():
            c = json.loads(line)
            if c["pool"] in METHODS:
                cands[(c["pool"], int(c["task_id"]))].append(c)
        for r in csv.DictReader((b / "per_task.csv").open()):
            pool, tid = r["pool"], int(r["task_id"])
            if pool not in METHODS or r["oracle"] != "True" or r["filter_majority_correct"] == "True":
                continue
            votes, first, plans, correct = Counter(), {}, {}, {}
            for c in cands[(pool, tid)]:
                if not c["hard0"]:
                    continue
                k = key(c["plan"])
                votes[k] += 1
                first[k] = min(first.get(k, 10 ** 9), int(c["candidate_id"]))
                plans[k], correct[k] = c["plan"], c["correct"]
            order = sorted(votes, key=lambda k: (-votes[k], first[k]))
            crank = next((i + 1 for i, k in enumerate(order) if correct[k]), None)
            if crank is None or crank > 3:
                continue
            top = [{"rank": i + 1, "plan": plans[k], "correct": bool(correct[k]), "votes": votes[k]}
                   for i, k in enumerate(order[:3])]
            cases.append({"run": run, "pool": pool, "method": METHODS[pool], "task_id": tid,
                          "task": descs[tid], "top": top, "correct_rank": crank,
                          "n_valid": sum(votes.values())})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--rng_seed", type=int, default=20261004)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    cases = load_cases()
    print(f"{len(cases)} cases", dict(Counter((c['run'], c['method']) for c in cases)), flush=True)
    if args.limit:
        cases = cases[: args.limit]
    client = OpenRouterClient(args.model)

    def run_one(c):
        rng = random.Random(args.rng_seed + RUN_CODE[c["run"]] * 100003 + POOL_CODE[c["pool"]] * 1009 + c["task_id"])
        perm = list(range(len(c["top"])))
        rng.shuffle(perm)
        labeled = [(LABELS[i], c["top"][j]) for i, j in enumerate(perm)]
        ck = f"{c['run']}:{c['pool']}:{c['task_id']}"
        if ck not in cache or cache[ck].get("choice") is None:
            prompt = build_prompt(c["task"], [(lab, t["plan"]) for lab, t in labeled])
            with call_context(task_id=c["task_id"], operator="TOP3_VERIFIER", run=c["run"], pool=c["pool"]):
                raw = client.generate(prompt, temperature=0.0, max_tokens=300)
            cache[ck] = {"raw": raw, "choice": parse_choice(raw, LABELS[: len(labeled)]),
                         "order": [t["rank"] for _, t in labeled]}
        ch = cache[ck]["choice"]
        pick = next((t for lab, t in labeled if lab == ch), None)
        return {"run": c["run"], "method": c["method"], "task_id": c["task_id"], "n_options": len(labeled),
                "n_valid": c["n_valid"], "correct_rank": c["correct_rank"],
                "label_order_ranks": "".join(str(t["rank"]) for _, t in labeled),
                "correct_label": "".join(lab for lab, t in labeled if t["correct"]),
                "choice": ch, "chosen_rank": pick["rank"] if pick else None,
                "recovered": bool(pick and pick["correct"]), "parse_fail": pick is None,
                "votes_top": "/".join(str(t["votes"]) for t in c["top"])}

    rows = []
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(run_one, c) for c in cases]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 25 == 0 or i == len(futs):
                cache_path.write_text(json.dumps(cache, ensure_ascii=False))
                print(f"[{i}/{len(futs)}]", flush=True)
    rows.sort(key=lambda r: (r["run"], r["method"], r["task_id"]))

    def summarize(label, rs):
        return {"pool": label, "cases": len(rs), "recovered": sum(r["recovered"] for r in rs),
                "recovered_pct": 100 * sum(r["recovered"] for r in rs) / len(rs) if rs else None,
                "chance_expected": sum(len(r["correct_label"]) / r["n_options"] for r in rs),
                "chose_majority_winner": sum(r["chosen_rank"] == 1 for r in rs),
                "rank2_cases": sum(r["correct_rank"] == 2 for r in rs),
                "rank2_recovered": sum(r["recovered"] and r["correct_rank"] == 2 for r in rs),
                "rank3_cases": sum(r["correct_rank"] == 3 for r in rs),
                "rank3_recovered": sum(r["recovered"] and r["correct_rank"] == 3 for r in rs),
                "two_option_cases": sum(r["n_options"] == 2 for r in rs),
                "parse_fail": sum(r["parse_fail"] for r in rs),
                "choice_dist": json.dumps(dict(Counter(r["choice"] for r in rs)))}

    summary = [summarize(f"{run}:{m}", [r for r in rows if r["run"] == run and r["method"] == m])
               for run in BUNDLES for m in METHODS.values()]
    summary.append(summarize("all", rows))
    for name, data in (("per_case.csv", rows), ("summary.csv", summary)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    (out / "usage.json").write_text(json.dumps({"usage": client.get_usage(), "args": {**vars(args), "out": os.path.relpath(args.out, ROOT)}}, indent=2))
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
