#!/usr/bin/env python3
"""
Audit of the final GPT-4o generic judge (run_locked_judge.py, GPT-4o-mini final pools, k = 8):
per pool, judge calls, finish_reason counts from the OpenRouter call log, completion-token max,
responses that parse to a valid candidate label, parse-failure fallbacks (fallback = the locked
Filter->Maj pick, counted in judge accuracy), and tasks with > k unique valid workflows (random
k-subset shown to the judge).

Asserts the call log has one ok call per cached judge decision, all with max_tokens = 300 and
temperature 0. Writes artifacts/final_primary/judge/truncation_audit.csv.

Qwen pools (artifacts/e2b_off_primary/judge): decisions come from the E2b-on locked judge cache
(artifacts/locked_judge, one logged call per entry in logs/locked_judge_calls.jsonl) plus the 88
calls re-run after E2b off (logs/e2b_off_judge_calls.jsonl). The older logs carry no prompt hash,
so for every reused decision the judge prompt is rebuilt from the E2b-on and the E2b-off pools and
asserted identical. Writes artifacts/e2b_off_primary/judge/truncation_audit.csv. No LLM calls.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_semantic_judge as base  # noqa: E402
from run_locked_judge import POOL_CODE, key, load_bundle  # noqa: E402

QJ = ROOT / "artifacts" / "e2b_off_primary" / "judge"
QLOGS = [ROOT / "logs" / "locked_judge_calls.jsonl", ROOT / "logs" / "e2b_off_judge_calls.jsonl"]
QOLD = {"qwen_b1": ROOT / "artifacts/qwen_nothink_main183/eval/b20260725",
        "qwen_b2": ROOT / "artifacts/qwen_nothink_main183/eval/b20260726"}
J = ROOT / "artifacts" / "final_primary" / "judge"
LOG = ROOT / "logs" / "final_primary_judge_calls.jsonl"


def main():
    calls = [json.loads(line) for line in LOG.open()]
    cache = json.loads((J / "cache_k8_openai_gpt-4o.json").read_text())
    per_task = list(csv.DictReader((J / "per_task.csv").open()))
    assert len(calls) == len(cache), (len(calls), len(cache))
    assert all(c["status"] == "ok" and str(c["max_tokens"]) == "300" and float(c["temperature"]) == 0.0 for c in calls)
    fin = Counter(r for c in calls for r in (c["finish_reasons"] if isinstance(c["finish_reasons"], list)
                                              else json.loads(c["finish_reasons"].replace("'", '"'))))
    rows = []
    for pool in ("BoN", "CoT", "Five"):
        rs = [r for r in per_task if r["pool"] == pool]
        keys = [k for k in cache if k.split(":")[1] == pool]
        modes = Counter(r["mode"] for r in rs)
        rows.append({"pool": pool, "tasks": len(rs), "judge_calls": len(keys),
                     "parsed_label": sum(cache[k]["chosen_label"] is not None for k in keys),
                     "parse_fail_fallback": modes.get("judge_parse_fail_fallback_filter_maj", 0),
                     "singleton_valid": modes.get("singleton_valid", 0),
                     "empty_valid_fallback": modes.get("empty_valid_fallback", 0),
                     "subsampled_gt_k8": sum(r["subsampled"] == "True" for r in rs)})
    rows.append({"pool": "all", **{k: sum(r[k] for r in rows) for k in rows[0] if k != "pool"},
                 "finish_reason_counts": json.dumps(dict(fin)),
                 "max_completion_tokens": max(int(c["completion_tokens"]) for c in calls)})
    with (J / "truncation_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    qwen()


def finish(c):
    fr = c["finish_reasons"]
    return fr if isinstance(fr, list) else json.loads(fr.replace("'", '"'))


def judge_prompt(cands, pool, tid, descs, card, k=8, seed=20260914):
    groups = {}
    for c in sorted(cands[(pool, tid)], key=lambda x: int(x["candidate_id"])):
        if c["hard0"]:
            groups.setdefault(key(c["plan"]), {"model_answer": json.dumps(c["plan"], ensure_ascii=False)})
    uniq = list(groups.values())
    rng = random.Random(seed + tid * 17 + POOL_CODE[pool] * 97 + k * 13)
    uniq = rng.sample(uniq, k) if len(uniq) > k else uniq
    if len(uniq) < 2:
        return None
    order = list(range(len(uniq)))
    rng.shuffle(order)
    return base.build_judge_prompt(descs.get(tid, ""), card, [(f"C{i}", uniq[j]) for i, j in enumerate(order)])


def qwen():
    calls = [json.loads(line) for f in QLOGS for line in f.open()]
    assert all(c["status"] == "ok" and int(c["max_tokens"]) == 300 for c in calls)
    locked = json.loads((ROOT / "artifacts/locked_judge/cache_k8_openai_gpt-4o.json").read_text())
    assert len(locked) == sum(1 for _ in QLOGS[0].open())
    recomputed = set(json.loads((QJ / "recomputed_tasks.json").read_text()))
    assert len(recomputed) == sum(1 for _ in QLOGS[1].open())
    cache = json.loads((QJ / "cache_k8_openai_gpt-4o.json").read_text())
    per_task = [r for r in csv.DictReader((QJ / "per_task.csv").open()) if r["bundle"] != "mini"]
    descs = base.load_task_descs()
    card = base.compact_tools_card(base.load_tools_manifest(base.TOOLS))
    rows = []
    for label, old_path in QOLD.items():
        new, per = load_bundle(ROOT / "artifacts" / "e2b_off_primary" / label)
        old, _ = load_bundle(old_path)
        for pool in ("BoN", "CoT", "Five"):
            rs = [r for r in per_task if r["bundle"] == label and r["pool"] == pool]
            modes = Counter(r["mode"] for r in rs)
            n_judge = reused = 0
            for r in rs:
                if r["mode"] != "judge":
                    continue
                tid = int(r["task_id"])
                ck = f"{label}:{pool}:{tid}:k8"
                assert cache[ck]["chosen_label"] is not None, ck
                n_judge += 1
                if ck not in recomputed:
                    reused += 1
                    assert ck in locked and locked[ck]["pick_key"] == cache[ck]["pick_key"], ck
                    assert judge_prompt(old, pool, tid, descs, card) == judge_prompt(new, pool, tid, descs, card), ck
            rows.append({"pool": f"{label}:{pool}", "tasks": len(rs), "judge_calls": n_judge, "parsed_label": n_judge,
                         "parse_fail_fallback": modes.get("judge_parse_fail_fallback_filter_maj", 0),
                         "singleton_valid": modes.get("singleton_valid", 0),
                         "empty_valid_fallback": modes.get("empty_valid_fallback", 0),
                         "subsampled_gt_k8": sum(r["subsampled"] == "True" for r in rs),
                         "reused_e2b_on_decisions_same_prompt": reused, "rerun_after_e2b_off": n_judge - reused})
    rows.append({"pool": "qwen all", **{k: sum(r[k] for r in rows) for k in rows[0] if k != "pool"},
                 "logged_calls_covering_these_decisions": len(calls),
                 "finish_reason_counts": json.dumps(dict(Counter(r for c in calls for r in finish(c)))),
                 "max_completion_tokens": max(int(c["completion_tokens"]) for c in calls)})
    with (QJ / "truncation_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
