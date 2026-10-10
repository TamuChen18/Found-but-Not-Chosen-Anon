#!/usr/bin/env python3
"""Per-transformation audit of the offline strict postprocess on the final primary pools
(GPT-4o-mini no-seed; Qwen batches 20260725 / 20260726) and on the historical August GPT-4o-mini
seeded pools.

For every candidate, starting from the stored (extracted / repaired) model_answer:
  E1  GENERATED reference canonicalization to <GENERATED>-{step}-<{key}>
  E2a dependence / dependence_content rebuilt from GENERATED references (all predecessors kept)
  E2b multi-predecessor collapse to the first predecessor        (disabled in the final protocol)
  E3  empty / missing outputs filled from the tool manifest
  E4  single-output key remap                                   (disabled in the final protocol)
  E5  non-dict inputs replaced by {}

Reports trigger counts, candidate-level Exact Match changes, and pool-level Oracle / unfiltered
majority / filtered majority / invalid % for: no normalization, final (E4 off, E2b off),
final + E4, final + E2b, and leave-one-out variants. No LLM calls.

Writes artifacts/final_primary/transformation_audit/{triggers,pool_variants}.csv and audit.json.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluators.evaluators import Evaluator  # noqa: E402
from freeze_strict_matched import TOOLS, TEST, load_tasks, majority_pick  # noqa: E402
from utils.answer_selector import hard_violation_count, load_tools_manifest, parse_plan  # noqa: E402

GEN_PAT = re.compile(r"^<GENERATED>-(\d+)-<?([^<>]+)>?$")
POOL_KEYS = {"BoN": "BoN-CoT-M", "CoT-tree": "CoT-only-M", "MO-Tree": "Five-M"}
BUNDLES = {
    "mini": ("artifacts/mini_noseed_trees_full/manifest_allnoseed.json", 20260808),
    "qwen_b1": ("artifacts/qwen_nothink_main183/manifest.json", 20260725),
    "qwen_b2": ("artifacts/qwen_nothink_main183/manifest.json", 20260726),
    "mini_seeded_aug": ("artifacts/phase2_fulltest/manifest_mini.json", 20260808),
}
VARIANTS = {
    "none": dict(e1=False, e2=False, e3=False, e4=False, e5=False, collapse=False),
    "final": dict(e1=True, e2=True, e3=True, e4=False, e5=True, collapse=False),
    "final+E4": dict(e1=True, e2=True, e3=True, e4=True, e5=True, collapse=False),
    "final+E2b": dict(e1=True, e2=True, e3=True, e4=False, e5=True, collapse=True),
    "final-E1": dict(e1=False, e2=True, e3=True, e4=False, e5=True, collapse=False),
    "final-E2a": dict(e1=True, e2=False, e3=True, e4=False, e5=True, collapse=False),
    "final-E3": dict(e1=True, e2=True, e3=False, e4=False, e5=True, collapse=False),
}


def postprocess(plan, agent_outputs, *, e1, e2, e3, e4, e5, collapse, tags=None):
    """Flag-controlled copy of freeze_strict_matched.strict_postprocess; with the `final` flags it is
    identical to strict_postprocess(..., enable_e4=False, collapse_multi=False)."""
    tags = tags if tags is not None else set()
    plan = json.loads(json.dumps(plan))
    step_to_outputs = {}
    for item in plan:
        if not isinstance(item, dict) or not isinstance(item.get("step"), int):
            continue
        outputs = item.get("outputs", [])
        if not isinstance(outputs, list) or len(outputs) == 0:
            agent = item.get("agent")
            fill = agent_outputs.get(agent, []) if isinstance(agent, str) else []
            if fill:
                tags.add("E3")
                if e3:
                    item["outputs"] = fill
        cur = item.get("outputs", [])
        step_to_outputs[item["step"]] = cur if isinstance(cur, list) else []
    for item in plan:
        if not isinstance(item, dict):
            continue
        inputs = item.get("inputs", {})
        if not isinstance(inputs, dict):
            tags.add("E5")
            if not e5:
                continue
            inputs = {}
            item["inputs"] = inputs
        preds, consumed = [], defaultdict(list)
        for name, value in list(inputs.items()):
            if not isinstance(value, str):
                continue
            m = GEN_PAT.match(value.strip())
            if not m:
                continue
            ref_step, orig_key = int(m.group(1)), m.group(2).strip()
            ref_key = orig_key
            valid = step_to_outputs.get(ref_step, [])
            if valid and ref_key not in valid and len(valid) == 1:
                tags.add("E4")
                if e4:
                    ref_key = valid[0]
            if f"<GENERATED>-{ref_step}-<{orig_key}>" != value:
                tags.add("E1")
            if e1 or ref_key != orig_key:
                inputs[name] = f"<GENERATED>-{ref_step}-<{ref_key}>"
            if ref_step not in preds:
                preds.append(ref_step)
            if ref_key not in consumed[ref_step]:
                consumed[ref_step].append(ref_key)
        if not preds:
            new_dep, new_dc = [-1], None
        elif collapse or len(preds) == 1:
            new_dep, new_dc = [preds[0]], {str(preds[0]): consumed[preds[0]]}
        else:
            new_dep, new_dc = list(preds), {str(p): consumed[p] for p in preds}
        if len(preds) > 1:
            tags.add("E2b")
        full_dep = list(preds) if preds else [-1]
        full_dc = {str(p): consumed[p] for p in preds} if preds else None
        if item.get("dependence") != full_dep or item.get("dependence_content") != full_dc:
            tags.add("E2a")
        if e2:
            item["dependence"] = new_dep
            item["dependence_content"] = new_dc
    return plan


def run_dir(manifest, key, seed):
    m = json.loads((ROOT / manifest).read_text())
    for r in m["runs"]:
        if r["key"] == key and int(r["seed"]) == seed:
            return ROOT / r["dir"]
    raise KeyError((manifest, key, seed))


def main():
    ev = Evaluator()
    ctx = load_tools_manifest(TOOLS)
    agent_outputs = {a: list(s.get("output", {}).keys()) for a, s in json.loads(Path(TOOLS).read_text()).items()}
    golds = {int(o["task_id"]): o["structured_plan"] for o in map(json.loads, Path(TEST).open()) if o}
    ids = [int(x) for x in (ROOT / "data" / "test_all183_ids.txt").read_text().split()]

    def correct(plan, gold):
        return plan is not None and bool(ev.check_answers_equivalence(json.dumps(plan), json.dumps(gold)))

    trig_rows, var_rows = [], []
    for bundle, (manifest, seed) in BUNDLES.items():
        for pool, key in POOL_KEYS.items():
            tasks = load_tasks(run_dir(manifest, key, seed))
            trig = defaultdict(int)
            em_change = defaultdict(lambda: [0, 0])
            agg = {v: {"oracle": 0, "unfiltered": 0, "filtered": 0, "n": 0, "valid": 0} for v in VARIANTS}
            for tid in ids:
                gold = golds[tid]
                per = {v: [] for v in VARIANTS}
                for c in sorted(tasks[tid].get("all_candidates") or [], key=lambda x: int(x.get("candidate_id", 0))):
                    cid = int(c.get("candidate_id", 0))
                    ans = c.get("model_answer")
                    text = ans if isinstance(ans, str) or ans is None else json.dumps(ans, ensure_ascii=False)
                    base = parse_plan(text) if text else None
                    tags = set()
                    for v, flags in VARIANTS.items():
                        plan = postprocess(base, agent_outputs, tags=tags if v == "final" else None, **flags) \
                            if base is not None else None
                        per[v].append({"candidate_id": cid, "plan": plan, "correct": correct(plan, gold),
                                       "hard0": plan is not None and hard_violation_count(plan, ctx) == 0})
                    trig["candidates"] += 1
                    for t in tags:
                        trig[t] += 1
                    c_none, c_final = per["none"][-1]["correct"], per["final"][-1]["correct"]
                    for v in VARIANTS:
                        if v in ("none", "final"):
                            continue
                        d = per[v][-1]["correct"]
                        if d != c_final:
                            em_change[v][0 if d else 1] += 1
                    if c_none != c_final:
                        em_change["none"][0 if c_none else 1] += 1
                for v, items in per.items():
                    hard = [x for x in items if x["hard0"]]
                    a = agg[v]
                    a["oracle"] += any(x["correct"] for x in items)
                    maj = majority_pick(items)
                    a["unfiltered"] += bool(maj and maj["correct"])
                    filt = majority_pick(hard) if hard else maj
                    a["filtered"] += bool(filt and filt["correct"])
                    a["n"] += len(items)
                    a["valid"] += len(hard)
            trig_rows.append({"bundle": bundle, "pool": pool, **{k: trig.get(k, 0) for k in
                              ["candidates", "E1", "E2a", "E2b", "E3", "E4", "E5"]},
                              **{f"em_{v}_vs_final_correct_only/final_only": f"{x[0]}/{x[1]}"
                                 for v, x in em_change.items()}})
            for v, a in agg.items():
                var_rows.append({"bundle": bundle, "pool": pool, "variant": v, "oracle": a["oracle"],
                                 "unfiltered_majority": a["unfiltered"], "filtered_majority": a["filtered"],
                                 "invalid_pct": round(100 * (1 - a["valid"] / a["n"]), 2)})
            print(trig_rows[-1], flush=True)
    out = ROOT / "artifacts" / "final_primary" / "transformation_audit"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("triggers.csv", trig_rows), ("pool_variants.csv", var_rows)):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    (out / "audit.json").write_text(json.dumps({"triggers": trig_rows, "variants": var_rows}, indent=1))


if __name__ == "__main__":
    main()
