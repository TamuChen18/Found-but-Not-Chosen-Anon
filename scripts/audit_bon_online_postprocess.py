#!/usr/bin/env python3
"""BoN stores model_answer after the online `_postprocess_structured_plan` (which includes the E4
single-output key remap and E2b collapse). Re-extract each BoN candidate from its raw_completion,
apply only the final strict postprocess (E4 off, E2b off), and compare with the stored-answer path.

Final primary BoN pools only; the August seeded BoN run did not store raw_completion.
Writes artifacts/final_primary/provenance/bon_online_postprocess.{csv,json}. No LLM calls.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluators.evaluators import Evaluator  # noqa: E402
from freeze_strict_matched import TOOLS, TEST, load_tasks, majority_pick, plan_key, strict_postprocess  # noqa: E402
from utils.answer_selector import hard_violation_count, load_tools_manifest, parse_plan  # noqa: E402

POOLS = {
    "mini_noseed": ("artifacts/mini_noseed_trees_full/manifest_allnoseed.json", "BoN-CoT-M", 20260808),
    "qwen_b1": ("artifacts/qwen_nothink_main183/manifest.json", "BoN-CoT-M", 20260725),
    "qwen_b2": ("artifacts/qwen_nothink_main183/manifest.json", "BoN-CoT-M", 20260726),
}


def run_dir(manifest, key, seed):
    m = json.loads((ROOT / manifest).read_text())
    for r in m["runs"]:
        if r["key"] == key and (seed is None or int(r["seed"]) == seed):
            return ROOT / r["dir"]
    raise KeyError((manifest, key, seed))


def main():
    ev = Evaluator()
    ctx = load_tools_manifest(TOOLS)
    agent_outputs = {a: list(s.get("output", {}).keys()) for a, s in json.loads(Path(TOOLS).read_text()).items()}
    golds = {int(o["task_id"]): o["structured_plan"] for o in map(json.loads, Path(TEST).open()) if o}
    ids = [int(x) for x in (ROOT / "data" / "test_all183_ids.txt").read_text().split()]

    def evaluate(text, gold):
        plan = parse_plan(strict_postprocess(text, agent_outputs, enable_e4=False, collapse_multi=False)) if text else None
        ok = plan is not None and bool(ev.check_answers_equivalence(json.dumps(plan), json.dumps(gold)))
        return plan, ok, plan is not None and hard_violation_count(plan, ctx) == 0

    rows = []
    for name, (manifest, key, seed) in POOLS.items():
        d = run_dir(manifest, key, seed)
        tasks = load_tasks(d)
        n_cand = n_diff = n_corr_diff = n_hard_diff = 0
        res = {"stored": {"oracle": 0, "selected": 0}, "raw": {"oracle": 0, "selected": 0}}
        task_changes = {"oracle": 0, "selected": 0}
        for tid in ids:
            gold = golds[tid]
            per = {"stored": [], "raw": []}
            for c in sorted(tasks[tid]["all_candidates"], key=lambda x: int(x["candidate_id"])):
                cid = int(c["candidate_id"])
                stored = c.get("model_answer")
                raw = c.get("raw_completion")
                re_ex = None
                if raw:
                    try:
                        re_ex = ev.extract_answer_from_model_completion(raw)
                    except Exception:
                        re_ex = None
                a = evaluate(stored, gold)
                b = evaluate(re_ex, gold)
                n_cand += 1
                n_diff += plan_key(a[0]) != plan_key(b[0])
                n_corr_diff += a[1] != b[1]
                n_hard_diff += a[2] != b[2]
                per["stored"].append({"candidate_id": cid, "plan": a[0], "correct": a[1], "hard0": a[2]})
                per["raw"].append({"candidate_id": cid, "plan": b[0], "correct": b[1], "hard0": b[2]})
            out = {}
            for k, items in per.items():
                hard = [x for x in items if x["hard0"]]
                pick = majority_pick(hard) if hard else majority_pick(items)
                out[k] = (any(x["correct"] for x in items), bool(pick and pick["correct"]))
                res[k]["oracle"] += out[k][0]
                res[k]["selected"] += out[k][1]
            task_changes["oracle"] += out["stored"][0] != out["raw"][0]
            task_changes["selected"] += out["stored"][1] != out["raw"][1]
        rows.append({
            "pool": name, "candidates": n_cand, "normalized_plan_differs": n_diff,
            "correctness_differs": n_corr_diff, "hard0_differs": n_hard_diff,
            "oracle_stored": res["stored"]["oracle"], "oracle_raw_reextract": res["raw"]["oracle"],
            "selected_stored": res["stored"]["selected"], "selected_raw_reextract": res["raw"]["selected"],
            "tasks_oracle_changed": task_changes["oracle"], "tasks_selected_changed": task_changes["selected"],
        })
        print(rows[-1], flush=True)
    out = ROOT / "artifacts" / "final_primary" / "provenance"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "bon_online_postprocess.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "bon_online_postprocess.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
