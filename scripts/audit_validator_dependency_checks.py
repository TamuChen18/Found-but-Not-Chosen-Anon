#!/usr/bin/env python3
"""
Does the structural validator's dependency-consistency check ever bind on the primary evaluation path?

Primary path (freeze_strict_matched.py --keep_multi_pred, i.e. E4 off, E2b off):
  raw candidate (extracted answer) -> strict_postprocess (E1 GENERATED canonicalization, E2 dependence /
  dependence_content rebuilt from the GENERATED references, E3 outputs fill) -> parse_plan ->
  hard_violation_count == 0.

hard_violation_count is re-implemented here with one counter per check (the total is asserted equal to the
library function for every candidate). The dependency-consistency checks are
  C1  a GENERATED source step is not listed in dependence
  C2  dependence lists predecessors but no input carries a GENERATED reference
  C3  dependence is [-1] but some input carries a GENERATED reference
For every candidate of the 11 final pools the raw plan is re-postprocessed and must equal the stored plan.

Output: artifacts/final_primary/validator_dependency_audit/{summary.csv,c_cases.csv}
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_vote_rank import BUNDLES  # noqa: E402
from freeze_strict_matched import TOOLS, strict_postprocess  # noqa: E402
from utils.answer_selector import (GENERATED_RE, _ctx_agents, _ctx_manifest, hard_violation_count,  # noqa: E402
                                   load_tools_manifest, parse_plan)

OUT = ROOT / "artifacts" / "final_primary" / "validator_dependency_audit"
CHECKS = ["non_dict_step", "unknown_agent", "bad_dependence_field", "bad_dependence_id", "dep_out_of_range",
          "missing_required_input", "malformed_generated", "generated_out_of_range", "generated_unknown_output",
          "C1_ref_not_in_dependence", "C2_dependence_without_ref", "C3_ref_without_dependence"]
CONSISTENCY = {"C1_ref_not_in_dependence", "C2_dependence_without_ref", "C3_ref_without_dependence"}


def violations(plan, ctx):
    c = Counter()
    if not plan:
        c["empty_plan"] = 99
        return c
    manifest, agents = _ctx_manifest(ctx), _ctx_agents(ctx)
    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            c["non_dict_step"] += 1
            continue
        agent = step.get("agent")
        if not isinstance(agent, str) or (agents is not None and agent not in agents):
            c["unknown_agent"] += 1
            continue
        deps = step.get("dependence", [-1])
        if deps is None:
            deps = [-1]
        if not isinstance(deps, list) or not deps:
            c["bad_dependence_field"] += 1
            deps = []
        dep_ids = []
        for d in deps:
            try:
                di = int(d)
            except (TypeError, ValueError):
                c["bad_dependence_id"] += 1
                continue
            dep_ids.append(di)
            if di != -1 and (di < 0 or di >= i):
                c["dep_out_of_range"] += 1
        inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
        spec = (manifest or {}).get(agent) if manifest else None
        expected = set((spec.get("input") or {}).keys()) if spec else None
        if expected is not None:
            c["missing_required_input"] += len(expected - set(inputs.keys()))
        non_neg = [d for d in dep_ids if d != -1]
        refs = []
        for val in inputs.values():
            if not isinstance(val, str) or "GENERATED" not in val:
                continue
            m = GENERATED_RE.match(val)
            if not m:
                c["malformed_generated"] += 1
                continue
            src, key = int(m.group(1)), m.group(2)
            refs.append(src)
            if src < 0 or src >= i:
                c["generated_out_of_range"] += 1
                continue
            if key is not None and manifest is not None:
                outs = set((((manifest.get(plan[src].get("agent")) or {}).get("output")) or {}).keys())
                so = plan[src].get("outputs")
                if isinstance(so, list):
                    outs |= set(str(x) for x in so)
                if key not in outs:
                    c["generated_unknown_output"] += 1
            if non_neg and src not in non_neg:
                c["C1_ref_not_in_dependence"] += 1
        if non_neg and not refs:
            c["C2_dependence_without_ref"] += 1
        if not non_neg and refs:
            c["C3_ref_without_dependence"] += 1
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = load_tools_manifest(TOOLS)
    agent_outputs = {a: list(s.get("output", {}).keys()) for a, s in json.loads(Path(TOOLS).read_text()).items()}
    seen, rows, cases = set(), [], []
    tot = Counter()
    for run, bundle, names in BUNDLES:
        for line in (ROOT / bundle / "per_candidate.jsonl").open():
            cand = json.loads(line)
            if cand["pool"] not in names:
                continue
            k = (run, cand["pool"], cand["task_id"], cand["candidate_id"])
            if k in seen:
                continue
            seen.add(k)
            raw, stored = cand.get("raw_plan"), cand.get("plan")
            if raw is not None:
                rebuilt = parse_plan(strict_postprocess(json.dumps(raw, ensure_ascii=False), agent_outputs,
                                                        enable_e4=False, collapse_multi=False))
                assert rebuilt == stored, k
            v = violations(stored, ctx)
            n = sum(v.values())
            assert n == hard_violation_count(stored, ctx), k
            assert (n == 0) == bool(cand["hard0"]), k
            pool = f"{run}:{names[cand['pool']]}"
            tot["candidates"] += 1
            tot[f"pool={pool}"] += 1
            tot["parsed"] += stored is not None
            tot["hard0"] += n == 0
            cons = {c: v[c] for c in CONSISTENCY if v[c]}
            if cons:
                tot["any_consistency_fired"] += 1
                other = {c: x for c, x in v.items() if c not in CONSISTENCY and x}
                tot["consistency_only"] += not other
                cases.append({"pool": pool, "task_id": cand["task_id"], "candidate_id": cand["candidate_id"],
                              **cons, "other_violations": json.dumps(other)})
            for c in CHECKS:
                tot[f"cands_with:{c}"] += v[c] > 0
    rows = [{"metric": k, "value": v} for k, v in tot.items()]
    with (OUT / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerows(rows)
    with (OUT / "c_cases.csv").open("w", newline="") as f:
        fields = ["pool", "task_id", "candidate_id", *sorted(CONSISTENCY), "other_violations"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(cases)
    for k, v in tot.items():
        if not k.startswith("pool="):
            print(f"{k}: {v}")
    print(f"pools: {sum(k.startswith('pool=') for k in tot)}")


if __name__ == "__main__":
    main()
