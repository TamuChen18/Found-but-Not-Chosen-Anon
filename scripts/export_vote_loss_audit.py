#!/usr/bin/env python3
"""
Export the winner-vs-gold audit sample (no LLM calls).

Audit unit: a pool-task instance with Oracle = 1, Selected = 0 and a non-tie vote loss under the
locked protocol: at least one exact-match correct candidate passes the structural filter
(so the loss is not caused by filtering) and the largest correct group has strictly fewer valid
votes than the winning wrong group. Votes are counted over valid candidates exactly as the locked
Filter->Maj does (tree pools: one vote per unique solution text).

Sampling: per (bundle, pool), the population sorted by task_id, a fixed-seed random sample of n
(random.Random(f"{seed}:{label}:{pool}")). Pools listed in --export_all are also exported in full.

Auto hints (objective, to speed up annotation; they are not labels):
  first_positional_diff   first step / field where winner and gold differ as strict EM compares them
  tools_missing / tools_extra   agent multiset differences (winner vs gold)
  n_steps_gold / n_steps_winner
  permutation_equivalent  winner equals gold after re-ordering steps and renumbering every step
                          reference consistently (candidate EM artifact; still needs human review)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = re.compile(r"^<GENERATED>-(\d+)-<(.+)>$")
ANNOTATION_COLS = ["primary_error_type", "secondary_error_type", "semantic_equivalent",
                   "first_substantive_difference", "changes_execution_semantics", "short_explanation",
                   "case_study_candidate"]
MODEL = {"mini": "GPT-4o-mini", "qwen_b1": "Qwen3.5-9B non-thinking (batch 20260725)"}
METHOD = {"BoN": "BoN", "CoT": "CoT-tree", "Five": "MO-Tree"}


def key(plan):
    return json.dumps(plan, sort_keys=True, ensure_ascii=False)


def load_bundle(path):
    cands = defaultdict(list)
    for line in (path / "per_candidate.jsonl").open():
        c = json.loads(line)
        cands[(c["pool"], int(c["task_id"]))].append(c)
    per_task = {(r["pool"], int(r["task_id"])): r for r in csv.DictReader((path / "per_task.csv").open())}
    return cands, per_task


def vote_loss(cs):
    """Return (winner_plan, win_votes, best_correct_plan, correct_votes, n_valid) for a non-tie vote loss."""
    valid = sorted((c for c in cs if c["hard0"]), key=lambda c: int(c["candidate_id"]))
    groups, first = Counter(), {}
    for c in valid:
        k = key(c["plan"])
        groups[k] += 1
        first.setdefault(k, c)
    if not groups:
        return None
    wk = max(groups, key=lambda k: (groups[k], -int(first[k]["candidate_id"])))
    if first[wk]["correct"]:
        return None
    correct = [k for k in groups if first[k]["correct"]]
    if not correct:
        return None
    ck = max(correct, key=lambda k: (groups[k], -int(first[k]["candidate_id"])))
    if groups[ck] >= groups[wk]:
        return None
    return first[wk]["plan"], groups[wk], first[ck]["plan"], groups[ck], len(valid)


def remap(plan, perm):
    """perm: old step id -> new step id. Renumber steps and every reference; reorder by new id."""
    def ref(v):
        if isinstance(v, str):
            m = GEN.match(v.strip())
            if m and int(m.group(1)) in perm:
                return f"<GENERATED>-{perm[int(m.group(1))]}-<{m.group(2)}>"
        return v
    out = []
    for s in plan:
        s = json.loads(json.dumps(s))
        if isinstance(s.get("step"), int) and s["step"] in perm:
            s["step"] = perm[s["step"]]
        if isinstance(s.get("dependence"), list):
            s["dependence"] = [perm.get(d, d) if isinstance(d, int) else d for d in s["dependence"]]
        if isinstance(s.get("dependence_content"), dict):
            s["dependence_content"] = {str(perm.get(int(k), k)) if str(k).lstrip("-").isdigit() else k: v
                                       for k, v in s["dependence_content"].items()}
        if isinstance(s.get("inputs"), dict):
            s["inputs"] = {k: ref(v) for k, v in s["inputs"].items()}
        out.append(s)
    return sorted(out, key=lambda s: s.get("step") if isinstance(s.get("step"), int) else 10**6)


def permutation_equivalent(winner, gold, limit=40320):
    if len(winner) != len(gold) or len(winner) > 8:
        return False
    if Counter(s.get("agent") for s in winner) != Counter(s.get("agent") for s in gold):
        return False
    steps = [s.get("step") for s in winner]
    gold_steps = [s.get("step") for s in gold]
    if not all(isinstance(x, int) for x in steps + gold_steps):
        return False
    target = key(gold)
    for n, p in enumerate(itertools.permutations(gold_steps)):
        if n >= limit:
            break
        perm = dict(zip(steps, p))
        if all(w.get("agent") == g.get("agent") for w, g in
               zip(sorted(winner, key=lambda s: perm[s["step"]]), sorted(gold, key=lambda s: s["step"]))):
            if key(remap(winner, perm)) == target:
                return True
    return False


def first_diff(winner, gold):
    if len(winner) != len(gold):
        return f"step count {len(winner)} vs gold {len(gold)}"
    for i, (w, g) in enumerate(zip(winner, gold)):
        for f in ("agent", "inputs", "dependence", "dependence_content", "outputs", "step"):
            if json.dumps(w.get(f), sort_keys=True) != json.dumps(g.get(f), sort_keys=True):
                if f == "inputs" and isinstance(w.get(f), dict) and isinstance(g.get(f), dict):
                    ks = sorted(set(w[f]) | set(g[f]))
                    bad = [k for k in ks if w[f].get(k) != g[f].get(k)]
                    return f"position {i}: inputs[{', '.join(bad)}]"
                return f"position {i}: {f}"
        extra = set(w) ^ set(g)
        if extra:
            return f"position {i}: keys {sorted(extra)}"
    return "none (identical as compared)"


def pretty(plan):
    lines = []
    for s in plan:
        dep = s.get("dependence")
        ins = ", ".join(f"{k}={v}" for k, v in (s.get("inputs") or {}).items())
        lines.append(f"s{s.get('step')}: {s.get('agent')} | inputs: {ins} | dep: {dep} | "
                     f"dep_content: {json.dumps(s.get('dependence_content'), ensure_ascii=False)} | "
                     f"outputs: {s.get('outputs')}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="append", required=True, help="label=bundle_dir")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--export_all", action="append", default=[], help="label:pool exported in full")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "vote_loss_audit"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = {int(o["task_id"]): o for o in map(json.loads, (ROOT / "data" / "test.jsonl").open()) if o}

    sample, full, pop_rows = [], defaultdict(list), []
    for spec in args.bundle:
        label, path = spec.split("=", 1)
        cands, per_task = load_bundle(ROOT / path)
        for pool in ("BoN", "CoT", "Five"):
            pop = []
            for (p, tid), pt in sorted(per_task.items()):
                if p != pool or pt["oracle"] != "True" or pt["filter_majority_correct"] == "True":
                    continue
                vl = vote_loss(cands[(p, tid)])
                if vl:
                    pop.append((tid, vl))
            rng = random.Random(f"{args.seed}:{label}:{pool}")
            picked = {tid for tid, _ in rng.sample(pop, min(args.n, len(pop)))}
            pop_rows.append({"model": MODEL[label], "method": METHOD[pool], "non_tie_vote_losses": len(pop),
                             "sampled": len(picked), "sampling": f"random.Random('{args.seed}:{label}:{pool}')"})
            for tid, (win, wv, cor, cv, nv) in pop:
                gold = tasks[tid]["structured_plan"]
                wa, ga = Counter(s.get("agent") for s in win), Counter(s.get("agent") for s in gold)
                row = {
                    "audit_id": f"{label}-{pool}-{tid}", "task_id": tid, "model": MODEL[label], "method": METHOD[pool],
                    "task_description": tasks[tid].get("task_desc", ""),
                    "gold_workflow": json.dumps(gold, ensure_ascii=False),
                    "winning_workflow": json.dumps(win, ensure_ascii=False), "winning_votes": wv,
                    "best_correct_workflow": json.dumps(cor, ensure_ascii=False), "correct_votes": cv,
                    "valid_candidates": nv, "structurally_valid_winner": True,
                    "auto_first_positional_diff": first_diff(win, gold),
                    "auto_tools_missing": json.dumps(sorted((ga - wa).elements())),
                    "auto_tools_extra": json.dumps(sorted((wa - ga).elements())),
                    "auto_n_steps_gold": len(gold), "auto_n_steps_winner": len(win),
                    "auto_permutation_equivalent": permutation_equivalent(win, gold),
                    "in_audit_sample": tid in picked,
                    **{c: "" for c in ANNOTATION_COLS},
                }
                if tid in picked:
                    sample.append(row)
                if f"{label}:{pool}" in args.export_all:
                    full[f"{label}_{pool}"].append(row)

    def write(path, rows):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write(out / "audit_sample.csv", sample)
    write(out / "population.csv", pop_rows)
    for name, rows in full.items():
        write(out / f"all_vote_losses_{name}.csv", rows)

    def cases_md(rows, title):
        md = [f"# {title}", ""]
        for r in rows:
            md += [f"## {r['audit_id']}  ({r['model']} · {r['method']} · task {r['task_id']})", "",
                   f"**Task.** {r['task_description']}", "",
                   f"**Votes.** winning wrong workflow: {r['winning_votes']} · best correct workflow: "
                   f"{r['correct_votes']} · valid candidates: {r['valid_candidates']}", "",
                   f"**Auto hints.** first positional diff: {r['auto_first_positional_diff']} · tools missing: "
                   f"{r['auto_tools_missing']} · tools extra: {r['auto_tools_extra']} · steps gold/winner: "
                   f"{r['auto_n_steps_gold']}/{r['auto_n_steps_winner']} · permutation-equivalent: "
                   f"{r['auto_permutation_equivalent']}", "",
                   "**Gold**", "```", pretty(json.loads(r["gold_workflow"])), "```",
                   "**Winning (wrong)**", "```", pretty(json.loads(r["winning_workflow"])), "```",
                   "**Best correct candidate**", "```", pretty(json.loads(r["best_correct_workflow"])), "```", ""]
        return "\n".join(md)

    (out / "audit_sample_cases.md").write_text(cases_md(sample, "Winner-vs-gold audit sample"))
    for name, rows in full.items():
        (out / f"all_vote_losses_{name}_cases.md").write_text(cases_md(rows, f"All non-tie vote losses: {name}"))
    for r in pop_rows:
        print(r)
    print(f"sample rows: {len(sample)}; permutation-equivalent winners in sample: "
          f"{sum(r['auto_permutation_equivalent'] for r in sample)}")
    for name, rows in full.items():
        print(f"{name}: {len(rows)} rows exported in full")


if __name__ == "__main__":
    main()
