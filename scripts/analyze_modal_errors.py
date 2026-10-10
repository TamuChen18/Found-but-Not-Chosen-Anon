#!/usr/bin/env python3
"""
Vote-loss winner consistency / modal-error analysis (offline, no LLM calls).

Population: pool-tasks with Oracle=1, Selected=0 and a non-tie vote loss (a correct valid group exists
and has strictly fewer votes than the winning wrong group), i.e. stage == lost_at_vote under the
bundle's protocol (hard0 filter, exact-JSON majority over normalized workflows, tie -> smallest
candidate_id).

Per case: winning wrong workflow (normalized), its votes, total valid votes, winner share, best correct
workflow (the most-voted correct group), its votes and share, vote margin.

Layer 1 (exact winner): for tasks that are vote losses in two (or three) methods of the same model run,
  is the winning wrong workflow identical (exact normalized JSON)?
Layer 2 (error signature): the set of atomic differences between the winner and the best correct
  workflow, expressed with tool names instead of step indices so that consistent renumbering does not
  change the signature:
    tool:missing=<X> / tool:extra=<X>                      (tool multiset difference)
    param:<tool>.<input>=<winner constant>                 (both constants, values differ)
    param_key:<tool>.<input>:missing|extra                 (input field present on one side only)
    dep:<tool>.<input>:expected=<src>,actual=<src>         (generated-reference source differs; a
                                                            source is <tool>.<output> or const)
    order_only                                             (same tools, inputs and data flow; only
                                                            the step order differs)
  Steps are paired by tool name (k-th occurrence with k-th occurrence). Two cases share the same
  signature when their difference sets are equal; `shares_atomic` = at least one common difference.

Usage:
  python3 scripts/analyze_modal_errors.py --bundle mini=... --bundle qwen_b1=... [--bundle qwen_b2=...] --out DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = re.compile(r"^<GENERATED>-(\d+)-<?([^<>]+)>?$")
METHODS = {"BoN": "BoN", "CoT": "CoT-tree", "Five": "MO-Tree"}


def key(plan):
    return json.dumps(plan, sort_keys=True, ensure_ascii=False)


def load_cases(bundle: Path, label: str):
    cands = defaultdict(list)
    for line in (bundle / "per_candidate.jsonl").open():
        c = json.loads(line)
        if c["pool"] in METHODS:
            cands[(c["pool"], int(c["task_id"]))].append(c)
    per_task = {(r["pool"], int(r["task_id"])): r for r in csv.DictReader((bundle / "per_task.csv").open())}
    cases, tops = {}, {}
    for (pool, tid), cs in cands.items():
        pt = per_task[(pool, tid)]
        valid = sorted((c for c in cs if c["hard0"]), key=lambda c: int(c["candidate_id"]))
        if not valid:
            continue
        groups, first, plans, correct = Counter(), {}, {}, {}
        for c in valid:
            k = key(c["plan"])
            groups[k] += 1
            first.setdefault(k, int(c["candidate_id"]))
            plans.setdefault(k, c["plan"])
            correct[k] = c["correct"]
        win = max(groups, key=lambda k: (groups[k], -first[k]))
        tops[(pool, tid)] = {"top_key": win, "top_correct": correct[win], "oracle": pt["oracle"] == "True",
                             "groups": groups, "n_valid": len(valid)}
        if pt["oracle"] != "True" or pt["filter_majority_correct"] == "True":
            continue
        ck = [k for k in groups if correct[k]]
        if not ck or correct[win]:
            continue
        best = max(ck, key=lambda k: (groups[k], -first[k]))
        if groups[best] == groups[win]:
            continue  # tie loss
        n = len(valid)
        cases[(pool, tid)] = {
            "run": label, "task_id": tid, "method": METHODS[pool], "pool": pool,
            "winner_key": win, "winner_plan": plans[win], "winning_wrong_votes": groups[win],
            "total_valid_votes": n, "winner_vote_share": groups[win] / n,
            "best_correct_key": best, "best_correct_plan": plans[best],
            "best_correct_votes": groups[best], "correct_vote_share": groups[best] / n,
            "vote_margin": groups[win] - groups[best],
        }
    return cases, tops


def src_of(value, steps_agent):
    if isinstance(value, str):
        m = GEN.match(value.strip())
        if m:
            s, o = int(m.group(1)), m.group(2).strip()
            return f"{steps_agent.get(s, f'step{s}')}.{o}"
    return None


def signature(win, ref):
    """Atomic differences of `win` relative to the correct workflow `ref`, tool-name based."""
    def index(plan):
        by_agent, agent_of = defaultdict(list), {}
        for st in plan:
            if isinstance(st, dict):
                by_agent[st.get("agent")].append(st)
                agent_of[st.get("step")] = st.get("agent")
        return by_agent, agent_of

    wa, w_agent = index(win)
    ra, r_agent = index(ref)
    diffs = set()
    for a in set(wa) | set(ra):
        nw, nr = len(wa.get(a, [])), len(ra.get(a, []))
        for _ in range(nr - nw):
            diffs.add(f"tool:missing={a}")
        for _ in range(nw - nr):
            diffs.add(f"tool:extra={a}")
        for sw, sr in zip(wa.get(a, []), ra.get(a, [])):
            iw, ir = sw.get("inputs") or {}, sr.get("inputs") or {}
            for f in set(iw) | set(ir):
                if f not in iw:
                    diffs.add(f"param_key:{a}.{f}:missing")
                    continue
                if f not in ir:
                    diffs.add(f"param_key:{a}.{f}:extra")
                    continue
                vw, vr = src_of(iw[f], w_agent), src_of(ir[f], r_agent)
                if vw is None and vr is None:
                    if iw[f] != ir[f]:
                        diffs.add(f"param:{a}.{f}={json.dumps(iw[f], ensure_ascii=False)}")
                elif vw != vr:
                    diffs.add(f"dep:{a}.{f}:expected={vr or 'const'},actual={vw or 'const'}")
    if not diffs:
        diffs.add("order_only")
    return tuple(sorted(diffs))


def category(sig):
    kinds = {d.split(":", 1)[0] for d in sig}
    for k, name in (("tool", "Tool"), ("dep", "Dependency"), ("param", "Parameter"), ("param_key", "Parameter"),
                    ("order_only", "Order-only")):
        if k in kinds:
            return name
    return "Other"


def boot_ci(xs, n_boot=10000, seed=0):
    if not xs:
        return (None, None)
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (100 * ms[int(0.025 * n_boot)], 100 * ms[int(0.975 * n_boot) - 1])


def stats(rs):
    def s(k, f):
        v = [r[k] for r in rs]
        return f(v) if v else None
    return {"n_cases": len(rs),
            "winner_share_mean": s("winner_vote_share", statistics.fmean),
            "winner_share_median": s("winner_vote_share", statistics.median),
            "correct_share_mean": s("correct_vote_share", statistics.fmean),
            "correct_share_median": s("correct_vote_share", statistics.median),
            "margin_mean": s("vote_margin", statistics.fmean),
            "margin_median": s("vote_margin", statistics.median),
            "winning_votes_mean": s("winning_wrong_votes", statistics.fmean),
            "correct_votes_mean": s("best_correct_votes", statistics.fmean)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", action="append", required=True, help="label=path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_cases, agree_rows, cross_rows = [], [], []
    for spec in args.bundle:
        label, path = spec.split("=", 1)
        cases, tops = load_cases(ROOT / path, label)
        for c in cases.values():
            c["signature"] = signature(c["winner_plan"], c["best_correct_plan"])
            c["signature_category"] = category(c["signature"])
            for p, m in METHODS.items():
                if p == c["pool"]:
                    continue
                t = tops.get((p, c["task_id"]))
                c[f"{m}_top_is_same_wrong"] = bool(t and t["top_key"] == c["winner_key"])
                c[f"{m}_same_wrong_share"] = (t["groups"].get(c["winner_key"], 0) / t["n_valid"]) if t else None
                c[f"{m}_top_correct"] = bool(t and t["top_correct"])
                c[f"{m}_oracle"] = bool(t and t["oracle"])
        for p, m in METHODS.items():
            src = [c for c in cases.values() if c["pool"] == p]
            for q, mq in METHODS.items():
                if q == p or not src:
                    continue
                xs = [int(c[f"{mq}_top_is_same_wrong"]) for c in src]
                present = [int((c[f"{mq}_same_wrong_share"] or 0) > 0) for c in src]
                lo, hi = boot_ci(xs)
                cross_rows.append({
                    "run": label, "vote_loss_method": m, "other_method": mq, "n_cases": len(src),
                    "other_top_is_same_wrong": sum(xs), "pct": 100 * sum(xs) / len(xs), "boot95": f"[{lo:.1f}, {hi:.1f}]",
                    "other_top_correct": sum(c[f"{mq}_top_correct"] for c in src),
                    "same_wrong_present_in_other_pool": sum(present),
                    "same_wrong_share_in_other_mean": statistics.fmean(c[f"{mq}_same_wrong_share"] or 0 for c in src),
                })
        by_task = defaultdict(dict)
        for (pool, tid), c in cases.items():
            by_task[tid][pool] = c
        for c in cases.values():
            others = [o for p, o in by_task[c["task_id"]].items() if p != c["pool"]]
            c["n_other_methods_lost"] = len(others)
            c["shared_exact_winner"] = any(o["winner_key"] == c["winner_key"] for o in others)
            c["shared_signature"] = any(o["signature"] == c["signature"] for o in others)
            c["shared_group"] = ("no_overlap" if not others else
                                 "shared_signature" if c["shared_signature"] else "overlap_different_error")
        combos = [(a, b) for a, b in combinations(["BoN", "CoT", "Five"], 2)] + [("BoN", "CoT", "Five")]
        for combo in combos:
            tids = sorted(t for t, d in by_task.items() if all(p in d for p in combo))
            same_w = [int(len({by_task[t][p]["winner_key"] for p in combo}) == 1) for t in tids]
            same_s = [int(len({by_task[t][p]["signature"] for p in combo}) == 1) for t in tids]
            share_a = [int(bool(set.intersection(*(set(by_task[t][p]["signature"]) for p in combo)))) for t in tids]
            same_c = [int(len({by_task[t][p]["signature_category"] for p in combo}) == 1) for t in tids]
            row = {"run": label, "methods": " & ".join(METHODS[p] for p in combo),
                   "n_vote_loss": {METHODS[p]: sum(1 for (q, _) in cases if q == p) for p in combo},
                   "overlap_tasks": len(tids)}
            for name, xs in (("exact_same_winner", same_w), ("same_signature", same_s),
                             ("shares_atomic_difference", share_a), ("same_signature_category", same_c)):
                lo, hi = boot_ci(xs)
                row[name] = sum(xs)
                row[name + "_pct"] = 100 * sum(xs) / len(xs) if xs else None
                row[name + "_boot95"] = f"[{lo:.1f}, {hi:.1f}]" if xs else ""
            row["n_vote_loss"] = json.dumps(row["n_vote_loss"])
            row["overlap_task_ids"] = " ".join(map(str, tids))
            agree_rows.append(row)
        all_cases += list(cases.values())

    strength = []
    for label in dict.fromkeys(c["run"] for c in all_cases):
        rs = [c for c in all_cases if c["run"] == label]
        strength.append({"run": label, "group": "all", **stats(rs)})
        for m in METHODS.values():
            strength.append({"run": label, "group": f"method={m}", **stats([c for c in rs if c["method"] == m])})
        for g in ("shared_signature", "overlap_different_error", "no_overlap"):
            strength.append({"run": label, "group": f"cross_method={g}", **stats([c for c in rs if c["shared_group"] == g])})
        for g in (True, False):
            strength.append({"run": label, "group": f"shared_exact_winner={g}",
                             **stats([c for c in rs if c["shared_exact_winner"] == g])})
    for g in ("shared_signature", "overlap_different_error", "no_overlap"):
        strength.append({"run": "all", "group": f"cross_method={g}", **stats([c for c in all_cases if c["shared_group"] == g])})
    strength.append({"run": "all", "group": "all", **stats(all_cases)})

    case_rows = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else
                      " | ".join(v) if isinstance(v, tuple) else v)
                  for k, v in c.items() if k not in ("winner_key", "best_correct_key")}
                 for c in sorted(all_cases, key=lambda c: (c["run"], c["task_id"], c["pool"]))]
    for c, r in zip(sorted(all_cases, key=lambda c: (c["run"], c["task_id"], c["pool"])), case_rows):
        r["winning_wrong_workflow_normalized"] = r.pop("winner_plan")
        r["best_correct_workflow"] = r.pop("best_correct_plan")

    def wcsv(name, rows):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)

    wcsv("modal_error_cases.csv", case_rows)
    wcsv("modal_error_agreement.csv", agree_rows)
    wcsv("modal_error_cross_pool_top.csv", cross_rows)
    wcsv("modal_error_vote_strength.csv", strength)
    cat = Counter((c["run"], c["method"], c["signature_category"]) for c in all_cases)
    wcsv("modal_error_signature_category.csv",
         [{"run": r, "method": m, "category": k, "n": n} for (r, m, k), n in sorted(cat.items())])
    (out / "modal_error_analysis.json").write_text(json.dumps(
        {"population": {l: len([c for c in all_cases if c["run"] == l]) for l in dict.fromkeys(c["run"] for c in all_cases)},
         "agreement": agree_rows, "cross_pool_top": cross_rows, "vote_strength": strength}, indent=1, ensure_ascii=False))
    for r in cross_rows:
        print(f"{r['run']:8s} lost in {r['vote_loss_method']:9s} -> {r['other_method']:9s} top is same wrong "
              f"{r['other_top_is_same_wrong']}/{r['n_cases']} ({r['pct']:.0f}%) {r['boot95']}; other top correct "
              f"{r['other_top_correct']}; same wrong present {r['same_wrong_present_in_other_pool']}")
    for r in agree_rows:
        print(f"{r['run']:8s} {r['methods']:30s} overlap={r['overlap_tasks']:3d}  exact={r['exact_same_winner']} "
              f"({r['exact_same_winner_pct'] or 0:.0f}%) {r['exact_same_winner_boot95']}  sig={r['same_signature']} "
              f"({r['same_signature_pct'] or 0:.0f}%) {r['same_signature_boot95']}  atomic={r['shares_atomic_difference_pct'] or 0:.0f}%")
    for r in strength:
        if r["n_cases"]:
            print(f"{r['run']:8s} {r['group']:38s} n={r['n_cases']:3d} win={r['winner_share_mean']:.2f}/"
                  f"{r['winner_share_median']:.2f} cor={r['correct_share_mean']:.2f}/{r['correct_share_median']:.2f} "
                  f"margin={r['margin_mean']:.1f}/{r['margin_median']:.1f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
