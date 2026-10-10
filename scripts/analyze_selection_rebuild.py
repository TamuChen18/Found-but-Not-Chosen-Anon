#!/usr/bin/env python3
"""
Selection-failure analysis on the final primary pools (E4 off, E2b off, hard0 filter, exact-JSON
majority; BoN occurrence voting, trees unique-text voting; tie -> smallest candidate_id). No LLM calls.

Primary pools:
  mini     artifacts/mini_noseed_trees_full/eval_e2b_off   (BoN / CoT-tree / MO-Tree, no request seed)
  qwen_b1  artifacts/e2b_off_primary/qwen_b1
  qwen_b2  artifacts/e2b_off_primary/qwen_b2
Reference only (different protocol: request seed sent): mini_seeded = artifacts/e2b_off_primary/mini

1. stage loss per pool (oracle, selected, filter / vote / tie loss)
2. per vote-loss case: wrong-winner share, strongest-correct share, margin (modal_error_cases.csv)
3. within-run method pairs: exact same wrong winner, same error signature (modal_error_agreement.csv)
4. cross-run, same method: qwen_b1 vs qwen_b2 (and mini_seeded vs mini as a reference):
   - both runs lose by vote on the task: exact same wrong winner / same signature
   - lost in run A: is run B's top workflow the same wrong one; is it present in B; is B's top correct

Outputs: artifacts/selection_rebuild/
"""
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_modal_errors import METHODS, boot_ci, category, load_cases, signature  # noqa: E402

OUT = ROOT / "artifacts/selection_rebuild"
PRIMARY = {"mini": "artifacts/mini_noseed_trees_full/eval_e2b_off",
           "qwen_b1": "artifacts/e2b_off_primary/qwen_b1",
           "qwen_b2": "artifacts/e2b_off_primary/qwen_b2"}
REFERENCE = {"mini_seeded": "artifacts/e2b_off_primary/mini"}
CROSS = [("qwen_b1", "qwen_b2", "primary"), ("mini_seeded", "mini", "reference: seeded vs no-seed")]


def wcsv(p, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def stage_row(label, p, m, pt, cands, cases):
    """Oracle-positive, not selected: vote loss if a valid correct workflow is strictly out-voted
    (load_cases), filter loss if no correct candidate survives hard0, otherwise tie loss."""
    rs = [r for r in pt if r["pool"] == p]
    n_filter = n_tie = n_vote = 0
    for r in rs:
        if r["oracle"] != "True" or r["filter_majority_correct"] == "True":
            continue
        k = (p, int(r["task_id"]))
        if k in cases:
            n_vote += 1
        elif not any(c["correct"] and c["hard0"] for c in cands.get(k, [])):
            n_filter += 1
        else:
            n_tie += 1
    cs = [c for c in cases.values() if c["pool"] == p]
    return {
        "run": label, "method": m, "n_tasks": len(rs),
        "oracle": sum(r["oracle"] == "True" for r in rs),
        "selected": sum(r["filter_majority_correct"] == "True" for r in rs),
        "filter_loss": n_filter, "vote_loss": n_vote, "tie_loss": n_tie,
        "winner_share_mean": statistics.fmean(c["winner_vote_share"] for c in cs) if cs else None,
        "winner_share_median": statistics.median(c["winner_vote_share"] for c in cs) if cs else None,
        "correct_share_mean": statistics.fmean(c["correct_vote_share"] for c in cs) if cs else None,
        "correct_share_median": statistics.median(c["correct_vote_share"] for c in cs) if cs else None,
        "margin_votes_mean": statistics.fmean(c["vote_margin"] for c in cs) if cs else None,
        "margin_votes_median": statistics.median(c["vote_margin"] for c in cs) if cs else None,
        "margin_share_mean": statistics.fmean(c["vote_margin"] / c["total_valid_votes"] for c in cs) if cs else None,
        "winner_share_ge_0.5": sum(c["winner_vote_share"] >= 0.5 for c in cs),
        "valid_votes_mean": statistics.fmean(c["total_valid_votes"] for c in cs) if cs else None,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, bundles in (("primary", PRIMARY), ("reference_mini_seeded", {**REFERENCE})):
        cmd = [sys.executable, str(ROOT / "scripts/analyze_modal_errors.py"), "--out", str(OUT / name)]
        for k, v in bundles.items():
            cmd += ["--bundle", f"{k}={v}"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    loaded = {}
    for label, b in {**PRIMARY, **REFERENCE}.items():
        cases, tops = load_cases(ROOT / b, label)
        for c in cases.values():
            c["signature"] = signature(c["winner_plan"], c["best_correct_plan"])
            c["signature_category"] = category(c["signature"])
        loaded[label] = (cases, tops)

    stage = []
    for label, b in {**PRIMARY, **REFERENCE}.items():
        cases, tops = loaded[label]
        pt = list(csv.DictReader(open(ROOT / b / "per_task.csv")))
        cands = {}
        for l in open(ROOT / b / "per_candidate.jsonl"):
            c = json.loads(l)
            cands.setdefault((c["pool"], int(c["task_id"])), []).append(c)
        for p, m in METHODS.items():
            stage.append(stage_row(label, p, m, pt, cands, cases))

    cross, cross_cases = [], []
    for a, b, kind in CROSS:
        (ca, ta), (cb, tb) = loaded[a], loaded[b]
        for p, m in METHODS.items():
            A = {t: c for (q, t), c in ca.items() if q == p}
            B = {t: c for (q, t), c in cb.items() if q == p}
            both = sorted(set(A) & set(B))
            same_w = [int(A[t]["winner_key"] == B[t]["winner_key"]) for t in both]
            same_s = [int(A[t]["signature"] == B[t]["signature"]) for t in both]
            same_cat = [int(A[t]["signature_category"] == B[t]["signature_category"]) for t in both]
            row = {"comparison": f"{a} vs {b}", "kind": kind, "method": m,
                   "vote_loss_a": len(A), "vote_loss_b": len(B), "both_vote_loss": len(both),
                   "jaccard_vote_loss_tasks": len(both) / len(set(A) | set(B)) if A or B else None}
            for nm, xs in (("exact_same_winner", same_w), ("same_signature", same_s), ("same_category", same_cat)):
                lo, hi = boot_ci(xs)
                row[nm] = sum(xs)
                row[nm + "_pct"] = 100 * sum(xs) / len(xs) if xs else None
                row[nm + "_boot95"] = f"[{lo:.1f}, {hi:.1f}]" if xs else ""
            for src, other, tops_o, tag in ((A, b, tb, "a_to_b"), (B, a, ta, "b_to_a")):
                top_same = present = top_corr = other_oracle = 0
                shares = []
                for t, c in src.items():
                    o = tops_o.get((p, t))
                    if o is None:
                        continue
                    top_same += o["top_key"] == c["winner_key"]
                    present += o["groups"].get(c["winner_key"], 0) > 0
                    top_corr += bool(o["top_correct"])
                    other_oracle += bool(o["oracle"])
                    shares.append(o["groups"].get(c["winner_key"], 0) / o["n_valid"])
                n = len(src)
                xs = [1] * top_same + [0] * (n - top_same)
                lo, hi = boot_ci(xs)
                row[f"{tag}_other_top_is_same_wrong"] = top_same
                row[f"{tag}_other_top_is_same_wrong_pct"] = 100 * top_same / n if n else None
                row[f"{tag}_boot95"] = f"[{lo:.1f}, {hi:.1f}]" if n else ""
                row[f"{tag}_same_wrong_present"] = present
                row[f"{tag}_other_top_correct"] = top_corr
                row[f"{tag}_same_wrong_share_in_other_mean"] = statistics.fmean(shares) if shares else None
            cross.append(row)
            for t in both:
                cross_cases.append({"comparison": f"{a} vs {b}", "method": m, "task_id": t,
                                    "same_winner": A[t]["winner_key"] == B[t]["winner_key"],
                                    "same_signature": A[t]["signature"] == B[t]["signature"],
                                    "share_a": A[t]["winner_vote_share"], "share_b": B[t]["winner_vote_share"],
                                    "category_a": A[t]["signature_category"], "category_b": B[t]["signature_category"],
                                    "signature_a": " | ".join(A[t]["signature"]),
                                    "signature_b": " | ".join(B[t]["signature"])})

    t22 = {}
    for label in loaded:
        cases, tops = loaded[label]
        res = {}
        for p, m in METHODS.items():
            t = tops.get((p, 22))
            if not t:
                continue
            res[m] = {"n_valid": t["n_valid"], "n_groups": len(t["groups"]), "top_votes": max(t["groups"].values()),
                      "top_is_correct": bool(t["top_correct"]),
                      "landslide_miswired_votes": sum(v for k, v in t["groups"].items()
                                                      if '"<GENERATED>-1-<segmentation_map_image_path>"' in k
                                                      and "Landslide_Segmentation" in k),
                      "stage": "vote_loss" if (p, 22) in cases else ("selected" if t["top_correct"] else "other")}
        t22[label] = res

    wcsv(OUT / "stage_loss.csv", stage)
    wcsv(OUT / "cross_run_recurrence.csv", cross)
    wcsv(OUT / "cross_run_cases.csv", cross_cases)
    (OUT / "task22.json").write_text(json.dumps(t22, indent=1))
    (OUT / "selection_rebuild.json").write_text(json.dumps(
        {"pools": {**PRIMARY, **REFERENCE}, "stage_loss": stage, "cross_run": cross, "task22": t22}, indent=1))
    for r in stage:
        print(f"{r['run']:12s} {r['method']:9s} O={r['oracle']} S={r['selected']} filt={r['filter_loss']} "
              f"vote={r['vote_loss']} tie={r['tie_loss']} win={r['winner_share_mean'] or 0:.2f}/{r['winner_share_median'] or 0:.2f} "
              f"cor={r['correct_share_mean'] or 0:.2f} margin={r['margin_votes_mean'] or 0:.1f} "
              f"(share {r['margin_share_mean'] or 0:.2f}) ge.5={r['winner_share_ge_0.5']} valid={r['valid_votes_mean'] or 0:.1f}")
    for r in cross:
        print(f"{r['comparison']:24s} {r['method']:9s} VL {r['vote_loss_a']}/{r['vote_loss_b']} both={r['both_vote_loss']} "
              f"exact={r['exact_same_winner']} ({r['exact_same_winner_pct'] or 0:.0f}%) {r['exact_same_winner_boot95']} "
              f"sig={r['same_signature']}; A->B top same {r['a_to_b_other_top_is_same_wrong']}/{r['vote_loss_a']} "
              f"present {r['a_to_b_same_wrong_present']} topcorr {r['a_to_b_other_top_correct']}; "
              f"B->A top same {r['b_to_a_other_top_is_same_wrong']}/{r['vote_loss_b']} topcorr {r['b_to_a_other_top_correct']}")
    print(json.dumps(t22))


if __name__ == "__main__":
    main()
