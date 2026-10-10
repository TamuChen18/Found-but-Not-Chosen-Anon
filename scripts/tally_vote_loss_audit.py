#!/usr/bin/env python3
"""Tally the annotated winner-vs-gold audit (artifacts/vote_loss_audit/audit_sample.csv).

Checks label validity, then prints / writes: primary error type x model (and x method), the
substantive planning error rate (Tool + Parameter/Binding + Dependency + Structure/Order) and the
possible EM-artifact rate (SemanticEquivalent / total), with Wilson 95% intervals.
"""
import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

LABELS = ["Tool", "Parameter/Binding", "Dependency", "Structure/Order", "SemanticEquivalent", "Other/Ambiguous"]
SUBSTANTIVE = {"Tool", "Parameter/Binding", "Dependency", "Structure/Order"}
SEM = {"yes", "no", "uncertain"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))

    problems = []
    for r in rows:
        p = r["primary_error_type"].strip()
        if p not in LABELS:
            problems.append(f"{r['audit_id']}: primary_error_type '{p}' not in {LABELS}")
        for s in filter(None, (x.strip() for x in r["secondary_error_type"].split(";"))):
            if s not in LABELS:
                problems.append(f"{r['audit_id']}: secondary label '{s}' not in {LABELS}")
        se = r["semantic_equivalent"].strip().lower()
        if se not in SEM:
            problems.append(f"{r['audit_id']}: semantic_equivalent '{se}' not in {sorted(SEM)}")
        elif p == "SemanticEquivalent" and se != "yes":
            problems.append(f"{r['audit_id']}: primary SemanticEquivalent requires semantic_equivalent = yes")
        elif p in SUBSTANTIVE and se == "yes":
            problems.append(f"{r['audit_id']}: substantive primary label but semantic_equivalent = yes")
    if problems:
        print(f"{len(problems)} annotation problems:\n  " + "\n  ".join(problems))
        sys.exit(1)

    models = list(dict.fromkeys(r["model"] for r in rows))
    table = []
    for lab in LABELS:
        row = {"primary_error_type": lab}
        for m in models:
            row[m] = sum(r["model"] == m and r["primary_error_type"] == lab for r in rows)
        row["Total"] = sum(r["primary_error_type"] == lab for r in rows)
        table.append(row)
    n = len(rows)
    sub = sum(r["primary_error_type"] in SUBSTANTIVE for r in rows)
    sem = sum(r["primary_error_type"] == "SemanticEquivalent" for r in rows)
    rates = []
    for scope, rs in [("Total", rows)] + [(m, [r for r in rows if r["model"] == m]) for m in models]:
        k1 = sum(r["primary_error_type"] in SUBSTANTIVE for r in rs)
        k2 = sum(r["primary_error_type"] == "SemanticEquivalent" for r in rs)
        lo1, hi1 = wilson(k1, len(rs))
        lo2, hi2 = wilson(k2, len(rs))
        rates.append({"scope": scope, "n": len(rs),
                      "substantive": k1, "substantive_pct": round(100 * k1 / len(rs), 1),
                      "substantive_wilson95": f"[{lo1:.1f}, {hi1:.1f}]",
                      "possible_em_artifact": k2, "em_artifact_pct": round(100 * k2 / len(rs), 1),
                      "em_artifact_wilson95": f"[{lo2:.1f}, {hi2:.1f}]"})
    by_method = Counter((r["model"], r["method"], r["primary_error_type"]) for r in rows)

    out = Path(args.csv).with_name("tally_primary_by_model.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    with Path(args.csv).with_name("tally_rates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rates[0].keys()))
        w.writeheader()
        w.writerows(rates)
    with Path(args.csv).with_name("tally_primary_by_method.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "primary_error_type", "n"])
        for (m, meth, lab), k in sorted(by_method.items()):
            w.writerow([m, meth, lab, k])

    for row in table:
        print(row)
    for r in rates:
        print(r)
    print(f"substantive {sub}/{n}, possible EM artifact {sem}/{n}; wrote tally_*.csv next to {args.csv}")


if __name__ == "__main__":
    main()
