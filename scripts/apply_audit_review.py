#!/usr/bin/env python3
"""
Merge LLM audit labels with review overrides into the final audit sheet.

Every row keeps its LLM labels in llm_* columns; the final label columns come from
review_overrides.csv when the case was reviewed, otherwise from the LLM. label_source records which.

Usage:
  python3 scripts/apply_audit_review.py --llm_csv artifacts/vote_loss_audit/audit_sample_llm_openai_gpt-6-astra-pro.csv \
      --out artifacts/vote_loss_audit/audit_sample_final.csv
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["primary_error_type", "secondary_error_type", "semantic_equivalent", "changes_execution_semantics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_csv", required=True)
    ap.add_argument("--overrides", default="artifacts/vote_loss_audit/review_overrides.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ov = {r["audit_id"]: r for r in csv.DictReader(open(ROOT / args.overrides))}
    rows = list(csv.DictReader(open(ROOT / args.llm_csv)))
    out = []
    for r in rows:
        r = dict(r)
        for k in FIELDS:
            r["llm_" + k] = r[k]
        o = ov.get(r["audit_id"])
        if o:
            for k in FIELDS:
                r[k] = o[k]
            r["label_source"] = f"llm+review:{o['decision']}"
            r["reviewer"] = o["reviewer"]
            r["review_note"] = o["review_note"]
        else:
            r["label_source"] = "llm"
            r["reviewer"] = ""
            r["review_note"] = ""
        out.append(r)
    with open(ROOT / args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    n = sum(r["label_source"] != "llm" for r in out)
    print(f"wrote {args.out}: {len(out)} rows, {n} reviewed")


if __name__ == "__main__":
    main()
