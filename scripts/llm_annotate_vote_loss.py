#!/usr/bin/env python3
"""
LLM annotation of the winner-vs-gold audit (artifacts/vote_loss_audit).

Each case is annotated independently with the taxonomy and rules of ANNOTATION_GUIDE.md. The
annotator sees the task description, gold / winning / best-correct workflows, vote counts, the
strict-EM comparison rule and the I/O specs of the agents involved. It does NOT see the auto_* hints.
Structured JSON output (json_schema); raw responses, usage and cost are cached per case, so
reruns only call the API for missing cases. Labels are written to a separate CSV with an
`annotator` column; the human sheet is not touched.

Usage:
  python3 scripts/llm_annotate_vote_loss.py --model openai/gpt-6-astra-pro \
      --csv artifacts/vote_loss_audit/audit_sample.csv \
      --csv artifacts/vote_loss_audit/all_vote_losses_mini_Five.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
OUT = ROOT / "artifacts" / "vote_loss_audit"
TOOLS = json.loads((ROOT / "src" / "interfaces" / "tools" / "tools_manifest.json").read_text())
LABELS = ["Tool", "Parameter/Binding", "Dependency", "Structure/Order", "SemanticEquivalent", "Other/Ambiguous"]

SYSTEM = """You are an expert annotator auditing errors of LLM-generated tool workflows for disaster-response tasks.

Each workflow is a JSON list of steps. A step has: agent (tool name), step (index), inputs (constant values or
references "<GENERATED>-k-<output_name>" to an output of step k), dependence (list of predecessor steps, [-1] = none),
dependence_content (which outputs are consumed from which predecessor) and outputs.

Evaluation is STRICT EXACT MATCH: steps are compared position by position and every field must be identical.
So a workflow can fail EM for reasons that do not change what it does (e.g. independent steps listed in a
different order with references renumbered consistently).

You receive the gold workflow, the WINNING workflow (wrong under EM, selected by majority vote) and the best
CORRECT candidate (exact match to gold, which received fewer votes). Explain why the winner is not the gold plan.

Answer three questions:
Q1. Where is the FIRST SUBSTANTIVE difference between winner and gold? Pure step renumbering / reordering is not
    substantive unless it changes the data flow. Be concrete (e.g. "s2 uses Urban_Anomaly_Detection instead of
    Anomaly_Detection_Forest", "s3.image_path references step 1 instead of step 2", "missing GeoChat step for the
    urban branch").
Q2. Would this difference change the execution semantics of the workflow under the task requirements?
    yes = genuine planning error; no = possible EM artifact; uncertain.
Q3. Note anything useful about why the correct workflow lost the vote (vote counts are given).

Primary error type (exactly one; the category of the first substantive difference), optional secondary types:
- Tool: wrong tool, a required tool missing, or a tool that should not be there.
- Parameter/Binding: right tool, but a constant parameter, input field, or generated-reference binding is wrong.
- Dependency: predecessor / generated-output reference wrong; the data-flow relation is incorrect.
- Structure/Order: overall step organization, branching or ordering differs substantively from what the task needs.
- SemanticEquivalent: accomplishes substantively the same plan as gold and fails strict EM only because of
  irrelevant ordering, equivalent representation, etc.
- Other/Ambiguous: cannot be clearly assigned; do not force a category.

SemanticEquivalent must be applied VERY strictly. Use it only if you can state explicitly that executing both
workflows would produce the same operational semantics under the task requirements, and the differences affect
no tool, no parameter and no dependency. "Looks similar" is not enough. If unsure, use Other/Ambiguous with
semantic_equivalent = "uncertain".

Consistency rules: primary SemanticEquivalent <=> semantic_equivalent = "yes". A substantive primary label
(Tool, Parameter/Binding, Dependency, Structure/Order) requires semantic_equivalent = "no".
case_study_candidate = "yes" only if the case clearly and compactly illustrates a wrong majority beating a
correct minority for a substantive reason."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "first_substantive_difference": {"type": "string"},
        "changes_execution_semantics": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "primary_error_type": {"type": "string", "enum": LABELS},
        "secondary_error_type": {"type": "array", "items": {"type": "string", "enum": LABELS}},
        "semantic_equivalent": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "vote_loss_note": {"type": "string"},
        "short_explanation": {"type": "string"},
        "case_study_candidate": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["first_substantive_difference", "changes_execution_semantics", "primary_error_type",
                 "secondary_error_type", "semantic_equivalent", "vote_loss_note", "short_explanation",
                 "case_study_candidate", "confidence"],
}


def agents_of(*plans):
    out = []
    for p in plans:
        for s in p:
            a = s.get("agent")
            if a and a not in out:
                out.append(a)
    return out


def user_prompt(r):
    gold, win, cor = (json.loads(r[k]) for k in ("gold_workflow", "winning_workflow", "best_correct_workflow"))
    specs = {a: TOOLS[a] for a in agents_of(gold, win, cor) if a in TOOLS}
    return (
        f"## Task description\n{r['task_description']}\n\n"
        f"## Agent specifications (tools used in these workflows)\n{json.dumps(specs, indent=1, ensure_ascii=False)}\n\n"
        f"## Gold workflow\n{json.dumps(gold, indent=1, ensure_ascii=False)}\n\n"
        f"## Winning workflow (wrong under strict EM; {r['winning_votes']} votes)\n"
        f"{json.dumps(win, indent=1, ensure_ascii=False)}\n\n"
        f"## Best correct candidate (exact match to gold; {r['correct_votes']} votes; "
        f"{r['valid_candidates']} valid candidates in the pool)\n{json.dumps(cor, indent=1, ensure_ascii=False)}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-6-astra-pro")
    ap.add_argument("--reasoning_effort", default="high")
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", default=None, help="cache file (default: artifacts/vote_loss_audit/llm_cache_<model>.jsonl)")
    args = ap.parse_args()

    rows = {}
    for p in args.csv:
        for r in csv.DictReader(open(ROOT / p)):
            rows.setdefault(r["audit_id"], r)
    ids = list(rows)[: args.limit] if args.limit else list(rows)

    tag = args.model.replace("/", "_")
    cache_path = ROOT / args.cache if args.cache else OUT / f"llm_cache_{tag}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.open():
            c = json.loads(line)
            cache[c["audit_id"]] = c

    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OpenRouter_API_KEY"),
                    max_retries=0, timeout=900)

    def call(aid):
        r = rows[aid]
        for attempt in range(1, 6):
            try:
                t0 = time.time()
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(r)}],
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "audit_label", "strict": True, "schema": SCHEMA}},
                    extra_body={"reasoning": {"effort": args.reasoning_effort}, "usage": {"include": True}},
                )
                text = resp.choices[0].message.content
                label = json.loads(text)
                u = resp.usage
                return {"audit_id": aid, "model": args.model, "label": label, "raw": text,
                        "prompt_sha256": hashlib.sha256(user_prompt(r).encode("utf-8")).hexdigest(),
                        "provider": getattr(resp, "provider", None),
                        "prompt_tokens": getattr(u, "prompt_tokens", None),
                        "completion_tokens": getattr(u, "completion_tokens", None),
                        "cost": getattr(u, "cost", None), "elapsed_s": round(time.time() - t0, 1),
                        "attempt": attempt}
            except Exception as e:  # transport / parse errors: retry the request, never edit the output
                err = f"{type(e).__name__}: {e}"
                time.sleep(min(60, 5 * 2 ** attempt))
        return {"audit_id": aid, "model": args.model, "error": err}

    todo = [a for a in ids if a not in cache or "error" in cache[a]]
    print(f"cases {len(ids)}, cached {len(ids) - len(todo)}, to annotate {len(todo)} with {args.model}", flush=True)
    with ThreadPoolExecutor(max_workers=args.threads) as ex, cache_path.open("a") as fc:
        futs = {ex.submit(call, a): a for a in todo}
        for i, f in enumerate(as_completed(futs), 1):
            c = f.result()
            cache[c["audit_id"]] = c
            fc.write(json.dumps(c, ensure_ascii=False) + "\n")
            fc.flush()
            print(f"[{i}/{len(todo)}] {c['audit_id']} "
                  f"{c.get('label', {}).get('primary_error_type', c.get('error'))} cost={c.get('cost')}", flush=True)

    for p in args.csv:
        src = list(csv.DictReader(open(ROOT / p)))
        out_rows = []
        for r in src:
            c = cache.get(r["audit_id"], {})
            lab = c.get("label") or {}
            r = dict(r)
            r.update({
                "primary_error_type": lab.get("primary_error_type", ""),
                "secondary_error_type": ";".join(lab.get("secondary_error_type", [])),
                "semantic_equivalent": lab.get("semantic_equivalent", ""),
                "first_substantive_difference": lab.get("first_substantive_difference", ""),
                "changes_execution_semantics": lab.get("changes_execution_semantics", ""),
                "short_explanation": lab.get("short_explanation", ""),
                "case_study_candidate": lab.get("case_study_candidate", ""),
                "vote_loss_note": lab.get("vote_loss_note", ""),
                "annotator_confidence": lab.get("confidence", ""),
                "annotator": args.model if lab else "",
            })
            out_rows.append(r)
        dst = (ROOT / p).with_name(Path(p).stem + f"_llm_{tag}.csv")
        with dst.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print("wrote", dst)
    done = [cache[a] for a in ids if a in cache and "label" in cache[a]]
    print(f"annotated {len(done)}/{len(ids)}; total cost ${sum(c.get('cost') or 0 for c in done):.2f}")


if __name__ == "__main__":
    main()
