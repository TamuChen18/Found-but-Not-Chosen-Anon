#!/usr/bin/env python3
"""
Export the LLM prompts used in the paper to prompts/*.md, read verbatim from the scripts that send them.

  top3_verifier.md          targeted Top-3 verifier and deployable Top-3 selector (GPT-4o)
  gpt4o_judge.md            GPT-4o judge over at most eight deduplicated workflows
  vote_loss_annotation.md   LLM first-pass labels for the 60-case winner-vs-gold audit
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "prompts"


def constant(script: str, name: str):
    tree = ast.parse((ROOT / "scripts" / script).read_text())
    env = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        try:
            env[node.targets[0].id] = eval(compile(ast.Expression(node.value), script, "eval"), {}, env)
        except Exception:
            continue
        if node.targets[0].id == name:
            return env[name]
    raise KeyError(f"{script}: {name}")


def section(title, body, lang=""):
    return f"## {title}\n\n~~~{lang}\n{body.rstrip()}\n~~~\n"


DOCS = {
    "top3_verifier.md": (
        "Targeted Top-3 verifier / deployable Top-3 selector",
        "Source: `scripts/run_top3_verifier.py` (`SYSTEM`, `build_prompt`); the selector "
        "(`scripts/run_top3_selector.py`) reuses the same prompt. Model `openai/gpt-4o`, temperature 0, "
        "max_tokens 300. The top-3 workflow groups in selector order are shown in a seeded random A/B/C order "
        "with the task and the full specifications of the tools they use; vote counts are not shown.",
        [("System / instruction", constant("run_top3_verifier.py", "SYSTEM"), ""),
         ("Layout of the rest of the prompt",
          "## Task\n{task description}\n\n## Tool specifications\n### {tool}\n{tool spec JSON}\n...\n\n"
          "## Candidates\n### Candidate A\n```json\n{workflow}\n```\n...\n\nReturn JSON only.", "")],
    ),
    "gpt4o_judge.md": (
        "GPT-4o judge",
        "Source: `scripts/run_semantic_judge.py` (`JUDGE_SYSTEM`, `build_judge_prompt`), applied to the final "
        "pools by `scripts/run_locked_judge.py`. Model `openai/gpt-4o`, temperature 0, max_tokens 300, k = 8: "
        "valid workflows are deduplicated, a seeded random subset of 8 is drawn when more exist, and shown in "
        "shuffled order with labels C0, C1, ...",
        [("System / instruction", constant("run_semantic_judge.py", "JUDGE_SYSTEM"), ""),
         ("Layout of the rest of the prompt",
          "## Task\n{task description}\n\n## Available agents (typed I/O)\n{compact tool catalog}\n\n"
          "## Candidates\n### Candidate C0\n```json\n{workflow}\n```\n...\n\nReturn JSON only.", "")],
    ),
    "vote_loss_annotation.md": (
        "Winner-vs-gold audit: LLM first-pass annotation",
        "Source: `scripts/llm_annotate_vote_loss.py` (`SYSTEM`, `user_prompt`, `SCHEMA`). First-pass labels for "
        "the 60-case audit, later reviewed by humans; final per-case labels are in "
        "`artifacts/vote_loss_audit_final/audit_sample_final.csv` (`llm_*` columns keep the first-pass labels).",
        [("System / rubric", constant("llm_annotate_vote_loss.py", "SYSTEM"), ""),
         ("User message layout",
          "## Task description\n{task}\n\n## Agent specifications (tools used in these workflows)\n{specs}\n\n"
          "## Gold workflow\n{gold}\n\n## Winning workflow (wrong under strict EM; {winning_votes} votes)\n{winner}\n\n"
          "## Best correct candidate (exact match to gold; {correct_votes} votes; {valid_candidates} valid "
          "candidates in the pool)\n{best_correct}", ""),
         ("Structured output schema (json_schema, strict)",
          json.dumps(constant("llm_annotate_vote_loss.py", "SCHEMA"), indent=2), "json")],
    ),
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, (title, intro, parts) in DOCS.items():
        text = f"# {title}\n\n{intro}\n\n" + "\n".join(section(t, b, lang) for t, b, lang in parts)
        (OUT / name).write_text(text)
        print(f"wrote prompts/{name}")


if __name__ == "__main__":
    main()
