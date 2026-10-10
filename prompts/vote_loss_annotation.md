# Winner-vs-gold audit: LLM first-pass annotation

Source: `scripts/llm_annotate_vote_loss.py` (`SYSTEM`, `user_prompt`, `SCHEMA`). First-pass labels for the 60-case audit, later reviewed by humans; final per-case labels are in `artifacts/vote_loss_audit_final/audit_sample_final.csv` (`llm_*` columns keep the first-pass labels).

## System / rubric

~~~
You are an expert annotator auditing errors of LLM-generated tool workflows for disaster-response tasks.

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
correct minority for a substantive reason.
~~~

## User message layout

~~~
## Task description
{task}

## Agent specifications (tools used in these workflows)
{specs}

## Gold workflow
{gold}

## Winning workflow (wrong under strict EM; {winning_votes} votes)
{winner}

## Best correct candidate (exact match to gold; {correct_votes} votes; {valid_candidates} valid candidates in the pool)
{best_correct}
~~~

## Structured output schema (json_schema, strict)

~~~json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "first_substantive_difference": {
      "type": "string"
    },
    "changes_execution_semantics": {
      "type": "string",
      "enum": [
        "yes",
        "no",
        "uncertain"
      ]
    },
    "primary_error_type": {
      "type": "string",
      "enum": [
        "Tool",
        "Parameter/Binding",
        "Dependency",
        "Structure/Order",
        "SemanticEquivalent",
        "Other/Ambiguous"
      ]
    },
    "secondary_error_type": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "Tool",
          "Parameter/Binding",
          "Dependency",
          "Structure/Order",
          "SemanticEquivalent",
          "Other/Ambiguous"
        ]
      }
    },
    "semantic_equivalent": {
      "type": "string",
      "enum": [
        "yes",
        "no",
        "uncertain"
      ]
    },
    "vote_loss_note": {
      "type": "string"
    },
    "short_explanation": {
      "type": "string"
    },
    "case_study_candidate": {
      "type": "string",
      "enum": [
        "yes",
        "no"
      ]
    },
    "confidence": {
      "type": "string",
      "enum": [
        "high",
        "medium",
        "low"
      ]
    }
  },
  "required": [
    "first_substantive_difference",
    "changes_execution_semantics",
    "primary_error_type",
    "secondary_error_type",
    "semantic_equivalent",
    "vote_loss_note",
    "short_explanation",
    "case_study_candidate",
    "confidence"
  ]
}
~~~
