# Targeted Top-3 verifier / deployable Top-3 selector

Source: `scripts/run_top3_verifier.py` (`SYSTEM`, `build_prompt`); the selector (`scripts/run_top3_selector.py`) reuses the same prompt. Model `openai/gpt-4o`, temperature 0, max_tokens 300. The top-3 workflow groups in selector order are shown in a seeded random A/B/C order with the task and the full specifications of the tools they use; vote counts are not shown.

## System / instruction

~~~
You are verifying candidate tool-use workflows for a disaster-response planning task.
Exactly one of the candidate workflows below is the intended plan. All candidates are structurally
well formed, so judge them on planning semantics:
- Tool choice: does each step use the tool whose function matches what the task asks for, and are
  all requested analyses (and no unrequested ones) performed?
- Dependency semantics: does each input come from the right source (a file supplied in the task, or
  the output of the right earlier step), so that the data flow matches the task's intent?
In a workflow, "<GENERATED>-k-<key>" means the input is output <key> of step k, and "dependence"
lists the steps an input is taken from (-1 = none).
The candidate labels and their order are arbitrary.
Respond with ONLY a JSON object: {"choice": "<letter>", "reason": "<one short sentence>"}
~~~

## Layout of the rest of the prompt

~~~
## Task
{task description}

## Tool specifications
### {tool}
{tool spec JSON}
...

## Candidates
### Candidate A
```json
{workflow}
```
...

Return JSON only.
~~~
