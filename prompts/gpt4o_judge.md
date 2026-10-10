# GPT-4o judge

Source: `scripts/run_semantic_judge.py` (`JUDGE_SYSTEM`, `build_judge_prompt`), applied to the final pools by `scripts/run_locked_judge.py`. Model `openai/gpt-4o`, temperature 0, max_tokens 300, k = 8: valid workflows are deduplicated, a seeded random subset of 8 is drawn when more exist, and shown in shuffled order with labels C0, C1, ...

## System / instruction

~~~
You are an expert at multi-agent geospatial workflow planning.
Given a natural-language disaster task and a catalog of typed agents (tools),
choose which candidate structured plan is most likely to be the correct workflow.

Rules:
- Prefer plans that use the right agents, wire inputs/outputs correctly via
  <GENERATED>-i-<output_key> references, and match the task requirements.
- Ignore how candidates are labeled; labels are arbitrary.
- Respond with ONLY a JSON object:
  {"chosen_id":"C<k>","confidence":0.0-1.0,"reason":"<one short sentence>"}
~~~

## Layout of the rest of the prompt

~~~
## Task
{task description}

## Available agents (typed I/O)
{compact tool catalog}

## Candidates
### Candidate C0
```json
{workflow}
```
...

Return JSON only.
~~~
