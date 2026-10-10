"""Best-of-N Chain-of-Thought: N independent CoT samples; majority = Selected, all = Oracle."""

from collections import defaultdict
import json
import traceback

from cot import ChainOfThoughts


class BestOfNCoT(ChainOfThoughts):
    """Independent CoT sampling (not Direct). Returns selected answer; use generate_with_candidates for Oracle."""

    def __init__(self, io, evaluator, use_fewshot, n=6, json_mode=True):
        super().__init__(io, evaluator, use_fewshot, num_chain_of_thought=1, json_mode=json_mode)
        self.n = n

    def _one_completion(self, user_question):
        if self.use_fewshot:
            examples_str = "Examples: "
            for example in self.prompt["examples"]:
                examples_str += "\n" + f"{example['input']}"
                examples_str += "\n" + f"{example['output']}"
        else:
            examples_str = ""

        system_prompt = self.prompt["system_prompt"].format(
            agents_desc=self.tools_desc, examples=examples_str
        )
        user_prompt = self.prompt["user_prompt"].format(task_desc=user_question)
        if self.json_mode:
            from cot import _JSON_MODE_SUFFIX

            user_prompt = user_prompt + _JSON_MODE_SUFFIX

        io_input = system_prompt + "\n" + user_prompt
        outs = self.io.generate(io_input, num_return=1)
        if not outs or not outs[0]:
            return None
        return outs[0].strip()

    def generate_with_candidates(self, user_question):
        raw_completions = []
        answers = []
        for _ in range(self.n):
            comp = self._one_completion(user_question)
            raw_completions.append(comp)
            if comp is None:
                answers.append(None)
                continue
            try:
                ans = self.evaluator.extract_answer_from_model_completion(comp)
                if ans is not None:
                    ans = self._postprocess_structured_plan(ans)
                answers.append(ans)
            except Exception:
                traceback.print_exc()
                answers.append(None)

        # majority over non-None answers
        answer2count = defaultdict(list)
        for i, ans in enumerate(answers):
            if ans is None:
                continue
            matched = None
            for existing in list(answer2count.keys()):
                if self.evaluator.check_answers_equivalence(ans, existing):
                    matched = existing
                    break
            if matched is None:
                answer2count[ans].append(i)
            else:
                answer2count[matched].append(i)

        if not answer2count:
            selected = None
        else:
            selected = max(answer2count.keys(), key=lambda a: len(answer2count[a]))

        candidates = []
        for i, ans in enumerate(answers):
            candidates.append(
                {
                    "candidate_id": i,
                    "model_answer": ans,
                    "raw_completion": raw_completions[i],
                    "is_selected": bool(
                        selected is not None
                        and ans is not None
                        and self.evaluator.check_answers_equivalence(ans, selected)
                    ),
                }
            )
        # mark only first majority member as selected for clarity
        if selected is not None:
            seen = False
            for c in candidates:
                if c["is_selected"]:
                    if seen:
                        c["is_selected"] = False
                    else:
                        seen = True

        return selected, candidates

    def generate(self, user_question):
        selected, _ = self.generate_with_candidates(user_question)
        return selected
