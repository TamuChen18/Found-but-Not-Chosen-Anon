# Portions of this file are adapted from rStar (https://github.com/zhentingqi/rStar),
# Copyright (c) 2024 rStar Team, and from ThoughtICR (https://github.com/jinyangwu/ThoughtICR),
# Copyright (c) 2025 Jinyang Wu. Both are licensed under the MIT License; see THIRD_PARTY_NOTICES.md,
# licenses/rStar-LICENSE and licenses/ThoughtICR-LICENSE.
import re
from typing import Dict, List, Tuple

from interfaces.IO_Interface import IO_Interface
from utils.agent_utils import concat_all_parent_steps, concat_direct_answers, concat_subquestions_and_subanswers, concat_subquestions_and_subanswers_as_da, reach_terminal_subquestion
from prompts.prompt_builder import PromptBuilder
from evaluators.evaluators import Evaluator
from interfaces.call_context import call_context

class MCTS_Generator:
    """Generator generates children nodes"""

    def __init__(self, args, model_name, tokenizer_name) -> None:
        generation_kwargs = {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        if args.api == "huggingface":
            generation_kwargs["top_k"] = args.top_k
        if getattr(args, "seed", None) is not None:
            generation_kwargs["seed"] = int(args.seed)

        self.io = IO_Interface(args.api, model_name, tokenizer_name, generation_kwargs)
        self.evaluator = Evaluator()

        self.num_divide_and_conquer = args.num_divide_and_conquer
        self.num_direct_answer = args.num_direct_answer  
        self.num_divide_and_conquer_votes = args.num_divide_and_conquer_votes       
        self.num_chain_of_thought = args.num_chain_of_thought

        # CoT-only / CoT+SR judgment controls: keep every CoT sample as its own
        # node via independent API calls (OpenRouter often ignores n>1).
        cot_only = (
            getattr(args, "disable_rephrase", False)
            and getattr(args, "disable_direct_answer", False)
            and getattr(args, "disable_divide_and_conquer", False)
            and getattr(args, "disable_self_refine", False)
            and not getattr(args, "disable_chain_of_thought", False)
        )
        # CoT+SR ablation: only COT+SR enabled (no RE/DA/DC)
        self.cot_sr_mode = (
            getattr(args, "disable_rephrase", False)
            and getattr(args, "disable_direct_answer", False)
            and getattr(args, "disable_divide_and_conquer", False)
            and not getattr(args, "disable_self_refine", False)
            and not getattr(args, "disable_chain_of_thought", False)
        )
        self.cot_expand_all_samples = bool(
            getattr(args, "cot_expand_all_samples", False) or cot_only or self.cot_sr_mode
        )
        
        self.promptbuilder = PromptBuilder(args.prompts_dir, args.tools_dir, args.use_fewshot)
        self.question_index = 3

    def generate_direct_answer(self, user_question: str, solution_trace: Dict[int, Dict[str, str]], paraphrased: bool, parent_is_subquestion: bool, parent_is_self_refine=False):
        direct_answer_list = []
        # For Self Refine
        if parent_is_self_refine:
            existing_direct_answer = solution_trace[0]['answers'][-1][1] + "\n"
            matches = re.findall(r'step \d+:', existing_direct_answer)
            next_answer_id = len(matches) + 1
        else:
            # For Divide and Conquer
            if parent_is_subquestion:
                existing_direct_answer, next_answer_id = concat_subquestions_and_subanswers_as_da(solution_trace)
            else:
                # For Root, Direct Answer, Rephrase and COT
                existing_direct_answer, next_answer_id = concat_direct_answers(solution_trace)
        
        io_input = self.promptbuilder.build_direct_answer_prompt(user_question, existing_direct_answer, next_answer_id)
        with call_context(operator="DIRECT_ANSWER"):
            io_output_list = self.io.generate(
                model_input=io_input,
                num_return=self.num_direct_answer,
                stop = [f"step {next_answer_id + 1}:"]
            )
        
        for x in io_output_list:
            if not isinstance(x, str):
                continue
            xs = x.strip()
            if xs and xs not in direct_answer_list:
                direct_answer_list.append(xs)

        # TODO: new added
        if len(direct_answer_list) == 0:
            first = io_output_list[0] if io_output_list else ""
            if isinstance(first, str):
                direct_answer_list = [first.strip()]
            else:
                direct_answer_list = [str(first)]

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  
        potential_answers_list = [None] * len(direct_answer_list)

        return direct_answer_list, potential_answers_list

    def _get_most_likely_answer(self, user_question, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1.0
        else:  
            _, most_confident_answer_full_completion, _, confidence = self.evaluator.find_most_confident_answer(
                user_question, 
                io_output_list, 
                self.promptbuilder,
                self.io
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence

    def _fewshot_cot_answer_question(self, question: str, paraphrased: bool, num_return: int, hint: str = None):  
        io_input = self.promptbuilder.build_chain_of_thought_prompt(question, paraphrased, hint)
        with call_context(operator="CHAIN_OF_THOUGHT"):
            io_output_list = self.io.generate(
                io_input,
                num_return=num_return
            )
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  
        return io_input, cleaned_io_output_list

    def generate_chain_of_thought(self, user_question: str, paraphrased: bool, hint: str):
        chain_of_thought_list, value_list = [], []
        num_return = max(1, int(self.num_chain_of_thought))

        if self.cot_expand_all_samples:
            # Independent CoT leaves (1 API call each) — matches BoN-CoT budgeting.
            for _ in range(num_return):
                _, outs = self._fewshot_cot_answer_question(
                    question=user_question,
                    paraphrased=paraphrased,
                    num_return=1,
                    hint=hint,
                )
                for o in outs:
                    if not o:
                        continue
                    chain_of_thought_list.append(o)
                    value_list.append(1.0)
            return chain_of_thought_list, value_list

        _, io_output_list = self._fewshot_cot_answer_question(
            question=user_question,
            paraphrased=paraphrased,
            num_return=num_return,
            hint=hint,
        )
        if not io_output_list:
            return [], []

        most_likely_answer, likelihood = self._get_most_likely_answer(
            user_question, io_output_list
        )
        chain_of_thought_list.append(most_likely_answer)
        value_list.append(likelihood)
        return chain_of_thought_list, value_list

    def generate_subquestions(self, user_question, solution_trace, paraphrased):
        subquestion_list, subanswer_list, value_list = [], [], []

        # concatenates only all the subquestions and subanswers in the prompt format
        existing_subquestions_and_subanswers, next_subquestion_id = concat_subquestions_and_subanswers(
            solution_trace, self.question_index
        )

        io_input = self.promptbuilder.build_divide_and_conquer_prompt(user_question, existing_subquestions_and_subanswers, self.question_index, next_subquestion_id)
        
        with call_context(operator="DC_SUBQUESTION"):
            io_output_list = self.io.generate(
                io_input,
                num_return=self.num_divide_and_conquer,
                stop=[f"Answer {self.question_index}.{next_subquestion_id}",],
            )

        subquestion_list = []
        for o in io_output_list:
            if o is None or not isinstance(o, str):
                continue
            subquestion_list.append(
                o.split(f"Question {self.question_index}.{next_subquestion_id}:")[-1].strip()
            )

        #! generate subanswers to the subquestions generated above
        io_input_list = []
        for subquestion in subquestion_list:
            io_input = self.promptbuilder.build_divide_and_conquer_prompt(user_question, existing_subquestions_and_subanswers, self.question_index, next_subquestion_id, subquestion)
            io_input_list.append(io_input)

        if not subquestion_list:
            return [], [], [], []

        # use last subquestion only to choose num_return (legacy behavior)
        if reach_terminal_subquestion(subquestion=subquestion_list[-1], user_question=user_question):
            num_return = self.num_chain_of_thought
        else:
            num_return = self.num_divide_and_conquer_votes
        
        with call_context(operator="DC_SUBANSWER"):
            io_output_list = self.io.generate(
                io_input_list,
                num_return=num_return,
                stop=[f"Question {self.question_index}.{next_subquestion_id + 1}"],
            )

        # io.generate(list) returns one result per prompt: str (n=1) or List[str] (n>1)
        grouped: List[List[str]] = []
        for item in io_output_list:
            if isinstance(item, list):
                grouped.append([x for x in item if isinstance(x, str)])
            elif isinstance(item, str):
                grouped.append([item])
            else:
                grouped.append([str(item)])

        cleaned_io_output_list = [
            [
                io_output.split(f"Answer {self.question_index}.{next_subquestion_id}:")[-1].strip()
                for io_output in io_output_group
            ]
            for io_output_group in grouped
        ]

        for cleaned_io_output_group in cleaned_io_output_list:
            try:
                most_likely_answer, likelihood = self._get_most_likely_answer(
                    user_question, cleaned_io_output_group
                )
            except Exception as e:
                print(f"Error in DC subanswer aggregation: {e}")
                most_likely_answer = cleaned_io_output_group[0] if cleaned_io_output_group else ""
                likelihood = 0.0

            subanswer_list.append(most_likely_answer)
            value_list.append(likelihood)

        assert len(subquestion_list) == len(subanswer_list) == len(value_list)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        potential_answers_list = [None] * len(subquestion_list)

        return subquestion_list, subanswer_list, value_list, potential_answers_list
    def generate_rephrased_user_question(self, user_question: str):
        rephrased_user_question_list = []
        io_input = self.promptbuilder.build_rephrase_prompt(user_question)
        with call_context(operator="REPHRASE"):
            io_output = self.io.generate(model_input=io_input, num_return=1)[0]

        io_output = io_output.split("?")[0] + "?"
        io_output = "Given a list of conditions, please answer the question. Condition 1: " + io_output
        rephrased_user_question_list.append(io_output)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  
        potential_answers_list = [None] * len(rephrased_user_question_list)

        return rephrased_user_question_list, potential_answers_list
    
    def generate_self_refine(self, user_question, solution_trace):
        self_reflection_and_refine_list = []
        existing_steps, next_step = concat_all_parent_steps(solution_trace)
        
        io_input = self.promptbuilder.build_self_refine_prompt(user_question=user_question, existing_steps=existing_steps, type="reflect")
        
        with call_context(operator="SELF_REFINE_REFLECT"):
            io_output_list = self.io.generate(
                model_input=io_input, num_return=3, stop=["<REFINE>", "</REFLECT>", f"step {next_step}"] 
            )

        io_output_reflect = []
        for io_output in io_output_list:
            feedback = io_output.strip().split("<REFLECT>")[-1].strip()
            if "All the above steps are correct" in feedback:
                io_output_reflect.append(existing_steps.strip())
            else:
                io_input = self.promptbuilder.build_self_refine_prompt(user_question=user_question, existing_steps = existing_steps, type="refine", feedback=feedback)
                with call_context(operator="SELF_REFINE_REFINE"):
                    io_output = self.io.generate(model_input=io_input, num_return=1, stop=["</REFINE>", f"step {next_step}"])[0].strip()
                io_output_reflect.append(io_output.split("<REFINE>")[-1].strip())
        
        if io_output_reflect:
            outputs = io_output_reflect
        else:
            outputs = [existing_steps.strip()]
        
        potential_answers_list = [None] * len(outputs)
        return outputs, potential_answers_list

    def get_usage(self) -> dict:
        return self.io.get_usage()

    def close(self):
        """
        Gracefully release GPU and model resources used by MCTS_Generator.
        """
        try:
            if hasattr(self.io, "close"):
                self.io.close()
        except Exception as e:
            print(f"[Warning] Failed to close IO_Interface: {e}")

        print("[MCTS_Generator] Cleanup completed successfully.")
