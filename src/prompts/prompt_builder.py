from typing import Dict
from utils.common_utils import read_yaml, read_json
import os

class PromptBuilder:
    def __init__(self, prompts_dir: str, tools_dir: str, use_fewshot: bool):

        prompts_dir = os.path.join(prompts_dir, "prompts.yaml")
        tools_dir = os.path.join(tools_dir, "tools_manifest.json")

        self.tools_list = read_json(tools_dir)
        prompts = read_yaml(prompts_dir)

        self.direct_answer_prompt = prompts["DIRECT_ANSWER"]
        self.chain_of_thought_prompt = prompts["CHAIN_OF_THOUGHT"]
        self.self_refine_prompt = prompts["SELF_REFINE"]
        self.divide_and_conquer_prompt = prompts["DIVIDE_AND_CONQUER"]
        self.rephrase_prompt = prompts["REPHRASE"]
        self.check_confidence_prompt = prompts["CHECK_CONFIDENCE"]

        self.use_fewshot = use_fewshot

    def process_fewshot_examples(self, examples, indexed=False):
        if not self.use_fewshot:
            return ""        
        examples_str = "Examples: "
        if indexed:
            for i, example in enumerate(examples):
                examples_str += "\n" + f"Instruction {i+1}: {example['input']}"
                examples_str += "\n" + f"Response {i+1}: {example['output']}"
        else:
            for example in examples:
                examples_str += "\n" + f"{example['input']}"
                examples_str += "\n" + f"{example['output']}"
        return examples_str
    
    def process_compare_examples(self, examples): 
        if not self.use_fewshot:
            return ""
           
        examples_str = "Examples: "
        for example in examples:
            examples_str += f"Question: {example['question']}"
            examples_str += "\n\n" + f"Output 1: {example['output1']}"
            examples_str += "\n\n" + f"Output 2: {example['output2']}"
            examples_str += "\n\n" + f"Answer: {example}"
        return examples_str

    def process_self_refine_examples(self, examples):
        if not self.use_fewshot:
            return ""
        example_str = "Examples: "
        for example in examples:
            example_str += "\n" + example["input"]
        return example_str + "\n"

    def build_direct_answer_prompt(self, user_question: str, existing_direct_answer: str, next_answer_id: int) -> str:

        examples_str = self.process_fewshot_examples(self.direct_answer_prompt["examples"], indexed=True).strip()
        
        system_prompt = self.direct_answer_prompt["system_prompt"].format(agents_desc = self.tools_list, examples=examples_str)
        user_prompt = self.direct_answer_prompt["user_prompt"].format(  question_index=3,
                                                                        task_desc=user_question, 
                                                                        existing_direct_answer=existing_direct_answer, 
                                                                        next_step=next_answer_id
                                                                      ).strip()

        return system_prompt + "\n" + user_prompt
              
    def build_chain_of_thought_prompt(self, user_question: str, paraphrased: bool, hint:str) -> str:
        
        examples_str = self.process_fewshot_examples(self.chain_of_thought_prompt["examples"], indexed=True)

        system_prompt = self.chain_of_thought_prompt["system_prompt"].format(agents_desc = self.tools_list, examples = examples_str)
        user_question += "\n\n" + hint if hint is not None else ""
        user_prompt = self.chain_of_thought_prompt["user_prompt"].format(task_desc = user_question)

        return system_prompt + "\n" + user_prompt
    
    def build_check_confidence_prompt(self, user_question, output1, output2):
        
        examples_str = self.process_compare_examples(self.check_confidence_prompt["examples"])
        system_prompt = self.check_confidence_prompt["system_prompt"].format(agents_desc = self.tools_list, examples = examples_str)
        user_prompt = self.check_confidence_prompt["user_prompt"].format(Question=user_question, 
                                                                         Output1 = output1, 
                                                                         Output2=output2
                                                                         )
        
        return system_prompt + "\n" + user_prompt

    def build_divide_and_conquer_prompt(self, user_question, existing_subquestions_and_subanswers, question_index, next_subquestion_id, subquestion=None) -> str:
        
        examples_str = self.process_fewshot_examples(self.divide_and_conquer_prompt["examples"])
        system_prompt = self.divide_and_conquer_prompt["system_prompt"].format(agents_desc = self.tools_list, examples = examples_str.strip()).strip()
        user_prompt = self.divide_and_conquer_prompt["user_prompt"].format(question_index = question_index,
                                                                           user_question = user_question.strip(), 
                                                                           existing_subquestions_and_subanswers = existing_subquestions_and_subanswers.strip()
                                                                           ).strip()
        if not subquestion:
            user_prompt += "\n" + f"Question {question_index}.{next_subquestion_id}: "
        else:
            user_prompt += "\n" + f"Question {question_index}.{next_subquestion_id}: " + subquestion
            user_prompt += "\n" + f"Answer {question_index}.{next_subquestion_id}: "

        return system_prompt + "\n\n" + user_prompt

    def build_rephrase_prompt(self, user_question) -> str:
        
        examples_str = self.process_fewshot_examples(self.rephrase_prompt["examples"])
        system_prompt = self.rephrase_prompt["system_prompt"].format(examples=examples_str).strip()
        user_prompt = self.rephrase_prompt["user_prompt"].format(task_desc=user_question).strip()
        return system_prompt + "\n" + user_prompt
    
    def build_self_refine_prompt(self, user_question: str, existing_steps: str, type: str = "reflect", feedback: str = "") -> str:
        
        examples_str = self.process_self_refine_examples(self.self_refine_prompt["examples"])
        system_prompt = self.self_refine_prompt["system_prompt"].format(examples = examples_str)
        
        if type == "reflect":
            user_prompt = self.self_refine_prompt["user_prompt_reflect"].format(agents_desc= self.tools_list,
                                                                                user_question=user_question, 
                                                                                existing_steps=existing_steps)
        elif type == "refine": 
            user_prompt = self.self_refine_prompt["user_prompt_refine"].format(user_question=user_question, 
                                                                               existing_steps=existing_steps, 
                                                                               feedback=feedback)
        else:
            user_prompt = ""

        return system_prompt + "\n" + user_prompt