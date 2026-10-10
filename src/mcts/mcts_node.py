# Portions of this file are adapted from rStar (https://github.com/zhentingqi/rStar),
# Copyright (c) 2024 rStar Team, and from ThoughtICR (https://github.com/jinyangwu/ThoughtICR),
# Copyright (c) 2025 Jinyang Wu. Both are licensed under the MIT License; see THIRD_PARTY_NOTICES.md,
# licenses/rStar-LICENSE and licenses/ThoughtICR-LICENSE.
from copy import deepcopy
from typing import List
import traceback

from mcts.mcts_base import MCTS_Node
from mcts.mcts_generator import MCTS_Generator
import numpy as np
from utils.agent_utils import Node_Type, reach_terminal_direct_answer, reach_terminal_subquestion

from utils.common_utils import verbose_print
from utils.agent_utils import make_hint

class Agent_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Agent_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        node_value: float = None,
        generator: MCTS_Generator = None,
        user_question: str = None,
        max_depth_allowed: int = None,
        disable_rephrase: bool = None,
        disable_direct_answer: bool = None,
        disable_chain_of_thought: bool = None,
        disable_divide_and_conquer: bool = None,
        disable_self_refine: bool = None,
        rephrase: str = None,
        chain_of_thought: str = None,
        subquestion: str = None,
        subanswer: str = None,
        is_new_subquestion: bool = None,
        self_refine: str = None,
        direct_answer: str = None,
        if_use_cards: bool = False,
        reasoning_path: list = []
    ) -> None:
        super().__init__()

        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None   
            if node_value is not None:
                assert node_value > 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:   
                assert depth == 0 
                assert all(   
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        rephrase,
                        chain_of_thought,
                        subquestion,
                        subanswer,
                        self_refine,
                        is_new_subquestion,
                        direct_answer,
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [generator, disable_rephrase, user_question, max_depth_allowed, disable_direct_answer]
                )
            elif node_type is Node_Type.REPHRASE:      
                assert depth == 1
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_rephrase,
                        user_question,
                        chain_of_thought,
                        subquestion,
                        subanswer,
                        self_refine,
                        is_new_subquestion,
                        direct_answer,
                        max_depth_allowed,
                        disable_direct_answer,
                    ]
                )
                assert all(attr is not None for attr in [parent, rephrase])
            elif node_type is Node_Type.CHAIN_OF_THOUGHT:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_rephrase,
                        user_question,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        self_refine,
                        direct_answer,
                        max_depth_allowed,
                        disable_direct_answer,
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, chain_of_thought])
            elif node_type is Node_Type.SELF_REFINE:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        node_value,
                        disable_rephrase,
                        user_question,
                        subquestion,
                        subanswer,
                        chain_of_thought,
                        is_new_subquestion,
                        direct_answer,
                        max_depth_allowed,
                        disable_direct_answer,
                    ]
                )
                assert all(attr is not None for attr in [parent, self_refine])
            elif node_type is Node_Type.DIVIDE_AND_CONQUER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_rephrase,
                        user_question,
                        chain_of_thought,
                        self_refine,
                        direct_answer,
                        max_depth_allowed,
                        disable_direct_answer,
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, node_value, subquestion, subanswer, is_new_subquestion]
                )
            elif node_type is Node_Type.DIRECT_ANSWER:                    
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_rephrase,
                        user_question,
                        rephrase,
                        chain_of_thought,
                        self_refine,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        max_depth_allowed,
                        disable_direct_answer,
                    ]
                )
                assert all(attr is not None for attr in [parent, direct_answer])
        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Agent_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.chain_of_thought = chain_of_thought
        self.self_refine = self_refine
        self.subquestion = subquestion
        self.subanswer = subanswer
        self.is_new_subquestion = is_new_subquestion
        self.direct_answer = direct_answer

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.generator = generator
            self.disable_direct_answer = disable_direct_answer
            self.disable_chain_of_thought = disable_chain_of_thought
            self.disable_divide_and_conquer = disable_divide_and_conquer
            self.disable_rephrase = disable_rephrase
            self.disable_self_refine = disable_self_refine
            self.question_index = generator.question_index
            self.max_depth_allowed = max_depth_allowed
            self.if_use_cards = if_use_cards
            self.reasoning_path = reasoning_path
        else:  # inherit from parent   
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.generator = parent.generator
            self.question_index = parent.generator.question_index
            self.max_depth_allowed = parent.max_depth_allowed
            self.disable_direct_answer = parent.disable_direct_answer
            self.disable_chain_of_thought = parent.disable_chain_of_thought
            self.disable_divide_and_conquer = parent.disable_divide_and_conquer
            self.disable_rephrase = parent.disable_rephrase
            self.disable_self_refine = parent.disable_self_refine
            self.if_use_cards = parent.if_use_cards
            self.reasoning_path = parent.reasoning_path

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.REPHRASE:
            self.paraphrased = True
            self.user_question = rephrase
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased

        #! record number of subquestions till now
        if parent is None:  # root
            self.subquestion_counter = 0
        else:
            if node_type is Node_Type.DIVIDE_AND_CONQUER and is_new_subquestion:
                self.subquestion_counter = parent.subquestion_counter + 1
            else:
                self.subquestion_counter = parent.subquestion_counter

        #! record number of one-step thought steps till now
        if parent is None:  # root
            self.direct_answer_counter = 0
        else:
            if node_type is Node_Type.DIRECT_ANSWER or node_type is Node_Type.SELF_REFINE:
                self.direct_answer_counter = parent.direct_answer_counter + 1
            else:
                self.direct_answer_counter = parent.direct_answer_counter

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "direct_answer": {}, "self_refine": {}, "path": [(self.node_type.value, self.id)], "answers": [(0, user_question)]}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)
            self.solution_trace[0]['path'].append((self.node_type.value, self.id))
            answer_id = self.solution_trace[0]['answers'][-1][0] + 1

            if node_type is Node_Type.REPHRASE: # rephrase the user question 
                self.solution_trace[0]["user_question"] = rephrase
                self.solution_trace[0]['answers'].append((answer_id, rephrase))

            elif node_type is Node_Type.CHAIN_OF_THOUGHT:
                assert self.subquestion_counter in self.solution_trace.keys()
                assert self.subquestion_counter == parent.subquestion_counter
                self.solution_trace[self.subquestion_counter]["chain_of_thought"] = {
                    "text": chain_of_thought,
                    "value": node_value,
                }
                self.solution_trace[0]['answers'].append((answer_id, chain_of_thought))
            
            elif node_type is Node_Type.DIVIDE_AND_CONQUER:
                assert is_new_subquestion and self.subquestion_counter == parent.subquestion_counter + 1
                self.solution_trace[self.subquestion_counter] = {
                    "subquestion": subquestion,
                    "subanswer": {"text": subanswer, "value": node_value},
                    "direct_answer": {},
                    "self_refine": {},
                }
                self.solution_trace[0]['answers'].append((answer_id, subanswer))

            elif node_type is Node_Type.DIRECT_ANSWER:
                assert "direct_answer" in self.solution_trace[self.subquestion_counter].keys()
                self.solution_trace[self.subquestion_counter]["direct_answer"][self.direct_answer_counter] = direct_answer
                self.solution_trace[0]['answers'].append((answer_id, direct_answer))

            elif node_type is Node_Type.SELF_REFINE:
                assert "self_refine" in self.solution_trace[self.subquestion_counter].keys()
                self.solution_trace[self.subquestion_counter]["self_refine"][self.direct_answer_counter] = self_refine
                self.solution_trace[0]['answers'].append((answer_id, self_refine))

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "UQ",
            Node_Type.REPHRASE: "RP",
            Node_Type.CHAIN_OF_THOUGHT: "COT",
            Node_Type.SELF_REFINE: "SR",
            Node_Type.DIVIDE_AND_CONQUER: "DC",
            Node_Type.DIRECT_ANSWER: "DA",
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def find_children(self, rollout_id: int):
        try:
            # self.children, terminate = self.children or self._create_children()
            if self.children:
                terminate = False
            else:
                self.children, terminate = self._create_children()
        except Exception as e:
            print(f"[find_children] partial failure on node {getattr(self, 'id', '?')}: {e}")
            traceback.print_exc()
            terminate = False
            if self.children is None:
                self.children = []
        
        for child in self.children:
            child.set_rollout_id(rollout_id)
        
        return self.children, terminate
    
    def _create_children(self):  
        def do_action_perform_direct_answer(parent_is_subquestion=False):
            verbose_print(f"---- Performing direct answer for node {self.id}...", self.verbose)

            parent_is_self_refine = True if self.node_type is Node_Type.SELF_REFINE else False
            ost_list, potential_answers_list = self.generator.generate_direct_answer(
                user_question=self.user_question, 
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
                parent_is_subquestion=parent_is_subquestion,
                parent_is_self_refine=parent_is_self_refine
            )
            for direct_answer, potential_answers in zip(ost_list, potential_answers_list):
                self.children.append(
                    Agent_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        direct_answer=direct_answer
                    )
                )        
        
        def do_action_perform_chain_of_thought():
            verbose_print(f"---- Performing chain-of-thought for node {self.id}...", self.verbose)

            if (
                self.node_type is not Node_Type.USER_QUESTION
                and self.node_type is not Node_Type.REPHRASE
            ):
                hint = make_hint(self.solution_trace, self.node_type)
            else:
                hint = None

            (chain_of_thought_list, value_list) = self.generator.generate_chain_of_thought(
                user_question=self.user_question, paraphrased=self.paraphrased, hint=hint
            )
            if len(chain_of_thought_list) == 0:  
                return
            for chain_of_thought, value in zip(chain_of_thought_list, value_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()  # this should not happen
                if chain_of_thought is None:
                    continue
                self.children.append(
                    Agent_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.CHAIN_OF_THOUGHT,
                        node_value=value,
                        chain_of_thought=chain_of_thought,
                    )
                )
            # CoT+SR: eagerly refine every CoT leaf so Oracle includes SR under one expand.
            if getattr(self.generator, "cot_sr_mode", False):
                for child in list(self.children):
                    if (
                        child.node_type is Node_Type.CHAIN_OF_THOUGHT
                        and not child.children
                    ):
                        child.find_children(rollout_id=getattr(self, "rollout_id", 0) or 0)
        
        def do_action_perform_divide_and_conquer():
            verbose_print(f"---- Performing divide and conquer for node {self.id}...", self.verbose)

            (subquestion_list, subanswer_list, value_list, potential_answers_list) = (
                self.generator.generate_subquestions(
                    user_question=self.user_question, solution_trace=self.solution_trace, paraphrased=self.paraphrased
                )
            )
            for subquestion, subanswer, value, potential_answers in zip(
                subquestion_list, subanswer_list, value_list, potential_answers_list
            ):
                if np.isnan(value) or value <= 0:
                    # value = 0.01
                    breakpoint()
                self.children.append(
                    Agent_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIVIDE_AND_CONQUER,
                        node_value=value,
                        subquestion=subquestion,
                        subanswer=subanswer,
                        is_new_subquestion=True
                    )
                )

        def do_action_perform_rephrase():
            verbose_print(f"---- Performing rephrase for node {self.id}...", self.verbose)

            rephrased_user_question_list, potential_answers_list = self.generator.generate_rephrased_user_question(
                user_question=self.user_question
            )
            for rephrased_user_question, potential_answers in zip(rephrased_user_question_list, potential_answers_list):
                self.children.append(
                    Agent_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASE,
                        rephrase=rephrased_user_question
                    )
                )

        def do_action_perform_self_refine(parent_is_subquestion=False):
            verbose_print(f"---- Performing self-reflection and refinement for node {self.id}...", self.verbose)

            self_reflection_and_refine_list, potential_answers_list = self.generator.generate_self_refine(
                user_question=self.user_question, 
                solution_trace=self.solution_trace,
            )
            if len(self_reflection_and_refine_list) == 0:
                return
            for self_reflection_and_refine, potential_answers in zip(self_reflection_and_refine_list, potential_answers_list):
                if self_reflection_and_refine is None:
                    continue
                self.children.append(
                    Agent_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.SELF_REFINE,
                        self_refine=self_reflection_and_refine
                    )
                )
        
        if not self.if_use_cards:
            # Full MCTS: always create the full sibling action set.
            # Do NOT skip DC/RE/SR when COT confidence > 0.8 — that early-stop
            # left only terminal DA/COT leaves, so later rollouts finished in 0s
            # with no extra API calls (broke budget-matched judgment runs).
            terminate = False

            def _try_action(name, fn):
                try:
                    fn()
                except Exception as e:
                    print(f"[create_children] {name} failed on node {self.id}: {e}")
                    traceback.print_exc()

            # Root Node allowed children: DA, COT, DC, RP
            if self.node_type is Node_Type.USER_QUESTION:
                if not self.disable_direct_answer:
                    _try_action("DA", do_action_perform_direct_answer)
                if not self.disable_chain_of_thought:
                    _try_action("COT", do_action_perform_chain_of_thought)
                if not self.disable_divide_and_conquer:
                    _try_action("DC", do_action_perform_divide_and_conquer)
                if not self.disable_rephrase:
                    _try_action("RE", do_action_perform_rephrase)

            # RP Node allowed children: DA, COT, DC
            elif self.node_type is Node_Type.REPHRASE:
                if not self.disable_direct_answer:
                    _try_action("DA", do_action_perform_direct_answer)
                if not self.disable_chain_of_thought:
                    _try_action("COT", do_action_perform_chain_of_thought)
                if not self.disable_divide_and_conquer:
                    _try_action("DC", do_action_perform_divide_and_conquer)

            # COT Node: terminal unless CoT+SR ablation (only COT+SR on) → allow SR
            elif self.node_type is Node_Type.CHAIN_OF_THOUGHT:
                cot_sr_mode = bool(getattr(self.generator, "cot_sr_mode", False))
                if cot_sr_mode and not self.disable_self_refine:
                    _try_action("SR", do_action_perform_self_refine)
                else:
                    raise ValueError("Chain-of-Thought node cannot create children!!")

            # DC Node allowed children: DA, COT, DC, SR
            elif self.node_type is Node_Type.DIVIDE_AND_CONQUER:
                if not self.disable_direct_answer:
                    _try_action("DA", lambda: do_action_perform_direct_answer(parent_is_subquestion=True))
                if not self.disable_chain_of_thought:
                    _try_action("COT", do_action_perform_chain_of_thought)
                if not self.disable_divide_and_conquer:
                    _try_action("DC", do_action_perform_divide_and_conquer)
                if not self.disable_self_refine:
                    _try_action("SR", lambda: do_action_perform_self_refine(parent_is_subquestion=True))

            # DA Node allowed children: DA, COT, SR
            elif self.node_type is Node_Type.DIRECT_ANSWER:
                if not self.disable_direct_answer:
                    _try_action("DA", do_action_perform_direct_answer)
                if not self.disable_chain_of_thought:
                    _try_action("COT", do_action_perform_chain_of_thought)
                if not self.disable_self_refine:
                    _try_action("SR", do_action_perform_self_refine)

            # SR Node allowed children: DA, COT
            elif self.node_type is Node_Type.SELF_REFINE:
                if not self.disable_direct_answer:
                    _try_action("DA", do_action_perform_direct_answer)
                if not self.disable_chain_of_thought:
                    _try_action("COT", do_action_perform_chain_of_thought)
        else:
            depth = self.depth
            try:
                cur_action = self.reasoning_path[depth+1]
            except:
                cur_action = Node_Type.CHAIN_OF_THOUGHT.value
            terminate = False
            parent_is_subquestion = self.node_type is Node_Type.DIVIDE_AND_CONQUER
            if cur_action == Node_Type.DIRECT_ANSWER.value:
                do_action_perform_direct_answer(parent_is_subquestion)
            elif cur_action == Node_Type.SELF_REFINE.value:
                do_action_perform_self_refine(parent_is_subquestion)
            elif cur_action == Node_Type.CHAIN_OF_THOUGHT.value:
                do_action_perform_chain_of_thought()
                if self.children[-1].node_type.value=="CHAIN_OF_THOUGHT" and self.children[-1].node_value > 0.8:
                    terminate = True
            elif cur_action == Node_Type.DIVIDE_AND_CONQUER.value:
                do_action_perform_divide_and_conquer()
            elif cur_action == Node_Type.REPHRASE.value:
                do_action_perform_rephrase()

        return self.children, terminate

    def is_valid_leaf_node(self):
        cot_sr_mode = bool(getattr(self.generator, "cot_sr_mode", False))
        cot_is_leaf = self.node_type is Node_Type.CHAIN_OF_THOUGHT and not cot_sr_mode
        sr_is_leaf = self.node_type is Node_Type.SELF_REFINE and cot_sr_mode
        return (
            self.node_type is Node_Type.DIRECT_ANSWER 
            and reach_terminal_direct_answer(self.direct_answer)
        ) or (
            self.node_type is Node_Type.DIVIDE_AND_CONQUER 
            and reach_terminal_subquestion(self.subquestion, self.user_question)
        ) or cot_is_leaf or sr_is_leaf

    def is_valid_solution_node(self):
        """Check if the node is a valid solution node."""
        cot_sr_mode = bool(getattr(self.generator, "cot_sr_mode", False))
        # Keep CoT answers even in CoT+SR; SR adds refined candidates on top.
        cot_is_sol = self.node_type is Node_Type.CHAIN_OF_THOUGHT
        sr_is_sol = self.node_type is Node_Type.SELF_REFINE and cot_sr_mode
        return (
            (
                self.node_type is Node_Type.DIVIDE_AND_CONQUER
                and reach_terminal_subquestion(self.subquestion, self.user_question)
            )
            or (self.node_type is Node_Type.DIRECT_ANSWER 
                and reach_terminal_direct_answer(self.direct_answer))
            or cot_is_sol
            or sr_is_sol
        )

    def set_potential_score(self, score: float):
        self.potential_score = score

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node(): 
            if self.node_value is None:
                return 0
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION or self.node_type is Node_Type.REPHRASE
