# Portions of this file are adapted from rStar (https://github.com/zhentingqi/rStar),
# Copyright (c) 2024 rStar Team, and from ThoughtICR (https://github.com/jinyangwu/ThoughtICR),
# Copyright (c) 2025 Jinyang Wu. Both are licensed under the MIT License; see THIRD_PARTY_NOTICES.md,
# licenses/rStar-LICENSE and licenses/ThoughtICR-LICENSE.
from enum import Enum, unique
import random
import re
from typing import Dict

@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    REPHRASE = "REPHRASE"
    CHAIN_OF_THOUGHT = "CHAIN_OF_THOUGHT"
    SELF_REFINE = "SELF_REFINE"
    DIVIDE_AND_CONQUER = "DIVIDE_AND_CONQUER"
    DIRECT_ANSWER = "DIRECT_ANSWER"

def concat_direct_answers(solution_trace):
    """
    Concatenate direct answers from the solution trace.
    Args:
        solution_trace (Dict[int, Dict[str, str]]): The solution trace containing answers.
    Returns:
        Tuple[str, int]: The concatenated direct answer and the next answer ID.
    """
    last_tuple = list(solution_trace.items())[-1]
    last_tuple_id, last_tuple_recording = last_tuple[0], last_tuple[1]
    assert "direct_answer" in last_tuple_recording.keys()
    if len(last_tuple_recording["direct_answer"]) > 0:
        solution_trace_str = ""
        for step_id, step_text in last_tuple_recording["direct_answer"].items():
            solution_trace_str += f"step {step_id}: " + step_text + "\n"
        return solution_trace_str, step_id + 1
    else:
        # no direct answer step yet
        return "", 1
    
def concat_subquestions_and_subanswers(solution_trace, question_index):
    solution_trace_str = ""

    for subquestion_id, solution_step in solution_trace.items():
        if subquestion_id == 0:
            continue

        assert subquestion_id > 0
        assert "subquestion" in solution_step.keys() and "subanswer" in solution_step.keys()

        solution_trace_str += f"Question {question_index}." + str(subquestion_id) + ": " + solution_step["subquestion"] + "\n"
        solution_trace_str += f"Answer {question_index}." + str(subquestion_id) + ": " + solution_step["subanswer"]["text"] + "\n"

    next_subquestion_id = int(sorted(solution_trace.keys())[-1]) + 1
    return solution_trace_str, next_subquestion_id
    
def concat_subquestions_and_subanswers_as_da(solution_trace):
    """
    Concatenate subquestions and subanswers from the solution trace for direct answer.
    Args:
        solution_trace (Dict[int, Dict[str, str]]): The solution trace containing subquestions and subanswers.
    Returns:
        Tuple[str, int]: The concatenated subquestions and subanswers, and the next answer ID.
    """
    subquestion_subanswer_as_da = ""
    step_id = 1 # Do not include the Root node
    while step_id in solution_trace:
        if "subanswer" in solution_trace[step_id]:
            text = solution_trace[step_id]["subanswer"]["text"]
            match = re.search(r"(.+?)\s*The structured task plan is:", text, re.DOTALL)
            if match:
                step_text = match.group(1).strip()
            else:
                step_text = text.strip()
            subquestion_subanswer_as_da += f"step {step_id}: " + step_text + "\n"
            step_id += 1
        else:
            # not subquestions yet
            return "", 1
    return subquestion_subanswer_as_da, step_id

def make_hint(
    solution_trace: Dict[int, Dict[str, str]], node_type: Node_Type
) -> str:
    if node_type in [Node_Type.DIVIDE_AND_CONQUER]:
        hint = ""

        for subquestion_id, solution_step in solution_trace.items():
            if subquestion_id == 0:
                continue

            assert subquestion_id > 0
            assert "subquestion" in solution_step.keys() and "subanswer" in solution_step.keys()

            hint += f"Hint " + str(subquestion_id) + " (You can partially refer to these imcomplete steps): " + solution_step["subquestion"]    
            hint += " "
            hint += solution_step["subanswer"]["text"]
            hint += "\n"

        hint = hint.strip("\n")
    elif node_type is Node_Type.DIRECT_ANSWER:
        hint = "Hint (You can partially refer to these imcomplete steps): "
        last_tuple = list(solution_trace.items())[-1]
        last_tuple_recording = last_tuple[1]
        assert last_tuple_recording["direct_answer"]
        for step_id, step_text in last_tuple_recording["direct_answer"].items():
            hint += step_text + " "

        hint = hint.strip(" ")
    
    elif node_type is Node_Type.SELF_REFINE:
        hint = "Hint (You can partially refer to these imcomplete steps):\n" + solution_trace[0]['answers'][-1][1]  
    else:
        raise ValueError(f"Invalid node type: {node_type}.")

    return hint

def reach_terminal_subquestion(subquestion, user_question, sim_model=None, sim_model_name=None):
    assert subquestion is not None

    if "now we can answer the question" in subquestion.lower():
        return True
    
    # experimental
    if sim_model is not None:
        phrases = re.split(r'(?<=[.?!])\s+', user_question.lower().strip())
        questions = [p for p in phrases if p.endswith('?')]

        if questions:      
            subquestion = subquestion.lower().split("now we can answer the question:")[-1]
            for question in questions:
                sentences = [subquestion, question]

                if sim_model_name == "bge":
                    embeddings = sim_model.encode(sentences, batch_size=2)['dense_vecs']
                    threshold = 0.6
                elif sim_model_name == "mpnet":
                    embeddings = sim_model.encode(sentences)
                    threshold = 0.5
                
                similarity = embeddings[0] @ embeddings[1].T

                if similarity > threshold:
                    return True
        else:
            subquestion = subquestion.lower().split("now we can answer the question:")[-1]
            for phrase in phrases:
                sentences = [subquestion, phrase]

                if sim_model_name == "bge":
                    embeddings = sim_model.encode(sentences, batch_size=2)['dense_vecs']
                    threshold = 0.6
                elif sim_model_name == "mpnet":
                    embeddings = sim_model.encode(sentences)
                    threshold = 0.5
                
                similarity = embeddings[0] @ embeddings[1].T

                if similarity > threshold:
                    return True    
    return False

def reach_terminal_direct_answer(direct_answer):
    pattern = re.compile(r"\[\s*(?:\{.*?\}\s*,?\s*)+\]", re.DOTALL)
    matches = pattern.findall(direct_answer)
    return "structured task plan" in direct_answer.lower() and matches

def split_user_question(user_question: str):
    user_question = user_question.strip().rstrip(".")
    last_period_id = user_question.rfind(".")
    assert last_period_id < len(user_question) - 1
    user_question_context = user_question[: last_period_id + 1].strip()
    user_question_problem = user_question[last_period_id + 1 :].strip()
    return user_question_context, user_question_problem

def concat_all_parent_steps(solution_trace):
    """Return: concatenated all parent steps"""
    solution_trace_str = ""
    for i, x in enumerate(solution_trace[0]['answers'][1:]):
        step_i = x[1].replace("\n", " ")
        solution_trace_str += f"step {i+1}: " + step_i + "\n"
    return solution_trace_str, (i+2)

_JSON_ARRAY_RE = re.compile(r'\[\s*(?:\{[\s\S]*?\}\s*,?\s*)+\]', re.DOTALL)

def extract_solution_from_node(node):
    """Extract the structured plan JSON text from a solution node, or None."""
    if node.node_type is Node_Type.DIVIDE_AND_CONQUER:
        return node.subanswer if _JSON_ARRAY_RE.findall(node.subanswer) else None
    elif node.node_type is Node_Type.CHAIN_OF_THOUGHT:
        return node.chain_of_thought if _JSON_ARRAY_RE.findall(node.chain_of_thought) else None
    elif node.node_type is Node_Type.DIRECT_ANSWER and 'structured task plan' in node.direct_answer.lower():
        return node.direct_answer if _JSON_ARRAY_RE.findall(node.direct_answer) else None
    elif node.node_type is Node_Type.SELF_REFINE:
        text = getattr(node, "self_refine", None) or ""
        return text if text and _JSON_ARRAY_RE.findall(text) else None
    else:
        return None


def stochastic_find_best_solution(
    root_node,
    evaluator,
    enable_potential_score,
):
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None, None, None, None, None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]

    def calculate_potential_score_for_solution_node(node):
        model_answer = evaluator.extract_answer_from_model_completion(extract_solution_from_node(node))
        potential_answers_history = node.potential_answers_history  # {depth -> [potential answers]}
        assert potential_answers_history[node.depth] is None

        potential_score = 1
        for depth, depth_potential_answers in potential_answers_history.items():
            if depth < node.depth:
                depth_score = sum(
                    evaluator.check_answers_equivalence(dpa, model_answer) for dpa in depth_potential_answers
                ) / len(depth_potential_answers)
                potential_score *= depth_score

        node.set_potential_score(potential_score)
        return potential_score

    prior_weights = (
        [calculate_potential_score_for_solution_node(node) for node in solution_nodes]
        if enable_potential_score
        else None
    )
    top_answer, top_completion, top_completion_id, top_confidence = evaluator.stochastic_find_most_confident_answer(
        completions=solutions, prior_weights=prior_weights
    )

    if top_answer is None or len(top_answer)==0:
        return None, None, None, None, None, None
    return top_answer, top_completion, top_confidence, solution_nodes[top_completion_id], solution_nodes, solutions


def find_valid_solution_nodes(root_node):  
    valid_solution_nodes = []

    def recursion(node):  
        # Collect solution nodes but still descend so CoT+SR can keep SR under COT.
        if node.is_valid_solution_node():
            valid_solution_nodes.append(node)

        if not node.children:
            return

        try:
            for child in node.children:
                recursion(child)
        except Exception:
            print(node.children, getattr(node.children, "node_type", None))

    recursion(root_node)

    return valid_solution_nodes