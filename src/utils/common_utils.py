# Portions of this file are adapted from rStar (https://github.com/zhentingqi/rStar).
# Copyright (c) 2024 rStar Team. Licensed under the MIT License; see THIRD_PARTY_NOTICES.md
# and licenses/rStar-LICENSE.
import os
import datetime
import json
from utils.agent_utils import Node_Type
import yaml
from colorama import Fore, Style

def read_jsonl(file_path):
    """
    Read a JSON Lines file and return a list of dictionaries.
    Args:
        file_path (str): Path to the JSON Lines file.
    Returns:
        list: A list of dictionaries representing the JSON objects in the file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_task_id_set(path):
    """Load task_ids from a txt (one per line) or judgment_set JSON."""
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return set()
    if path.endswith(".json") or text.startswith("{"):
        obj = json.loads(text)
        ids = obj.get("task_ids") or [t["task_id"] for t in obj.get("tasks", [])]
        return {int(x) for x in ids}
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(int(line.split(",")[0].strip()))
    return set(ids)


def filter_data_by_task_ids(data_item_list, task_ids):
    if not task_ids:
        return data_item_list
    wanted = {int(x) for x in task_ids}
    return [d for d in data_item_list if int(d.get("task_id")) in wanted]

def read_json(file_path):
    """
    Read a JSON file and return its content.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        dict: The content of the JSON file as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def read_yaml(file_path):
    """
    Read a YAML file and return its content.
    Args:
        file_path (str): Path to the YAML file.
    Returns:
        dict: The content of the YAML file as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_json(js_obj, file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(js_obj, f, indent=2, default=str)

def read_txt(file_path):
    assert str(file_path).endswith(".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def verbose_print(message, verbose):
    """
    Prints if verbose flag is true.
    """
    if verbose:
        print(message)

def print_tree_from_root(mcts_searcher, rollout_id, root_node, chosen_node=None, file=None):
    color_print = False if file else True

    def my_print(text):
        if file:
            file.write(text + "\n")
        else:
            print(text)

    def print_tree(parent_node, node, file, rollout_id):
        to_print = ""

        num_indent = 4
        dash = "-" * num_indent * node.depth
        space = " " * num_indent * node.depth

        attributes = f"Q: {round(mcts_searcher.Q[node], 2)}" + "; " + f"N: {mcts_searcher.N[node]}" + "; "
        attributes += f"V: {round(node.node_value, 2)}" if node.node_value is not None else "V: None"

        uct_value = "UCT: " + str(
            round(mcts_searcher._compute_uct(parent_node=parent_node, node=node, rollout_id=rollout_id), 2)
        )
        attributes += "; " + uct_value

        solution_marker = "(T) " if node.is_valid_solution_node() else "" 

        node_info = "[" + solution_marker + node.__str__() + ": " + attributes + "]"
        if chosen_node and node == chosen_node:
            node_info = "[" + node_info + "]"
        node_info += " "

        if color_print and node.is_valid_solution_node():
            node_details = Fore.RED + Style.BRIGHT + node_info + Fore.RESET + Style.RESET_ALL
        else:
            node_details = node_info

        if node.node_type is Node_Type.USER_QUESTION:
            gt = node.expected_answer.replace("\n", " ")
            node_details += f"User: {node.user_question}" + "\n" + space + " " * len(node_info) + f"Ground truth: {gt}"
        elif node.node_type is Node_Type.REPHRASE:
            node_details += f"Reph-User: {node.user_question}"
        elif node.node_type is Node_Type.CHAIN_OF_THOUGHT:
            node_details += f"Ans: {node.chain_of_thought}"
        elif node.node_type is Node_Type.SELF_REFINE:
            node_details += f"Refine-Summary: {node.self_refine}"
        elif node.node_type is Node_Type.DIVIDE_AND_CONQUER:
            node_details += f"Q: {node.subquestion}" + "\n" + space + " " * len(node_info) + f"A: {node.subanswer}"
        elif node.node_type is Node_Type.DIRECT_ANSWER:
            node_details += f"DA: {node.direct_answer}"

        to_print += dash + node_details

        my_print(to_print)

        for child in node.children:
            print_tree(node, child, file, rollout_id)

        if node.depth == 0:
            my_print("\n" + "=" * 50 + "\n")

    print_tree(parent_node=None, node=root_node, file=file, rollout_id=rollout_id)