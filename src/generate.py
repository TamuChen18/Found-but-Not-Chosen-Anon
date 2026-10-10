# Portions of this file are adapted from rStar (https://github.com/zhentingqi/rStar),
# Copyright (c) 2024 rStar Team, and from ThoughtICR (https://github.com/jinyangwu/ThoughtICR),
# Copyright (c) 2025 Jinyang Wu. Both are licensed under the MIT License; see THIRD_PARTY_NOTICES.md,
# licenses/rStar-LICENSE and licenses/ThoughtICR-LICENSE.
import time
import traceback
from dotenv import load_dotenv
import json
import torch
from mcts.mcts_generator import MCTS_Generator
from mcts.mcts_search import search_for_answers
from utils.common_utils import read_json, read_jsonl, load_task_id_set, filter_data_by_task_ids
from utils.logger import setup_main_logger, get_worker_logger
from config.args import parse_args, post_process_args
from evaluators.evaluators import Evaluator
import concurrent.futures
import os
import math
from utils.wandb_utils import init_wandb, finish_wandb

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import warnings
warnings.filterwarnings("ignore")

def worker_process(args, data_item, model_name, tokenizer_name, log_queue):
    """
    Function executed in each worker process.
    """
    try:
       
        task_id = data_item.get("task_id")
        task_desc = data_item.get("task_desc")
        gt_solution = data_item.get("structured_plan")

        evaluator = Evaluator()
        gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)

        js = {
            "task_id": task_id,
            "problem": task_desc,
            "model_completion": None,
            "model_answer": None,
            "all_model_completions": {},
            "gold_solution": gt_solution,
            "gold_answer": gt_answer,
        }

        return search_for_answers(
            args, 
            evaluator, 
            js, 
            task_desc, 
            task_id, 
            gt_answer,
            model_name, 
            tokenizer_name, 
            log_queue
        )

    except Exception as e:
        raise RuntimeError(f"Worker failed on task {data_item.get('task_id')}: {e}")

def worker_thread(args, data_item, model_name, tokenizer_name, log_queue):
    """
    Thread worker for OpenAI API models.
    """
    task_id = data_item.get("task_id")
    task_desc = data_item.get("task_desc")
    gt_solution = data_item.get("structured_plan")

    evaluator = Evaluator()
    gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)

    js = {
        "task_id": task_id,
        "problem": task_desc,
        "model_completion": None,
        "model_answer": None,
        "all_model_completions": {},
        "gold_solution": gt_solution,
        "gold_answer": gt_answer,
    }

    return search_for_answers(
        args, evaluator, js, task_desc, task_id, gt_answer,
        model_name=model_name, tokenizer_name=tokenizer_name, log_queue = log_queue
    )

def main(args):
    """
    Main function to handle the execution of the script based on parsed arguments.
    Args:
        args (Namespace): Parsed command line arguments.
    Returns:
        None
    """
    main_logger, log_queue, log_listener = setup_main_logger(args.logdir)

    run = init_wandb(args)

    assert args.api in ["openai", "openrouter", "huggingface", "vllm"], (
        "Only OpenAI, OpenRouter, vLLM and HuggingFace models are supported."
    )

    if args.api in ["openai", "openrouter"]:
        model_name = args.model_ckpt
        tokenizer_name = None
        ExecutorClass = concurrent.futures.ThreadPoolExecutor
        num_workers = args.max_threads if args.max_threads > 0 else 1

    elif args.api in ["huggingface", "vllm"]:
        model_name = args.model_ckpt
        tokenizer_name = args.tokenizer_ckpt or args.model_ckpt
        num_workers = 30 
        # num_workers = 1
        ExecutorClass = concurrent.futures.ProcessPoolExecutor

    test_file = os.path.join(args.data_root, args.train_test_json_data)
    data_item_list = read_jsonl(test_file)
    if getattr(args, "task_ids_file", None):
        task_ids = load_task_id_set(args.task_ids_file)
        data_item_list = filter_data_by_task_ids(data_item_list, task_ids)
        print(f"[INFO] Filtered to {len(data_item_list)} tasks from {args.task_ids_file}")
    elif args.task_num != 0:
        data_item_list = data_item_list[args.task_num - 1 : args.task_num]
    elif args.max_tasks > 0:
        data_item_list = data_item_list[: args.max_tasks]
    
    if args.if_use_cards:
        args.reason_structure = read_json(args.reuse_dir)

    total_correct = 0
    total_correct_tools = 0
    total_correct_parameters = 0
    total_correct_dependencies = 0
    num_tested = 0
    
    start_time = time.time()
    
    with ExecutorClass(max_workers=num_workers) as executor:
        futures = {}
        for i, data_item in enumerate(data_item_list):
            if args.api in ["huggingface", "vllm"]:
                futures[executor.submit(worker_process, args, data_item, model_name, tokenizer_name, log_queue)] = data_item
            else:
                futures[executor.submit(worker_thread, args, data_item, model_name, tokenizer_name, log_queue)] = data_item

        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            correct = False
            tools_correct = False
            parameters_correct = False
            dependencies_correct = False

            data_item = futures[future]
            
            try:
                result = future.result(timeout = 300)
                if result:
                    js, _, _, _, correct, tools_correct, parameters_correct, dependencies_correct = result
                else:
                    continue
                main_logger.info(f"Task {data_item.get('task_id')} finished with result: {result is not None}")
            except Exception as e:
                main_logger.error(f"Future {i} failed on task {data_item.get('task_id')}: {e}")
                main_logger.error(traceback.print_exc())
                continue
            
            main_logger.info(f"get result for {data_item.get('task_id')}!")
            
            if js:
                with open(os.path.join(args.answer_sheets_dir, f"Task {js['task_id']} - Legacy.json"), "w") as f:
                    json.dump(js, f)
            
            num_tested += 1
            total_correct += int(correct)
            total_correct_tools += int(tools_correct)
            total_correct_parameters += int(parameters_correct)
            total_correct_dependencies += int(dependencies_correct)

            main_logger.info(f"Overall Accuracy: {(total_correct/(num_tested))*100:.2f}\n")
            main_logger.info(f"Tools Accuracy: {(total_correct_tools/(num_tested))*100:.2f}\n")
            main_logger.info(f"Parameter Accuracy: {(total_correct_parameters/(num_tested))*100:.2f}\n")
            main_logger.info(f"Dependency Accuracy: {(total_correct_dependencies/(num_tested))*100:.2f}\n") 

            with open(os.path.join(args.run_outputs_dir, "MCTS_result.txt"), "w") as f:
                if not args.disable_answer_selection:
                    f.write(f"Num tested: {num_tested}\n")
                    f.write(f"Num correct: {total_correct}\n")
                    f.write(f"Overall Accuracy: {(total_correct/(num_tested))*100:.2f}\n")
                    f.write(f"Tools Accuracy: {(total_correct_tools/(num_tested))*100:.2f}\n")
                    f.write(f"Parameter Accuracy: {(total_correct_parameters/(num_tested))*100:.2f}\n")
                    f.write(f"Dependency Accuracy: {(total_correct_dependencies/(num_tested))*100:.2f}\n") 
            
            if args.if_use_cards:
                main_logger.info(f"model: {model_name} | file: {args.file} | difficulty: {args.attribute_type}\n")
            else:
                main_logger.info(f"model: {model_name} | file: {args.file}\n")

    log_listener.stop()
    end_time = time.time()

    if num_tested == 0:
        num_tested+=1
    
    elapsed_time = end_time - start_time
    average_time = elapsed_time / num_tested
    
    minutes, seconds = divmod(elapsed_time, 60)
    average_mins, average_secs = divmod(average_time, 60)

    finish_wandb(
        run,
        {
            "tools_accuracy": f"{(total_correct_tools/(num_tested))*100:.2f}",
            "parameters_accuracy": f"{(total_correct_parameters/(num_tested))*100:.2f}",
            "dependencies_accuracy": f"{(total_correct_dependencies/(num_tested))*100:.2f}",
            "overall_accuracy": f"{(total_correct/(num_tested))*100:.2f}",
            "total_execution_time": f"{minutes} min {seconds: .2f} sec",
            "average_execution_time": f"{average_mins} min {average_secs: .2f} sec",
        },
    )

    main_logger.info("Execution completed successfully.")
    main_logger.info(f"Time taken to complete execution: {minutes} min {seconds: .2f} sec")
    main_logger.info(f"Average time taken: {average_mins} min {average_secs: .2f} sec")
    
    with open(os.path.join(args.run_outputs_dir, "MCTS_result.txt"), "a") as f:
        f.write(f"Time taken to complete execution: {minutes} min {seconds: .2f} sec\n")
        f.write(f"Average time taken: {average_mins} min {average_secs: .2f} sec")

if __name__ == "__main__":
    args = parse_args()
    args = post_process_args(args)
    main(args)