#python test/test_generate.py --train_test_json_data train.jsonl --max_threads 1 --num_chain_of_thought 2 --use_fewshot True

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import gc
import time
import traceback
from dotenv import load_dotenv

from best_of_n_cot import BestOfNCoT
from cot import ChainOfThoughts

from utils.common_utils import read_jsonl, load_task_id_set, filter_data_by_task_ids
from config.args import parse_args, post_process_args
from interfaces.IO_Interface import IO_Interface
from evaluators.evaluators import Evaluator
from utils.wandb_utils import init_wandb, finish_wandb

import multiprocessing
import concurrent.futures
import torch
import copy


def _create_baseline_model(args, io, evaluator):
    if args.test_type == "bon_cot":
        n = getattr(args, "num_bon_samples", 0) or args.num_rollouts
        return BestOfNCoT(io, evaluator, args.use_fewshot, n=n)
    elif args.test_type == "cot":
        return ChainOfThoughts(io, evaluator, args.use_fewshot, args.num_chain_of_thought)
    else:
        raise ValueError(f"Test type {args.test_type} is not implemented.")


def _eval_answer(evaluator, model_answer, gt_answer):
    correct = evaluator.check_answers_equivalence(model_answer, gt_answer)
    tools_correct = evaluator.check_tools_correctness(model_answer, gt_answer)
    parameters_correct = evaluator.check_parameters_correctness(model_answer, gt_answer)
    dependencies_correct = evaluator.check_dependencies_correctness(model_answer, gt_answer)
    return correct, tools_correct, parameters_correct, dependencies_correct


def _run_baseline_once(args, data_item, model_name, tokenizer_name, process_mode=False):
    local_args = copy.deepcopy(args)
    generation_kwargs = {
        "max_tokens": local_args.max_tokens,
        "temperature": local_args.temperature,
        "top_p": local_args.top_p,
    }
    if getattr(local_args, "seed", None) is not None:
        generation_kwargs["seed"] = int(local_args.seed)
    if not process_mode:
        generation_kwargs["timeout"] = getattr(local_args, "api_timeout", 120)
        generation_kwargs["max_retries"] = getattr(local_args, "api_max_retries", 4)

    io = IO_Interface(local_args.api, model_name, tokenizer_name, generation_kwargs)
    evaluator = Evaluator()
    baseline_model = _create_baseline_model(local_args, io, evaluator)

    task_id = data_item.get("task_id")
    task_desc = data_item.get("task_desc")
    gt_solution = data_item.get("structured_plan")
    gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)

    try:
        candidates = None
        if args.test_type == "bon_cot" and hasattr(baseline_model, "generate_with_candidates"):
            model_answer, cand_raw = baseline_model.generate_with_candidates(task_desc)
            candidates = []
            for c in cand_raw:
                ans = c.get("model_answer")
                correct, tools_correct, parameters_correct, dependencies_correct = _eval_answer(
                    evaluator, ans, gt_answer
                )
                candidates.append(
                    {
                        "candidate_id": c["candidate_id"],
                        "model_answer": ans,
                        "raw_completion": c.get("raw_completion"),
                        "is_selected": bool(c.get("is_selected")),
                        "correct": bool(correct),
                        "tools_correct": bool(tools_correct),
                        "parameters_correct": bool(parameters_correct),
                        "dependencies_correct": bool(dependencies_correct),
                    }
                )
        else:
            model_answer = baseline_model.generate(task_desc)

        correct, tools_correct, parameters_correct, dependencies_correct = _eval_answer(
            evaluator, model_answer, gt_answer
        )
        usage = io.get_usage()
        return (
            model_answer,
            correct,
            tools_correct,
            parameters_correct,
            dependencies_correct,
            usage,
            candidates,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"Error in baseline task {task_id}: {e}")
        usage = io.get_usage()
        return None, False, False, False, False, usage, None


def worker_process(args, data_item, model_name, tokenizer_name):
    return _run_baseline_once(args, data_item, model_name, tokenizer_name, process_mode=True)


def worker_thread(args, data_item, model_name, tokenizer_name):
    return _run_baseline_once(args, data_item, model_name, tokenizer_name, process_mode=False)
def main(args):
    """
    Main function to handle the execution of the script based on parsed arguments.
    Args:
        args (Namespace): Parsed command line arguments.
    Returns:
        None
    """

    load_dotenv()

    run = init_wandb(args)

    assert args.api in ["openai", "openrouter", "huggingface", "vllm"], (
        "Only OpenAI, OpenRouter, vLLM and HuggingFace models are supported."
    )
    
    num_gpus = torch.cuda.device_count() if args.api in ["huggingface"] and torch.cuda.is_available() else 0

    if args.api in ["openai", "openrouter"]:
        model_name = args.model_ckpt
        tokenizer_name = None
        ExecutorClass = concurrent.futures.ThreadPoolExecutor
        num_workers = args.max_threads if args.max_threads > 0 else 1

    elif args.api in ["huggingface", "vllm"]:
        multiprocessing.set_start_method("spawn", force=True)
        model_name = args.model_ckpt
        tokenizer_name = args.tokenizer_ckpt or args.model_ckpt
        num_workers = 20
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
    
    total_correct = 0
    total_correct_tools = 0
    total_correct_parameters = 0
    total_correct_dependencies = 0
    num_tested = 0
    aggregated_usage = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    import json as _json
    answer_sheets_dir = args.answer_sheets_dir

    start_time = time.time()
    
    with ExecutorClass(max_workers=num_workers) as executor:
        futures = {}
        for i, data_item in enumerate(data_item_list):
            try:
                if args.api in ["huggingface", "vllm"]:
                    futures[executor.submit(worker_process, args, data_item, model_name, tokenizer_name)] = data_item
                else:
                    futures[executor.submit(worker_thread, args, data_item, model_name, tokenizer_name)] = data_item
            except Exception as e:
                print(f"Error on item {i}: {e}")
                continue
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            task_usage = {}
            try:
                result = future.result()
                candidates = None
                if len(result) == 7:
                    (
                        model_answer,
                        correct,
                        tools_correct,
                        parameters_correct,
                        dependencies_correct,
                        task_usage,
                        candidates,
                    ) = result
                elif len(result) == 6:
                    model_answer, correct, tools_correct, parameters_correct, dependencies_correct, task_usage = result
                else:
                    model_answer, correct, tools_correct, parameters_correct, dependencies_correct = result
                    task_usage = {}
            except Exception as e:
                print(f"Future {i} failed: {e}")
                traceback.print_exc()
                continue

            data_item = futures[future]
            task_id = data_item.get("task_id")

            print("="*50)
            print(f"get result for {task_id}!\n")
            print(f"Task {task_id} Ground Truth: {data_item.get('structured_plan')}\n")
            print(f"Task {task_id} Model Answer: {model_answer}\n")

            num_tested += 1
            total_correct += int(correct)
            total_correct_tools += int(tools_correct)
            total_correct_parameters += int(parameters_correct)
            total_correct_dependencies += int(dependencies_correct)

            if task_usage:
                for k in aggregated_usage:
                    aggregated_usage[k] += task_usage.get(k, 0)

            if candidates:
                all_candidates = candidates
                oracle_correct = any(c.get("correct") for c in candidates)
            else:
                all_candidates = [{
                    "candidate_id": 0,
                    "model_answer": model_answer,
                    "is_selected": True,
                    "correct": bool(correct),
                    "tools_correct": bool(tools_correct),
                    "parameters_correct": bool(parameters_correct),
                    "dependencies_correct": bool(dependencies_correct),
                }]
                oracle_correct = bool(correct)

            answer_payload = {
                "task_id": task_id,
                "method": args.test_type,
                "selected_best": {
                    "answer": model_answer,
                    "metrics": {
                        "correct": bool(correct),
                        "tools_correct": bool(tools_correct),
                        "parameters_correct": bool(parameters_correct),
                        "dependencies_correct": bool(dependencies_correct),
                    },
                },
                "oracle_correct": bool(oracle_correct),
                "all_candidates": all_candidates,
                "usage": task_usage or {},
            }
            with open(os.path.join(answer_sheets_dir, f"Task {task_id} - Answer.json"), "w") as f:
                _json.dump(answer_payload, f, indent=2, default=str)

            print(f"Overall Accuracy: {(total_correct/(num_tested))*100:.2f}\n")
            print(f"Tools Accuracy: {(total_correct_tools/(num_tested))*100:.2f}\n")
            print(f"Parameter Accuracy: {(total_correct_parameters/(num_tested))*100:.2f}\n")
            print(f"Dependency Accuracy: {(total_correct_dependencies/(num_tested))*100:.2f}\n") 

            print("="*50)

            with open(os.path.join(args.run_outputs_dir, f"{args.test_type}_result.txt"), "w") as f:
                if not args.disable_answer_selection:
                    f.write(f"Num tested: {num_tested}\n")
                    f.write(f"Num correct: {total_correct}\n")
                    f.write(f"Overall Accuracy: {(total_correct/(num_tested))*100:.2f}\n")
                    f.write(f"Tools Accuracy: {(total_correct_tools/(num_tested))*100:.2f}\n")
                    f.write(f"Parameter Accuracy: {(total_correct_parameters/(num_tested))*100:.2f}\n")
                    f.write(f"Dependency Accuracy: {(total_correct_dependencies/(num_tested))*100:.2f}\n")   
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    average_time = elapsed_time / max(num_tested, 1)
    
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

    print("Execution completed successfully.")
    print(f"Time taken to complete execution: {minutes} min {seconds: .2f} sec")
    print(f"Average time taken: {average_mins} min {average_secs: .2f} sec")
    print(f"\n--- Aggregated API Usage ---")
    for k, v in aggregated_usage.items():
        print(f"  {k}: {v}")

    with open(os.path.join(args.run_outputs_dir, f"{args.test_type}_result.txt"), "a") as f:
        f.write(f"Time taken to complete execution: {minutes} min {seconds: .2f} sec\n")
        f.write(f"Average time taken: {average_mins} min {average_secs: .2f} sec\n")
        f.write(f"\n--- Aggregated API Usage ---\n")
        for k, v in aggregated_usage.items():
            f.write(f"  {k}: {v}\n")                

if __name__ == "__main__":
    args = parse_args()
    args = post_process_args(args)
    print(args)
    main(args)