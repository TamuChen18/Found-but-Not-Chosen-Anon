# Portions of this file are adapted from rStar (https://github.com/zhentingqi/rStar).
# Copyright (c) 2024 rStar Team. Licensed under the MIT License; see THIRD_PARTY_NOTICES.md
# and licenses/rStar-LICENSE.
import os
import ast
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from copy import deepcopy
import threading

from interfaces.call_context import call_context
from mcts.mcts_generator import MCTS_Generator
from mcts.mcts_base import MCTS_Searcher
from mcts.mcts_node import Agent_MCTS_Node
from utils.agent_utils import (
    Node_Type,
    stochastic_find_best_solution,
    find_valid_solution_nodes,
    extract_solution_from_node,
)
from utils.answer_selector import load_tools_manifest, select_answer
from utils.common_utils import save_json
from utils.logger import setup_main_logger, get_worker_logger


def _save_rollout_artifacts(args_local, task_id, model_rollout_nodes, model_all_solution_nodes):
    os.makedirs(args_local.answer_sheets_dir, exist_ok=True)

    save_json(
        [
            {
                "rollout_id": getattr(n, "rollout_id", None),
                "trace": getattr(n, "solution_trace", None),
            }
            for n in model_all_solution_nodes
            if n is not None
        ],
        os.path.join(args_local.answer_sheets_dir, f"Task {task_id} - Final Solutions.json"),
    )
    save_json(
        [
            {
                "rollout_id": i,
                "trace": getattr(n, "solution_trace", None),
                "node_type": getattr(getattr(n, "node_type", None), "value", None),
            }
            for i, n in enumerate(model_rollout_nodes)
            if n is not None
        ],
        os.path.join(args_local.answer_sheets_dir, f"Task {task_id} - Rollout Solutions.json"),
    )
    save_json(
        [
            {
                "rollout_id": i,
                "path": n.solution_trace[0].get("path", None),
                "node_type": getattr(getattr(n, "node_type", None), "value", None),
            }
            for i, n in enumerate(model_rollout_nodes)
            if n is not None and getattr(n, "solution_trace", None)
        ],
        os.path.join(args_local.answer_sheets_dir, f"Task {task_id} - Rollout Path.json"),
    )


def _save_answer_artifact(args_local, task_id, best_answer, best_solution,
                          all_rollout_evaluations, overall_metrics, usage=None):
    """Save the 4th artifact: Answer.json with all candidates and metrics for RL."""
    os.makedirs(args_local.answer_sheets_dir, exist_ok=True)
    payload = {
        "task_id": task_id,
        "selected_best": {
            "answer": best_answer,
            "solution": best_solution,
            "metrics": overall_metrics,
        },
        "all_candidates": all_rollout_evaluations,
    }
    if usage is not None:
        payload["usage"] = usage
    save_json(
        payload,
        os.path.join(args_local.answer_sheets_dir, f"Task {task_id} - Answer.json"),
    )

def generator_thread_or_process(
    j,
    train_path,
    args_local,
    user_question,
    task_id,
    gt_answer,
    generator_info,
    log_queue
):
    """
    Worker function for a single MCTS rollout path.
    Supports both multiprocessing (HuggingFace/vLLM) and multithreading (OpenAI API).
    """

    if isinstance(threading.current_thread(), threading._MainThread):
        worker_id = f"proc_{os.getpid()}"
    else:
        worker_id = f"thread_{threading.get_ident()}"
    logger = get_worker_logger(log_queue, f"worker_{worker_id}")

    # Recreate generator locally if multiprocessing
    if isinstance(generator_info, dict):
        generator = MCTS_Generator(
            args=args_local,
            model_name=generator_info["model_name"],
            tokenizer_name=generator_info["tokenizer_name"]
        )
    else:
        generator = generator_info  # Thread-safe shared instance

    model_solutions, model_all_solutions, model_rollout_nodes, model_best_path, model_all_solution_nodes = (
        [],
        [],
        [],
        [],
        [],
    )
    root_node = None
    mcts_searcher = None

    try:
        if args_local.if_use_cards:
            args_local.max_depth_allowed = len(train_path)
            args_local.num_rollouts = args_local.reuse_rollouts

        # Initialize MCTS searcher
        mcts_searcher = MCTS_Searcher(
            exploration_weight=args_local.mcts_exploration_weight,
            weight_scheduler=args_local.mcts_weight_scheduler,
            num_rollouts=args_local.num_rollouts,
            discount=args_local.mcts_discount_factor,
            verbose=args_local.verbose,
        )

        # Initialize root node
        root_node = Agent_MCTS_Node(
            parent=None,
            depth=0,
            node_type=Node_Type.USER_QUESTION,
            verbose=args_local.verbose,
            generator=generator,
            user_question=user_question,
            max_depth_allowed=args_local.max_depth_allowed,
            disable_rephrase=args_local.disable_rephrase,
            disable_direct_answer=args_local.disable_direct_answer,
            disable_chain_of_thought=args_local.disable_chain_of_thought,
            disable_divide_and_conquer=args_local.disable_divide_and_conquer,
            disable_self_refine=args_local.disable_self_refine,
            if_use_cards=args_local.if_use_cards,
            reasoning_path=train_path,
        )

        # Perform rollouts
        for i in range(args_local.num_rollouts):
            try:
                with call_context(task_id=task_id, rollout=i):
                    logger.info(
                        f"[Task {task_id}] Rollout {i} (path {j}) started."
                    )
                    rollout_node = mcts_searcher.do_rollout(root_node, i)
                    model_rollout_nodes.append(rollout_node)

                    # Select best answer among rollouts
                    if not args_local.disable_answer_selection:
                        _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = stochastic_find_best_solution(
                            root_node,
                            generator.evaluator,
                            enable_potential_score=args_local.enable_potential_score,
                        )
                        if best_solution is None:
                            continue
                        model_solutions.append(best_solution)
                        model_all_solutions.append(all_solutions)
                        if (
                            chosen_node.solution_trace
                            and "path" in chosen_node.solution_trace[0]
                        ):
                            model_best_path.append(chosen_node.solution_trace[0]["path"])
                        model_all_solution_nodes.extend(all_solution_nodes)

                    logger.info(
                        f"[Task {task_id}] Rollout {i} (path {j}) completed."
                    )
                    print(f"{i} completed")
            except Exception as e:
                logger.error(f"Error in rollout {i} (path {j}): {e}")
                traceback.print_exc()

    except Exception as e:
        logger.error(f"Exception in train_path {j}: {e}")
        traceback.print_exc()
    finally:
        worker_usage = generator.get_usage() if hasattr(generator, "get_usage") else None
        if hasattr(generator, "close"):
            generator.close()

    # Collect per-candidate metadata from the final tree
    candidate_info = []
    final_solution_nodes = find_valid_solution_nodes(root_node) if root_node else []
    seen_solutions = set()
    for node in final_solution_nodes:
        sol = extract_solution_from_node(node)
        if sol is None:
            continue
        sol_key = sol.strip() if isinstance(sol, str) else json.dumps(sol, sort_keys=True, ensure_ascii=False)
        if sol_key in seen_solutions:
            continue
        seen_solutions.add(sol_key)
        path = node.solution_trace[0].get("path") if node.solution_trace else None
        candidate_info.append({
            "solution": sol,
            "path": path,
            "node_type": node.node_type.name if node.node_type else None,
            "depth": getattr(node, "depth", None),
            "node_value": getattr(node, "node_value", None),
            "potential_score": getattr(node, "potential_score", None),
            "Q": mcts_searcher.Q.get(node, 0.0) if mcts_searcher else None,
            "N": mcts_searcher.N.get(node, 0) if mcts_searcher else None,
        })

    if args_local.mode != 'eval':
        _save_rollout_artifacts(
            args_local,
            task_id,
            model_rollout_nodes,
            model_all_solution_nodes,
        )

    return (j, model_solutions, model_all_solutions, model_best_path, worker_usage, candidate_info)

def search_for_answers(
    args, 
    evaluator, 
    original_js, 
    user_question: str, 
    task_id: int, 
    gt_answer: str, 
    model_name: str, 
    tokenizer_name:str, 
    log_queue: object
):
    """
    Perform MCTS-based answer search with parallel rollouts.
    Uses threads for OpenAI API and processes for local models (HuggingFace/vLLM).
    """
    logger = get_worker_logger(log_queue, f"search_task_{task_id}")
    logger.info(f"Starting search for task {task_id}: {user_question}")

    # Prepare train paths
    if args.if_use_cards:
        try:
            train_paths = args.reason_structure[user_question][: args.num_cards]
            train_paths = [
                list(ast.literal_eval(x)) if not isinstance(x, list) else x
                for x in train_paths
            ]
        except KeyError as e:
            logger.error("Key not found in distilled paths")
    else:
        train_paths = [None]

    print(train_paths)

    # Select executor type
    if args.api in ["huggingface", "vllm"]:
        Executor = ProcessPoolExecutor
        generator_info = {
            "model_name": args.model_ckpt,
            "tokenizer_name": args.tokenizer_ckpt or args.model_ckpt
        }
        logger.info("Using multiprocessing executor for local inference.")
    else:
        Executor = ThreadPoolExecutor
        generator_info = MCTS_Generator(args, args.model_ckpt, None)
        logger.info("Using multithreading executor for API-based inference.")

    # Run rollouts in parallel
    path_threads = 2
    results = []

    with Executor(max_workers=path_threads) as executor:
        futures = [
            executor.submit(
                generator_thread_or_process,
                j,
                train_path,
                deepcopy(args),
                user_question,
                task_id,
                gt_answer,
                generator_info,
                log_queue
            )
            for j, train_path in enumerate(train_paths)
        ]

        try:
            for future in as_completed(futures):
                results.append(future.result())
        except Exception as e:
            logger.error(f"Worker crashed: {e}")
            traceback.print_exc()

    # Sort and aggregate results
    results.sort(key=lambda x: x[0])
    model_solutions, model_all_solutions, model_best_path = [], [], []

    aggregated_usage = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_candidate_info = []
    for _, sols, all_sols, paths, worker_usage, cand_info in results:
        model_solutions.extend(sols)
        model_all_solutions.extend(all_sols)
        model_best_path.extend(paths)
        if worker_usage:
            for k in aggregated_usage:
                aggregated_usage[k] += worker_usage.get(k, 0)
        all_candidate_info.extend(cand_info)

    logger.info(f"Collected {len(model_solutions)} solutions, {len(all_candidate_info)} unique candidates.")

    # Default evaluation results
    correct = tools_correct = parameters_correct = dependencies_correct = None

    # ── Evaluate every unique candidate individually ──
    all_candidates_evaluated = []
    if args.mode != "eval":
        for cid, cand in enumerate(all_candidate_info):
            cand_answer = evaluator.extract_answer_from_model_completion(cand["solution"])
            if cand_answer:
                c_correct = evaluator.check_answers_equivalence(cand_answer, gt_answer)
                c_tools = evaluator.check_tools_correctness(cand_answer, gt_answer)
                c_params = evaluator.check_parameters_correctness(cand_answer, gt_answer)
                c_deps = evaluator.check_dependencies_correctness(cand_answer, gt_answer)
            else:
                c_correct = c_tools = c_params = c_deps = False

            all_candidates_evaluated.append({
                "candidate_id": cid,
                "solution": cand.get("solution"),
                "model_answer": cand_answer,
                "model_path": cand.get("path"),
                "node_type": cand.get("node_type"),
                "depth": cand.get("depth"),
                "node_value": cand.get("node_value"),
                "potential_score": cand.get("potential_score"),
                "Q": cand.get("Q"),
                "N": cand.get("N"),
                "correct": c_correct,
                "tools_correct": c_tools,
                "parameters_correct": c_params,
                "dependencies_correct": c_deps,
            })

    # ── Select best answer ──
    if not args.disable_answer_selection and model_solutions:
        selector = getattr(args, "answer_selector", "majority") or "majority"
        if selector == "majority":
            best_answer, best_solution, _, _ = evaluator.stochastic_find_most_confident_answer(model_solutions)
        else:
            tools_path = os.path.join(args.tools_dir, "tools_manifest.json")
            ctx = load_tools_manifest(tools_path)
            chosen = select_answer(selector, all_candidates_evaluated, ctx)
            if chosen is None:
                best_answer, best_solution, _, _ = evaluator.stochastic_find_most_confident_answer(model_solutions)
                logger.warning(
                    f"Task {task_id}: answer_selector={selector} returned None; fell back to majority."
                )
            else:
                best_answer = chosen.get("model_answer")
                best_solution = chosen.get("solution")
                if best_solution is None and best_answer is not None:
                    # Fallback: recover a matching completion string
                    for sol in model_solutions:
                        try:
                            if evaluator.extract_answer_from_model_completion(sol) == best_answer:
                                best_solution = sol
                                break
                        except Exception:
                            continue
                    if best_solution is None:
                        best_solution = best_answer
                logger.info(
                    f"Task {task_id}: selected via {selector} "
                    f"(candidate_id={chosen.get('candidate_id')}, depth={chosen.get('depth')})"
                )

        if args.mode != 'eval':
            correct = evaluator.check_answers_equivalence(best_answer, gt_answer)
            tools_correct = evaluator.check_tools_correctness(best_answer, gt_answer)
            parameters_correct = evaluator.check_parameters_correctness(best_answer, gt_answer)
            dependencies_correct = evaluator.check_dependencies_correctness(best_answer, gt_answer)

            logger.info(f"GT {task_id}: {gt_answer}")
            logger.info(f"Best {task_id}: {best_answer}")
            logger.info(
                f"Final evaluation for task {task_id}: Correct={correct}, Tools={tools_correct}, "
                f"Params={parameters_correct}, Deps={dependencies_correct}"
            )

            overall_metrics = {
                "correct": correct,
                "tools_correct": tools_correct,
                "parameters_correct": parameters_correct,
                "dependencies_correct": dependencies_correct,
            }

            _save_answer_artifact(
                args, task_id, best_answer, best_solution,
                all_candidates_evaluated, overall_metrics,
                usage=aggregated_usage,
            )

            original_js["model_completion"] = best_solution
            original_js["model_answer"] = best_answer
            original_js["model_all_answer"] = model_all_solutions

            return (
                original_js,
                model_solutions,
                model_all_solutions,
                model_best_path,
                correct,
                tools_correct,
                parameters_correct,
                dependencies_correct,
            )
        else:
            return best_answer, best_solution

    if args.mode != "eval":
        _save_answer_artifact(
            args, task_id,
            best_answer=None, best_solution=None,
            all_rollout_evaluations=all_candidates_evaluated,
            overall_metrics={
                "correct": False, "tools_correct": False,
                "parameters_correct": False, "dependencies_correct": False,
                "failure_reason": "No valid structured_plan solution extracted from any rollout.",
            },
            usage=aggregated_usage,
        )

        original_js["failure_reason"] = "No valid structured_plan solution extracted from any rollout."
        original_js["model_completion"] = None
        original_js["model_answer"] = None
        original_js["model_all_answer"] = model_all_solutions
        original_js["all_model_completions"] = original_js.get("all_model_completions", {})
        logger.warning(
            f"Task {task_id} produced no valid solutions. Rollout traces were still saved."
        )
        return (
            original_js,
            model_solutions,
            model_all_solutions,
            model_best_path,
            False,
            False,
            False,
            False,
        )
