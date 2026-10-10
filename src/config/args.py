# Portions of this file are adapted from rStar (https://github.com/zhentingqi/rStar),
# Copyright (c) 2024 rStar Team, and from ThoughtICR (https://github.com/jinyangwu/ThoughtICR),
# Copyright (c) 2025 Jinyang Wu. Both are licensed under the MIT License; see THIRD_PARTY_NOTICES.md,
# licenses/rStar-LICENSE and licenses/ThoughtICR-LICENSE.
import os
from datetime import datetime
from argparse import ArgumentParser
from utils.common_utils import save_json

def parse_args():
    """Function to parse command line arguments.
    Returns:
        Namespace: Parsed command line arguments.
    """
    
    parser = ArgumentParser()
    
    #general arguments
    parser.add_argument("--mode", type=str, default="mcts", choices=["mcts", "baseline"])
    parser.add_argument("--max_threads", type=int, default=32, help="Maximum number of threads per process")
    parser.add_argument("--if_use_cards", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--task_num", type=int, default=0)
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=0,
        help="Run only the first N tasks (0 = all). Ignored when task_num != 0.",
    )
    parser.add_argument(
        "--task_ids_file",
        type=str,
        default=None,
        help="Text/JSON file of task_ids to run (one id per line, or judgment_set JSON with task_ids).",
    )
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--train_test_json_data", type=str, default="train.jsonl", choices=["train.jsonl", "test.jsonl"])
    parser.add_argument("--verbose", type=lambda x: (str(x).lower() == 'true'), default=False)

    #model arguments
    allowed_apis = ["openai", "openrouter", "huggingface", "vllm"]
    parser.add_argument("--api", type=str, choices=allowed_apis, default="openai", help=f"API to use: Choose from {allowed_apis}.")
    parser.add_argument("--model_ckpt", type=str, default="gpt-4o-mini", help="path of the model to use")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None, help="path of the tokenizer to use")
    parser.add_argument("--sim_model", default="sentence-transformers/all-mpnet-base-v2") 
    parser.add_argument("--max_tokens", type=int, default=2048, help="max_tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--top_k", type=int, default=20, help="top_k")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG + API sampling seed. When set, included in run output path.",
    )

    #directory arguments
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save model artifacts")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory to save model outputs")
    parser.add_argument("--prompts_dir", type=str, default="src/prompts", help="Directory to load prompts")
    parser.add_argument("--tools_dir", type=str, default="src/interfaces/tools", help="Directory to load tools")

    #mcts generator arguments
    parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts for MCTS")
    parser.add_argument("--max_depth_allowed", type=int, default=5)
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)  
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const") 

    #reasoning arguments
    parser.add_argument("--disable_rephrase", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 1: rephrase")
    parser.add_argument("--disable_direct_answer", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 2: direct answer")
    parser.add_argument("--disable_chain_of_thought", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 3: chain-of-thought")
    parser.add_argument("--disable_divide_and_conquer", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 4: divide and conquer")
    parser.add_argument("--disable_self_refine", type=lambda x: (str(x).lower() == 'true'), default=False, help="action 5: self-reflection and refinement")

    parser.add_argument("--num_direct_answer", type=int, default=10)
    parser.add_argument("--num_chain_of_thought", type=int, default=16)
    parser.add_argument(
        "--cot_expand_all_samples",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="If true, each CoT sample becomes its own MCTS leaf (independent API calls). "
        "Auto-enabled when only CoT action is on (CoT-only MCTS).",
    )
    parser.add_argument("--num_divide_and_conquer", type=int, default=10, help="Number of trials for 'divide and conquer'.")
    parser.add_argument("--num_divide_and_conquer_votes", type=int, default=10, help="Number of trials for subquestions of each question.")
    parser.add_argument("--use_fewshot", type=lambda x: (str(x).lower() == 'true'), default=True)

    parser.add_argument("--enable_potential_score", default=False)
    parser.add_argument("--disable_answer_selection", default=False)
    parser.add_argument(
        "--answer_selector",
        type=str,
        default="majority",
        choices=["majority", "filter_then_majority", "filter_then_short", "filter_then_nodup"],
        help="How to pick selected_best among tree candidates. "
        "'majority' = legacy exact-JSON vote; filter_* = gold-free hard-filter strategies.",
    )

    # eval mode args
    parser.add_argument(
        "--test_type",
        choices=["cot", "bon_cot"],
        default="cot",
    )
    parser.add_argument(
        "--num_bon_samples",
        type=int,
        default=0,
        help="N for bon / bon_cot (0 = use num_rollouts).",
    ) 

    # TOT args
    parser.add_argument("--num_generate_sample", type=int, default=7)
    parser.add_argument("--num_evaluate_sample", type=int, default=5)
    parser.add_argument("--n_select_sample", type=int, default=1)

    # RAP args
    parser.add_argument("--n_sample_subquestion", type=int, default=16)
    parser.add_argument("--n_sample_confidence", type=int, default=32)
    parser.add_argument("--w_exp", type=int, default=1)
    parser.add_argument("--r_alpha", type=int, default=0.5)
    parser.add_argument("--r1_default", type=int, default=1)

    # Distillation args
    parser.add_argument("--k", type=float, default=0.8)
    parser.add_argument("--attribute_type", type=str, default='semantic', choices=['condition', 'semantic', 'rerank'])
    parser.add_argument("--similarity_type", type=str, default='average', choices=['average', 'max'])
    parser.add_argument("--structure_dir", type=str, default="structure")
    parser.add_argument("--file_dir", type=str, default="artifacts/mcts/")

    #Distilled MCTS args
    parser.add_argument("--reuse_dir", type=str, default="artifacts/structure/")
    parser.add_argument("--reuse_rollouts", type=int, default=1)
    parser.add_argument("--num_cards", type=int, default=10)

    return parser.parse_args()

def post_process_args(args):
    model_name = args.model_ckpt.split("/")[-1]
    sim_model = args.sim_model.split("/")[-1]
    args.file = args.train_test_json_data.split(".")[0]
    suffix = f"{args.file}_|_rolls_{args.num_rollouts}_|_reuse_train_{args.if_use_cards}_|_re_{not args.disable_rephrase}_|_da_{not args.disable_direct_answer}_n_{args.num_direct_answer}_|_cot_{not args.disable_chain_of_thought}_|_dc_{not args.disable_divide_and_conquer}_|_sr_{not args.disable_self_refine}_|_{sim_model}"
    
    # Ensure directories exist
    args.artifacts_dir = os.path.join(args.artifacts_dir)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    if args.mode == "mcts":
        stamp = f"{datetime.now().strftime('%m-%d_%H-%M')}"
        if getattr(args, "seed", None) is not None:
            stamp = os.path.join(f"seed_{args.seed}", stamp)
        args.run_outputs_dir = os.path.join(
            args.artifacts_dir,
            args.mode,
            model_name,
            suffix,
            stamp,
        )
        os.makedirs(args.run_outputs_dir, exist_ok=True)
        args.logdir = os.path.join(args.run_outputs_dir, "mcts_logs.log")
        args.answer_sheets_dir = os.path.join(args.run_outputs_dir, "answer_sheets")
        os.makedirs(args.answer_sheets_dir, exist_ok=True)
        save_json(vars(args), os.path.join(args.run_outputs_dir, "config.json"))

    elif args.mode == "baseline":
        stamp = f"{datetime.now().strftime('%m-%d_%H-%M')}"
        if getattr(args, "seed", None) is not None:
            stamp = os.path.join(f"seed_{args.seed}", stamp)
        args.run_outputs_dir = os.path.join(
            args.artifacts_dir,
            args.mode,
            model_name,
            args.test_type,
            suffix,
            stamp,
        )
        os.makedirs(args.run_outputs_dir, exist_ok=True)
        args.answer_sheets_dir = os.path.join(args.run_outputs_dir, "answer_sheets")
        os.makedirs(args.answer_sheets_dir, exist_ok=True)
        # args.train_test_json_data is controlled by CLI (default: train.jsonl)
        if args.test_type == "cot":
            args.num_chain_of_thought = 1 
        args.logdir = os.path.join(args.run_outputs_dir, f"{args.mode}_logs.log")
        save_json(vars(args), os.path.join(args.run_outputs_dir, "config.json"))

    else:
        raise ValueError(f"Invalid mode: {args.mode}")       

    return args
