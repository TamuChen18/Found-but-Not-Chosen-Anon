#!/usr/bin/env python3
"""
Prompt-matched BoN (CoT-prompt BoN) on GPT-4o-mini.

Per task: the CoT-tree root expansion prompt (PromptBuilder.build_chain_of_thought_prompt with
paraphrased=False, hint=None, i.e. exactly what MCTS_Generator._fewshot_cot_answer_question sends
from the USER_QUESTION root), sampled N times as independent n=1 requests with no tree state.
Before any request, every prompt's sha256 is checked against the CoT-tree call log so the prompt
is byte-identical to the one the no-seed CoT-tree run used.

Answer sheets follow the BoN layout (all_candidates with candidate_id / model_answer /
raw_completion, plus usage). model_answer = Evaluator.extract_answer_from_model_completion(raw),
the same extraction the tree applies to its leaves (no legacy online postprocess).

Usage:
  set -a; source .env; set +a
  OPENROUTER_SEND_SEED=0 OPENROUTER_CALL_LOG=logs/calllogs_prompt_matched_bon/pmbon.jsonl \
  OPENROUTER_LOG_METHOD=PM-BoN python3 scripts/run_prompt_matched_bon.py --out_dir artifacts/prompt_matched_bon/run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from evaluators.evaluators import Evaluator  # noqa: E402
from interfaces.OpenRouter_client import OpenRouterClient  # noqa: E402
from interfaces.call_context import call_context  # noqa: E402
from prompts.prompt_builder import PromptBuilder  # noqa: E402
from utils.agent_utils import _JSON_ARRAY_RE  # noqa: E402

MODEL = "openai/gpt-4o-mini"
GEN = {"max_tokens": 2048, "temperature": 0.7, "top_p": 0.95}


def ref_hashes(path: Path):
    out = {}
    for line in path.open():
        o = json.loads(line)
        if o.get("operator") == "CHAIN_OF_THOUGHT" and o.get("task_id") is not None:
            out.setdefault(int(o["task_id"]), set()).add(o["prompt_sha256"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts/prompt_matched_bon/run")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--task_ids_file", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ref_calllog", default="logs/calllogs_mini_noseed_trees_full/cot.jsonl",
                    help="CoT-tree call log with prompt_sha256 to verify prompts against; '' to skip")
    ap.add_argument("--verify_only", action="store_true")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open("data/test.jsonl") if l.strip()]
    if args.task_ids_file:
        keep = {int(x) for x in Path(args.task_ids_file).read_text().split() if x.strip().isdigit()}
        tasks = [t for t in tasks if int(t["task_id"]) in keep]

    pb = PromptBuilder("src/prompts", "src/interfaces/tools", True)
    refs = ref_hashes(Path(args.ref_calllog)) if args.ref_calllog else None
    prompts, bad = {}, []
    for t in tasks:
        tid = int(t["task_id"])
        p = pb.build_chain_of_thought_prompt(t["task_desc"], False, None)
        h = hashlib.sha256(p.encode("utf-8")).hexdigest()
        if refs is not None and refs.get(tid) != {h}:
            bad.append(tid)
        prompts[tid] = p
    if refs is not None:
        print(f"prompt hash check: {len(tasks) - len(bad)}/{len(tasks)} match CoT-tree call log", flush=True)
    if bad:
        sys.exit(f"prompt mismatch for task_ids {bad[:20]}")
    if args.verify_only:
        return

    out = Path(args.out_dir) / "answer_sheets"
    out.mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "config.json").write_text(json.dumps({
        "method": "prompt_matched_bon", "model": args.model, "n": args.n, **GEN, "seed": None,
        "send_seed": os.getenv("OPENROUTER_SEND_SEED"), "enable_thinking": os.getenv("OPENROUTER_ENABLE_THINKING"),
        "base_url": os.getenv("OPENROUTER_BASE_URL"), "prompt": "build_chain_of_thought_prompt(task_desc, False, None)",
        "ref_calllog": args.ref_calllog,
    }, indent=2))
    ev = Evaluator()
    lock = threading.Lock()
    done = [0]

    def run(tid):
        path = out / f"Task {tid} - Answer.json"
        if path.exists():
            return tid, "skip"
        client = OpenRouterClient(args.model)
        cands = []
        for i in range(args.n):
            with call_context(task_id=tid, operator="CHAIN_OF_THOUGHT", sample=i):
                text = (client.generate(prompts[tid], **GEN) or "").strip()
            has_arr = bool(text) and bool(_JSON_ARRAY_RE.findall(text))
            cands.append({
                "candidate_id": i,
                "model_answer": ev.extract_answer_from_model_completion(text) if text else None,
                "raw_completion": text,
                "has_json_array": has_arr,
            })
        sheet = {"task_id": tid, "method": "prompt_matched_bon", "all_candidates": cands,
                 "usage": client.get_usage()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sheet, ensure_ascii=False))
        tmp.replace(path)
        with lock:
            done[0] += 1
            print(f"task {tid} done ({done[0]}) tokens={sheet['usage'].get('total_tokens')}", flush=True)
        return tid, "ok"

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(run, int(t["task_id"])) for t in tasks]
        for f in as_completed(futs):
            f.result()
    print("finished", len(list(out.glob("Task * - Answer.json"))), "sheets", flush=True)


if __name__ == "__main__":
    main()
