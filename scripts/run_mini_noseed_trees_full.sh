#!/usr/bin/env bash
# GPT-4o-mini CoT-tree (CoT-only-M) and MO-Tree (Five-M) on all 183 test tasks without a per-request
# seed. The commands are the locked August ones (scripts/run_phase2_mini_fulltest.sh); the only change
# is OPENROUTER_SEND_SEED=0. --seed 20260808 still seeds Python-side RNG and the run path.
# Every API request is logged with provenance (task_id, operator, prompt hash, sent parameters).
#
# Usage: bash scripts/run_mini_noseed_trees_full.sh            # both, in parallel
#        METHODS=cot bash scripts/run_mini_noseed_trees_full.sh
#        TASK_IDS_FILE=data/x.txt OUT_TAG=smoke bash scripts/run_mini_noseed_trees_full.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
set -a; source .env 2>/dev/null || true; set +a
PY="${PYTHON:-/usr/bin/python3}"
if ! curl -sf -o /dev/null -H "Authorization: Bearer ${OPENROUTER_API_KEY:-${OpenRouter_API_KEY:-}}" \
    https://openrouter.ai/api/v1/key; then
  echo "OpenRouter API key check failed; aborting." >&2
  exit 1
fi

METHODS="${METHODS:-cot five}"
THREADS="${THREADS:-8}"
TAG="${OUT_TAG:-full}"
OUT="artifacts/mini_noseed_trees_$TAG"
LOGD="logs/calllogs_mini_noseed_trees_$TAG"
mkdir -p "$OUT" "$LOGD"
TASKS=()
[[ -n "${TASK_IDS_FILE:-}" ]] && TASKS=(--task_ids_file "$TASK_IDS_FILE")
mini=(--api openrouter --model_ckpt openai/gpt-4o-mini --train_test_json_data test.jsonl --disable_wandb --temperature 0.7)

COT_PAT="artifacts/mcts/gpt-4o-mini/test_|_rolls_1_|_reuse_train_False_|_re_False_|_da_False_n_10_|_cot_True_|_dc_False_|_sr_False_|_all-mpnet-base-v2/seed_20260808"
FIVE_PAT="artifacts/mcts/gpt-4o-mini/test_|_rolls_2_|_reuse_train_False_|_re_True_|_da_True_n_10_|_cot_True_|_dc_True_|_sr_True_|_all-mpnet-base-v2/seed_20260808"
list_dirs() { ls -d "$1"/*/ 2>/dev/null | sed 's:/*$::' | sort || true; }

start() {
  local m="$1" key pat before dir
  if [[ "$m" == cot ]]; then
    key=CoT-only-M; pat="$COT_PAT"
    args=(--num_rollouts 1 --num_chain_of_thought 32
          --disable_rephrase true --disable_direct_answer true
          --disable_chain_of_thought false --disable_divide_and_conquer true
          --disable_self_refine true)
  else
    key=Five-M; pat="$FIVE_PAT"
    args=(--num_rollouts 2 --num_direct_answer 10)
  fi
  before=$(list_dirs "$pat")
  OPENROUTER_SEND_SEED=0 OPENROUTER_MAX_ATTEMPTS=8 OPENROUTER_LOG_METHOD="$key" \
  OPENROUTER_CALL_LOG="$LOGD/$m.jsonl" \
  "$PY" main.py --mode mcts --seed 20260808 "${args[@]}" \
    --max_threads "$THREADS" "${mini[@]}" "${TASKS[@]}" \
    > "logs/mini_noseed_${m}_$TAG.log" 2>&1 &
  PIDS+=($!)
  dir=""
  for _ in $(seq 1 120); do
    dir=$(comm -13 <(echo "$before") <(list_dirs "$pat") | tail -1)
    [[ -n "$dir" ]] && break
    sleep 2
  done
  printf '{"model":"gpt-4o-mini","key":"%s","seed":20260808,"send_seed":false,"dir":"%s"}\n' \
    "$key" "$dir" > "$OUT/run_$m.json"
  echo "[$(date -Iseconds)] STARTED $m ($key) -> $dir"
}

PIDS=()
for m in $METHODS; do start "$m"; done
wait "${PIDS[@]}"
echo "[$(date -Iseconds)] NOSEED_TREES_DONE ($METHODS)"
