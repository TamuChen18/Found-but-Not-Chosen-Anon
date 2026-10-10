#!/usr/bin/env bash
# GPT-4o-mini BoN-32 on all 183 test tasks. Every setting is the locked main-run command
# (scripts/run_phase2_mini_fulltest.sh): same prompt, few-shot, manifest, temperature 0.7,
# max_tokens 2048 (args default), --seed 20260808. Only the per-request seed is switched:
#   SEND_SEED=0 (default): no seed is sent             -> artifacts/mini_noseed_bon_full
#   SEND_SEED=1: seed sent, as in the August main run   -> artifacts/mini_seeded_bon_rerun
#
# Usage: bash scripts/run_mini_noseed_bon_full.sh ; SEND_SEED=1 bash scripts/run_mini_noseed_bon_full.sh
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

SEND_SEED="${SEND_SEED:-0}"
if [[ "$SEND_SEED" == "1" ]]; then TAG=seeded_bon_rerun; KEY=BoN-CoT-M-seeded-rerun; SENT=true
else TAG=noseed_bon_full; KEY=BoN-CoT-M-noseed; SENT=false; fi
OUT=artifacts/mini_$TAG
LOGD=logs/calllogs_mini_$TAG
[[ "$SEND_SEED" == "1" ]] || LOGD=logs/calllogs_mini_noseed_full
THREADS="${THREADS:-8}"
mkdir -p "$OUT" "$LOGD"
BON_PAT="artifacts/baseline/gpt-4o-mini/bon_cot/test_|_rolls_32_|_reuse_train_False_|_re_True_|_da_True_n_10_|_cot_True_|_dc_True_|_sr_True_|_all-mpnet-base-v2/seed_20260808"
list_dirs() { ls -d "$BON_PAT"/*/ 2>/dev/null | sed 's:/*$::' | sort || true; }

before=$(list_dirs)
OPENROUTER_SEND_SEED="$SEND_SEED" OPENROUTER_MAX_ATTEMPTS=8 \
OPENROUTER_CALL_LOG="$LOGD/bon32.jsonl" \
"$PY" main.py --mode baseline --test_type bon_cot --seed 20260808 \
  --num_bon_samples 32 --num_rollouts 32 \
  --max_threads "$THREADS" \
  --api openrouter --model_ckpt openai/gpt-4o-mini --train_test_json_data test.jsonl \
  --disable_wandb --temperature 0.7 \
  > "logs/mini_$TAG.log" 2>&1 &
PID=$!
dir=""
for _ in $(seq 1 120); do
  dir=$(comm -13 <(echo "$before") <(list_dirs) | tail -1)
  [[ -n "$dir" ]] && break
  sleep 2
done
printf '{"model":"gpt-4o-mini","key":"%s","seed":20260808,"send_seed":%s,"dir":"%s"}\n' "$KEY" "$SENT" "$dir" > "$OUT/run.json"
echo "[$(date -Iseconds)] STARTED pid=$PID send_seed=$SEND_SEED -> $dir"
wait "$PID"
echo "[$(date -Iseconds)] BON_DONE send_seed=$SEND_SEED"
