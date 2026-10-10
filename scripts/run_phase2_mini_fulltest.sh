#!/usr/bin/env bash
# Phase 2 confirmation: GPT-4o-mini FULL test set (~183).
# Four configs: CoT-only / Five-M / Five-H24 (for r*=0.75) / BoN-CoT
# Default: 1 seed probe. Do NOT expand to other models until mini result is clear.
#
# Usage:
#   bash scripts/run_phase2_mini_fulltest.sh
#   SEEDS="20260808 20260809" bash scripts/run_phase2_mini_fulltest.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
set -a; source .env 2>/dev/null || true; set +a
PY="${PYTHON:-/usr/bin/python3}"
SEEDS="${SEEDS:-20260808}"
THREADS="${THREADS:-4}"
N=32
# Locked H24 from mini real-allocation calib (r*=0.75)
H24_ROLLS=2 H24_COT=4 H24_DA=2 H24_DC=2 H24_DEPTH=3
MANIFEST_DIR=artifacts/phase2_fulltest
mkdir -p "$MANIFEST_DIR" logs
MANIFEST="$MANIFEST_DIR/manifest_mini.json"

latest_seed_run() {
  local pattern="$1" seed="$2"
  # Quote paths: artifact dirs contain literal '|' characters.
  ls -d "${pattern}/seed_${seed}"/*/ 2>/dev/null | sort | tail -1 | sed 's:/*$::'
}

n_sheets() {
  local dir="$1"
  ls "$dir"/answer_sheets/Task\ *\ -\ Answer.json 2>/dev/null | wc -l | tr -d ' '
}

append_manifest() {
  local key="$1" dir="$2" seed="$3" note="$4"
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("$MANIFEST")
data = json.loads(p.read_text()) if p.exists() else {"runs": [], "protocol": "fulltest_confirmation_v1"}
data["runs"] = [r for r in data.get("runs", []) if not (r.get("key")=="$key" and str(r.get("seed"))==str("$seed"))]
data["runs"].append({
    "model": "gpt-4o-mini",
    "key": "$key",
    "seed": int("$seed"),
    "dir": "$dir",
    "n_sheets": int("$(n_sheets "$dir")"),
    "note": "$note",
})
p.write_text(json.dumps(data, indent=2))
print("manifest+", "$key", "$seed", "$dir", "n=$(n_sheets "$dir")")
PY
}

mini=(--api openrouter --model_ckpt openai/gpt-4o-mini --train_test_json_data test.jsonl --disable_wandb --temperature 0.7)

for seed in $SEEDS; do
  echo "[$(date -Iseconds)] === FULLTEST mini seed=$seed START (no task_ids_file = all test.jsonl) ===" | tee -a logs/phase2_driver.log

  echo "[$(date -Iseconds)] BoN-$N seed=$seed"
  "$PY" main.py --mode baseline --test_type bon_cot --seed "$seed" \
    --num_bon_samples "$N" --num_rollouts "$N" \
    --max_threads "$THREADS" "${mini[@]}" \
    2>&1 | tee "logs/phase2_mini_bon_seed${seed}.log"
  append_manifest "BoN-CoT-M" "$(latest_seed_run artifacts/baseline/gpt-4o-mini/bon_cot/test_|_rolls_${N}_|_reuse_train_False_|_re_True_|_da_True_n_10_|_cot_True_|_dc_True_|_sr_True_|_all-mpnet-base-v2 "$seed")" "$seed" "n=$N"

  echo "[$(date -Iseconds)] CoT-only-$N seed=$seed"
  "$PY" main.py --mode mcts --seed "$seed" --num_rollouts 1 --num_chain_of_thought "$N" \
    --disable_rephrase true --disable_direct_answer true \
    --disable_chain_of_thought false --disable_divide_and_conquer true \
    --disable_self_refine true \
    --max_threads "$THREADS" "${mini[@]}" \
    2>&1 | tee "logs/phase2_mini_cotonly_seed${seed}.log"
  append_manifest "CoT-only-M" "$(latest_seed_run artifacts/mcts/gpt-4o-mini/test_|_rolls_1_|_reuse_train_False_|_re_False_|_da_False_n_10_|_cot_True_|_dc_False_|_sr_False_|_all-mpnet-base-v2 "$seed")" "$seed" "n=$N"

  echo "[$(date -Iseconds)] Five-H24 (r*=0.75 hetero slice) seed=$seed"
  "$PY" main.py --mode mcts --seed "$seed" \
    --num_rollouts "$H24_ROLLS" --num_chain_of_thought "$H24_COT" \
    --num_direct_answer "$H24_DA" --num_divide_and_conquer "$H24_DC" \
    --max_depth_allowed "$H24_DEPTH" \
    --max_threads "$THREADS" "${mini[@]}" \
    2>&1 | tee "logs/phase2_mini_h24_seed${seed}.log"
  append_manifest "Five-H24" "$(latest_seed_run artifacts/mcts/gpt-4o-mini/test_|_rolls_${H24_ROLLS}_|_reuse_train_False_|_re_True_|_da_True_n_${H24_DA}_|_cot_True_|_dc_True_|_sr_True_|_all-mpnet-base-v2 "$seed")" "$seed" "r_star=0.75 lock=H24b"

  echo "[$(date -Iseconds)] Five-M seed=$seed"
  "$PY" main.py --mode mcts --seed "$seed" --num_rollouts 2 --num_direct_answer 10 \
    --max_threads "$THREADS" "${mini[@]}" \
    2>&1 | tee "logs/phase2_mini_five_seed${seed}.log"
  append_manifest "Five-M" "$(latest_seed_run artifacts/mcts/gpt-4o-mini/test_|_rolls_2_|_reuse_train_False_|_re_True_|_da_True_n_10_|_cot_True_|_dc_True_|_sr_True_|_all-mpnet-base-v2 "$seed")" "$seed" "rolls=2 da=10"

  echo "[$(date -Iseconds)] FULLTEST_SEED_DONE $seed" | tee -a logs/phase2_driver.log
  "$PY" scripts/summarize_phase2_fulltest.py || true
done

echo "[$(date -Iseconds)] PHASE2_MINI_DONE" | tee -a logs/phase2_driver.log
"$PY" scripts/summarize_phase2_fulltest.py || true
