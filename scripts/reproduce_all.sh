#!/usr/bin/env bash
# Rebuild every number, table and figure of the paper from the raw candidate pools
# (download the data package into runs/, artifacts/ and logs/ first; see README).
# Offline: GPT-4o judge / Top-3 verifier / Top-3 selector decisions are replayed from their caches;
# any cache miss fails instead of calling an API (unreachable base URL, placeholder key).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
export OPENROUTER_API_KEY=offline-replay
export OPENROUTER_BASE_URL="http://127.0.0.1:9/v1"
export OPENROUTER_MAX_ATTEMPTS=1
export SKIP_JUDGE=0
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1791763200}"
[[ -f data/test.jsonl && -f third_party/DisasterBench_Open/evaluators/evaluators.py ]] || "$PY" scripts/prepare_disasterbench.py

MMAN=artifacts/mini_noseed_trees_full/manifest_allnoseed.json
AMAN=artifacts/phase2_fulltest/manifest_mini.json
QMAN=artifacts/qwen_nothink_main183/manifest.json
E=artifacts/e2b_off_primary

echo "== [1/5] candidate bundles (strict postprocess, hard0, exact-JSON majority)"
# final primary pools, E2b off
"$PY" scripts/freeze_strict_matched.py --keep_multi_pred --manifest "$MMAN" --out artifacts/mini_noseed_trees_full/eval_e2b_off
"$PY" scripts/freeze_strict_matched.py --keep_multi_pred --manifest "$QMAN" --seed 20260725 --out "$E/qwen_b1"
"$PY" scripts/freeze_strict_matched.py --keep_multi_pred --manifest "$QMAN" --seed 20260726 --out "$E/qwen_b2"
# sensitivity bundles: E2b on, and the earlier fixed-seed GPT-4o-mini pools
"$PY" scripts/freeze_strict_matched.py --manifest "$MMAN" --out artifacts/mini_noseed_trees_full/eval_e2b_on
"$PY" scripts/freeze_strict_matched.py --keep_multi_pred --manifest "$AMAN" --seed 20260808 --out "$E/mini"
"$PY" scripts/freeze_strict_matched.py --manifest "$AMAN" --seed 20260808 --out artifacts/phase2_fulltest/FINAL_LOCKED/review_bundle
"$PY" scripts/freeze_strict_matched.py --manifest "$QMAN" --seed 20260725 --out artifacts/qwen_nothink_main183/eval/b20260725
"$PY" scripts/freeze_strict_matched.py --manifest "$QMAN" --seed 20260726 --out artifacts/qwen_nothink_main183/eval/b20260726

echo "== [2/5] E2b-off bundle diagnostics and cached judge (Qwen + fixed-seed GPT-4o-mini)"
"$PY" scripts/analyze_pool_concentration.py --bundle mini="$E/mini" --bundle qwen_b1="$E/qwen_b1" --bundle qwen_b2="$E/qwen_b2" --out "$E/concentration"
"$PY" scripts/analyze_vote_multiplicity.py --keep_multi_pred --manifest "mini=$AMAN:20260808:$E/mini" \
  --manifest "qwen_b1=$QMAN:20260725:$E/qwen_b1" --manifest "qwen_b2=$QMAN:20260726:$E/qwen_b2" --out "$E/vote_multiplicity"
"$PY" scripts/run_locked_judge.py --bundle mini="$E/mini" --bundle qwen_b1="$E/qwen_b1" --bundle qwen_b2="$E/qwen_b2" --out "$E/judge"

echo "== [3/5] seed sensitivity (no-seed vs fixed-seed GPT-4o-mini)"
"$PY" scripts/analyze_noseed_bon.py --rerun_dir "$("$PY" -c "import json;print(json.load(open('artifacts/mini_seeded_bon_rerun/run.json'))['dir'])")"
"$PY" scripts/analyze_noseed_trees.py

echo "== [4/5] Top-3 verifier and deployable selector (cached GPT-4o decisions)"
"$PY" scripts/run_top3_verifier.py
"$PY" scripts/run_top3_selector.py

echo "== [5/5] final primary analyses, tables and figures"
bash scripts/run_final_primary_offline.sh
"$PY" scripts/audit_historical_tables.py
"$PY" scripts/tally_vote_loss_audit.py --csv artifacts/vote_loss_audit_final/audit_sample_final.csv

echo "== check against the numbers reported in the paper"
"$PY" scripts/check_paper_numbers.py
echo "done"
