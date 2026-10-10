#!/usr/bin/env bash
# Offline analyses on the final primary pools (E4 off, E2b off):
#   GPT-4o-mini  all-no-seed BoN / CoT-tree / MO-Tree  (artifacts/mini_noseed_trees_full/manifest_allnoseed.json)
#   Qwen         existing b20260725 / b20260726 pools  (artifacts/qwen_nothink_main183/manifest.json)
# Outputs go to artifacts/final_primary/. Only the GPT-4o judge step calls an API (SKIP_JUDGE=1 skips it).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
O=artifacts/final_primary
MINI=artifacts/mini_noseed_trees_full/eval_e2b_off
MMAN=artifacts/mini_noseed_trees_full/manifest_allnoseed.json
Q1=artifacts/e2b_off_primary/qwen_b1
Q2=artifacts/e2b_off_primary/qwen_b2
QMAN=artifacts/qwen_nothink_main183/manifest.json
mkdir -p "$O"

# primary bundles
"$PY" scripts/freeze_strict_matched.py --keep_multi_pred --manifest "$MMAN" --out "$MINI"
"$PY" scripts/freeze_strict_matched.py --keep_multi_pred --manifest "$QMAN" --seed 20260725 --out "$Q1"
"$PY" scripts/freeze_strict_matched.py --keep_multi_pred --manifest "$QMAN" --seed 20260726 --out "$Q2"

# Qwen MO-Tree breadth variants (b20260725 BoN / CoT + the breadth Five run), for the budget figure
for w in medium orig; do
  "$PY" - "$w" <<'PY'
import json, sys
w = sys.argv[1]
q = [r for r in json.load(open("artifacts/qwen_nothink_main183/manifest.json"))["runs"]
     if r["seed"] == 20260725 and r["key"] != "Five-M"]
f = json.load(open(f"artifacts/qwen_nothink_breadth/{w}/manifest.json"))["runs"]
json.dump({"protocol": f"qwen breadth {w}, E2b off", "runs": q + f},
          open(f"artifacts/final_primary/breadth_{w}_manifest.json", "w"), indent=1)
PY
  "$PY" scripts/freeze_strict_matched.py --keep_multi_pred --manifest "$O/breadth_${w}_manifest.json" \
    --seed 20260725 --out "$O/breadth/$w"
done

# pool statistics / stage loss
"$PY" scripts/analyze_pool_concentration.py --bundle mini="$MINI" --bundle qwen_b1="$Q1" --bundle qwen_b2="$Q2" \
  --out "$O/concentration"

# ties and gold-topology breakdown
for b in mini:"$MINI" qwen_b1:"$Q1" qwen_b2:"$Q2"; do
  "$PY" scripts/analyze_ties_and_structure.py --bundle "${b#*:}" --out "$O/topology/${b%%:*}"
done

# Prompt-matched BoN control (generation: scripts/run_prompt_matched_bon.py)
"$PY" scripts/eval_prompt_matched_bon.py

# budget curves (BoN, prompt-matched BoN and CoT-tree are non-adaptive n=1 call sequences)
qdir() { "$PY" -c "import json,sys;print([r['dir'] for r in json.load(open('$QMAN'))['runs'] if r['key']==sys.argv[1] and r['seed']==int(sys.argv[2])][0])" "$1" "$2"; }
mdir() { "$PY" -c "import json,sys;print([r['dir'] for r in json.load(open('$MMAN'))['runs'] if r['key']==sys.argv[1]][0])" "$1"; }
for b in 1:20260725 2:20260726; do
  "$PY" scripts/eval_prompt_matched_bon.py --run artifacts/prompt_matched_bon_qwen/run \
    --locked "artifacts/e2b_off_primary/qwen_b${b%%:*}" --bon_run "$(qdir BoN-CoT-M "${b#*:}")" \
    --label "qwen_b${b%%:*}" --out "artifacts/prompt_matched_bon_qwen/eval_vs_b${b%%:*}"
done
RUNS=(--run "mini=BoN:$(mdir BoN-CoT-M):$MINI" --run "mini=CoT:$(mdir CoT-only-M):$MINI"
      --run "qwen_b1=BoN:$(qdir BoN-CoT-M 20260725):$Q1" --run "qwen_b1=CoT:$(qdir CoT-only-M 20260725):$Q1"
      --run "qwen_b2=BoN:$(qdir BoN-CoT-M 20260726):$Q2" --run "qwen_b2=CoT:$(qdir CoT-only-M 20260726):$Q2"
      --run "mini=PMBoN:artifacts/prompt_matched_bon/run:artifacts/prompt_matched_bon/eval"
      --run "qwen_b1=PMBoN:artifacts/prompt_matched_bon_qwen/run:artifacts/prompt_matched_bon_qwen/eval_vs_b1")
"$PY" scripts/analyze_budget_curve.py --keep_multi_pred --mode prefix --ns 1,2,4,8,16,full "${RUNS[@]}" --out "$O/budget_curve"
"$PY" scripts/analyze_budget_curve.py --keep_multi_pred --mode random --ns 4,8,16,full "${RUNS[@]}" --out "$O/budget_curve"
"$PY" scripts/plot_budget_curve.py --final --mode prefix
"$PY" scripts/plot_budget_curve.py --final --mode random
"$PY" scripts/plot_budget_curve.py --final --mode prefix --panel gpt

# deterministic selector sensitivity
"$PY" scripts/analyze_vote_multiplicity.py --keep_multi_pred \
  --manifest "mini=$MMAN:20260808:$MINI" --out "$O/vote_multiplicity"

# GPT-4o judge (fresh cache: the cache key is label:pool:task, not candidate content)
if [[ "${SKIP_JUDGE:-0}" != 1 ]]; then
  [[ -n "${OPENROUTER_API_KEY:-}" ]] || { set -a; source .env 2>/dev/null || true; set +a; }
  OPENROUTER_CALL_LOG=logs/final_primary_judge_calls.jsonl \
    "$PY" scripts/run_locked_judge.py --bundle mini="$MINI" --threads 8 --out "$O/judge"
fi

# selection-failure analysis, assembled tables
"$PY" scripts/analyze_selection_rebuild.py
"$PY" scripts/build_final_primary.py
"$PY" scripts/plot_vote_concentration.py
"$PY" scripts/plot_task22_workflow.py

# postprocess transformation audit; BoN online-postprocess check
"$PY" scripts/audit_transformations_final.py
"$PY" scripts/audit_bon_online_postprocess.py

# calibration-task exclusion (test 2, 3, 13, 16, 20), GPT-4o-mini final pools
"$PY" scripts/freeze_strict_matched.py --manifest "$MMAN" --keep_multi_pred \
  --exclude_ids_file data/judgment_calib5_ids.txt --out "$O/excl_calib5_178"

"$PY" scripts/analyze_vote_rank.py
"$PY" scripts/build_fig1_panels.py
"$PY" scripts/analyze_voteloss_partial_credit.py
"$PY" scripts/audit_partial_credit_definition.py
"$PY" scripts/analyze_task22_votes.py
"$PY" scripts/audit_backend_sampling.py
"$PY" scripts/audit_judge_truncation.py
"$PY" scripts/analyze_margin_gated_selector.py
"$PY" scripts/analyze_verifier_failure_modes.py
"$PY" scripts/audit_validator_dependency_checks.py
"$PY" scripts/audit_normalization_em_flips.py
"$PY" scripts/audit_final_consistency.py
