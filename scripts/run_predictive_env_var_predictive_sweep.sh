#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_var_predictive_k3}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-results/predictive_env_decomposition_comparison_k3/shared_reference.pt}"
PREVIOUS_ROOT="${PREVIOUS_ROOT:-results/predictive_env_future_var_k3}"

cd "$PROJECT_DIR"
if [[ ! -f "$REFERENCE_CHECKPOINT" ]]; then
  echo "Required shared reference is missing: $REFERENCE_CHECKPOINT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

for lambda_var_predictive in 0 0.01 0.05 0.1 0.2; do
  lambda_tag="${lambda_var_predictive/./p}"
  echo "conditional-Zvar A2 K=3 lambda_var_predictive=$lambda_var_predictive"
  GPU="${GPU:-0}" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
  REPRESENTATION_CONSTRAINT=classification \
  DECOMPOSITION_TYPE=complementary_gate \
  VARIANT_FUSION_MODE=direct_gated \
  LAMBDA_INVPRED=1.0 \
  LAMBDA_VAR_PREDICTIVE="$lambda_var_predictive" \
  LAMBDA_VAR_UTILITY=0.0 \
  LAMBDA_FUTURE_VAR=0.0 \
  LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 \
  ENVIRONMENT_MATCHING=overlap REQUIRE_REFERENCE_CHECKPOINT=1 \
  REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
  OUTPUT="$OUTPUT_ROOT/lambda_$lambda_tag" \
  bash scripts/run_predictive_env_iv_patchtst.sh
done

"$PYTHON" tools/summarize_predictive_env_var_predictive_sweep.py \
  "$OUTPUT_ROOT" --previous_root "$PREVIOUS_ROOT"
cat "$OUTPUT_ROOT/var_predictive_summary.txt"
