#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_reference_refactor_k3}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-results/predictive_env_decomposition_comparison_k3/shared_reference.pt}"

cd "$PROJECT_DIR"
if [[ ! -f "$REFERENCE_CHECKPOINT" ]]; then
  echo "Required shared reference is missing: $REFERENCE_CHECKPOINT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

run_variant() {
  local label="$1"
  local refactor_mode="$2"
  local scale_calibration="$3"
  echo "$label mode=$refactor_mode scale=$scale_calibration"
  GPU="${GPU:-0}" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
  REPRESENTATION_CONSTRAINT=classification \
  DECOMPOSITION_TYPE=complementary_gate \
  VARIANT_FUSION_MODE=direct_gated \
  PREDICTIVE_ENV_REFACTOR_MODE="$refactor_mode" \
  FUSION_SCALE_CALIBRATION="$scale_calibration" \
  STAGE_EPOCHS=2 LAMBDA_H_ANCHOR=0.0 \
  LAMBDA_INVPRED=1.0 \
  LAMBDA_VAR_CONDITIONAL_GAIN=0.2 \
  VAR_CONDITIONAL_MARGIN=0.0 \
  LAMBDA_VAR_PREDICTIVE=0.0 \
  LAMBDA_VAR_UTILITY=0.0 \
  LAMBDA_FUTURE_VAR=0.0 \
  LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 \
  ENVIRONMENT_MATCHING=overlap REQUIRE_REFERENCE_CHECKPOINT=1 \
  REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
  OUTPUT="$OUTPUT_ROOT/$label" \
  bash scripts/run_predictive_env_iv_patchtst.sh
}

run_variant A_current current none
run_variant B_h_reference h_reference none
run_variant C_gradient_isolated h_reference_grad_isolated none
run_variant D_isolated_rms h_reference_grad_isolated rms

"$PYTHON" tools/summarize_predictive_env_reference_refactor.py "$OUTPUT_ROOT"
cat "$OUTPUT_ROOT/reference_refactor_summary.txt"
