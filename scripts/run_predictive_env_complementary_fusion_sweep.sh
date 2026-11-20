#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_complementary_fusion_k3}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-results/predictive_env_decomposition_comparison_k3/shared_reference.pt}"
PREVIOUS_ROOT="${PREVIOUS_ROOT:-results/predictive_env_complementary_gate_k3}"

cd "$PROJECT_DIR"
if [[ ! -f "$REFERENCE_CHECKPOINT" ]]; then
  echo "Required shared reference is missing: $REFERENCE_CHECKPOINT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

for lambda_invpred in 0.5 1.0 2.0; do
  lambda_tag="${lambda_invpred/./p}"
  for fusion in off direct_gated; do
    echo "complementary A2 K=3 lambda_invpred=$lambda_invpred fusion=$fusion"
    GPU="${GPU:-0}" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
    REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate \
    VARIANT_FUSION_MODE="$fusion" \
    LAMBDA_INVPRED="$lambda_invpred" \
    LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 \
    ENVIRONMENT_MATCHING=overlap REQUIRE_REFERENCE_CHECKPOINT=1 \
    REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
    OUTPUT="$OUTPUT_ROOT/lambda_$lambda_tag/$fusion" \
    bash scripts/run_predictive_env_iv_patchtst.sh
  done
done


"$PYTHON" tools/summarize_predictive_env_complementary_fusion_sweep.py \
  "$OUTPUT_ROOT" --previous_root "$PREVIOUS_ROOT"
cat "$OUTPUT_ROOT/complementary_fusion_summary.txt"
