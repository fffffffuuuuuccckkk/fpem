#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_complementary_gate_k3}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-results/predictive_env_decomposition_comparison_k3/shared_reference.pt}"
PREVIOUS_ROOT="${PREVIOUS_ROOT:-results/predictive_env_decomposition_comparison_k3}"

cd "$PROJECT_DIR"
if [[ ! -f "$REFERENCE_CHECKPOINT" ]]; then
  echo "Required existing shared reference is missing: $REFERENCE_CHECKPOINT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_ROOT"

GPU="${GPU:-0}" SEED=2021 ENV_NUM=3 EXPERIMENTS=A0 \
REPRESENTATION_CONSTRAINT=contrastive DECOMPOSITION_TYPE=projection \
REQUIRE_REFERENCE_CHECKPOINT=1 REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
OUTPUT="$OUTPUT_ROOT/baseline" \
bash scripts/run_predictive_env_iv_patchtst.sh

for constraint in contrastive classification; do
  for decomposition in projection signed_gate complementary_gate; do
    echo "A2 K=3 constraint=$constraint decomposition=$decomposition"
    GPU="${GPU:-0}" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
    REPRESENTATION_CONSTRAINT="$constraint" \
    DECOMPOSITION_TYPE="$decomposition" \
    LAMBDA_GATE_ACTIVITY=0.0001 \
    LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 \
    ENVIRONMENT_MATCHING=overlap REQUIRE_REFERENCE_CHECKPOINT=1 \
    REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
    OUTPUT="$OUTPUT_ROOT/$constraint/$decomposition" \
    bash scripts/run_predictive_env_iv_patchtst.sh
  done
done

"$PYTHON" tools/summarize_predictive_env_complementary_comparison.py \
  "$OUTPUT_ROOT" --previous_root "$PREVIOUS_ROOT"
cat "$OUTPUT_ROOT/complementary_gate_summary.txt"
