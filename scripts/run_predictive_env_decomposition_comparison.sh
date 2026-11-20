#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_decomposition_comparison_k3}"
REFERENCE_CHECKPOINT="$OUTPUT_ROOT/shared_reference.pt"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"

if [[ ! -f "$REFERENCE_CHECKPOINT" ]]; then
  GPU="${GPU:-0}" SEED=2021 ENV_NUM=3 PREPARE_REFERENCE_ONLY=1 \
  REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
  OUTPUT="$OUTPUT_ROOT/reference_build" \
  bash scripts/run_predictive_env_iv_patchtst.sh
fi

GPU="${GPU:-0}" SEED=2021 ENV_NUM=3 EXPERIMENTS=A0 \
REPRESENTATION_CONSTRAINT=contrastive DECOMPOSITION_TYPE=projection \
REQUIRE_REFERENCE_CHECKPOINT=1 REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
OUTPUT="$OUTPUT_ROOT/baseline" \
bash scripts/run_predictive_env_iv_patchtst.sh

for constraint in contrastive classification; do
  for decomposition in projection signed_gate; do
    echo "K=3 constraint=$constraint decomposition=$decomposition"
    GPU="${GPU:-0}" SEED=2021 ENV_NUM=3 EXPERIMENTS="A2,A4" \
    REPRESENTATION_CONSTRAINT="$constraint" \
    DECOMPOSITION_TYPE="$decomposition" \
    ENVIRONMENT_MATCHING=overlap REQUIRE_REFERENCE_CHECKPOINT=1 \
    REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
    OUTPUT="$OUTPUT_ROOT/$constraint/$decomposition" \
    bash scripts/run_predictive_env_iv_patchtst.sh
  done
done

"$PYTHON" tools/summarize_predictive_env_decomposition_comparison.py "$OUTPUT_ROOT"
cat "$OUTPUT_ROOT/decomposition_comparison_summary.txt"
