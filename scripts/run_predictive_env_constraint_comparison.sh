#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_constraint_comparison_k3}"
REFERENCE_CHECKPOINT="$OUTPUT_ROOT/shared_reference.pt"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"

if [[ ! -f "$REFERENCE_CHECKPOINT" ]]; then
  GPU="${GPU:-0}" \
  SEED=2021 \
  ENV_NUM=3 \
  PREPARE_REFERENCE_ONLY=1 \
  REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
  OUTPUT="$OUTPUT_ROOT/reference_build" \
  bash scripts/run_predictive_env_iv_patchtst.sh
fi

for constraint in contrastive classification; do
  echo "K=3 representation constraint: $constraint"
  GPU="${GPU:-0}" \
  SEED=2021 \
  ENV_NUM=3 \
  EXPERIMENTS="A2,A4" \
  REPRESENTATION_CONSTRAINT="$constraint" \
  ENVIRONMENT_MATCHING="${ENVIRONMENT_MATCHING:-overlap}" \
  REQUIRE_REFERENCE_CHECKPOINT=1 \
  REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
  OUTPUT="$OUTPUT_ROOT/$constraint" \
  bash scripts/run_predictive_env_iv_patchtst.sh
done

"$PYTHON" tools/summarize_predictive_env_constraint_comparison.py "$OUTPUT_ROOT"
cat "$OUTPUT_ROOT/representation_constraint_summary.txt"
