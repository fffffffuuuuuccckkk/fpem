#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
SWEEP_ROOT="${SWEEP_ROOT:-results/predictive_env_count_sweep_no_mapping}"
ENV_COUNTS="${ENV_COUNTS:-2 3 4 6 8}"
EXPERIMENTS="${EXPERIMENTS:-A2,A4}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-$SWEEP_ROOT/shared_reference.pt}"

cd "$PROJECT_DIR"
mkdir -p "$SWEEP_ROOT"
if [[ ! -f "$REFERENCE_CHECKPOINT" ]]; then
  echo "creating shared reference once with seed=${SEED:-2021}"
  GPU="${GPU:-0}" \
  SEED="${SEED:-2021}" \
  PREPARE_REFERENCE_ONLY=1 \
  REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
  OUTPUT="$SWEEP_ROOT/reference_build" \
  bash scripts/run_predictive_env_iv_patchtst.sh
fi

first=1
for env_num in $ENV_COUNTS; do
  run_experiments="$EXPERIMENTS"
  if [[ "$first" == "1" ]]; then
    run_experiments="A0,$EXPERIMENTS"
    first=0
  fi
  echo "environment-count sweep: K=$env_num experiments=$run_experiments"
  GPU="${GPU:-0}" \
  SEED="${SEED:-2021}" \
  ENV_NUM="$env_num" \
  EXPERIMENTS="$run_experiments" \
  EPOCHS="${EPOCHS:-10}" \
  WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}" \
  BATCH_SIZE="${BATCH_SIZE:-32}" \
  DATA_ROOT="${DATA_ROOT:-./dataset/ETT-small/}" \
  REQUIRE_REFERENCE_CHECKPOINT=1 \
  REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
  OUTPUT="$SWEEP_ROOT/env_k$env_num" \
  bash scripts/run_predictive_env_iv_patchtst.sh
done

"$PYTHON" tools/summarize_predictive_env_count_sweep.py "$SWEEP_ROOT"
cat "$SWEEP_ROOT/environment_count_summary.txt"
