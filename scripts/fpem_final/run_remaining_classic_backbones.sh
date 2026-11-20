#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
METHOD="${METHOD:?set METHOD=TimesNet or Autoformer}"
case "$METHOD" in TimesNet|Autoformer) ;; *) echo "unsupported METHOD=$METHOD" >&2; exit 2 ;; esac
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_classic_baselines_completion_20261013}"
DRY_RUN="${DRY_RUN:-0}"
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status"

# Global inventory across the three servers shows that ETTh1/ETTh2/ExchangeRate
# have 48/96/192/336/720 results; ETTm1/ETTm2 and all pred_24 are missing.
run_group() {
  local datasets="$1" horizons="$2"
  env PYTHON="$PYTHON" OUTPUT_ROOT="$OUTPUT_ROOT" GPU_LIST="$GPU_LIST" \
    DRY_RUN="$DRY_RUN" METHODS="$METHOD" DATASETS="$datasets" \
    PRED_LENGTHS="$horizons" TRAIN_EPOCHS=10 \
    RUN_LABEL="FPEMBaseComplete20261013" \
    bash scripts/fpem_final/run_classic_baselines_completion_matrix.sh
}

run_group "ETTm1,ETTm2" "24,48,96,192,336,720"
run_group "ETTh1,ETTh2,ExchangeRate" "24"
