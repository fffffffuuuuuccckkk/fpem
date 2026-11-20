#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
ROLE="${SERVER_ROLE:?set SERVER_ROLE=primary|server2|server3}"
RESUME="${RESUME:-1}"
SEARCH_GPUS="${SEARCH_GPUS:-0 1}"
BASELINE_GPUS="${BASELINE_GPUS:-2 3}"

case "$ROLE" in
  primary)
    manifest=scripts/fpem_final/manifests/h48_h192_primary.csv
    shard=0
    python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
    ;;
  server2)
    manifest=scripts/fpem_final/manifests/h48_h192_server2.csv
    shard=1
    python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
    ;;
  server3)
    manifest=scripts/fpem_final/manifests/h48_h192_server3.csv
    shard=2
    python=/data/OuXiaoyu/miniconda3/envs/tslib/bin/python
    ;;
  *) echo "invalid SERVER_ROLE=$ROLE" >&2; exit 2 ;;
esac

cd "$PROJECT_DIR"
mkdir -p results/fpem_no_future_anchor_patchtst_search_h48_h192/logs \
  results/fpem_classic_baselines_h48_h192/logs

search_screen="fpem_h48_h192_search"
baseline_screen="fpem_h48_h192_baselines"
if screen -list 2>/dev/null | grep -q "[.]$search_screen"; then
  echo "search screen already running: $search_screen" >&2; exit 3
fi
if screen -list 2>/dev/null | grep -q "[.]$baseline_screen"; then
  echo "baseline screen already running: $baseline_screen" >&2; exit 3
fi

# Each case prepares or finds one compatible shared reference/A0. All LR
# candidates in that case reuse it; existing compatible artifacts are copied.
screen -dmS "$search_screen" bash -lc "cd '$PROJECT_DIR' && \
  PYTHON='$python' MANIFEST='$manifest' GPU_LIST='$SEARCH_GPUS' RESUME='$RESUME' \
  OUTPUT_ROOT='results/fpem_no_future_anchor_patchtst_search_h48_h192' \
  bash scripts/fpem_final/run_fpem_head_lr_search_matrix.sh \
  > 'results/fpem_no_future_anchor_patchtst_search_h48_h192/${ROLE}.log' 2>&1"

# The 96 baseline jobs (6 methods x 8 datasets x 2 horizons) are deterministically
# sharded 32/32/32 across the three hosts. Traffic/Electricity are consequently
# spread across all hosts instead of forming a single tail.
screen -dmS "$baseline_screen" bash -lc "cd '$PROJECT_DIR' && \
  PYTHON='$python' GPU_LIST='$BASELINE_GPUS' PRED_LENGTHS='48,192' \
  SHARD_COUNT=3 SHARD_INDEX='$shard' RUN_LABEL='FPEMH48H192' \
  OUTPUT_ROOT='results/fpem_classic_baselines_h48_h192' \
  bash scripts/fpem_final/run_classic_baselines_matrix.sh \
  > 'results/fpem_classic_baselines_h48_h192/${ROLE}.log' 2>&1"

echo "role=$ROLE FPEM_manifest=$manifest FPEM_GPUs=$SEARCH_GPUS baseline_shard=$shard/3 baseline_GPUs=$BASELINE_GPUS"
screen -list | grep -E "${search_screen}|${baseline_screen}" || true
