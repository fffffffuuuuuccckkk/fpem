#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
ROLE="${ROLE:-primary}"
SESSION="timefilter_fpem_${ROLE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_no_future_anchor_timefilter_search}"
if [[ "$ROLE" == "server3" ]]; then
  DEFAULT_PYTHON=/data/OuXiaoyu/miniconda3/envs/tslib/bin/python
else
  DEFAULT_PYTHON=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
fi

run_lane() {
  local role="$1" gpu="$2"
  local manifest="$PROJECT_DIR/scripts/fpem_final/manifests/timefilter_${role}.csv"
  local lane_log="$PROJECT_DIR/$OUTPUT_ROOT/launcher_${role}_gpu${gpu}.log"
  mkdir -p "$(dirname "$lane_log")"
  while IFS=, read -r assigned_gpu dataset pred_len; do
    [[ -n "$assigned_gpu" ]] || continue
    [[ "$assigned_gpu" == "$gpu" ]] || continue
    echo "[$(date -Is)] start $dataset pred=$pred_len gpu=$gpu" | tee -a "$lane_log"
    env PROJECT_DIR="$PROJECT_DIR" PYTHON="${PYTHON:-$DEFAULT_PYTHON}" \
      BACKBONE=timefilter DATASET="$dataset" PRED_LEN="$pred_len" GPU="$gpu" \
      OUTPUT_ROOT="$OUTPUT_ROOT" FIXED_LAMBDA_FUTURE_H=0.0 \
      FIXED_LAMBDA_VARIANT_ANCHOR=0.0 \
      bash "$PROJECT_DIR/scripts/fpem_final/run_fpem_search_case.sh" \
      >>"$lane_log" 2>&1
    echo "[$(date -Is)] complete $dataset pred=$pred_len gpu=$gpu" | tee -a "$lane_log"
  done < "$manifest"
  echo "[$(date -Is)] lane complete gpu=$gpu" | tee -a "$lane_log"
}

if [[ "${1:-}" == "--lane" ]]; then
  run_lane "${2:?role}" "${3:?gpu}"
  exit 0
fi

cd "$PROJECT_DIR"
manifest="$PROJECT_DIR/scripts/fpem_final/manifests/timefilter_${ROLE}.csv"
[[ -s "$manifest" ]] || { echo "missing manifest: $manifest" >&2; exit 2; }
if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "screen already running: $SESSION"
  exit 0
fi

screen -dmS "$SESSION" -t gpu0 bash "$0" --lane "$ROLE" 0
for gpu in 1 2 3; do
  screen -S "$SESSION" -X screen -t "gpu${gpu}" bash "$0" --lane "$ROLE" "$gpu"
done
echo "started screen=$SESSION role=$ROLE manifest=$manifest output=$OUTPUT_ROOT"
