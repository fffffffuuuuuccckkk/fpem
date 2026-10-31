#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
DATASET="${DATASET:?set DATASET}"
HORIZONS="${HORIZONS:-48 96 336 720}"
ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/results/fpem_probabilistic/frozen_ema_heads}"
mkdir -p "$ROOT/queue_logs"
declare -A running=()
failed=0

poll_finished() {
  local gpu pid
  for gpu in "${!running[@]}"; do
    pid="${running[$gpu]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      if ! wait "$pid"; then
        echo "FAILED dataset=$DATASET gpu=$gpu pid=$pid" >&2
        failed=1
      fi
      unset "running[$gpu]"
    fi
  done
}

gpu_is_free() {
  local gpu="$1" used
  used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ -z "${running[$gpu]:-}" ]] || return 1
  [[ "$used" =~ ^[0-9]+$ ]] && ((used < 64))
}

for horizon in $HORIZONS; do
  if [[ -s "$ROOT/$DATASET/pred_$horizon/summary.csv" ]]; then
    echo "SKIP complete: $DATASET/$horizon"
    continue
  fi
  while true; do
    poll_finished
    chosen=""
    for gpu in 0 1 2 3; do
      if gpu_is_free "$gpu"; then
        chosen="$gpu"
        break
      fi
    done
    if [[ -n "$chosen" ]]; then
      echo "START dataset=$DATASET horizon=$horizon gpu=$chosen"
      DATASET="$DATASET" PRED_LEN="$horizon" GPU="$chosen" \
        OUTPUT_ROOT="$ROOT" \
        bash "$PROJECT_DIR/scripts/fpem_probabilistic/run_frozen_ema_heads.sh" \
        >"$ROOT/queue_logs/${DATASET}_${horizon}.log" 2>&1 &
      running[$chosen]="$!"
      break
    fi
    sleep 30
  done
done

while ((${#running[@]})); do
  poll_finished
  ((${#running[@]})) && sleep 30 || true
done
echo "QUEUE_DONE dataset=$DATASET failed=$failed"
exit "$failed"
