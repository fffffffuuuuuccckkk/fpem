#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
SERVER_INDEX="${SERVER_INDEX:?set SERVER_INDEX to 0, 1, or 2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_PROJECT/results/foil_shifts_a0_protocol_20261011}"
SCREEN_NAME="foilshifts_itr_s${SERVER_INDEX}"

cd "$PROJECT_DIR"
test -x "$PYTHON"
command -v screen >/dev/null
mkdir -p "$OUTPUT_ROOT/queue_logs"
export PROJECT_DIR DATA_PROJECT PYTHON SERVER_INDEX OUTPUT_ROOT
export BACKBONES=itransformer GPU_LIST="${GPU_LIST:-0 1 2 3}"
export WAIT_SCREENS_PATTERN=fpem_itr

if [[ "${RUN_IN_SCREEN:-0}" != 1 ]]; then
  if screen -ls | grep -Eq "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "already queued: $SCREEN_NAME"
    exit 0
  fi
  screen -dmS "$SCREEN_NAME" env RUN_IN_SCREEN=1 \
    bash scripts/fpem_probabilistic/launch_itransformer_foil_shifts.sh
  sleep 2
  screen -ls | grep -Eq "[.]${SCREEN_NAME}[[:space:]]"
  echo "queued: $SCREEN_NAME; waiting for current iTransformer screens"
  exit 0
fi

exec >> "$OUTPUT_ROOT/queue_logs/itransformer_launcher_server${SERVER_INDEX}.log" 2>&1
echo "QUEUE_START server=$SERVER_INDEX GPUs=$GPU_LIST"
bash scripts/fpem_probabilistic/run_foil_shifts_matrix.sh
echo "QUEUE_COMPLETE server=$SERVER_INDEX"
