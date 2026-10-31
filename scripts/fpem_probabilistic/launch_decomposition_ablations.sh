#!/usr/bin/env bash
# Launch independent short probabilistic and long deterministic queues.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
SHORT_GPUS="${SHORT_GPUS:-0}"
LONG_GPUS="${LONG_GPUS:-1 2 3}"
SHORT_ROOT="$DATA_PROJECT/results/fpem_probabilistic/no_domain_ablations_20261012"
LONG_ROOT="$DATA_PROJECT/results/fpem_decomposition_controls_20261012"

mkdir -p "$SHORT_ROOT" "$LONG_ROOT"
if screen -ls | grep -q '[.]fpem_nodom_short'; then
  echo "short queue already running"
else
  screen -dmS fpem_nodom_short -L -Logfile "$SHORT_ROOT/screen.log" \
    env PROJECT_DIR="$PROJECT_DIR" DATA_PROJECT="$DATA_PROJECT" PYTHON="$PYTHON" \
      BASE_ROOT="$DATA_PROJECT/results/fpem_probabilistic/unified_lr_search_20261010" \
      OUTPUT_ROOT="$SHORT_ROOT" GPU_LIST="$SHORT_GPUS" \
      bash "$PROJECT_DIR/scripts/fpem_probabilistic/run_decomposition_ablations.sh" queue
  echo "started short probability-head queue on GPUs $SHORT_GPUS"
fi
if screen -ls | grep -q '[.]fpem_nodom_long'; then
  echo "long queue already running"
else
  screen -dmS fpem_nodom_long -L -Logfile "$LONG_ROOT/screen.log" \
    env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
      OUTPUT_ROOT="$LONG_ROOT" GPU_LIST="$LONG_GPUS" \
      bash "$PROJECT_DIR/scripts/fpem_probabilistic/run_remaining_length_ablations.sh" queue
  echo "started long-horizon queue on GPUs $LONG_GPUS"
fi
