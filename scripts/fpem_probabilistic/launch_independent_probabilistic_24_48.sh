#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
ROLE="${ROLE:?set ROLE=s2 or s3}"
ROOT="$PROJECT_DIR/results/fpem_probabilistic/independent_baselines_24_48"
mkdir -p "$ROOT/logs"
if [[ "$ROLE" == s2 ]]; then
  PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
  queues=(
    'tsflow:ETTh1:24 tsflow:ETTh1:48 tsflow:ETTh2:24 tsflow:ETTh2:48 tsflow:ExchangeRate:24'
    'tsflow:ETTm1:24 tsflow:ETTm1:48 tsflow:ETTm2:24 tsflow:ETTm2:48 tsflow:ExchangeRate:48'
    'timegrad:ETTh1:24 timegrad:ETTh1:48 timegrad:ETTh2:24 timegrad:ETTh2:48'
    'tsdiff:ETTh1:24 tsdiff:ETTh1:48 tsdiff:ETTh2:24 tsdiff:ETTh2:48'
  )
  gpus=(0 1 2 3)
elif [[ "$ROLE" == s3 ]]; then
  PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/tslib/bin/python}"
  queues=(
    'timegrad:ETTm1:24 timegrad:ETTm1:48 timegrad:ETTm2:24 timegrad:ETTm2:48'
    'tsdiff:ETTm1:24 tsdiff:ETTm1:48 tsdiff:ETTm2:24 tsdiff:ETTm2:48'
    'timegrad:ExchangeRate:24 timegrad:ExchangeRate:48 tsflow:ExchangeRate:24 tsflow:ExchangeRate:48 tsdiff:ExchangeRate:24 tsdiff:ExchangeRate:48'
  )
  gpus=(1 2 3)
else
  echo "ROLE must be s2 or s3" >&2
  exit 2
fi
for idx in "${!gpus[@]}"; do
  gpu="${gpus[$idx]}"
  name="indprob_${ROLE}_g${gpu}"
  if screen -ls | grep -Fq ".$name"; then
    echo "already running $name"
    continue
  fi
  read -r -a cases <<<"${queues[$idx]}"
  screen -L -Logfile "$ROOT/logs/$name.log" -dmS "$name" \
    env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
    bash "$PROJECT_DIR/scripts/fpem_probabilistic/run_independent_probabilistic_queue.sh" \
    "$gpu" "${cases[@]}"
  echo "started $name GPU=$gpu cases=${#cases[@]}"
done
