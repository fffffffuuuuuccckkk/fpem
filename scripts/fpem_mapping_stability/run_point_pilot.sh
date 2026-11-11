#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PYTHON=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
CASE=${1:?case required}
GPU=${2:?gpu required}
OUT="$ROOT/results/fpem_mapping_stability/pilot_20261027/$CASE"
mkdir -p "$OUT"
cd "$ROOT" || exit 1

attempt=1
while (( attempt <= 4 )); do
  if [[ -f "$OUT/point_pilot_complete" ]]; then
    echo "complete $CASE"
    exit 0
  fi
  echo "SMRVA case=$CASE gpu=$GPU attempt=$attempt"
  if "$PYTHON" -u tools/run_fpem_smrva.py "$CASE" --gpu "$GPU" \
      --output results/fpem_mapping_stability/pilot_20261027 \
      >> "$OUT/run.log" 2>&1; then
    if [[ -f "$OUT/point_pilot_complete" ]]; then
      echo "complete $CASE"
      exit 0
    fi
  fi
  attempt=$((attempt + 1))
  if (( attempt <= 4 )); then
    echo "retry after 60s $CASE"
    sleep 60
  fi
done
echo "failed after four attempts: $CASE" >&2
exit 1
