#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PYTHON=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
CASE=${1:?case required}
GPU=${2:?gpu required}
OUT="$ROOT/results/fpem_dual_consensus_retrieval/point_pilot/$CASE"
mkdir -p "$OUT"
cd "$ROOT" || exit 1

attempt=1
while (( attempt <= 4 )); do
  if [[ -f "$OUT/point_pilot_complete" ]]; then
    echo "DCRF complete $CASE"
    exit 0
  fi
  echo "DCRF case=$CASE gpu=$GPU attempt=$attempt"
  if "$PYTHON" -u tools/run_fpem_dcrf.py "$CASE" --gpu "$GPU" \
       --output results/fpem_dual_consensus_retrieval/point_pilot \
       >> "$OUT/run.log" 2>&1; then
    if [[ -f "$OUT/point_pilot_complete" ]]; then
      echo "DCRF complete $CASE"
      exit 0
    fi
  fi
  attempt=$((attempt + 1))
  if (( attempt <= 4 )); then
    echo "DCRF retry after 60s $CASE"
    sleep 60
  fi
done
echo "DCRF failed after four attempts $CASE" >&2
exit 1
