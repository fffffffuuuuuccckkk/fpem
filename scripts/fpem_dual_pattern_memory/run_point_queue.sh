#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PYTHON=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
CASE=${1:?case required}
GPU=${2:?GPU required}
OUT="$ROOT/results/fpem_dual_pattern_memory/point_pilot/$CASE"
mkdir -p "$OUT"
cd "$ROOT"
if [[ -f "$OUT/point_pilot_complete" ]]; then
  echo "Already complete: $CASE"
  exit 0
fi
for attempt in 1 2 3; do
  echo "$(date -Is) DPPM $CASE GPU=$GPU attempt=$attempt" | tee -a "$OUT/queue.log"
  if CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/run_fpem_dppm.py "$CASE" \
      --gpu 0 --output results/fpem_dual_pattern_memory/point_pilot \
      >>"$OUT/run.log" 2>&1; then
    echo "$(date -Is) complete" | tee -a "$OUT/queue.log"
    exit 0
  fi
  echo "$(date -Is) failed attempt=$attempt" | tee -a "$OUT/queue.log"
  if (( attempt < 3 )); then sleep 60; fi
done
exit 1
