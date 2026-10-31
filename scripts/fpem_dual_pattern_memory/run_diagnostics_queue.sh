#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PYTHON=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
CASE=${1:?case required}
GPU=${2:?gpu required}
OUT="$ROOT/results/fpem_dual_pattern_memory/diagnostics/$CASE"
mkdir -p "$OUT"
cd "$ROOT" || exit 1
if [[ -f "$OUT/diagnostic_complete" ]]; then echo "already complete: $CASE"; exit 0; fi
for attempt in 1 2 3; do
  echo "$(date -Is) diagnostic $CASE GPU=$GPU attempt=$attempt" | tee -a "$OUT/queue.log"
  if CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/diagnose_fpem_dppm.py "$CASE" \
     --gpu 0 --output results/fpem_dual_pattern_memory/diagnostics \
     >>"$OUT/run.log" 2>&1; then
    echo "$(date -Is) complete" | tee -a "$OUT/queue.log"
    exit 0
  fi
  echo "$(date -Is) failed attempt=$attempt" | tee -a "$OUT/queue.log"
  if (( attempt < 3 )); then sleep 60; fi
done
exit 1
