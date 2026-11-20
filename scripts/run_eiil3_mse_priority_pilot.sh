#!/usr/bin/env bash
# One resumable EIIL3 MSE-priority pilot. A0 and feature identity are audited.
set -euo pipefail
gpu="$1"
case_key="$2"
root=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
output="${EIIL3_OUTPUT_ROOT:-$root/results/fpem_eiil3_mse_priority_20261112}"
manifest="${EIIL3_MANIFEST:-results/fpem_eiil3_20261110/manifest.jsonl}"
lambda_inv="${EIIL3_LAMBDA_INV:-0.0001}"
log="$output/queue_logs/${case_key//\//_}.log"
mkdir -p "$output/queue_logs"
cd "$root"
for attempt in 1 2; do
  if [[ -s "$output/$case_key/point_complete" ]]; then
    echo "REUSED $case_key" >> "$log"
    exit 0
  fi
  echo "START $case_key attempt=$attempt gpu=$gpu" >> "$log"
  if CUDA_VISIBLE_DEVICES="$gpu" "$python" tools/run_fpem_eiil3_mse_priority.py \
      --manifest "$manifest" \
      --case-key "$case_key" --output-root "$output" --gpu 0 \
      --lambda-inv "$lambda_inv" --stop-on-gain --resume >> "$log" 2>&1; then
    echo "COMPLETE $case_key" >> "$log"
    exit 0
  fi
  echo "FAILED $case_key attempt=$attempt" >> "$log"
  if [[ "$attempt" -eq 1 ]]; then sleep 60; fi
done
exit 1
