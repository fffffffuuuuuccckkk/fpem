#!/usr/bin/env bash
# Resumable one-case A0-safe interpolation, fresh detached R3 and verified source cleanup.
set -euo pipefail
gpu="$1"
case_key="$2"
root=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
old="$root/results/fpem_eiil3_mse_priority_20261112/$case_key"
new="$root/results/fpem_eiil3_f03_safe_20261112/$case_key"
manifest="$root/results/fpem_eiil3_20261110/manifest.jsonl"
binding="$root/results/fpem_eiil3_mse_priority_20261112/asset_bindings/patchtst_ETTh1_720/case_manifest.jsonl"
if [[ "$case_key" == "patchtst/ETTh1/pred_720/seed_2021" && -s "$binding" ]]; then
  manifest="$binding"
fi
log="$root/results/fpem_eiil3_f03_safe_20261112/queue_logs/${case_key//\//_}.log"
mkdir -p "$(dirname "$log")"
cd "$root"
if [[ -s "$new/complete" ]]; then
  echo "REUSED $case_key" >> "$log"
  exit 0
fi
for attempt in 1 2; do
  echo "START $case_key gpu=$gpu attempt=$attempt" >> "$log"
  if CUDA_VISIBLE_DEVICES="$gpu" "$python" tools/run_fpem_f03_safe.py \
      --source-case "$old" --output-case "$new" --gpu 0 \
      --manifest "$manifest" \
      --fit-r3 --cleanup-source --resume >> "$log" 2>&1; then
    echo "COMPLETE $case_key" >> "$log"
    exit 0
  fi
  echo "FAILED $case_key attempt=$attempt" >> "$log"
  if [[ "$attempt" -eq 1 ]]; then sleep 60; fi
done
exit 1
