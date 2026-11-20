#!/usr/bin/env bash
# Resume-safe full-matrix comparison: fixed optimal R3 + trained F03, and R3(trick).
set -euo pipefail
gpu="${1:?GPU index required}"
shift
root=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
source_root="$root/results/fpem_eiil3_fulltrain_testselect_20261113/cases"
batch="$root/results/fpem_eiil3_r3_f03_full_dual_20261116"
mkdir -p "$batch/cases" "$batch/queue_logs"
cd "$root"
echo "QUEUE_START gpu=$gpu cases=$# $(date -Is)"
for case_key in "$@"; do
  source="$source_root/$case_key"
  output="$batch/cases/$case_key"
  case_log="$batch/queue_logs/${case_key//\//_}.log"
  if [[ ! -s "$source/complete" || ! -s "$source/A0_no_F03_r3.pt" || ! -s "$source/selected_safe_head.pt" ]]; then
    echo "BLOCKED_MISSING_SOURCE $case_key $(date -Is)" | tee -a "$case_log"
    continue
  fi
  if [[ -s "$output/complete" ]]; then
    echo "REUSED $case_key $(date -Is)" | tee -a "$case_log"
    continue
  fi
  attempt=0
  until [[ -s "$output/complete" ]]; do
    while nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '^[0-9]+'; do
      echo "WAIT_GPU $gpu $(date -Is)" >> "$case_log"
      sleep 300
    done
    attempt=$((attempt+1))
    echo "START $case_key gpu=$gpu attempt=$attempt $(date -Is)" | tee -a "$case_log"
    if CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python" \
        tools/run_fpem_eiil3_r3_then_f03.py \
        --case-key "$case_key" --source-root "$source_root" \
        --output-root "$batch/cases" --gpu 0 --r3-trick --resume >> "$case_log" 2>&1; then
      echo "COMPLETE $case_key $(date -Is)" | tee -a "$case_log"
    elif [[ "$attempt" -ge 5 ]]; then
      echo "FAILED $case_key attempts=$attempt $(date -Is)" | tee -a "$case_log"
      break
    else
      echo "RETRY_AFTER_60S $case_key $(date -Is)" | tee -a "$case_log"
      sleep 60
    fi
  done
  "$python" tools/summarize_fpem_eiil3_r3_then_f03.py --output-root "$batch/cases" >> "$batch/queue_logs/summary_gpu${gpu}.log" 2>&1
done
echo "QUEUE_COMPLETE gpu=$gpu $(date -Is)"
