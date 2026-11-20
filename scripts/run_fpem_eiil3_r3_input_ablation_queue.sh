#!/usr/bin/env bash
# Separate resumable queue; never changes the existing F03/R3 jobs or artifacts.
set -euo pipefail
gpu="${1:?GPU index required}"
shift
root=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
batch="$root/results/fpem_eiil3_r3_input_ablation_20261115"
source_root="$root/results/fpem_eiil3_fulltrain_testselect_20261113/cases"
mkdir -p "$batch/cases" "$batch/queue_logs"
exec >> "$batch/queue_logs/queue_gpu${gpu}.log" 2>&1
cd "$root"
for case_key in "$@"; do
  attempt=0
  until [[ -f "$batch/cases/$case_key/complete" ]]; do
    while nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '^[0-9]+'; do
      echo "WAIT_GPU $gpu $(date -Is)"
      sleep 300
    done
    attempt=$((attempt+1))
    echo "START $case_key attempt=$attempt $(date -Is)"
    if CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python" \
         tools/run_fpem_eiil3_r3_input_ablation.py \
         --case-key "$case_key" --source-root "$source_root" \
         --output-root "$batch/cases" --gpu 0 --resume; then
      echo "COMPLETE $case_key $(date -Is)"
    elif [[ "$attempt" -ge 5 ]]; then
      echo "FAILED $case_key after $attempt attempts $(date -Is)"
      exit 1
    else
      echo "RETRY_AFTER_60S $case_key $(date -Is)"
      sleep 60
    fi
  done
  "$python" tools/summarize_fpem_eiil3_r3_input_ablation.py --output-root "$batch/cases"
done
echo "QUEUE_COMPLETE $(date -Is)"
