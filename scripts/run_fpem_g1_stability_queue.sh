#!/usr/bin/env bash
set -uo pipefail

# Usage: PYTHON=/absolute/python bash this_script GPU backbone:dataset:horizon ...
# Idempotent case outputs allow restart after a host memory interruption.
gpu="${1:?GPU index required}"
shift
project=/data/OuXiaoyu/Time-Series-Library-FPEM
cd "$project" || exit 2
python_bin="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
root=results/fpem_g1_stability_search_20261102
mkdir -p "$root/queue_logs"
for case_id in "$@"; do
  IFS=: read -r backbone dataset horizon <<< "$case_id"
  case_dir="$root/$backbone/$dataset/pred_$horizon"
  mkdir -p "$case_dir"
  if [[ -f "$case_dir/complete" ]]; then
    continue
  fi
  for attempt in 1 2 3; do
    echo "START $case_id attempt=$attempt gpu=$gpu" | tee -a "$root/queue_logs/gpu${gpu}.log"
    if PYTHONPATH=. "$python_bin" tools/run_fpem_g1_stability_search.py \
      --backbone "$backbone" --dataset "$dataset" --pred_len "$horizon" --gpu "$gpu" \
      >> "$case_dir/queue.log" 2>&1; then
      if [[ -f "$case_dir/complete" ]]; then
        echo "DONE $case_id" | tee -a "$root/queue_logs/gpu${gpu}.log"
        break
      fi
    fi
    if [[ "$attempt" -eq 3 ]]; then
      echo "FAILED $case_id after three attempts; see $case_dir/queue.log" | tee -a "$root/queue_logs/gpu${gpu}.log"
      exit 1
    fi
    echo "RETRY $case_id after 60 seconds" | tee -a "$root/queue_logs/gpu${gpu}.log"
    sleep 60
  done
done
PYTHONPATH=. "$python_bin" tools/analyze_fpem_g1_stability.py --result_root "$root" \
  >> "$root/queue_logs/gpu${gpu}.log" 2>&1
echo "QUEUE COMPLETE gpu=$gpu" | tee -a "$root/queue_logs/gpu${gpu}.log"
