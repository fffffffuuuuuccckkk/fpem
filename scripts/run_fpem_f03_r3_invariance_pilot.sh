#!/usr/bin/env bash
set -u
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
base_config="$1"
result_dir="$2"
gpu="$3"
seed="$4"
mkdir -p "$result_dir"
attempt=0
while [ ! -f "$result_dir/complete" ] && [ "$attempt" -lt 3 ]; do
  attempt=$((attempt+1))
  /data/OuXiaoyu/miniconda3/envs/basicts/bin/python -u tools/run_fpem_f03_r3_invariance.py \
    --base_run_config "$base_config" --output "$result_dir" --gpu "$gpu" \
    --seed "$seed" --epochs 10 --train_stride 16 --val_stride 4 --test_stride 1 \
    --lrs 0.0003,0.001 --lambdas 0.001,0.01
  code=$?
  if [ "$code" -eq 0 ] && [ -f "$result_dir/complete" ]; then
    exit 0
  fi
  echo "FAILED attempt=$attempt code=$code; waiting 60 s before resumable retry" >&2
  sleep 60
done
if [ -f "$result_dir/complete" ]; then exit 0; fi
exit 1
