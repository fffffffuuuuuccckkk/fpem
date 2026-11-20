#!/usr/bin/env bash
set -u
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
base_config="$1"
old_result="$2"
result_dir="$3"
gpu="$4"
seed="$5"
mkdir -p "$result_dir"
if [ ! -f "$old_result/r3.pt" ] || [ ! -f "$old_result/F03/complete" ]; then
  echo "Missing matched R3/F03 source: $old_result" >&2
  exit 2
fi
if [ ! -f "$result_dir/r3.pt" ]; then cp "$old_result/r3.pt" "$result_dir/r3.pt" || exit 2; fi
if [ ! -f "$result_dir/F03/complete" ]; then cp -a "$old_result/F03" "$result_dir/F03" || exit 2; fi
attempt=0
while [ ! -f "$result_dir/F03_I3_BLEND/complete" ] && [ "$attempt" -lt 3 ]; do
  attempt=$((attempt+1))
  /data/OuXiaoyu/miniconda3/envs/basicts/bin/python -u tools/run_fpem_f03_r3_invariance.py \
    --base_run_config "$base_config" --output "$result_dir" --gpu "$gpu" \
    --seed "$seed" --epochs 10 --train_stride 16 --val_stride 4 --test_stride 1 \
    --methods F03,F03_I3_BLEND --lrs 0.0003 --lambdas 0.001
  code=$?
  if [ "$code" -eq 0 ] && [ -f "$result_dir/F03_I3_BLEND/complete" ]; then exit 0; fi
  echo "Blend failed seed=$seed attempt=$attempt code=$code; retry after 60s" >&2
  sleep 60
done
test -f "$result_dir/F03_I3_BLEND/complete"
