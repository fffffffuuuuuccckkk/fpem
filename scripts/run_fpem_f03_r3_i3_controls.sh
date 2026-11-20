#!/usr/bin/env bash
set -u
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
seed="$1"
gpu="$2"
root=results/fpem_performance_optimization_20261103/f03_r3_prototype_invariance
source_dir="$root/moderntcn/ETTm2/pred_720/seed_$seed"
target_dir="$root/i3_controls_full/moderntcn/ETTm2/pred_720/seed_$seed"
mkdir -p "$target_dir"
if [ ! -f "$target_dir/r3.pt" ]; then
  cp "$source_dir/r3.pt" "$target_dir/r3.pt" || exit 3
fi
attempt=0
while [ ! -f "$target_dir/complete" ] && [ "$attempt" -lt 3 ]; do
  attempt=$((attempt+1))
  /data/OuXiaoyu/miniconda3/envs/basicts/bin/python -u tools/run_fpem_f03_r3_invariance.py \
    --base_run_config results/fpem_dual_pattern_memory/g1_r3_full_point/moderntcn/ETTm2/pred_720/base_run_config.json \
    --output "$target_dir" --gpu "$gpu" --seed "$seed" --epochs 10 \
    --train_stride 16 --val_stride 4 --test_stride 1 \
    --lrs 0.0003,0.001 --lambdas 0.001,0.01 \
    --methods F03_I3_shuffled,F03_I3_random,F03_I3_time
  code=$?
  if [ "$code" -eq 0 ] && [ -f "$target_dir/complete" ]; then exit 0; fi
  echo "I3 CONTROL FAILED seed=$seed attempt=$attempt code=$code" >&2
  sleep 60
done
if [ -f "$target_dir/complete" ]; then exit 0; fi
exit 1
