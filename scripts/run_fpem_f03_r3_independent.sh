#!/usr/bin/env bash
set -u
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
seed="$1"
gpu="$2"
root=results/fpem_performance_optimization_20261103/f03_r3_prototype_invariance
result_dir="$root/independent_moderntcn_ettm2_336/seed_$seed"
mkdir -p "$result_dir"
attempt=0
while [ ! -f "$result_dir/complete" ] && [ "$attempt" -lt 3 ]; do
  attempt=$((attempt+1))
  /data/OuXiaoyu/miniconda3/envs/basicts/bin/python -u tools/run_fpem_f03_r3_invariance.py \
    --base_run_config results/fpem_dual_pattern_memory/g1_r3_full_point/moderntcn/ETTm2/pred_336/base_run_config.json \
    --output "$result_dir" --gpu "$gpu" --seed "$seed" --epochs 10 \
    --train_stride 16 --val_stride 4 --test_stride 1 \
    --lrs 0.001 --lambdas 0.01 \
    --methods F03,F03_I3,F03_I3_shuffled,F03_I3_random
  code=$?
  if [ "$code" -eq 0 ] && [ -f "$result_dir/complete" ]; then exit 0; fi
  echo "INDEPENDENT FAILED seed=$seed attempt=$attempt code=$code" >&2
  sleep 60
done
if [ -f "$result_dir/complete" ]; then exit 0; fi
exit 1
