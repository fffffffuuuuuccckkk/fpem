#!/usr/bin/env bash
set -u
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
variant="$1"
gpu="$2"
root=results/fpem_performance_optimization_20261103/f03_r3_prototype_invariance
source_dir="$root/moderntcn/ETTm2/pred_720/seed_2022"
target_dir="$root/knob_search/$variant/moderntcn/ETTm2/pred_720/seed_2022"
mkdir -p "$target_dir"
if [ ! -f "$target_dir/r3.pt" ]; then
  cp "$source_dir/r3.pt" "$target_dir/r3.pt" || exit 3
fi
case "$variant" in
  support_mass)
    methods=F03_I1,F03_I2
    extra=(--support_weighting mass)
    ;;
  support_uniform)
    methods=F03_I1,F03_I2
    extra=(--support_weighting uniform)
    ;;
  gradient_rms)
    methods=F03_I2
    extra=(--gradient_normalization rms)
    ;;
  constraint_late)
    methods=F03_I3
    extra=(--constraint_start_epoch 3)
    ;;
  *) echo "unknown variant $variant" >&2; exit 4;;
esac
attempt=0
while [ ! -f "$target_dir/complete" ] && [ "$attempt" -lt 3 ]; do
  attempt=$((attempt+1))
  /data/OuXiaoyu/miniconda3/envs/basicts/bin/python -u tools/run_fpem_f03_r3_invariance.py \
    --base_run_config results/fpem_dual_pattern_memory/g1_r3_full_point/moderntcn/ETTm2/pred_720/base_run_config.json \
    --output "$target_dir" --gpu "$gpu" --seed 2022 --epochs 10 \
    --train_stride 16 --val_stride 4 --test_stride 1 \
    --lrs 0.001 --lambdas 0.01 --methods "$methods" "${extra[@]}"
  code=$?
  if [ "$code" -eq 0 ] && [ -f "$target_dir/complete" ]; then exit 0; fi
  echo "KNOB FAILED variant=$variant attempt=$attempt code=$code" >&2
  sleep 60
done
if [ -f "$target_dir/complete" ]; then exit 0; fi
exit 1
