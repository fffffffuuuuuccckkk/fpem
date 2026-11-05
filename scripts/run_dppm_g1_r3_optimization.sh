#!/usr/bin/env bash
set -u -o pipefail
case_name="${1:?case name required}"
gpu="${2:?gpu index required}"
repo=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
root=results/fpem_dual_pattern_memory/g1_r3_optimization_pilot
cd "$repo" || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4
mkdir -p "$root/$case_name"
if [ -f "$root/$case_name/complete" ]; then
  echo "SKIP $case_name complete"
  exit 0
fi
for attempt in 1 2 3; do
  echo "START $case_name attempt=$attempt"
  if "$python" tools/run_fpem_dppm_g1_r3_optimization.py "$case_name" \
      --gpu 0 --seed 2021 --eval_stride 4 --max_lr_trials 24 \
      --output "$root"; then
    echo "COMPLETE $case_name"
    exit 0
  fi
  if [ "$attempt" -lt 3 ]; then
    echo "RETRY $case_name after 60 seconds"
    sleep 60
  fi
done
echo "FAILED $case_name after three attempts"
exit 1
