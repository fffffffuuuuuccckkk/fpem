#!/usr/bin/env bash
set -u -o pipefail
case_name="${1:?case name required}"
gpu="${2:?gpu index required}"
repo=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
output=results/fpem_dual_pattern_memory/g1_plus_r3_pilot
cd "$repo" || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4
mkdir -p "$output/$case_name"
if [ -f "$output/$case_name/R3/complete" ]; then
  echo "SKIP $case_name R3 complete"
  exit 0
fi
for attempt in 1 2; do
  echo "START $case_name G1+R3 attempt=$attempt"
  if "$python" tools/run_fpem_dppm_rare.py "$case_name" \
      --methods R3 --stable_mode g1 --gpu 0 --patterns 8 --rank 8 \
      --epochs 10 --eval_stride 4 --controls --output "$output"; then
    echo "DONE $case_name G1+R3"
    exit 0
  fi
  if [ "$attempt" -eq 1 ]; then
    echo "RETRY $case_name after 60s"
    sleep 60
  fi
done
echo "FAILED $case_name G1+R3"
exit 1
