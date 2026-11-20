#!/usr/bin/env bash
set -u -o pipefail
case_name="${1:?case name required}"
gpu="${2:?gpu index required}"
repo=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
output=results/fpem_dual_pattern_memory/rare_learning_pilot
cd "$repo" || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4
mkdir -p "$output/$case_name"
for method in R0 R1 R2 R3 R4 R5 R6 R7; do
  if [ -f "$output/$case_name/$method/complete" ]; then
    echo "SKIP $case_name $method complete"
    continue
  fi
  attempt=0
  while [ "$attempt" -lt 2 ]; do
    attempt=$((attempt+1))
    echo "START $case_name $method attempt=$attempt"
    if "$python" tools/run_fpem_dppm_rare.py "$case_name" \
        --methods "$method" --gpu 0 --patterns 8 --rank 8 --epochs 10 \
        --eval_stride 4 --controls --output "$output"; then
      break
    fi
    if [ "$attempt" -lt 2 ]; then
      echo "RETRY after 60s: $case_name $method"
      sleep 60
    else
      echo "FAILED twice: $case_name $method; continuing other methods"
      printf '%s\n' "$method" >> "$output/$case_name/failed_methods.txt"
    fi
  done
done
"$python" tools/summarize_fpem_dppm_rare.py --output "$output" || true
echo "CASE FINISHED $case_name"
