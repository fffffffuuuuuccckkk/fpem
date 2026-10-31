#!/usr/bin/env bash
set -u -o pipefail
repo=/data/OuXiaoyu/Time-Series-Library-FPEM
root="$repo/results/fpem_dual_pattern_memory/g1_r3_optimization_pilot"
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
for case_name in patchtst_ETTh1_720 itransformer_ETTm1_720 moderntcn_ETTm2_720; do
  while [ ! -f "$root/$case_name/complete" ]; do
    sleep 30
  done
done
cd "$repo" || exit 2
"$python" tools/summarize_fpem_dppm_g1_r3_optimization.py
