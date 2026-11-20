#!/usr/bin/env bash
set -u -o pipefail
repo=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
root=results/fpem_dual_pattern_memory/rare_learning_pilot
cd "$repo" || exit 2
export CUDA_VISIBLE_DEVICES=3
for case_name in patchtst_ETTh1_720 itransformer_ETTm1_720 moderntcn_ETTm2_720; do
  while :; do
    count=$(find "$root/$case_name" -mindepth 2 -maxdepth 2 -name complete -type f | wc -l)
    if [ "$count" -ge 8 ]; then break; fi
    if [ -f "$root/$case_name/failed_methods.txt" ]; then
      echo "Case $case_name failed methods; cannot reconcile full set"
      exit 3
    fi
    sleep 60
  done
  "$python" tools/reconcile_dppm_rare_diagnostics.py "$case_name" --gpu 0 --output "$root" || exit 4
done
"$python" tools/summarize_fpem_dppm_rare.py --output "$root"
echo "RECONCILIATION FINISHED"
