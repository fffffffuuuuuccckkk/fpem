#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
OUT="$ROOT/results/fpem_dual_pattern_memory/next_pilot"
for (( attempt=0;attempt<240;attempt++ ));do
  if [[ -f "$OUT/patchtst_ETTh1_720/next_pilot_complete" &&
        -f "$OUT/itransformer_ETTm1_720/next_pilot_complete" &&
        -f "$OUT/moderntcn_ETTm2_720/next_pilot_complete" ]];then
    cd "$ROOT" || exit 1
    /data/OuXiaoyu/miniconda3/envs/basicts/bin/python tools/summarize_fpem_dppm_next.py \
      >"$OUT/summarize.log" 2>&1
    exit $?
  fi
  sleep 60
done
echo "not all cases complete within four hours" >"$OUT/summarize.log"
exit 1
