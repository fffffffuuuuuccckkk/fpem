#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
OUT="$ROOT/results/fpem_dual_pattern_memory/diagnostics"
for (( attempt=0; attempt<120; attempt++ )); do
  if [[ -f "$OUT/patchtst_ETTh1_720/diagnostic_complete" &&
        -f "$OUT/itransformer_ETTm1_720/diagnostic_complete" &&
        -f "$OUT/moderntcn_ETTm2_720/diagnostic_complete" ]]; then
    cd "$ROOT" || exit 1
    /data/OuXiaoyu/miniconda3/envs/basicts/bin/python tools/summarize_fpem_dppm_diagnostics.py \
      >"$OUT/summarize.log" 2>&1
    exit $?
  fi
  sleep 60
done
echo "diagnostic cases did not all finish within 120 minutes" >"$OUT/summarize.log"
exit 1
