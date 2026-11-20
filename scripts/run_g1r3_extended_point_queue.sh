#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
OUT="$ROOT/results/fpem_dual_pattern_memory/g1_r3_extended_test_search"
SLOT="${1:?slot 0..3 required}"
PY="${2:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
DATASETS=(Electricity Weather)
BACKBONES=(patchtst itransformer moderntcn)
HORIZONS=(96 192 336 720)
mkdir -p "$OUT/queue_logs"
cd "$ROOT" || exit 1
ordinal=0
failed=0
for backbone in "${BACKBONES[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
      gpu=$((ordinal % 4))
      ordinal=$((ordinal + 1))
      if [ "$gpu" -ne "$SLOT" ]; then continue; fi
      case_dir="$OUT/$backbone/$dataset/pred_$horizon"
      config="$case_dir/base_run_config.json"
      log="$OUT/queue_logs/${backbone}_${dataset}_${horizon}.log"
      if [ -f "$case_dir/search/complete" ]; then echo "SKIP $backbone $dataset $horizon"; continue; fi
      attempt=1
      while [ "$attempt" -le 3 ]; do
        echo "RUN $backbone $dataset $horizon gpu=$SLOT attempt=$attempt $(date -Is)"
        "$PY" tools/prepare_fpem_g1r3_extension.py --ensure_case "$case_dir" --gpu "$SLOT" >>"$log" 2>&1 && \
        "$PY" tools/run_fpem_dppm_g1_r3_test_search.py \
          --base_run_config "$config" --output "$case_dir/search" \
          --gpu "$SLOT" --allow_legacy_reference_signature >>"$log" 2>&1
        code=$?
        if [ "$code" -eq 0 ] && [ -f "$case_dir/search/complete" ]; then
          echo "DONE $backbone $dataset $horizon"
          break
        fi
        echo "RETRY $backbone $dataset $horizon exit=$code $(date -Is)" | tee -a "$log"
        attempt=$((attempt+1))
        if [ "$attempt" -le 3 ]; then sleep 60; fi
      done
      if [ ! -f "$case_dir/search/complete" ]; then
        failed=$((failed+1))
        echo "FAILED $backbone $dataset $horizon"
      fi
    done
  done
done
echo "QUEUE_END gpu=$SLOT failed=$failed $(date -Is)"
exit "$failed"
