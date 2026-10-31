#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
OUT="$ROOT/results/fpem_dual_pattern_memory/g1_r3_extended_test_search"
DATASETS_CSV="${1:?datasets CSV required}"
SLOT="${2:?0 or 1 required}"
GPU_BASE="${3:?gpu base required}"
PY="${4:?python required}"
IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"
mkdir -p "$OUT/queue_logs"
cd "$ROOT" || exit 1
ordinal=0
failed=0
for dataset in "${DATASETS[@]}"; do
  for horizon in 24 48; do
    assigned=$((ordinal % 2))
    ordinal=$((ordinal + 1))
    if [ "$assigned" -ne "$SLOT" ]; then continue; fi
    gpu=$((GPU_BASE + SLOT))
    case_dir="$OUT/patchtst/$dataset/pred_$horizon"
    log="$OUT/queue_logs/prob_crps_${dataset}_${horizon}.log"
    if [ -f "$case_dir/probability_search_ema_lr/complete" ]; then
      echo "SKIP $dataset $horizon"; continue
    fi
    wait_count=0
    while [ ! -f "$case_dir/search/complete" ] && [ "$wait_count" -lt 360 ]; do
      sleep 60
      wait_count=$((wait_count + 1))
    done
    if [ ! -f "$case_dir/search/complete" ]; then
      echo "MISSING point anchor $dataset $horizon" | tee -a "$log"
      failed=$((failed + 1)); continue
    fi
    attempt=1
    while [ "$attempt" -le 3 ]; do
      echo "RUN $dataset $horizon gpu=$gpu attempt=$attempt $(date -Is)" | tee -a "$log"
      "$PY" tools/run_fpem_g1r3_probability_search.py \
        --case_dir "$case_dir" --gpu "$gpu" \
        --output "$case_dir/probability_search_ema_lr" \
        --samples 100 --epochs 10 --train_stride 32 --test_stride 16 \
        >>"$log" 2>&1
      status=$?
      if [ "$status" -eq 0 ] && [ -f "$case_dir/probability_search_ema_lr/complete" ]; then
        echo "DONE $dataset $horizon" | tee -a "$log"
        break
      fi
      echo "RETRY $dataset $horizon exit=$status" | tee -a "$log"
      attempt=$((attempt + 1))
      if [ "$attempt" -le 3 ]; then sleep 60; fi
    done
    if [ ! -f "$case_dir/probability_search_ema_lr/complete" ]; then
      failed=$((failed + 1)); echo "FAILED $dataset $horizon" | tee -a "$log"
    fi
  done
done
echo "QUEUE_END datasets=$DATASETS_CSV slot=$SLOT failed=$failed $(date -Is)"
exit "$failed"
