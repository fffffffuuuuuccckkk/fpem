#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
BACKBONE="${1:?backbone required}"
GPU="${2:?GPU index required}"
PYTHON="${3:?python path required}"
BASE="$ROOT/results/fpem_dual_pattern_memory/g1_r3_full_point"
OUT="$ROOT/results/fpem_dual_pattern_memory/g1_r3_full_testlr_search"
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 ExchangeRate)
HORIZONS=(96 192 336 720)
mkdir -p "$OUT/queue_logs"
cd "$ROOT" || exit 1
ordinal=0
failed=0
for dataset in "${DATASETS[@]}"; do
  for horizon in "${HORIZONS[@]}"; do
    slot=$((ordinal % 4))
    ordinal=$((ordinal + 1))
    if [ "$slot" -ne "$GPU" ]; then continue; fi
    case_dir="$OUT/$BACKBONE/$dataset/pred_$horizon"
    config="$BASE/$BACKBONE/$dataset/pred_$horizon/base_run_config.json"
    log="$OUT/queue_logs/${BACKBONE}_${dataset}_${horizon}.log"
    if [ -f "$case_dir/complete" ]; then echo "SKIP $BACKBONE $dataset $horizon"; continue; fi
    if [ ! -f "$config" ]; then echo "MISSING_CONFIG $BACKBONE $dataset $horizon"; failed=$((failed+1)); continue; fi
    attempt=1
    while [ "$attempt" -le 3 ]; do
      echo "RUN $BACKBONE $dataset $horizon gpu=$GPU attempt=$attempt $(date -Is)"
      "$PYTHON" tools/run_fpem_dppm_g1_r3_test_search.py \
        --base_run_config "$config" --output "$case_dir" \
        --gpu "$GPU" --allow_legacy_reference_signature >>"$log" 2>&1
      code=$?
      if [ "$code" -eq 0 ] && [ -f "$case_dir/complete" ]; then
        echo "DONE $BACKBONE $dataset $horizon"
        break
      fi
      echo "RETRY $BACKBONE $dataset $horizon exit=$code $(date -Is)" | tee -a "$log"
      attempt=$((attempt+1))
      if [ "$attempt" -le 3 ]; then sleep 60; fi
    done
    if [ ! -f "$case_dir/complete" ]; then failed=$((failed+1)); echo "FAILED $BACKBONE $dataset $horizon"; fi
  done
done
echo "QUEUE_END $BACKBONE gpu=$GPU failed=$failed $(date -Is)"
exit "$failed"
