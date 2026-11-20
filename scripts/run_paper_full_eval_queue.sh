#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PY="${1:?python path required}"
GPU="${2:?gpu required}"
shift 2
cd "$ROOT" || exit 1
failed=0
for item in "$@"; do
  dataset="${item%%:*}"
  horizon="${item##*:}"
  CASE="$ROOT/results/fpem_dual_pattern_memory/g1_r3_extended_test_search/patchtst/$dataset/pred_$horizon"
  OUT="$ROOT/results/fpem_paper_completion_20261101/patchtst/$dataset/pred_$horizon/full_test_eval"
  mkdir -p "$OUT"
  if [ -s "$OUT/verified.complete" ]; then echo "SKIP $item"; continue; fi
  if [ ! -s "$CASE/search/model.pt" ] || \
     [ ! -s "$CASE/probability_search_ema_lr/metrics.json" ]; then
    echo "MISSING SOURCE $item" | tee -a "$OUT/queue.log"
    failed=$((failed+1))
    continue
  fi
  success=0
  for attempt in 1 2 3; do
    echo "RUN $item GPU$GPU attempt=$attempt $(date -Is)" | tee -a "$OUT/queue.log"
    "$PY" tools/fpem_paper_full_eval.py \
      --case_dir "$CASE" --prob_dir "$CASE/probability_search_ema_lr" \
      --output "$OUT" --gpu "$GPU" --samples 100 --batch_size 2 \
      --val_stride 4 >> "$OUT/queue.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ] && [ -s "$OUT/verified.complete" ]; then success=1; break; fi
    echo "RETRY $item code=$code" | tee -a "$OUT/queue.log"
    sleep 60
  done
  if [ "$success" -ne 1 ]; then failed=$((failed+1)); fi
done
echo "QUEUE END failed=$failed $(date -Is)"
exit "$failed"
