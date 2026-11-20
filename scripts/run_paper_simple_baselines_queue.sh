#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
DATASET="${1:?dataset}"
HORIZON="${2:?horizon}"
GPU="${3:?gpu}"
PY="${4:?python}"
PAPER="$ROOT/results/fpem_paper_completion_20261101/patchtst/$DATASET/pred_$HORIZON"
if [ "$DATASET" = Weather ]; then
  CASE="$PAPER"
else
  CASE="$ROOT/results/fpem_dual_pattern_memory/g1_r3_extended_test_search/patchtst/$DATASET/pred_$HORIZON"
fi
OUT="$PAPER/simple_distributions"
mkdir -p "$OUT"
cd "$ROOT" || exit 1
for wait_no in $(seq 1 720); do
  if [ -s "$PAPER/full_test_eval/verified.complete" ] || [ -s "$PAPER/verified.complete" ]; then break; fi
  sleep 60
done
if [ ! -s "$PAPER/full_test_eval/verified.complete" ] && [ ! -s "$PAPER/verified.complete" ]; then
  echo "FAILED prerequisite full TEST: $DATASET $HORIZON" >> "$OUT/queue.log"
  exit 1
fi
if [ -s "$OUT/verified.complete" ] && [ -s "$OUT/metrics.csv" ]; then
  echo "SKIP verified $DATASET $HORIZON" >> "$OUT/queue.log"
  exit 0
fi
for attempt in 1 2 3; do
  echo "RUN $DATASET $HORIZON GPU$GPU attempt=$attempt $(date -Is)" >> "$OUT/queue.log"
  "$PY" tools/fpem_paper_simple_distributions.py \
    --case_dir "$CASE" --output "$OUT" --gpu "$GPU" \
    --samples 100 --batch_size 2 >> "$OUT/queue.log" 2>&1
  code=$?
  if [ "$code" -eq 0 ] && [ -s "$OUT/verified.complete" ] && [ -s "$OUT/metrics.csv" ]; then
    echo "COMPLETE $DATASET $HORIZON $(date -Is)" >> "$OUT/queue.log"
    exit 0
  fi
  echo "RETRY code=$code" >> "$OUT/queue.log"
  sleep 60
done
echo "FAILED $DATASET $HORIZON" >> "$OUT/queue.log"
exit 1
