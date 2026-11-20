#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
DATASET="${1:?dataset}"
HORIZON="${2:?horizon}"
GPU="${3:?gpu}"
PY="${4:?python}"
CASE="$ROOT/results/fpem_paper_completion_20261101/patchtst/$DATASET/pred_$HORIZON"
LOG="$CASE/independent_baselines/queue.log"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT" || exit 1
for wait_no in $(seq 1 720); do
  if [ -s "$CASE/simple_distributions/verified.complete" ]; then break; fi
  sleep 60
done
if [ ! -s "$CASE/simple_distributions/verified.complete" ]; then
  echo "FAILED prerequisite simple baseline $DATASET $HORIZON" >> "$LOG"
  exit 1
fi
for method in timegrad tsflow tsdiff; do
  success=0
  for attempt in 1 2 3; do
    suffix=""
    [ "$attempt" -gt 1 ] && suffix="_retry_$attempt"
    OUT="$CASE/independent_baselines/${method}${suffix}"
    if [ -s "$OUT/complete" ] && [ -s "$OUT/test_metrics.json" ]; then
      echo "SKIP $method verified file exists" >> "$LOG"
      success=1
      break
    fi
    extra=""
    if [ -s "$OUT/final_epoch.pt" ]; then extra="--resume_eval"; fi
    echo "RUN $method $DATASET $HORIZON GPU$GPU attempt=$attempt $(date -Is)" >> "$LOG"
    "$PY" tools/run_independent_probabilistic_baseline.py \
      --method "$method" --dataset "$DATASET" --pred_len "$HORIZON" \
      --seq_len 96 --output "$OUT" --gpu "$GPU" \
      --epochs 13 --batch_size 32 --num_samples 100 $extra >> "$LOG" 2>&1
    code=$?
    if [ "$code" -eq 0 ] && [ -s "$OUT/test_metrics.json" ]; then
      "$PY" -c 'import json,sys,math;p=json.load(open(sys.argv[1]));assert p["evaluated_windows"]==p["total_test_windows"] and p["num_samples"]>=100 and all(math.isfinite(p[k]) for k in ("CRPS","ProbTS_CRPS","PICP_95","MPIW_95"))' "$OUT/test_metrics.json"
      if [ "$?" -eq 0 ]; then
        printf 'verified\n' > "$OUT/paper_verified.complete"
        success=1
        break
      fi
    fi
    echo "RETRY $method code=$code" >> "$LOG"
    sleep 60
  done
  if [ "$success" -ne 1 ]; then
    echo "FAILED $method $DATASET $HORIZON" >> "$LOG"
    exit 1
  fi
done
echo "COMPLETE independent baselines $DATASET $HORIZON $(date -Is)" >> "$LOG"
