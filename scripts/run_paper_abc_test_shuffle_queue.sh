#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
PY=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
GPU="${1:?gpu}"
shift
cd "$ROOT" || exit 1
for triplet in "$@"; do
  IFS=: read -r backbone dataset horizon <<< "$triplet"
  if [ "$backbone" = patchtst ] && [ "$horizon" = 96 ]; then
    CASE="$ROOT/results/fpem_paper_completion_20261101/abc_patchtst_etth1_96"
  else
    CASE="$ROOT/results/fpem_abc_mechanism_search_20261031/$backbone/$dataset/pred_$horizon"
  fi
  OUT="$ROOT/results/fpem_paper_completion_20261101/abc_test_shuffle/$backbone/$dataset/pred_$horizon"
  mkdir -p "$OUT"
  if [ -s "$OUT/verified.complete" ] && [ -s "$OUT/metrics.csv" ]; then
    echo "SKIP $triplet" >> "$OUT/queue.log"
    continue
  fi
  success=0
  for attempt in 1 2 3; do
    echo "RUN $triplet GPU$GPU attempt=$attempt $(date -Is)" >> "$OUT/queue.log"
    "$PY" tools/fpem_paper_abc_test_shuffle.py --case "$CASE" \
      --output "$OUT" --gpu "$GPU" >> "$OUT/queue.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ] && [ -s "$OUT/verified.complete" ] && [ -s "$OUT/metrics.csv" ]; then
      success=1
      break
    fi
    echo "RETRY code=$code" >> "$OUT/queue.log"
    sleep 60
  done
  if [ "$success" -ne 1 ]; then
    echo "FAILED $triplet" >> "$OUT/queue.log"
    exit 1
  fi
done
