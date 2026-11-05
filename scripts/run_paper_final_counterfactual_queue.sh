#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
OUT="$ROOT/results/fpem_paper_final_evidence_20261101"
PY=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
GPU="${1:?gpu}"
shift
cd "$ROOT" || exit 1
for triplet in "$@"; do
  IFS=: read -r backbone dataset horizon <<< "$triplet"
  if [ "$backbone" = patchtst ] && [ "$horizon" = 96 ]; then
    case_dir="$ROOT/results/fpem_paper_completion_20261101/abc_patchtst_etth1_96"
  else
    case_dir="$ROOT/results/fpem_abc_mechanism_search_20261031/$backbone/$dataset/pred_$horizon"
  fi
  output="$OUT/condition_counterfactual/$backbone/$dataset/pred_$horizon"
  mkdir -p "$output"
  if [ -s "$output/verified.complete" ] && [ -s "$output/condition_counterfactual.csv" ]; then
    echo "SKIP $triplet" >> "$output/queue.log"
    continue
  fi
  if [ ! -s "$case_dir/C12_all_fused.pt" ] || [ ! -s "$case_dir/C_risk_cache.pt" ]; then
    echo "MISSING prerequisite $triplet" >> "$output/queue.log"
    exit 1
  fi
  success=0
  for attempt in 1 2 3; do
    echo "RUN $triplet GPU$GPU attempt=$attempt $(date -Is)" >> "$output/queue.log"
    "$PY" tools/fpem_paper_final_counterfactual.py \
      --case "$case_dir" --output "$output" --gpu "$GPU" \
      >> "$output/queue.log" 2>&1
    code=$?
    if [ "$code" -eq 0 ] && [ -s "$output/verified.complete" ]; then success=1; break; fi
    echo "RETRY $triplet code=$code" >> "$output/queue.log"
    sleep 60
  done
  if [ "$success" -ne 1 ]; then echo "FAILED $triplet" >> "$output/queue.log"; exit 1; fi
done
