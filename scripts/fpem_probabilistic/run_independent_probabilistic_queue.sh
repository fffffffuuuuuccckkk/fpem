#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
ROOT="$PROJECT_DIR/results/fpem_probabilistic/independent_baselines_24_48"
GPU="${1:?GPU required}"
shift
cd "$PROJECT_DIR"
for case in "$@"; do
  IFS=: read -r method dataset horizon <<<"$case"
  output="$ROOT/$method/$dataset/pred_$horizon"
  if [[ -s "$output/complete" && -s "$output/test_metrics.json" ]]; then
    echo "REUSE $case $output"
    continue
  fi
  extra=()
  if [[ -e "$output" ]]; then
    if [[ -s "$output/final_epoch.pt" ]]; then
      extra+=(--resume_eval)
    else
      echo "INCOMPLETE, requires inspection: $output" >&2
      exit 3
    fi
  fi
  chunk=100
  if [[ "$method" == tsdiff ]]; then chunk=20; fi
  echo "START $case gpu=$GPU output=$output"
  "$PYTHON" -u tools/run_independent_probabilistic_baseline.py \
    --method "$method" --dataset "$dataset" --pred_len "$horizon" \
    --output "$output" --gpu "$GPU" --seed 2021 --epochs 13 \
    --batch_size 32 --lr 1e-4 --hidden 64 --num_samples 100 \
    --sample_chunk "$chunk" --max_train_batches 0 --max_test_windows 0 \
    "${extra[@]}"
  echo "FINISH $case"
done
