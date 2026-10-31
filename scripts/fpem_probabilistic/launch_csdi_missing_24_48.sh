#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
RESULT_ROOT="$PROJECT_DIR/results/fpem_probabilistic/independent_baselines_24_48"
mkdir -p "$RESULT_ROOT/logs"
for spec in '0 ETTm1' '2 ETTm2' '3 ExchangeRate'; do
  read -r gpu dataset <<<"$spec"
  screen_name="csdi_${dataset,,}_24_48"
  if screen -ls | grep -Fq ".$screen_name"; then
    echo "already running: $screen_name"
    continue
  fi
  screen -L -Logfile "$RESULT_ROOT/logs/${screen_name}.log" -dmS "$screen_name" bash -lc '
    set -euo pipefail
    project="$1"; root="$2"; gpu="$3"; dataset="$4"
    for horizon in 24 48; do
      existing="$project/results/fpem_probabilistic/unified_lr_search_20261010/csdi/$dataset/pred_$horizon/test_metrics.json"
      target="$root/csdi/$dataset/pred_$horizon"
      if [[ -s "$existing" || -s "$target/test_metrics.json" ]]; then
        echo "reuse complete $dataset $horizon"
        continue
      fi
      if [[ -e "$target" ]]; then
        echo "incomplete target requires inspection: $target" >&2
        exit 3
      fi
      DATASET="$dataset" PRED_LEN="$horizon" GPU="$gpu" \
        OUTPUT_ROOT="$root/csdi" \
        bash "$project/scripts/fpem_probabilistic/run_csdi_aligned.sh"
    done
  ' _ "$PROJECT_DIR" "$RESULT_ROOT" "$gpu" "$dataset"
  echo "started $screen_name GPU=$gpu"
done
