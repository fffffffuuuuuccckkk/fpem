#!/usr/bin/env bash
# One source case had completed trials but no best_config.json.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
cd "$PROJECT_DIR"
while screen -ls | grep -q '[.]fpem_nodom_short'; do sleep 60; done
while true; do
  used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 256 )) && break
  sleep 30
done
bash scripts/fpem_probabilistic/run_remaining_length_ablations.sh case \
  results/fpem_probabilistic/moderntcn_weather_336_recovered_best.json 0
