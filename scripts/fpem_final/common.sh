#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"
SEED="${SEED:-2021}"

dataset_config() {
  case "$1" in
    ETTh1)        echo "./dataset/all_datasets/ETT-small ett_hour ETTh1.csv h 32 24 7" ;;
    ETTh2)        echo "./dataset/all_datasets/ETT-small ett_hour ETTh2.csv h 32 24 7" ;;
    ETTm1)        echo "./dataset/all_datasets/ETT-small ett_minute ETTm1.csv 15min 32 96 7" ;;
    ETTm2)        echo "./dataset/all_datasets/ETT-small ett_minute ETTm2.csv 15min 32 96 7" ;;
    ExchangeRate) echo "./dataset/all_datasets/exchange_rate custom exchange_rate.csv d 32 7 8" ;;
    Weather)      echo "./dataset/all_datasets/weather custom weather.csv 10min 16 144 21" ;;
    Electricity)  echo "./dataset/all_datasets/electricity custom electricity.csv h 4 168 321" ;;
    Traffic)      echo "./dataset/all_datasets/traffic custom traffic.csv h 2 168 862" ;;
    *) echo "unknown dataset: $1" >&2; return 2 ;;
  esac
}

backbone_lr() {
  case "$1" in
    patchtst|itransformer|cyclenet) echo "1e-4" ;;
    *) echo "unknown FPEM backbone: $1" >&2; return 2 ;;
  esac
}

result_complete() {
  local root="$1"
  [[ -s "$root/A2/metrics_and_diagnostics.json" &&
     -s "$root/run_config.json" &&
     -s "$root/protocol.txt" &&
     -e "$root/run_complete" ]]
}

archive_incomplete_result() {
  local root="$1"
  if [[ -e "$root" ]] && ! result_complete "$root"; then
    local archived="${root}.incomplete.$(date +%Y%m%d_%H%M%S)"
    mv "$root" "$archived"
    echo "archived incomplete result: $root -> $archived"
  fi
}

json_metric() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
v = d
for part in sys.argv[2].split('.'):
    v = v[part]
print(v)
PY
}

shell_quote_command() {
  printf '%q ' "$@"
  printf '\n'
}
