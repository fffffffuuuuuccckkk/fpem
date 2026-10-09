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
    patchtst|itransformer|cyclenet|timefilter|moderntcn) echo "1e-4" ;;
    *) echo "unknown FPEM backbone: $1" >&2; return 2 ;;
  esac
}

modern_tcn_config() {
  # patch_size patch_stride num_stages ffn_ratio dropout head_dropout
  # ModernTCN's upstream kernels are 51/5 and all stage dimensions are 64.
  # Keep the common seq_len=96 and search protocol of the PatchTST matrix.
  case "$1" in
    ETTh1)        echo "8 4 1 1 0.3 0.0" ;;
    ETTh2)        echo "8 4 1 1 0.3 0.0" ;;
    ETTm1)        echo "8 4 3 8 0.3 0.0" ;;
    ETTm2)        echo "8 4 3 8 0.3 0.1" ;;
    ExchangeRate) echo "1 1 1 1 0.2 0.6" ;;
    Weather)      echo "8 4 1 8 0.4 0.0" ;;
    Electricity)  echo "8 4 1 8 0.3 0.0" ;;
    Traffic)      echo "8 4 1 8 0.3 0.0" ;;
    *) echo "unknown ModernTCN dataset: $1" >&2; return 2 ;;
  esac
}

timefilter_config() {
  # patch_len d_model d_ff n_heads e_layers dropout positional_embedding
  # pred_len=48 follows the upstream 96-step setting.
  local dataset="$1" horizon="$2"
  case "$dataset" in
    ETTh1)
      [[ "$horizon" == 720 ]] && echo "2 128 128 4 2 0.8 0" || echo "2 128 256 4 2 0.8 0" ;;
    ETTh2)
      case "$horizon" in
        48|96) echo "4 128 256 4 1 0.8 1" ;;
        192)   echo "4 128 256 4 1 0.6 1" ;;
        336)   echo "8 256 256 4 2 0.7 1" ;;
        720)   echo "8 256 256 4 2 0.3 1" ;;
      esac ;;
    ETTm1)
      if [[ "$horizon" == 720 ]]; then
        echo "16 256 256 4 2 0.7 1"
      elif [[ "$horizon" == 48 || "$horizon" == 96 ]]; then
        echo "8 256 256 4 2 0.3 1"
      else
        echo "8 256 256 4 2 0.5 1"
      fi ;;
    ETTm2)
      [[ "$horizon" == 720 ]] && echo "16 128 128 4 2 0.8 1" || echo "16 128 128 4 2 0.6 1" ;;
    Weather)      echo "48 128 256 4 2 0.3 1" ;;
    ExchangeRate) echo "16 128 256 4 2 0.3 1" ;;
    Electricity)  echo "16 128 256 4 2 0.3 1" ;;
    Traffic)      echo "16 128 256 4 2 0.3 1" ;;
    *) echo "unknown TimeFilter dataset: $dataset" >&2; return 2 ;;
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
