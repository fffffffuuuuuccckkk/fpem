#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-stage0full_v1}"
PYTHON_EXEC="${PYTHON_BIN:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
A0_MSE="0.3908533454"

latest_metric() {
  local ablation="$1"
  find results -mindepth 2 -maxdepth 2 -type f -name metrics.npy \
    -path "*_${ablation}_*fpem_pmg_${ablation}_${EXPERIMENT_TAG}_0/metrics.npy" \
    -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {$1=""; sub(/^ /,""); print}'
}

read_mse() {
  "$PYTHON_EXEC" - "$1" <<'PY'
import numpy as np
import sys
print(float(np.load(sys.argv[1], allow_pickle=False)[1]))
PY
}

less_than() {
  "$PYTHON_EXEC" - "$1" "$2" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)
PY
}

not_materially_worse() {
  "$PYTHON_EXEC" - "$1" "$2" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= float(sys.argv[2]) + 0.002 else 1)
PY
}

echo "[stage0-validation] Running A1 first"
ABLATION=A1 bash scripts/run_fpem_pmg.sh
A1_METRIC="$(latest_metric A1)"
A1_MSE="$(read_mse "$A1_METRIC")"
echo "[stage0-validation] A0 MSE=${A0_MSE}; A1_new MSE=${A1_MSE}"

if ! less_than "$A1_MSE" "$A0_MSE"; then
  echo "[stage0-validation] A1_new did not beat A0; stopping before A3/A6 as requested."
  exit 0
fi

echo "[stage0-validation] A1_new beat A0; running A3"
ABLATION=A3 bash scripts/run_fpem_pmg.sh
A3_METRIC="$(latest_metric A3)"
A3_MSE="$(read_mse "$A3_METRIC")"
echo "[stage0-validation] A1_new MSE=${A1_MSE}; A3_new MSE=${A3_MSE}"

if ! not_materially_worse "$A3_MSE" "$A1_MSE"; then
  echo "[stage0-validation] A3_new is materially worse than A1_new; stopping before A6."
  exit 0
fi

echo "[stage0-validation] A3_new is reasonable; running A6"
ABLATION=A6 bash scripts/run_fpem_pmg.sh
echo "[stage0-validation] Completed A1 -> A3 -> A6 sequence."
