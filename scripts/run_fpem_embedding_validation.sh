#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PMG_SPACE=embedding
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-dualspace_v1}"
PYTHON_EXEC="${PYTHON_BIN:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
A1_HIDDEN_MSE="0.3984408081"

latest_metric() {
  local ablation="$1"
  find results -mindepth 2 -maxdepth 2 -type f -name metrics.npy \
    -path "*fpem_pmg_emb_${ablation}_cinv_B0_product_${EXPERIMENT_TAG}_0/metrics.npy" \
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

echo "[embedding-validation] Running A1_embedding first"
ABLATION=A1 bash scripts/run_fpem_pmg.sh
A1_METRIC="$(latest_metric A1)"
A1_EMBEDDING_MSE="$(read_mse "$A1_METRIC")"
echo "[embedding-validation] A1_hidden=${A1_HIDDEN_MSE}; A1_embedding=${A1_EMBEDDING_MSE}"

if ! less_than "$A1_EMBEDDING_MSE" "$A1_HIDDEN_MSE"; then
  echo "[embedding-validation] A1_embedding did not improve over A1_hidden; stopping before A2/A3/A5/A6."
  exit 0
fi

echo "[embedding-validation] A1_embedding improved over hidden; running A2, A3, A5, A6"
for ablation in A2 A3 A5 A6; do
  echo "[embedding-validation] Running ${ablation}_embedding"
  ABLATION="$ablation" bash scripts/run_fpem_pmg.sh
done
echo "[embedding-validation] Completed embedding sequence."
