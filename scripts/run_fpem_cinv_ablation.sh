#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PMG_SPACE=embedding
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-cinv_v1}"

for mode in soft_mapping pattern_only; do
  echo "[cinv-ablation] Running A3 with CINV_MODE=${mode}"
  ABLATION=A3 CINV_MODE="$mode" bash scripts/run_fpem_pmg.sh
done
echo "[cinv-ablation] Completed B1 and B2."
