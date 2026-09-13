#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

for ZINV_MODE in hidden_only stable_relation_correction; do
  echo "Starting FPem-PMG Z_inv ablation: ${ZINV_MODE}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  PMG_SPACE=embedding \
  ABLATION=A3 \
  CINV_MODE=pattern_only \
  ZINV_MODE="$ZINV_MODE" \
  EXPERIMENT_TAG="${EXPERIMENT_TAG:-zinv_v1}" \
  bash scripts/run_fpem_pmg.sh
done
