#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
echo "starting original deterministic FPEM for ${DATASET:?} / ${PRED_LEN:?}"
env VARIANT_FUSION_MODE=horizon_future_var \
  bash "$PROJECT_DIR/scripts/fpem_probabilistic/run_pilot_matrix.sh"
for ablation in invariant_only deterministic_affine_center affine_flow full \
                shuffled_zvar unconditional_flow gaussian_baseline; do
  echo "starting $ablation for ${DATASET:?} / ${PRED_LEN:?}"
  env PROB_ABLATION="$ablation" \
    bash "$PROJECT_DIR/scripts/fpem_probabilistic/run_pilot_matrix.sh"
done
