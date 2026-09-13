#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
TAG="${EXPERIMENT_TAG:-predmap_v1}"

for PRED_STAB_MODE in per_future_znorm train_global relative_future; do
  echo "Starting predictive-stability ablation: ${PRED_STAB_MODE}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PMG_SPACE=raw ABLATION=A3 \
    CINV_MODE=pattern_only ZINV_MODE=stable_relation_correction \
    FREEZE_PATCH_STAGE2=true VARIANT_INPUT_MODE=latent_only \
    PRED_STAB_MODE="$PRED_STAB_MODE" MAPPING_USE_MODE=full \
    EXPERIMENT_TAG="${TAG}_pred" bash scripts/run_fpem_pmg.sh
done

# P1 is the requested main non-collapsed definition. P0/P1/P2 diagnostics
# remain separate so the final choice is not made from MSE alone. The P1 run
# above is already the full (M3) mapping run, so do not train that exact
# configuration twice or overwrite/reuse its result directory.
for MAPPING_USE_MODE in none context_only delta_only; do
  echo "Starting mapping-contribution ablation: ${MAPPING_USE_MODE}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PMG_SPACE=raw ABLATION=A3 \
    CINV_MODE=pattern_only ZINV_MODE=stable_relation_correction \
    FREEZE_PATCH_STAGE2=true VARIANT_INPUT_MODE=latent_only \
    PRED_STAB_MODE=train_global MAPPING_USE_MODE="$MAPPING_USE_MODE" \
    EXPERIMENT_TAG="${TAG}_map" bash scripts/run_fpem_pmg.sh
done

echo "Mapping M3/full is supplied by the P1/train_global predictive run above."
