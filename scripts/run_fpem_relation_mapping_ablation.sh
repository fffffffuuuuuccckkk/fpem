#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PMG_SPACE=raw
export FREEZE_PATCH_STAGE2=true
export ABLATION=A5
export CINV_MODE=pattern_only
export ZINV_MODE=stable_relation_correction
export PRED_STAB_MODE=relative_future
export MAPPING_USE_MODE=delta_only
export VARIANT_INPUT_MODE=latent_only
export FUSION_MODE=typed_raw
export TYPED_COMPONENTS=full
export EXPERIMENT_TAG="relation_mapping_v1"

for mode in legacy decomposed_no_variant decomposed_full variant_only; do
  export RELATION_MAPPING_MODE="$mode"
  bash scripts/run_fpem_pmg.sh
done
