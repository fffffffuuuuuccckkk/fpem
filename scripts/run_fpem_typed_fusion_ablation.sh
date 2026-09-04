#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
TAG="${EXPERIMENT_TAG:-typedfusion_v1}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

run_fusion() {
  local fusion_mode="$1"
  local components="$2"
  local suffix="$3"
  CUDA_VISIBLE_DEVICES="$GPU" PMG_SPACE=raw ABLATION=A5 \
    CINV_MODE=pattern_only ZINV_MODE=stable_relation_correction \
    FREEZE_PATCH_STAGE2=true PRED_STAB_MODE=relative_future \
    MAPPING_USE_MODE=delta_only VARIANT_INPUT_MODE=latent_only \
    FUSION_MODE="$fusion_mode" TYPED_COMPONENTS="$components" \
    EXPERIMENT_TAG="${TAG}_${suffix}" bash scripts/run_fpem_pmg.sh
}

echo "Starting F0 invariant-only"
run_fusion invariant_only full F0

echo "Starting F1 legacy latent environment (reliability off)"
run_fusion legacy_environment full F1

echo "Starting F2 typed shape residual"
run_fusion typed_raw shape F2

echo "Starting F3 typed geometry: multiplicative scale + additive shift"
run_fusion typed_raw geometry F3

echo "Starting F4 typed full: shape -> scale -> shift"
run_fusion typed_raw full F4

echo "Completed typed-fusion F0-F4 ablation."
