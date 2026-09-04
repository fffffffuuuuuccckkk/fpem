#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
TAG="${EXPERIMENT_TAG:-factorized_var_v1}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

common_run() {
  local ablation="$1"
  local variant_mode="$2"
  local suffix="$3"
  CUDA_VISIBLE_DEVICES="$GPU" PMG_SPACE=raw ABLATION="$ablation" \
    CINV_MODE=pattern_only ZINV_MODE=stable_relation_correction \
    FREEZE_PATCH_STAGE2=true PRED_STAB_MODE=relative_future \
    MAPPING_USE_MODE=delta_only VARIANT_INPUT_MODE="$variant_mode" \
    EXPERIMENT_TAG="${TAG}_${suffix}" bash scripts/run_fpem_pmg.sh
}

echo "Starting I0: P2 + M2 + A3 invariant baseline"
common_run A3 latent_only invariant_baseline

echo "Starting V0: latent-only environment candidate, reliability off"
common_run A5 latent_only V0

echo "Starting Vshape: latent + factorized shape residual"
common_run A5 factorized_shape Vshape

echo "Starting Vgeo: latent + factorized level/scale residual"
common_run A5 factorized_geometry Vgeo

echo "Starting Vfull: latent + factorized shape/level/scale residual"
common_run A5 factorized_full Vfull

echo "Completed I0 and factorized variation component ablations."
