#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting R1-FP: raw graph with frozen PatchEmbedding"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PMG_SPACE=raw ABLATION=A3 \
  CINV_MODE=pattern_only ZINV_MODE=stable_relation_correction \
  FREEZE_PATCH_STAGE2=true VARIANT_INPUT_MODE=latent_only \
  EXPERIMENT_TAG="${EXPERIMENT_TAG:-rawvar_v1}" bash scripts/run_fpem_pmg.sh

echo "Starting V0: latent-only Variant with Environment"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PMG_SPACE=raw ABLATION=A5 \
  CINV_MODE=pattern_only ZINV_MODE=stable_relation_correction \
  FREEZE_PATCH_STAGE2=auto VARIANT_INPUT_MODE=latent_only \
  EXPERIMENT_TAG="${EXPERIMENT_TAG:-rawvar_v1}" bash scripts/run_fpem_pmg.sh

echo "Starting V1: raw-deviation Variant with Environment"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PMG_SPACE=raw ABLATION=A5 \
  CINV_MODE=pattern_only ZINV_MODE=stable_relation_correction \
  FREEZE_PATCH_STAGE2=auto VARIANT_INPUT_MODE=raw_deviation \
  EXPERIMENT_TAG="${EXPERIMENT_TAG:-rawvar_v1}" bash scripts/run_fpem_pmg.sh
