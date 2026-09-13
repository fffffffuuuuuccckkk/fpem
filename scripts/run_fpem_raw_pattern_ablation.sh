#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GPU="${GPU:-2}"
SMOKE="${SMOKE:-0}"
RUN_SET="${RUN_SET:-all}"

run_raw() {
  local name="$1" mapping="$2" shape="$3" scale="$4" shift="$5"
  echo "Launching ${name}: mapping=${mapping}, shape=${shape}, scale=${scale}, shift=${shift}"
  CUDA_VISIBLE_DEVICES="$GPU" \
  SMOKE="$SMOKE" PMG_SPACE=raw ABLATION=A5 CINV_MODE=pattern_only \
  ZINV_MODE=stable_relation_correction FREEZE_PATCH_STAGE2=true \
  PRED_STAB_MODE=relative_future MAPPING_USE_MODE=delta_only \
  VARIANT_INPUT_MODE=latent_only FUSION_MODE=typed_raw TYPED_COMPONENTS=full \
  RELATION_MAPPING_MODE=legacy FORECAST_MODE=raw_pattern \
  RAW_EXPERIMENT="$name" RAW_MAPPING_MODE="$mapping" \
  RAW_SHAPE_MODE="$shape" RAW_SCALE_MODE="$scale" RAW_SHIFT_MODE="$shift" \
  RAW_DECODER_CONTEXT="${RAW_DECODER_CONTEXT:-pattern_plus_hidden}" \
  PATTERN_SELECTION="${PATTERN_SELECTION:-soft}" \
  PATTERN_TOPK="${PATTERN_TOPK:-3}" \
  PATTERN_TEMPERATURE="${PATTERN_TEMPERATURE:-1.0}" \
  PATTERN_STRAIGHT_THROUGH="${PATTERN_STRAIGHT_THROUGH:-1}" \
  bash scripts/run_fpem_pmg.sh
}

case "$RUN_SET" in
  core)
    run_raw RP0 stable_only off off off
    run_raw RP4_RM0 stable_only gated gated gated
    ;;
  RP0) run_raw RP0 stable_only off off off ;;
  RP1) run_raw RP1 stable_only gated off off ;;
  RP2) run_raw RP2 stable_only off gated gated ;;
  RP3) run_raw RP3 stable_only unit unit unit ;;
  RP4|RM0) run_raw RP4_RM0 stable_only gated gated gated ;;
  RM1) run_raw RM1 variant_unit gated gated gated ;;
  RM2) run_raw RM2 variant_gated gated gated gated ;;
  all)
    run_raw RP0 stable_only off off off
    run_raw RP1 stable_only gated off off
    run_raw RP2 stable_only off gated gated
    run_raw RP3 stable_only unit unit unit
    run_raw RP4_RM0 stable_only gated gated gated
    run_raw RM1 variant_unit gated gated gated
    run_raw RM2 variant_gated gated gated gated
    ;;
  *) echo "Unknown RUN_SET=$RUN_SET (all, core, RP0..RP4, RM0..RM2)" >&2; exit 2 ;;
esac
