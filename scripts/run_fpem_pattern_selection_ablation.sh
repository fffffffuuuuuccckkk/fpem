#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GPU="${GPU:-3}"
SMOKE="${SMOKE:-0}"
RUN_SET="${RUN_SET:-all}"

run_selection() {
  local name="$1" selection="$2" topk="$3"
  echo "Launching ${name}: selector=${selection}, topk=${topk}, variant_mapping=stable_only"
  CUDA_VISIBLE_DEVICES="$GPU" SMOKE="$SMOKE" \
  PMG_SPACE=raw ABLATION=A5 CINV_MODE=pattern_only \
  ZINV_MODE=stable_relation_correction FREEZE_PATCH_STAGE2=true \
  PRED_STAB_MODE=relative_future MAPPING_USE_MODE=delta_only \
  VARIANT_INPUT_MODE=latent_only FUSION_MODE=typed_raw TYPED_COMPONENTS=full \
  RELATION_MAPPING_MODE=legacy FORECAST_MODE=raw_pattern \
  RAW_EXPERIMENT="$name" RAW_MAPPING_MODE=stable_only \
  RAW_SHAPE_MODE=gated RAW_SCALE_MODE=gated RAW_SHIFT_MODE=gated \
  RAW_DECODER_CONTEXT=pattern_plus_hidden \
  PATTERN_SELECTION="$selection" PATTERN_TOPK="$topk" \
  PATTERN_TEMPERATURE=1.0 PATTERN_STRAIGHT_THROUGH=1 \
  bash scripts/run_fpem_pmg.sh
}

case "$RUN_SET" in
  PS0) run_selection PS0 soft 3 ;;
  PS1) run_selection PS1 top1 1 ;;
  PS2) run_selection PS2 topk 3 ;;
  all)
    run_selection PS0 soft 3
    run_selection PS1 top1 1
    run_selection PS2 topk 3
    ;;
  *) echo "Unknown RUN_SET=$RUN_SET (all, PS0, PS1, PS2)" >&2; exit 2 ;;
esac
