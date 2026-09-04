#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXEC="$PYTHON_BIN"
elif [[ -x /data/OuXiaoyu/miniconda3/envs/basicts/bin/python ]]; then
  PYTHON_EXEC=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
else
  PYTHON_EXEC=python3
fi

ABLATION="${ABLATION:-A6}"
SMOKE="${SMOKE:-0}"
PMG_SPACE="${PMG_SPACE:-embedding}"
CINV_MODE="${CINV_MODE:-pattern_only}"
ZINV_MODE="${ZINV_MODE:-stable_relation_correction}"
FREEZE_PATCH_STAGE2="${FREEZE_PATCH_STAGE2:-auto}"
VARIANT_INPUT_MODE="${VARIANT_INPUT_MODE:-latent_only}"
PRED_STAB_MODE="${PRED_STAB_MODE:-train_global}"
MAPPING_USE_MODE="${MAPPING_USE_MODE:-full}"
FUSION_MODE="${FUSION_MODE:-legacy_environment}"
TYPED_COMPONENTS="${TYPED_COMPONENTS:-full}"
RELATION_MAPPING_MODE="${RELATION_MAPPING_MODE:-legacy}"
FUTURE_MAPPING_MODE="${FUTURE_MAPPING_MODE:-off}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-dualspace_v1}"
A0_CHECKPOINT="${A0_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/long_term_forecast_ETTh1_96_96_A0_PatchTST_ETTh1_ftM_sl96_ll48_pl96_dm64_nh4_el1_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_fpem_pmg_A0_0/checkpoint.pth}"
COMMON_FLAGS=(
  --fpem_pmg_enabled 1
  --fpem_pmg_use_pattern 1
  --fpem_pmg_use_mapping 1
  --fpem_pmg_use_mapping_delta 1
  --fpem_pmg_use_variant 1
  --fpem_pmg_use_env 1
  --fpem_pmg_use_reliability 1
  --fpem_pmg_use_retrospective_validity 0
  --fpem_pmg_ablation "$ABLATION"
  --fpem_pmg_representation_space "$PMG_SPACE"
  --fpem_pmg_cinv_mode "$CINV_MODE"
  --fpem_pmg_zinv_mode "$ZINV_MODE"
  --fpem_pmg_freeze_patch_embedding_stage2 "$FREEZE_PATCH_STAGE2"
  --fpem_pmg_variant_input_mode "$VARIANT_INPUT_MODE"
  --fpem_pmg_predictive_stability_mode "$PRED_STAB_MODE"
  --fpem_pmg_mapping_use_mode "$MAPPING_USE_MODE"
  --fpem_pmg_fusion_mode "$FUSION_MODE"
  --fpem_pmg_typed_fusion_components "$TYPED_COMPONENTS"
  --fpem_pmg_relation_mapping_mode "$RELATION_MAPPING_MODE"
  --fpem_pmg_future_mapping_mode "$FUTURE_MAPPING_MODE"
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  COMMON_FLAGS+=(--fpem_pmg_resume_checkpoint "$RESUME_CHECKPOINT")
fi

case "$ABLATION" in
  A0) MODEL=PatchTST; COMMON_FLAGS=() ;;
  A1) MODEL=PatchTST_FPEM; COMMON_FLAGS+=(--fpem_pmg_use_mapping 0 --fpem_pmg_use_env 0) ;;
  A2) MODEL=PatchTST_FPEM; COMMON_FLAGS+=(--fpem_pmg_use_mapping_delta 0 --fpem_pmg_use_env 0) ;;
  A3) MODEL=PatchTST_FPEM; COMMON_FLAGS+=(--fpem_pmg_use_env 0) ;;
  A4) MODEL=PatchTST_FPEM; COMMON_FLAGS+=(--fpem_pmg_use_variant 0 --fpem_pmg_use_env 0) ;;
  A5) MODEL=PatchTST_FPEM; COMMON_FLAGS+=(--fpem_pmg_use_reliability 0) ;;
  A6) MODEL=PatchTST_FPEM ;;
  A7) MODEL=PatchTST_FPEM; COMMON_FLAGS+=(--fpem_pmg_use_retrospective_validity 0) ;;
  *) echo "Unknown ABLATION=$ABLATION (expected A0..A7)" >&2; exit 2 ;;
esac

case "$PRED_STAB_MODE" in
  per_future_znorm) PRED_TAG="P0" ;;
  train_global) PRED_TAG="P1" ;;
  relative_future) PRED_TAG="P2" ;;
  *) echo "Unknown PRED_STAB_MODE=$PRED_STAB_MODE" >&2; exit 2 ;;
esac

case "$MAPPING_USE_MODE" in
  none) MAPPING_TAG="M0" ;;
  context_only) MAPPING_TAG="M1" ;;
  delta_only) MAPPING_TAG="M2" ;;
  full) MAPPING_TAG="M3" ;;
  *) echo "Unknown MAPPING_USE_MODE=$MAPPING_USE_MODE" >&2; exit 2 ;;
esac

case "$FUSION_MODE" in
  invariant_only) FUSION_TAG="fusion_F0_inv" ;;
  legacy_environment) FUSION_TAG="fusion_F1_legacy" ;;
  typed_raw)
    case "$TYPED_COMPONENTS" in
      shape) FUSION_TAG="fusion_F2_shape" ;;
      geometry) FUSION_TAG="fusion_F3_geometry" ;;
      full) FUSION_TAG="fusion_F4_full" ;;
      *) echo "Unknown TYPED_COMPONENTS=$TYPED_COMPONENTS" >&2; exit 2 ;;
    esac
    ;;
  *) echo "Unknown FUSION_MODE=$FUSION_MODE" >&2; exit 2 ;;
esac

case "$RELATION_MAPPING_MODE" in
  legacy) RELATION_TAG="relation_R0" ;;
  decomposed_no_variant) RELATION_TAG="relation_R1" ;;
  decomposed_full) RELATION_TAG="relation_R2" ;;
  variant_only) RELATION_TAG="relation_R3" ;;
  *) echo "Unknown RELATION_MAPPING_MODE=$RELATION_MAPPING_MODE" >&2; exit 2 ;;
esac

case "$FUTURE_MAPPING_MODE" in
  off) FUTURE_TAG="HF0" ;;
  stable_only) FUTURE_TAG="HF1" ;;
  adaptive_unit) FUTURE_TAG="HF2" ;;
  adaptive_gated) FUTURE_TAG="HF3" ;;
  *) echo "Unknown FUTURE_MAPPING_MODE=$FUTURE_MAPPING_MODE" >&2; exit 2 ;;
esac

case "$FREEZE_PATCH_STAGE2" in
  auto) PATCH_TAG="patchauto" ;;
  true) PATCH_TAG="raw_freezepatch" ;;
  false) PATCH_TAG="trainpatch" ;;
  *) echo "Unknown FREEZE_PATCH_STAGE2=$FREEZE_PATCH_STAGE2" >&2; exit 2 ;;
esac

case "$VARIANT_INPUT_MODE" in
  latent_only) VARIANT_TAG="V0" ;;
  raw_deviation) VARIANT_TAG="V1" ;;
  factorized_shape) VARIANT_TAG="Vshape" ;;
  factorized_geometry) VARIANT_TAG="Vgeo" ;;
  factorized_full) VARIANT_TAG="Vfull" ;;
  *) echo "Unknown VARIANT_INPUT_MODE=$VARIANT_INPUT_MODE" >&2; exit 2 ;;
esac

case "$CINV_MODE" in
  product) CINV_TAG="B0" ;;
  soft_mapping) CINV_TAG="B1" ;;
  pattern_only) CINV_TAG="B2" ;;
  *) echo "Unknown CINV_MODE=$CINV_MODE" >&2; exit 2 ;;
esac

case "$ZINV_MODE" in
  interpolate) ZINV_TAG="C0" ;;
  hidden_only) ZINV_TAG="C1" ;;
  stable_relation_correction) ZINV_TAG="C2" ;;
  *) echo "Unknown ZINV_MODE=$ZINV_MODE" >&2; exit 2 ;;
esac

if [[ "$SMOKE" == "1" ]]; then
  EPOCHS=1
  D_MODEL=16
  D_FF=32
  BATCH=256
  SEQ_LEN=32
  LABEL_LEN=16
  PRED_LEN=8
  SPACE_TAG="hidden"
  [[ "$PMG_SPACE" == "embedding" ]] && SPACE_TAG="emb"
  [[ "$PMG_SPACE" == "raw" ]] && SPACE_TAG="raw"
  DESCRIPTION="fpem_pmg_${SPACE_TAG}_${PATCH_TAG}_${ABLATION}_${CINV_TAG}_${ZINV_TAG}_${PRED_TAG}_${MAPPING_TAG}_${VARIANT_TAG}_${FUSION_TAG}_${RELATION_TAG}_${FUTURE_TAG}_${EXPERIMENT_TAG}_smoke"
else
  EPOCHS=10
  D_MODEL=64
  D_FF=128
  BATCH=32
  SEQ_LEN=96
  LABEL_LEN=48
  PRED_LEN=96
  SPACE_TAG="hidden"
  [[ "$PMG_SPACE" == "embedding" ]] && SPACE_TAG="emb"
  [[ "$PMG_SPACE" == "raw" ]] && SPACE_TAG="raw"
  DESCRIPTION="fpem_pmg_${SPACE_TAG}_${PATCH_TAG}_${ABLATION}_${CINV_TAG}_${ZINV_TAG}_${PRED_TAG}_${MAPPING_TAG}_${VARIANT_TAG}_${FUSION_TAG}_${RELATION_TAG}_${FUTURE_TAG}_${EXPERIMENT_TAG}"
  if [[ "$MODEL" == "PatchTST_FPEM" && -f "$A0_CHECKPOINT" ]]; then
    COMMON_FLAGS+=(--fpem_pmg_a0_checkpoint "$A0_CHECKPOINT")
  fi
fi

"$PYTHON_EXEC" -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id "ETTh1_${SEQ_LEN}_${PRED_LEN}_${ABLATION}" \
  --model "$MODEL" \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len "$SEQ_LEN" \
  --label_len "$LABEL_LEN" \
  --pred_len "$PRED_LEN" \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model "$D_MODEL" \
  --n_heads 4 \
  --e_layers 1 \
  --d_layers 1 \
  --d_ff "$D_FF" \
  --batch_size "$BATCH" \
  --train_epochs "$EPOCHS" \
  --patience 3 \
  --learning_rate 0.0001 \
  --num_workers 2 \
  --des "$DESCRIPTION" \
  "${COMMON_FLAGS[@]}"
