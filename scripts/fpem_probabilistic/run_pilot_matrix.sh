#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
source "$PROJECT_DIR/scripts/fpem_final/common.sh"
DATASET="${DATASET:?set DATASET to ETTh1, Weather, or ExchangeRate}"
PRED_LEN="${PRED_LEN:?set PRED_LEN}"
GPU="${GPU:-0}"
BACKBONE="${BACKBONE:-patchtst}"
PROB_ABLATION="${PROB_ABLATION:-full}"
FUSION_MODE="${VARIANT_FUSION_MODE:-prob_affine_flow}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_probabilistic/pilot}"
OUTPUT="$OUTPUT_ROOT/$BACKBONE/$DATASET/pred_$PRED_LEN/$PROB_ABLATION"
if [[ "$FUSION_MODE" != "prob_affine_flow" ]]; then
  OUTPUT="$OUTPUT_ROOT/$BACKBONE/$DATASET/pred_$PRED_LEN/original_${FUSION_MODE}"
fi
cd "$PROJECT_DIR"
if [[ -s "$OUTPUT/A2/metrics_and_diagnostics.json" ]]; then
  echo "already complete: $OUTPUT"
  exit 0
fi
read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS \
  <<<"$(dataset_config "$DATASET")"
D_MODEL=512
D_FF=2048
DROPOUT=0.1
MODERN_TCN_PATCH_SIZE=8
MODERN_TCN_PATCH_STRIDE=4
MODERN_TCN_NUM_STAGES=1
MODERN_TCN_FFN_RATIO=1
MODERN_TCN_HEAD_DROPOUT=0.0
if [[ "$BACKBONE" == "moderntcn" ]]; then
  D_MODEL=64
  read -r MODERN_TCN_PATCH_SIZE MODERN_TCN_PATCH_STRIDE MODERN_TCN_NUM_STAGES \
    MODERN_TCN_FFN_RATIO DROPOUT MODERN_TCN_HEAD_DROPOUT \
    <<<"$(modern_tcn_config "$DATASET")"
  D_FF=$((D_MODEL * MODERN_TCN_FFN_RATIO))
fi
ARCHIVE_SHA256="$(sha256sum dataset/all_datasets.zip | awk '{print $1}')"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-}"
if [[ -z "$REFERENCE_CHECKPOINT" ]]; then
  REFERENCE_CHECKPOINT="$($PYTHON - "$PROJECT_DIR/results" "$BACKBONE" "$DATASET" "$PRED_LEN" "$ARCHIVE_SHA256" "$D_MODEL" "$D_FF" "$DROPOUT" <<'PY'
import sys
from pathlib import Path
import torch
root, backbone, dataset, horizon, archive, d_model, d_ff, dropout = sys.argv[1:]
for path in sorted(Path(root).rglob('shared_reference.pt')):
    if path.parent.name != f'pred_{horizon}' or path.parent.parent.name != dataset:
        continue
    if path.parent.parent.parent.name != backbone:
        continue
    try:
        signature = torch.load(path, map_location='cpu')['signature']
    except Exception:
        continue
    if (signature.get('backbone') == backbone and
        signature.get('dataset_archive_sha256') == archive and
        signature.get('d_model') == int(d_model) and
        signature.get('d_ff') == int(d_ff) and
        abs(float(signature.get('dropout', -1)) - float(dropout)) < 1e-12 and
        signature.get('warmup_epochs') == 3 and
        signature.get('seed') == 2021):
        print(path)
        break
PY
)"
fi
if [[ -z "$REFERENCE_CHECKPOINT" || ! -s "$REFERENCE_CHECKPOINT" ]]; then
  echo "No compatible shared reference for $BACKBONE $DATASET/$PRED_LEN" >&2
  echo "Set REFERENCE_CHECKPOINT explicitly after verifying its signature." >&2
  exit 2
fi
mkdir -p "$OUTPUT"
env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" GPU="$GPU" BACKBONE="$BACKBONE" \
  DATASET_NAME="$DATASET" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
  DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" \
  FREQ="$FREQ" CYCLENET_CYCLE_LEN="$CYCLE_LEN" SEQ_LEN=96 PRED_LEN="$PRED_LEN" \
  BATCH_SIZE="$BATCH_SIZE" D_MODEL="$D_MODEL" D_FF="$D_FF" N_HEADS=2 E_LAYERS=1 \
  DROPOUT="$DROPOUT" MODERN_TCN_PATCH_SIZE="$MODERN_TCN_PATCH_SIZE" \
  MODERN_TCN_PATCH_STRIDE="$MODERN_TCN_PATCH_STRIDE" \
  MODERN_TCN_NUM_STAGES="$MODERN_TCN_NUM_STAGES" \
  MODERN_TCN_FFN_RATIO="$MODERN_TCN_FFN_RATIO" \
  MODERN_TCN_HEAD_DROPOUT="$MODERN_TCN_HEAD_DROPOUT" \
  EXPERIMENTS=A2 EPOCHS="${EPOCHS:-10}" WARMUP_EPOCHS=3 \
  STAGE_EPOCHS=2 ENV_NUM=3 NUM_WORKERS=0 \
  REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
  VARIANT_FUSION_MODE="$FUSION_MODE" PROB_ABLATION="$PROB_ABLATION" \
  PROB_LOSS_WEIGHT_MODE="${PROB_LOSS_WEIGHT_MODE:-gradient_relative_maturity}" \
  PROB_LOSS_MIN_WEIGHT="${PROB_LOSS_MIN_WEIGHT:-0.05}" \
  FUTURE_PATCH_LEN=16 USE_STOCHASTIC_INNOVATION="${USE_STOCHASTIC_INNOVATION:-1}" \
  PROB_NUM_SAMPLES="${PROB_NUM_SAMPLES:-100}" PROB_FLOW_STEPS="${PROB_FLOW_STEPS:-12}" \
  MAX_PROB_EVAL_BATCHES="${MAX_PROB_EVAL_BATCHES:-0}" \
  LAMBDA_FUTURE_H=0 LAMBDA_VARIANT_ANCHOR=0 \
  DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 \
  LR=1e-4 LR_BACKBONE=2e-5 LR_INV_HEAD=5e-5 LR_DECOMPOSER=1e-4 \
  LR_ENV_HEAD=1e-4 LR_VARIANT=1e-4 \
  REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" REQUIRE_REFERENCE_CHECKPOINT=1 \
  SAVE_FINAL_CHECKPOINT=1 OUTPUT="$OUTPUT" \
  bash scripts/run_predictive_env_iv_patchtst.sh >"$OUTPUT/run.log" 2>&1
echo "complete: $OUTPUT"
