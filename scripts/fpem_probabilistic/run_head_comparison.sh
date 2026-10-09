#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
source "$PROJECT_DIR/scripts/fpem_final/common.sh"
DATASET="${DATASET:?set DATASET}"
PRED_LEN="${PRED_LEN:?set PRED_LEN}"
GPU="${GPU:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/results/fpem_probabilistic/head_comparison}"
OUTPUT="$OUTPUT_ROOT/$DATASET/pred_$PRED_LEN"
cd "$PROJECT_DIR"
COMPLETE_MARKER="summary.csv"
if [[ "${PROB_VALIDATION_ONLY:-0}" == "1" ]]; then
  COMPLETE_MARKER="validation_complete"
fi
if [[ -s "$OUTPUT/$COMPLETE_MARKER" ]]; then
  echo "already complete: $OUTPUT"
  exit 0
fi
if [[ -e "$OUTPUT/run_config.json" ]]; then
  echo "existing incomplete comparison; inspect before retry: $OUTPUT" >&2
  exit 3
fi
read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS \
  <<<"$(dataset_config "$DATASET")"
ARCHIVE_SHA256="$(sha256sum dataset/all_datasets.zip | awk '{print $1}')"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-}"
if [[ -z "$REFERENCE_CHECKPOINT" ]]; then
  REFERENCE_CHECKPOINT="$($PYTHON - "$PROJECT_DIR/results" "$DATASET" "$PRED_LEN" "$ARCHIVE_SHA256" <<'PY'
import sys
from pathlib import Path
import torch
root, dataset, horizon, archive = sys.argv[1:]
for path in sorted(Path(root).rglob('shared_reference.pt')):
    if path.parent.name != f'pred_{horizon}' or path.parent.parent.name != dataset:
        continue
    if path.parent.parent.parent.name != 'patchtst':
        continue
    try:
        signature = torch.load(path, map_location='cpu')['signature']
    except Exception:
        continue
    if (signature.get('backbone') == 'patchtst' and
        signature.get('dataset_archive_sha256') == archive and
        signature.get('d_model') == 512 and signature.get('d_ff') == 2048 and
        signature.get('dropout') == 0.1 and
        signature.get('warmup_epochs') == 3 and
        signature.get('seed') == 2021):
        print(path)
        break
PY
)"
fi
if [[ ! -s "$REFERENCE_CHECKPOINT" ]]; then
  echo "No compatible shared reference: $DATASET/$PRED_LEN" >&2
  exit 2
fi
mkdir -p "$OUTPUT"
env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
  PREDICTIVE_ENV_RUNNER=tools/run_fpem_head_comparison.py \
  DATASET_NAME="$DATASET" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
  DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" \
  FREQ="$FREQ" CYCLENET_CYCLE_LEN="$CYCLE_LEN" \
  SEQ_LEN=96 PRED_LEN="$PRED_LEN" BATCH_SIZE="$BATCH_SIZE" \
  D_MODEL=512 D_FF=2048 N_HEADS=2 E_LAYERS=1 DROPOUT=0.1 \
  GPU="$GPU" EXPERIMENTS=A2 EPOCHS="${EPOCHS:-10}" WARMUP_EPOCHS=3 \
  STAGE_EPOCHS=2 ENV_NUM=3 NUM_WORKERS=0 \
  REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
  VARIANT_FUSION_MODE=prob_affine_flow \
  PROB_HEAD="${PROB_HEAD:-all_shared}" \
  PROB_CONDITION_MODE="${PROB_CONDITION_MODE:-full_shape}" \
  PROB_HEAD_LR="${PROB_HEAD_LR:-0.0001}" \
  PROB_LOSS_WEIGHT_MODE="${PROB_LOSS_WEIGHT_MODE:-gradient_relative_maturity}" \
  PROB_LOSS_MIN_WEIGHT="${PROB_LOSS_MIN_WEIGHT:-0.05}" \
  PROB_NUM_SAMPLES="${PROB_NUM_SAMPLES:-100}" \
  PROB_FLOW_STEPS="${PROB_FLOW_STEPS:-12}" \
  MAX_PROB_EVAL_BATCHES="${MAX_PROB_EVAL_BATCHES:-0}" \
  FUTURE_PATCH_LEN=16 LAMBDA_FUTURE_H=0 LAMBDA_VARIANT_ANCHOR=0 \
  DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 \
  LR=1e-4 LR_BACKBONE=2e-5 LR_INV_HEAD=5e-5 \
  LR_DECOMPOSER="${LR_DECOMPOSER:-1e-4}" \
  LR_ENV_HEAD="${LR_ENV_HEAD:-1e-4}" \
  LR_VARIANT="${LR_VARIANT:-1e-4}" \
  PROB_VALIDATION_ONLY="${PROB_VALIDATION_ONLY:-0}" \
  REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" REQUIRE_REFERENCE_CHECKPOINT=1 \
  OUTPUT="$OUTPUT" \
  bash scripts/run_predictive_env_iv_patchtst.sh >"$OUTPUT/run.log" 2>&1
echo "complete: $OUTPUT"
