#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
source "$PROJECT_DIR/scripts/fpem_final/common.sh"
DATASET="${DATASET:?set DATASET}"
PRED_LEN="${PRED_LEN:?set PRED_LEN}"
GPU="${GPU:?set GPU}"
ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/results/fpem_probabilistic/frozen_ema_heads}"
RUN_DIR="$ROOT/$DATASET/pred_$PRED_LEN"
ANCHOR_DIR="$RUN_DIR/anchor"
ANCHOR_CHECKPOINT="$ANCHOR_DIR/A2/trained_checkpoint.pt"
mkdir -p "$ANCHOR_DIR"
cd "$PROJECT_DIR"
read -r DATA_ROOT DATA_CLASS DATA_PATH FREQ BATCH_SIZE CYCLE_LEN CHANNELS \
  <<<"$(dataset_config "$DATASET")"
ARCHIVE_SHA256="$(sha256sum dataset/all_datasets.zip | awk '{print $1}')"

if [[ ! -s "$ANCHOR_CHECKPOINT" ]]; then
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
        signature = torch.load(path, map_location='cpu', weights_only=False)['signature']
    except Exception:
        continue
    if (signature.get('backbone') == 'patchtst' and
        signature.get('dataset_archive_sha256') == archive and
        signature.get('d_model') == 512 and signature.get('d_ff') == 2048 and
        signature.get('dropout') == 0.1 and signature.get('warmup_epochs') == 3 and
        signature.get('seed') == 2021):
        print(path)
        break
PY
)"
  fi
  if [[ -n "$REFERENCE_CHECKPOINT" ]]; then
    REQUIRE_REFERENCE_CHECKPOINT=1
    echo "reusing reference: $REFERENCE_CHECKPOINT"
  else
    REFERENCE_CHECKPOINT="$ANCHOR_DIR/shared_reference.pt"
    REQUIRE_REFERENCE_CHECKPOINT=0
    echo "no compatible reference; creating once: $REFERENCE_CHECKPOINT"
  fi
  env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
    DATASET_NAME="$DATASET" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
    DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" \
    FREQ="$FREQ" CYCLENET_CYCLE_LEN="$CYCLE_LEN" \
    SEQ_LEN=96 PRED_LEN="$PRED_LEN" BATCH_SIZE="$BATCH_SIZE" \
    D_MODEL=512 D_FF=2048 N_HEADS=2 E_LAYERS=1 DROPOUT=0.1 \
    GPU="$GPU" EXPERIMENTS=A2 EPOCHS="${ANCHOR_EPOCHS:-10}" \
    WARMUP_EPOCHS=3 STAGE_EPOCHS=2 ENV_NUM=3 NUM_WORKERS=0 \
    REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate \
    VARIANT_FUSION_MODE=prob_affine_flow PROB_ABLATION=invariant_only \
    PROB_LOSS_WEIGHT_MODE=historical_ema_maturity \
    PROB_NUM_SAMPLES=8 MAX_PROB_EVAL_BATCHES=1 \
    FUTURE_PATCH_LEN=16 LAMBDA_FUTURE_H=0 LAMBDA_VARIANT_ANCHOR=0 \
    DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 \
    LR=1e-4 LR_BACKBONE=2e-5 LR_INV_HEAD=5e-5 \
    LR_DECOMPOSER=1e-4 LR_ENV_HEAD=1e-4 LR_VARIANT=1e-4 \
    REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" \
    REQUIRE_REFERENCE_CHECKPOINT="$REQUIRE_REFERENCE_CHECKPOINT" \
    SAVE_FINAL_CHECKPOINT=1 OUTPUT="$ANCHOR_DIR" \
    bash scripts/run_predictive_env_iv_patchtst.sh >"$ANCHOR_DIR/run.log" 2>&1
fi
if [[ ! -s "$ANCHOR_CHECKPOINT" ]]; then
  echo "anchor checkpoint missing after training: $ANCHOR_CHECKPOINT" >&2
  exit 2
fi
echo "frozen anchor: $ANCHOR_CHECKPOINT"
FROZEN_ANCHOR_CHECKPOINT="$ANCHOR_CHECKPOINT" \
  OUTPUT_ROOT="$ROOT" PROB_LOSS_WEIGHT_MODE=historical_ema_maturity \
  DATASET="$DATASET" PRED_LEN="$PRED_LEN" GPU="$GPU" \
  bash scripts/fpem_probabilistic/run_head_comparison.sh
echo "completed frozen EMA probability heads: $DATASET/$PRED_LEN"
