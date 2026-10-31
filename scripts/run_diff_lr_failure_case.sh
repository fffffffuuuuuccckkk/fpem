#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
DATASET="${DATASET:?set DATASET}"
PRED_LEN="${PRED_LEN:?set PRED_LEN}"
ENV_NUM="${ENV_NUM:?set ENV_NUM}"
GPU="${GPU:-0}"
A0_MSE="${A0_MSE:?set A0_MSE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_diff_lr_l2_failure_cases}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"

cd "$PROJECT_DIR"
ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
case "$DATASET" in
  ETTm2)
    DATA_ROOT=./dataset/all_datasets/ETT-small; DATA_CLASS=ett_minute
    DATA_PATH=ETTm2.csv; FREQ=15min; BATCH_SIZE=32; CYCLE_LEN=96; CHUNK=32 ;;
  ETTh1)
    DATA_ROOT=./dataset/all_datasets/ETT-small; DATA_CLASS=ett_hour
    DATA_PATH=ETTh1.csv; FREQ=h; BATCH_SIZE=32; CYCLE_LEN=24; CHUNK=32 ;;
  ExchangeRate)
    DATA_ROOT=./dataset/all_datasets/exchange_rate; DATA_CLASS=custom
    DATA_PATH=exchange_rate.csv; FREQ=d; BATCH_SIZE=32; CYCLE_LEN=7; CHUNK=32 ;;
  *) echo "unsupported failure case: $DATASET" >&2; exit 2 ;;
esac

if [[ "$PRED_LEN" == 96 ]]; then
  REFERENCE="results/predictive_env_featurewise_gate_all_backbones_k3/patchtst/$DATASET/shared_reference.pt"
else
  REFERENCE="results/predictive_env_patchtst_multihorizon_96/pred_$PRED_LEN/$DATASET/shared_reference.pt"
fi
if [[ ! -s "$REFERENCE" ]]; then
  ALT="results/predictive_env_patchtst_film_featurewise_multihorizon_k3/pred_$PRED_LEN/$DATASET/shared_reference.pt"
  [[ -s "$ALT" ]] && REFERENCE="$ALT"
fi
[[ -s "$REFERENCE" ]] || { echo "missing shared reference: $REFERENCE" >&2; exit 3; }

DESTINATION="$OUTPUT_ROOT/${DATASET}_pred${PRED_LEN}_k${ENV_NUM}"
mkdir -p "$DESTINATION"
cat > "$DESTINATION/protocol.txt" <<EOF
dataset=$DATASET
pred_len=$PRED_LEN
environment_count=$ENV_NUM
A0_MSE=$A0_MSE
training=single-stage end-to-end differential LR
backbone_lr=2e-5
inv_head_lr=5e-5
decomposer_lr=1e-4
env_head_lr=1e-4
variant_lr=1e-4
reliability_lr=3e-4
gamma_beta_lr=1e-4*(0.2+0.8*maturity)
maturity=G_var_L2/(G_inv_L2+G_var_L2+1e-8)
reference=$REFERENCE
EOF

GPU="$GPU" BACKBONE=patchtst SEED=2021 ENV_NUM="$ENV_NUM" EXPERIMENTS=A2 \
DATASET_NAME="$DATASET" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" \
TARGET=OT FREQ="$FREQ" CYCLENET_CYCLE_LEN="$CYCLE_LEN" SEQ_LEN=96 \
PRED_LEN="$PRED_LEN" BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS=0 \
REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
VARIANT_FUSION_MODE=horizon_future_var VARIANT_FUSION_GATE_TYPE=feature \
EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 LAMBDA_INVPRED=1.0 \
LAMBDA_VAR_GAIN=0.0 LAMBDA_VAR_CONDITIONAL_GAIN=0.0 LAMBDA_FUTURE_VAR=0.0 \
LAMBDA_FUTURE_H=0.1 LAMBDA_HORIZON_RELIABILITY=0.0 \
FUTURE_PATCH_LEN=16 FUTURE_TEACHER_PATCHES_PER_BATCH=2 \
FUTURE_TEACHER_EVAL_PATCH_COUNT=3 HORIZON_FUTURE_DIM=32 \
HORIZON_FUTURE_CHUNK_SIZE="$CHUNK" HORIZON_FUTURE_GAMMA_SCALE=0.1 \
HORIZON_FUTURE_BETA_SCALE=0.1 PREDICTIVE_ENV_REFACTOR_MODE=current \
REFERENCE_CHECKPOINT="$REFERENCE" REQUIRE_REFERENCE_CHECKPOINT=1 \
DIFFERENTIAL_LR=1 LR_BACKBONE=2e-5 LR_INV_HEAD=5e-5 \
LR_DECOMPOSER=1e-4 LR_ENV_HEAD=1e-4 LR_VARIANT=1e-4 \
LR_GAMMA_BETA_BASE=1e-4 LR_RELIABILITY=3e-4 \
OUTPUT="$DESTINATION" SAVE_FINAL_CHECKPOINT=1 \
bash scripts/run_predictive_env_iv_patchtst.sh 2>&1 | tee "$DESTINATION/run.log"

"$PYTHON" - "$DESTINATION/A2/metrics_and_diagnostics.json" "$A0_MSE" \
  "$DESTINATION/comparison.txt" <<'PY'
import json, sys
from pathlib import Path
metrics = json.loads(Path(sys.argv[1]).read_text())
a0 = float(sys.argv[2])
lines = [
    f"A0: {a0:.9f}",
    f"Zinv: {metrics['inv_MSE']:.9f}",
    f"raw_gamma_beta_full: {metrics['raw_full_MSE']:.9f}",
    f"r_gated_full: {metrics['full_MSE']:.9f}",
    f"raw_variant_gain: {metrics['raw_variant_gain']:.9f}",
    f"gated_variant_gain: {metrics['gated_variant_gain']:.9f}",
    f"beats_A0: {metrics['full_MSE'] < a0}",
]
Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
print(" | ".join(lines))
PY
