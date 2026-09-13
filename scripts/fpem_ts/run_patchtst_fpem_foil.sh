#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
cd "$PROJECT_DIR"

if [ -f /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh ]; then
  source /data/OuXiaoyu/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-basicts}"
fi

DATA_ROOT="${DATA_ROOT:-/data/OuXiaoyu/datasets/ExchangeRate}"
DATA_PATH="${DATA_PATH:-}"
DATA_NAME="${DATA_NAME:-split_npy}"
TARGET="${TARGET:-OT}"
FEATURES="${FEATURES:-M}"
ENC_IN="${ENC_IN:-8}"
GPU="${GPU:-0}"
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
PRED_LEN="${PRED_LEN:-96}"
ENV_NUM="${ENV_NUM:-6}"
MODEL_ID="${MODEL_ID:-exchange_foil_fpem_patchtst_${PRED_LEN}}"

mkdir -p logs foil_env checkpoints

python fpem_ts/foil_shift.py \
  --root_path "$DATA_ROOT" \
  --data_path "$DATA_PATH" \
  --target "$TARGET" \
  --seq_len "$SEQ_LEN" \
  --pred_len "$PRED_LEN" \
  --env_num "$ENV_NUM" \
  --out_dir foil_env

python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id "$MODEL_ID" \
  --model PatchTST_FPEM \
  --data "$DATA_NAME" \
  --root_path "$DATA_ROOT/" \
  --data_path "$DATA_PATH" \
  --features "$FEATURES" \
  --target "$TARGET" \
  --freq "${FREQ:-h}" \
  --seq_len "$SEQ_LEN" \
  --label_len "$LABEL_LEN" \
  --pred_len "$PRED_LEN" \
  --enc_in "$ENC_IN" \
  --dec_in "$ENC_IN" \
  --c_out "$ENC_IN" \
  --d_model "${D_MODEL:-128}" \
  --n_heads "${N_HEADS:-8}" \
  --e_layers "${E_LAYERS:-2}" \
  --d_layers 1 \
  --d_ff "${D_FF:-256}" \
  --dropout "${DROPOUT:-0.1}" \
  --factor 1 \
  --patch_len "${PATCH_LEN:-16}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --learning_rate "${LR:-0.0001}" \
  --train_epochs "${EPOCHS:-10}" \
  --patience "${PATIENCE:-3}" \
  --gpu "$GPU" \
  --num_workers "${NUM_WORKERS:-4}" \
  --fpem_lambda_inv "${FPEM_LAMBDA_INV:-0.2}" \
  --fpem_lambda_env "${FPEM_LAMBDA_ENV:-0.2}" \
  --fpem_lambda_sep "${FPEM_LAMBDA_SEP:-0.01}" \
  --fpem_lambda_delta_sparse "${FPEM_LAMBDA_DELTA_SPARSE:-0.0}" \
  --fpem_use_gate "${FPEM_USE_GATE:-1}" \
  --foil_env_num "$ENV_NUM" \
  --des foil_fpem
