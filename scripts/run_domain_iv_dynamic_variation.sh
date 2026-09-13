#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
"$PYTHON" -u tools/run_domain_iv_dynamic_variation.py \
  --root_path "${DATA_ROOT:-./dataset/ETT-small/}" \
  --experiments "${EXPERIMENTS:-V0,V1,V2,V3,V4,V5}" \
  --epochs "${EPOCHS:-10}" --batch_size "${BATCH_SIZE:-32}" \
  --d_model "${D_MODEL:-512}" --d_ff "${D_FF:-2048}" \
  --n_heads "${N_HEADS:-2}" --e_layers "${E_LAYERS:-1}" \
  --env_num "${ENV_NUM:-6}" --env_ema_beta "${ENV_EMA_BETA:-0.9}" \
  --gpu "${GPU:-0}" --output "${OUTPUT:-results/domain_iv_dynamic_ETTh1_96_96}"
