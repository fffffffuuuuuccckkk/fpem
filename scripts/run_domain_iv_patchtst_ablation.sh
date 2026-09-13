#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"

"$PYTHON" -u tools/run_domain_iv_patchtst_ablation.py \
  --root_path "${DATA_ROOT:-./dataset/ETT-small/}" \
  --experiments "${EXPERIMENTS:-D0,D1,D2,D3,D4,D5,D6}" \
  --epochs "${EPOCHS:-10}" --batch_size "${BATCH_SIZE:-32}" \
  --d_model "${D_MODEL:-512}" --d_ff "${D_FF:-2048}" \
  --n_heads "${N_HEADS:-2}" --e_layers "${E_LAYERS:-1}" \
  --foil_env_path "${FOIL_ENV_PATH:-foil_env/ETTh1_foil_env_k6.npz}" \
  --gpu "${GPU:-0}" --output "${OUTPUT:-results/domain_iv_patchtst_ETTh1_96_96}"
