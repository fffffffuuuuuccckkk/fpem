#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
cd "$PROJECT_DIR"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"

"$PYTHON" -u tools/run_eivpm_patchtst_ablation.py \
  --root_path "${DATA_ROOT:-./dataset/ETT-small/}" \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --experiments "${EXPERIMENTS:-E0,E1,E2,E3,E4,E5,E6,E7,E8,E9}" \
  --epochs "${EPOCHS:-10}" --warmup_epochs "${WARMUP_EPOCHS:-3}" --batch_size "${BATCH_SIZE:-32}" \
  --d_model "${D_MODEL:-512}" --d_ff "${D_FF:-2048}" \
  --n_heads "${N_HEADS:-2}" --e_layers "${E_LAYERS:-1}" \
  --pattern_count "${PATTERN_COUNT:-32}" --mapping_rank "${MAPPING_RANK:-8}" \
  --env_num "${ENV_NUM:-6}" --gpu "${GPU:-0}" \
  --foil_env_path "${FOIL_ENV_PATH:-foil_env/ETTh1_foil_env_k6.npz}" \
  --output "${OUTPUT:-results/eivpm_patchtst}"
