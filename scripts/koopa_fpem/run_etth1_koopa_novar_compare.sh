#!/usr/bin/env bash
set -euo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
mkdir -p ./dataset/ETT-small
if [ ! -f ./dataset/ETT-small/ETTh1.csv ]; then
  for url in \
    https://cdn.jsdelivr.net/gh/zhouhaoyi/ETDataset@main/ETT-small/ETTh1.csv \
    https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
  do
    curl -L --connect-timeout 12 --max-time 120 --retry 2 -o ./dataset/ETT-small/ETTh1.csv "$url" && break || true
  done
  test -s ./dataset/ETT-small/ETTh1.csv
fi

EPOCHS="${EPOCHS:-10}"
BS="${BS:-32}"
PYTHON_BIN="${PYTHON_BIN:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
FULL_TAG="${FULL_TAG:-koopa_full_samplewise_0822}"
NOVAR_TAG="${NOVAR_TAG:-koopa_novar_from_start_0822}"
FULL_MODEL_ID="ETTh1_96_48_${FULL_TAG}"
NOVAR_MODEL_ID="ETTh1_96_48_${NOVAR_TAG}"
FULL_SETTING="long_term_forecast_${FULL_MODEL_ID}_Koopa_ETTh1_ftM_sl96_ll48_pl48_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_${FULL_TAG}_0"
NOVAR_SETTING="long_term_forecast_${NOVAR_MODEL_ID}_Koopa_NoVar_ETTh1_ftM_sl96_ll48_pl48_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_${NOVAR_TAG}_0"
FULL_CKPT="./checkpoints/${FULL_SETTING}/checkpoint.pth"
NOVAR_CKPT="./checkpoints/${NOVAR_SETTING}/checkpoint.pth"
OUT="./case_outputs/${NOVAR_SETTING}_vs_full"

if [ ! -f "${FULL_CKPT}" ]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" EPOCHS="${EPOCHS}" BS="${BS}" TAG="${FULL_TAG}" \
    PYTHON_BIN="${PYTHON_BIN}" bash scripts/koopa_fpem/run_etth1_koopa_full_samplewise.sh
fi

"${PYTHON_BIN}" -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id "${NOVAR_MODEL_ID}" --model Koopa_NoVar --data ETTh1 --features M \
  --seq_len 96 --label_len 48 --pred_len 48 \
  --e_layers 2 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --des "${NOVAR_TAG}" --learning_rate 0.001 \
  --train_epochs "${EPOCHS}" --batch_size "${BS}" --patience 3 --num_workers 0 --itr 1

"${PYTHON_BIN}" -u tools/koopa_fpem/analyze_koopa_samplewise.py \
  --ckpt "${FULL_CKPT}" --out_dir "${OUT}" \
  --model Koopa --model_id "${FULL_MODEL_ID}" --des "${FULL_TAG}" \
  --compare_model Koopa_NoVar --compare_ckpt "${NOVAR_CKPT}" \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 48 \
  --enc_in 7 --dec_in 7 --c_out 7 --factor 3 --batch_size "${BS}" --num_workers 0

echo "OUT=${OUT}"
