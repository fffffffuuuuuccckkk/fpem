#!/usr/bin/env bash
set -euo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
TAG="${TAG:-koopa_full_samplewise_0822}"
MODEL_ID="ETTh1_96_48_${TAG}"
SETTING="long_term_forecast_${MODEL_ID}_Koopa_ETTh1_ftM_sl96_ll48_pl48_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_${TAG}_0"
CKPT="./checkpoints/${SETTING}/checkpoint.pth"
OUT="./case_outputs/${SETTING}"

"${PYTHON_BIN}" -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id "${MODEL_ID}" --model Koopa --data ETTh1 --features M \
  --seq_len 96 --label_len 48 --pred_len 48 \
  --e_layers 2 --d_layers 1 --factor 3 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --des "${TAG}" --learning_rate 0.001 \
  --train_epochs "${EPOCHS}" --batch_size "${BS}" --patience 3 --num_workers 0 --itr 1

"${PYTHON_BIN}" -u tools/koopa_fpem/analyze_koopa_samplewise.py \
  --ckpt "${CKPT}" --out_dir "${OUT}" \
  --model Koopa --model_id "${MODEL_ID}" --des "${TAG}" \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 48 \
  --enc_in 7 --dec_in 7 --c_out 7 --factor 3 --batch_size "${BS}" --num_workers 0

echo "OUT=${OUT}"
