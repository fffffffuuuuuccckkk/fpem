#!/usr/bin/env bash
set -euo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
PY="${PY:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
mkdir -p dataset/ETT-small logs decomp_shift_outputs/knf_full_0823 decomp_shift_outputs/knf_nocontrol_0823
if [ ! -s dataset/ETT-small/ETTh1.csv ]; then
  curl -L --connect-timeout 12 --max-time 120 --retry 2 -o dataset/ETT-small/ETTh1.csv https://cdn.jsdelivr.net/gh/zhouhaoyi/ETDataset@main/ETT-small/ETTh1.csv
fi
$PY tools/decomp_shift/knf_etth1_runner.py --mode full --out decomp_shift_outputs/knf_full_0823 --epochs ${EPOCHS:-10} --batch ${BS:-32}
$PY tools/decomp_shift/knf_etth1_runner.py --mode nocontrol --out decomp_shift_outputs/knf_nocontrol_0823 --epochs ${EPOCHS:-10} --batch ${BS:-32}