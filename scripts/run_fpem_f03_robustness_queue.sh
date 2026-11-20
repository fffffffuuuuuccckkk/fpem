#!/usr/bin/env bash
set -u -o pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2

dataset="${1:?dataset}"
horizon="${2:?horizon}"
gpu="${3:?gpu}"
python_bin="${PYTHON_BIN:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
root="results/fpem_performance_optimization_20261103/f03_robustness/itransformer/${dataset}/pred_${horizon}"
base="results/fpem_dual_pattern_memory/g1_r3_full_point/itransformer/${dataset}/pred_${horizon}/base_run_config.json"
mkdir -p "${root}"
if [[ -f "${root}/queue_complete" ]]; then exit 0; fi
if [[ ! -f "${base}" ]]; then
  echo "Missing exact reusable A0 config: ${base}" >> "${root}/queue.log"
  exit 2
fi
for attempt in 1 2 3; do
  echo "attempt=${attempt} case=itransformer/${dataset}/${horizon} gpu=${gpu}" >> "${root}/queue.log"
  if PYTHONPATH=. "${python_bin}" -u tools/run_fpem_f03_seed_lr_robustness.py \
      --base_run_config "${base}" --output "${root}" --gpu "${gpu}" \
      --train_stride 16 --val_stride 4 --test_stride 1 --epochs 10 \
      --seeds 2022,2023,2024 --lrs 1e-4,3e-4,1e-3 \
      --allow_legacy_reference_signature >> "${root}/queue.log" 2>&1; then
    if [[ -f "${root}/complete" ]]; then
      touch "${root}/queue_complete"
      exit 0
    fi
  fi
  echo "attempt=${attempt} failed; retry after 60s" >> "${root}/queue.log"
  sleep 60
done
echo "failed after three attempts" >> "${root}/queue.log"
exit 1
