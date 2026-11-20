#!/usr/bin/env bash
set -u -o pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
backbone="${1:?backbone}"
dataset="${2:?dataset}"
horizon="${3:?horizon}"
gpu="${4:?gpu}"
python_bin="${PYTHON_BIN:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
root="results/fpem_performance_optimization_20261103/f03_anchor_search/${backbone}/${dataset}/pred_${horizon}"
base="results/fpem_dual_pattern_memory/g1_r3_full_point/${backbone}/${dataset}/pred_${horizon}/base_run_config.json"
mkdir -p "${root}"
if [[ -f "${root}/queue_complete" ]]; then exit 0; fi
if [[ ! -f "${base}" ]]; then echo "missing reusable A0: ${base}" >> "${root}/queue.log"; exit 2; fi
for attempt in 1 2 3; do
  echo "attempt=${attempt} case=${backbone}/${dataset}/${horizon} gpu=${gpu}" >> "${root}/queue.log"
  if PYTHONPATH=. "${python_bin}" -u tools/run_fpem_f03_anchor_search.py \
    --base_run_config "${base}" --output "${root}" --gpu "${gpu}" \
    --train_stride 16 --val_stride 4 --test_stride 1 --epochs 10 \
    --allow_legacy_reference_signature >> "${root}/queue.log" 2>&1; then
      if [[ -f "${root}/complete" ]]; then touch "${root}/queue_complete"; exit 0; fi
  fi
  echo "retry after 60 seconds" >> "${root}/queue.log"
  sleep 60
done
echo "failed after three attempts" >> "${root}/queue.log"
exit 1
