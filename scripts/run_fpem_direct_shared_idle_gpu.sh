#!/usr/bin/env bash
set -u -o pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2

backbone="${1:?backbone}"
dataset="${2:?dataset}"
horizon="${3:?horizon}"
gpu="${4:?gpu}"
python_bin="${PYTHON_BIN:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
root=results/fpem_performance_optimization_20261103/direct_shared_pilot
case_dir="${root}/${backbone}/${dataset}/pred_${horizon}"
base="results/fpem_dual_pattern_memory/g1_r3_full_point/${backbone}/${dataset}/pred_${horizon}/base_run_config.json"
mkdir -p "${case_dir}"

if [[ ! -f "${base}" ]]; then
  echo "Missing reusable A0 configuration: ${base}" >> "${case_dir}/queue.log"
  exit 2
fi
if [[ -f "${case_dir}/queue_complete" ]]; then
  exit 0
fi

for attempt in 1 2 3; do
  echo "attempt=${attempt} gpu=${gpu} case=${backbone}/${dataset}/${horizon}" >> "${case_dir}/queue.log"
  if PYTHONPATH=. "${python_bin}" -u tools/run_fpem_direct_invariant_search.py \
      --base_run_config "${base}" --output "${case_dir}" --gpu "${gpu}" \
      --train_stride 16 --val_stride 4 --test_stride 1 --epochs 10 \
      --methods F00,F03,F01,P01 --allow_legacy_reference_signature \
      >> "${case_dir}/queue.log" 2>&1; then
    if [[ -f "${case_dir}/F00/complete" && -f "${case_dir}/F03/complete" && \
          -f "${case_dir}/F01/complete" && -f "${case_dir}/P01/complete" ]]; then
      touch "${case_dir}/queue_complete"
      exit 0
    fi
  fi
  echo "attempt=${attempt} failed; retry after 60s" >> "${case_dir}/queue.log"
  sleep 60
done
echo "failed after three attempts" >> "${case_dir}/queue.log"
exit 1
