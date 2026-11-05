#!/usr/bin/env bash
set -u -o pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
dataset="${1:?dataset}"
horizon="${2:?horizon}"
gpu="${3:?gpu}"
python_bin="${PYTHON_BIN:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
root="results/fpem_performance_optimization_20261103/f03_backbone_continuation/itransformer/${dataset}/pred_${horizon}"
shared="results/fpem_performance_optimization_20261103/f03_robustness/itransformer/${dataset}/pred_${horizon}"
base="results/fpem_dual_pattern_memory/g1_r3_full_point/itransformer/${dataset}/pred_${horizon}/base_run_config.json"
mkdir -p "${root}"
if [[ -f "${root}/queue_complete" ]]; then exit 0; fi
for seed in 2022 2023 2024; do
  selection="${shared}/seed_${seed}/selection.json"
  if [[ ! -f "${base}" || ! -f "${selection}" ]]; then
    echo "Missing exact A0 config or F03 selection: ${seed}" >> "${root}/queue.log"
    exit 2
  fi
  for mode in head_only joint; do
    output="${root}/seed_${seed}/${mode}"
    if [[ -f "${output}/complete" ]]; then continue; fi
    for attempt in 1 2 3; do
      echo "seed=${seed} mode=${mode} attempt=${attempt} gpu=${gpu}" >> "${root}/queue.log"
      if PYTHONPATH=. "${python_bin}" -u tools/run_fpem_f03_backbone_continuation.py \
          --base_run_config "${base}" --shared_selection "${selection}" \
          --output "${output}" --mode "${mode}" --seed "${seed}" \
          --gpu "${gpu}" --epochs 5 --head_lr_multiplier 0.1 \
          --backbone_lr 2e-5 --train_stride 16 --val_stride 4 --test_stride 1 \
          --allow_legacy_reference_signature >> "${root}/queue.log" 2>&1; then
        if [[ -f "${output}/complete" ]]; then break; fi
      fi
      echo "seed=${seed} mode=${mode} attempt=${attempt} failed; retry after 60s" >> "${root}/queue.log"
      sleep 60
    done
    if [[ ! -f "${output}/complete" ]]; then exit 1; fi
  done
done
touch "${root}/queue_complete"
