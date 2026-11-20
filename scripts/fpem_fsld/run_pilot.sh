#!/usr/bin/env bash
# Four independent GPUs; identical A0 checkpoint per case, no A0 retraining.
# VAL selects rank/checkpoint; TEST is read only after selection.
set -euo pipefail
cd "${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
ROOT="${FSLD_ROOT:-results/fpem_fsld_pilot_20261016}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
mkdir -p "$ROOT"

CASES=(
  'patchtst_ETTh1_96|results/fpem_no_future_anchor_patchtst_search/patchtst/ETTh1/pred_96/A0/run_config.json'
  'patchtst_ETTh1_720|results/fpem_latent_decomposition_search/pilot_20261015/patchtst_ETTh1_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json'
  'moderntcn_ETTm2_720|results/fpem_latent_decomposition_search/pilot_20261015/moderntcn_ETTm2_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json'
  'itransformer_ETTm1_720|results/fpem_latent_decomposition_search/pilot_20261015/itransformer_ETTm1_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json'
)

run_case() {
  local lane="$1" identifier config direct method trial output
  IFS='|' read -r identifier config <<< "${CASES[$lane]}"
  if [[ ! -f "$config" ]]; then
    echo "MISSING_CONFIG $identifier $config" >> "$ROOT/queue.log"
    return 1
  fi
  direct="$ROOT/$identifier/direct"
  mkdir -p "$direct"
  if [[ ! -f "$direct/direct_complete" ]]; then
    echo "START_DIRECT $identifier $(date -Is)" >> "$ROOT/queue.log"
    if ! CUDA_VISIBLE_DEVICES="$lane" "$PYTHON" tools/run_fpem_fsld.py \
      --base_run_config "$config" --output "$direct" --phase direct --gpu 0 \
      > "$direct/run.log" 2>&1; then
      echo "FAILED_DIRECT $identifier $(date -Is)" >> "$ROOT/queue.log"
      return 1
    fi
    echo "DONE_DIRECT $identifier $(date -Is)" >> "$ROOT/queue.log"
  fi
  for method in A1 D0 F1 F2 F3 R1 R2; do
    output="$ROOT/$identifier/$method"
    mkdir -p "$output"
    [[ -f "$output/run_complete" ]] && continue
    echo "START_TRAIN $identifier $method $(date -Is)" >> "$ROOT/queue.log"
    if CUDA_VISIBLE_DEVICES="$lane" "$PYTHON" tools/run_fpem_fsld.py \
      --base_run_config "$config" --output "$output" --phase train \
      --method "$method" --epochs 10 --seed 2021 --gpu 0 \
      --lr_head 0.00005 --lr_backbone 0.00002 --lr_decomposer 0.0001 \
      > "$output/run.log" 2>&1; then
      echo "DONE_TRAIN $identifier $method $(date -Is)" >> "$ROOT/queue.log"
    else
      echo "FAILED_TRAIN $identifier $method $(date -Is)" >> "$ROOT/queue.log"
    fi
  done
  output="$ROOT/$identifier/probe"
  mkdir -p "$output"
  if [[ ! -f "$output/run_complete" ]]; then
    echo "START_PROBE $identifier $(date -Is)" >> "$ROOT/queue.log"
    if CUDA_VISIBLE_DEVICES="$lane" "$PYTHON" tools/run_fpem_fsld.py \
      --base_run_config "$config" --output "$output" --phase probe \
      --epochs 10 --seed 2021 --gpu 0 --lr_head 0.00005 \
      > "$output/run.log" 2>&1; then
      echo "DONE_PROBE $identifier $(date -Is)" >> "$ROOT/queue.log"
    else
      echo "FAILED_PROBE $identifier $(date -Is)" >> "$ROOT/queue.log"
    fi
  fi
}

for lane in 0 1 2 3; do
  run_case "$lane" &
done
wait
"$PYTHON" tools/summarize_fpem_fsld.py --root "$ROOT" >> "$ROOT/queue.log" 2>&1
