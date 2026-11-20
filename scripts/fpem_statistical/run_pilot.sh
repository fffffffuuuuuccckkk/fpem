#!/usr/bin/env bash
# Four independent case lanes, one GPU each.  Restartable; never retrains A0.
set -euo pipefail
cd "${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
ROOT="${STAT_ROOT:-results/fpem_statistical_pilot_20261017}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
mkdir -p "$ROOT"
CASES=(
  'patchtst_ETTh1_96|results/fpem_no_future_anchor_patchtst_search/patchtst/ETTh1/pred_96/A0/run_config.json'
  'patchtst_ETTh1_720|results/fpem_latent_decomposition_search/pilot_20261015/patchtst_ETTh1_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json'
  'moderntcn_ETTm2_720|results/fpem_latent_decomposition_search/pilot_20261015/moderntcn_ETTm2_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json'
  'itransformer_ETTm1_720|results/fpem_latent_decomposition_search/pilot_20261015/itransformer_ETTm1_720_A0_complementary_gate_predictive_sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json'
)
SPECS=(
  'a0|4' 'a1|4' 'gate|4' 'layernorm|1' 'groupnorm|1'
  'linear|4' 'mlp|4' 't1_group|4' 't1_group|8'
  't2_orthogonal|4' 't2_orthogonal|8' 'c3_dual|4' 'c3_dual|8'
  't0_stnorm|4' 'c1_covariance|4' 't4_learned|4' 'c8_mixed|4'
)

run_one() {
  local lane="$1" config="$2" case_root="$3" method="$4" groups="$5" protocol="$6"
  local output="$case_root/${protocol}_${method}_g${groups}"
  mkdir -p "$output"
  [[ -f "$output/run_complete" ]] && return 0
  local extra=()
  [[ "$method" == t2_orthogonal ]] && extra+=(--stat_use_orthogonal_basis)
  [[ "$method" == c1_covariance || "$method" == c3_dual ]] && extra+=(--stat_whitening full)
  [[ "$protocol" == joint ]] && extra+=(--train_backbone)
  echo "START $case_root $protocol $method G=$groups $(date -Is)" >> "$ROOT/queue.log"
  if CUDA_VISIBLE_DEVICES="$lane" "$PYTHON" tools/run_fpem_statistical.py \
    --base_run_config "$config" --output "$output" \
    --decomposition_type latent_statistics --stat_method "$method" --stat_groups "$groups" \
    --epochs 10 --seed 2021 --gpu 0 --lr_head 0.00005 --lr_backbone 0.00002 \
    --lr_stat 0.0001 "${extra[@]}" > "$output/run.log" 2>&1; then
    echo "DONE $case_root $protocol $method G=$groups $(date -Is)" >> "$ROOT/queue.log"
  else
    echo "FAILED $case_root $protocol $method G=$groups $(date -Is)" >> "$ROOT/queue.log"
    tail -20 "$output/run.log" >> "$ROOT/failures.log"
  fi
}

run_case() {
  local lane="$1" id config case_root method groups
  IFS='|' read -r id config <<< "${CASES[$lane]}"
  case_root="$ROOT/$id"
  mkdir -p "$case_root"
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r method groups <<< "$spec"
    run_one "$lane" "$config" "$case_root" "$method" "$groups" frozen
  done
  # Only the two best statistical methods on VAL receive joint fine-tuning.
  while read -r method groups; do
    [[ -n "$method" ]] || continue
    run_one "$lane" "$config" "$case_root" "$method" "$groups" joint
  done < <("$PYTHON" tools/select_fpem_stat_candidates.py --case_root "$case_root")
}

for lane in 0 1 2 3; do
  run_case "$lane" &
done
wait
"$PYTHON" tools/summarize_fpem_statistical.py --root "$ROOT" >> "$ROOT/queue.log" 2>&1
