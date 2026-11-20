#!/usr/bin/env bash
set -euo pipefail

case_name="${1:?case_name required}"
gpu="${2:?gpu index required}"
repo=/data/OuXiaoyu/Time-Series-Library-FPEM
root="$repo/results/fpem_periodic_token_a0_20261118"
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python

case "$case_name" in
  patchtst_etth1_720)
    base="$repo/results/fpem_eiil3_fulltrain_testselect_20261113/asset_audit/asset_bindings/patchtst/ETTh1/pred_720/seed_2021/base_run_config.json"
    old="$repo/results/server2_imports/gpu-41/fpem_final_patchtst_search/patchtst/ETTh1/pred_720/shared_reference.pt"
    ;;
  moderntcn_ettm2_720)
    base="$root/inputs/moderntcn_ETTm2_720_base_run_config.json"
    old="$root/inputs/moderntcn_ETTm2_720_old_reference.pt"
    ;;
  itransformer_etth1_96)
    base="$repo/results/fpem_eiil3_fulltrain_testselect_20261113/asset_audit/asset_bindings/itransformer/ETTh1/pred_96/seed_2021/base_run_config.json"
    old="$repo/results/server2_imports/gpu-41/fpem_final_patchtst_search/itransformer/ETTh1/pred_96/shared_reference.pt"
    ;;
  *) echo "unsupported case $case_name" >&2; exit 2 ;;
esac

out="$root/pilot_$case_name"
if [[ ! -f "$out/complete" ]]; then
  "$python" "$repo/tools/run_fpem_periodic_token_a0.py" \
    --repo-root "$repo" --token-stage-root "$repo" \
    --base-run-config "$base" --old-a0-checkpoint "$old" \
    --output-dir "$out" --gpu "$gpu"
fi
"$python" "$repo/tools/aggregate_fpem_periodic_token_a0.py" --root "$root"
