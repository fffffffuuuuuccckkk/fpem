#!/usr/bin/env bash
set -u -o pipefail
case_name="${1:?case required}"
gpu="${2:?gpu required}"
repo=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
pilot=results/fpem_dual_pattern_memory/rare_learning_pilot
controls=results/fpem_dual_pattern_memory/rare_learning_controls
cd "$repo" || exit 2
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4
mkdir -p "$controls/$case_name"
while :; do
  count=$(find "$pilot/$case_name" -mindepth 2 -maxdepth 2 -name complete -type f | wc -l)
  if [ "$count" -ge 8 ]; then break; fi
  if [ -f "$pilot/$case_name/failed_methods.txt" ]; then
    echo "Pilot has failed methods; controls deferred"
    exit 3
  fi
  sleep 60
done
"$python" tools/reconcile_dppm_rare_diagnostics.py "$case_name" --gpu 0 --output "$pilot" || exit 8
"$python" tools/summarize_fpem_dppm_rare.py --output "$pilot" || true
selected=$("$python" tools/plan_dppm_rare_controls.py "$case_name" --root "$pilot") || exit 4
echo "VAL-selected positive OOS methods: $selected"
for method in $selected; do
  for shuffle in 3021 3022 3023; do
    "$python" tools/run_fpem_dppm_rare.py "$case_name" --methods "$method" \
      --gpu 0 --patterns 8 --rank 8 --epochs 10 --eval_stride 4 \
      --shuffle_train_seed "$shuffle" --output "$controls/shuffle_$shuffle" || exit 5
  done
  "$python" tools/run_fpem_dppm_rare.py "$case_name" --methods "$method" \
    --gpu 0 --patterns 8 --rank 8 --epochs 10 --eval_stride 4 \
    --freeze_router --output "$controls/history_only_router" || exit 6
  "$python" tools/run_fpem_dppm_rare.py "$case_name" --methods "$method" \
    --gpu 0 --patterns 8 --rank 8 --epochs 10 --eval_stride 4 \
    --lambda_sparse 0 --output "$controls/no_sparsity" || exit 7
done
echo "VALIDATIONS FINISHED $case_name"
