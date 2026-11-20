#!/usr/bin/env bash
set -euo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM
py=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
root=results/fpem_visible_variation
mkdir -p "$root/queue_logs"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for spec in 'ETTh1 96' 'ETTm2 336' 'Weather 96'; do
  read -r dataset horizon <<< "$spec"
  name="${dataset}_${horizon}_seed2021"
  if [[ -s "$root/$name/metrics.csv" && -s "$root/$name/classification_ranking_summary.pdf" ]]; then
    echo "REUSED $name"
    continue
  fi
  echo "START $name"
  "$py" tools/plot_fpem_visible_variation.py --dataset "$dataset" --horizon "$horizon" \
    --seed 2021 --output "$root" > "$root/queue_logs/$name.log" 2>&1
  "$py" tools/aggregate_fpem_visible_variation.py --root "$root"
  echo "DONE $name"
done
"$py" tools/aggregate_fpem_visible_variation.py --root "$root"
echo 'PILOT_COMPLETE'
