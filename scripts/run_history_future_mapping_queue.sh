#!/usr/bin/env bash
set -u
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
py=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
root=results/history_future_mapping_validation
mkdir -p "$root/queue_logs"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Stage 1 pilot first; later cases test generality and K/H sensitivity.
plan=(
  'ETTh1 OT 96 96 10'
  'ETTm2 OT 96 96 10'
  'Weather OT 96 96 10'
  'Electricity auto 96 96 10'
  'Traffic auto 96 96 10'
  'ETTh1 OT 96 96 5'
  'ETTh1 OT 96 96 20'
  'ETTh1 OT 96 192 10'
  'ETTh1 OT 96 720 10'
)
for spec in "${plan[@]}"; do
  read -r dataset channel L H K <<< "$spec"
  key="${dataset}_${channel}_L${L}_H${H}_K${K}_seed2022"
  if [[ "$channel" == auto ]]; then
    done_file=$(find "$root" -maxdepth 2 -path "*/${dataset}_*_L${L}_H${H}_K${K}_seed2022/config.json" -type f -print -quit)
    if [[ -n "$done_file" ]] && grep -q '"permutations": 999' "$done_file" && [[ -s "${done_file%config.json}summary.csv" ]]; then
      echo "REUSED $key";continue
    fi
  elif [[ -s "$root/$key/summary.csv" && -s "$root/$key/Figure_E_mapping_swap.pdf" ]] && \
       grep -q '"permutations": 999' "$root/$key/config.json"; then
    echo "REUSED $key";continue
  fi
  ok=0
  for attempt in 1 2; do
    echo "START $key attempt=$attempt"
    if "$py" tools/history_future_mapping_validation.py --dataset "$dataset" --channel "$channel" \
         --L "$L" --H "$H" --K "$K" --seed 2022 --permutations 999 \
         > "$root/queue_logs/${key}.log" 2>&1; then ok=1;break;fi
    echo "FAILED $key attempt=$attempt"
    if [[ $attempt -eq 1 ]]; then sleep 60;fi
  done
  if [[ $ok -eq 1 ]]; then echo "COMPLETE $key";else echo "PERMANENT_FAILURE $key";fi
  "$py" tools/aggregate_history_future_mapping.py --root "$root" || true
done
"$py" tools/aggregate_history_future_mapping.py --root "$root"
echo 'QUEUE_COMPLETE'
