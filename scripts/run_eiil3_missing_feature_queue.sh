#!/usr/bin/env bash
# Follow an A0 queue and materialize EIIL3 q/features only after exact A0 exists.
set -euo pipefail
gpu="$1"; plan="$2"; wait_screen="$3"; max_cache_gb="$4"
root=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
asset_root="$root/results/fpem_eiil3_mse_priority_20261112/a0_assets"
cd "$root"
while screen -ls 2>/dev/null | grep -q "$wait_screen"; do sleep 300; done
while IFS=, read -r backbone dataset horizon; do
  [[ -z "$backbone" || "$backbone" == \#* ]] && continue
  output="$asset_root/$backbone/$dataset/pred_$horizon/seed_2021"
  manifest="$output/case_manifest.jsonl"
  log="$asset_root/queue_logs/${backbone}_${dataset}_${horizon}_features.log"
  if [[ ! -s "$manifest" ]]; then
    echo "BLOCKED missing manifest $backbone/$dataset/$horizon" >> "$log"
    continue
  fi
  if [[ -s "$output/assets_status.json" ]]; then
    echo "REUSED $backbone/$dataset/$horizon" >> "$log"
    continue
  fi
  for attempt in 1 2; do
    echo "START $backbone/$dataset/$horizon attempt=$attempt" >> "$log"
    if CUDA_VISIBLE_DEVICES="$gpu" "$python" tools/prepare_fpem_eiil3_features.py \
      --manifest "$manifest" --case-key "$backbone/$dataset/pred_$horizon/seed_2021" \
      --output "$output" --gpu 0 --max-cache-gb "$max_cache_gb" >> "$log" 2>&1; then
      echo "COMPLETE $backbone/$dataset/$horizon" >> "$log"
      break
    fi
    echo "FAILED $backbone/$dataset/$horizon attempt=$attempt" >> "$log"
    [[ "$attempt" -eq 1 ]] && sleep 60
  done
done < "$plan"
