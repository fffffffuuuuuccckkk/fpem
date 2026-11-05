#!/usr/bin/env bash
set -euo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM
base_config="$1"
old_base="$2"
new_base="$3"
gpu="$4"
seeds="$5"
for seed in ${seeds//,/ }; do
  bash scripts/run_fpem_f03_r3_consensus_blend.sh \
    "$base_config" "$old_base/seed_$seed" "$new_base/seed_$seed" "$gpu" "$seed"
done
