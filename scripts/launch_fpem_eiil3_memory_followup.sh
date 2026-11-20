#!/usr/bin/env bash
# Recover Traffic-sized cases after the original sequential retry has exited.
set -euo pipefail
root=/data/OuXiaoyu/Time-Series-Library-FPEM
batch="$root/results/fpem_eiil3_fulltrain_testselect_20261113"
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
name=fpem_testselect_memory_followup_s1
cd "$root"
if screen -ls | grep -Fq ".$name"; then
  echo "REUSED_SCREEN $name"
  exit 0
fi
task="while screen -ls | grep -Fq '.fpem_testselect_memory_retry_s1'; do sleep 300; done; while nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '^[0-9]+'; do sleep 300; done; MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1 '$python' tools/run_fpem_eiil3_safe_full_queue.py --plan '$batch/plans/server1_retry_memory.csv' --output-root '$batch/cases' --gpu 0 --full-train --selection-split test --exhaustive-r3 --a0-wait-hours 0"
screen -dmS "$name" bash -lc "cd '$root'; { $task; } >> '$batch/queue_logs/$name.log' 2>&1"
echo "START_SCREEN $name"
