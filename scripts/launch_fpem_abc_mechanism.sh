#!/usr/bin/env bash
# Four independent GPU screens on the primary host. Full six run first;
# remaining existing 60 cases receive no-retrain A/B diagnostics.
set -euo pipefail
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
SOURCE="$ROOT/results/fpem_dual_pattern_memory/g1_r3_full_testlr_search"
RESULT="$ROOT/results/fpem_abc_mechanism_search_20261031"
cd "$ROOT"
mkdir -p "$RESULT/queue_logs"
gpu0=(patchtst/ETTh1/pred_720 patchtst/ETTh1/pred_96)
gpu1=(itransformer/ETTm1/pred_720 itransformer/ETTm1/pred_96)
gpu2=(moderntcn/ETTm2/pred_720)
gpu3=(moderntcn/ETTm2/pred_96)
index=0
while IFS= read -r case_path; do
  case "$case_path" in
    patchtst/ETTh1/pred_720|patchtst/ETTh1/pred_96|\
    itransformer/ETTm1/pred_720|itransformer/ETTm1/pred_96|\
    moderntcn/ETTm2/pred_720|moderntcn/ETTm2/pred_96) continue ;;
  esac
  case $((index % 4)) in
    0) gpu0+=("low:$case_path") ;;
    1) gpu1+=("low:$case_path") ;;
    2) gpu2+=("low:$case_path") ;;
    3) gpu3+=("low:$case_path") ;;
  esac
  index=$((index+1))
done < <(find "$SOURCE" -mindepth 3 -maxdepth 3 -type d -name 'pred_*' \
  -printf '%P\n' | sort)
for gpu in 0 1 2 3; do
  session="fpem_abc_g${gpu}"
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "ALREADY RUNNING $session"
    continue
  fi
  case "$gpu" in
    0) screen -dmS "$session" bash scripts/run_fpem_abc_queue.sh 0 "${gpu0[@]}" ;;
    1) screen -dmS "$session" bash scripts/run_fpem_abc_queue.sh 1 "${gpu1[@]}" ;;
    2) screen -dmS "$session" bash scripts/run_fpem_abc_queue.sh 2 "${gpu2[@]}" ;;
    3) screen -dmS "$session" bash scripts/run_fpem_abc_queue.sh 3 "${gpu3[@]}" ;;
  esac
  echo "STARTED $session"
done
echo "PLANNED six full ABC, ${index} additional A/B low-cost cases"
screen -ls
