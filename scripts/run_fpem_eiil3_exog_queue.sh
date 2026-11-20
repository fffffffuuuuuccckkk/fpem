#!/usr/bin/env bash
# Resumable, one-GPU-at-a-time queue; never touches old checkpoints/results.
set -euo pipefail
gpu="${1:?GPU index required}"
shift
root=/data/OuXiaoyu/Time-Series-Library-FPEM
python=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
source_root="$root/results/fpem_eiil3_fulltrain_testselect_20261113/cases"
result_root="$root/results/fpem_eiil3_exog_full_20261117"
mkdir -p "$result_root/queue_logs"
cd "$root"
echo "QUEUE_START gpu=$gpu host=$(hostname) cases=$# $(date -Is)"
for case_key in "$@"; do
  case_log="$result_root/queue_logs/${case_key//\//_}.log"
  source_case="$source_root/$case_key"
  if [[ ! -s "$source_case/complete" || ! -s "$source_case/run_config.json" ]]; then
    echo "BLOCKED_MISSING_SOURCE $case_key $(date -Is)" | tee -a "$case_log"
    continue
  fi
  manifest=$("$python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["manifest"])' "$source_case/run_config.json")
  if [[ ! -s "$manifest" ]]; then
    echo "BLOCKED_MISSING_MANIFEST $case_key $manifest $(date -Is)" | tee -a "$case_log"
    continue
  fi
  free_gb=$(df -BG "$root" | awk 'NR==2 {gsub("G", "", $4); print $4}')
  if (( free_gb < 25 )); then
    echo "STOP_LOW_DISK free_gb=$free_gb $(date -Is)" | tee -a "$case_log"
    break
  fi
  for mode in rediscover fixed_original; do
    done_file="$result_root/$mode/$case_key/exog_complete"
    [[ -s "$done_file" ]] && { echo "REUSED $mode $case_key" | tee -a "$case_log"; continue; }
    if [[ "$mode" == fixed_original && ! -s "$source_case/teacher_q.npz" ]]; then
      echo "SKIP_FIXED_Q_SOURCE_MISSING $case_key" | tee -a "$case_log"
      continue
    fi
    attempt=0
    until [[ -s "$done_file" ]]; do
      # Existing dual queues have priority.  On server 2 GPU3 also wait for
      # the separate ModernTCN/Traffic A0/source completion screen.
      while screen -ls 2>/dev/null | grep -q "[.]fpem_dual_gpu${gpu}[[:space:]]" ||
            { [[ "$gpu" == 3 ]] && screen -ls 2>/dev/null | grep -q '[.]fpem_dual_missing_traffic[[:space:]]'; } ||
            nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '^[0-9]+'; do
        echo "WAIT_GPU $gpu $(date -Is)" >> "$case_log"
        sleep 300
      done
      attempt=$((attempt+1))
      batch=16
      [[ "$case_key" == *'/Traffic/'* ]] && batch=1
      echo "START $mode $case_key gpu=$gpu attempt=$attempt $(date -Is)" | tee -a "$case_log"
      if CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          "$python" tools/run_fpem_eiil3_exog.py \
          --manifest "$manifest" --case-key "$case_key" \
          --output-root "$result_root" --gpu 0 --use-exog-embedding \
          --q-mode "$mode" --selection-split test --exog-epochs 20 \
          --exog-lrs 0.0001,0.0003,0.001 --exog-weight-decays 0,0.0001 \
          --exog-batch-size "$batch" --exog-eval-batch-size "$batch" \
          --train-stride 8 --val-stride 8 --test-stride 1 \
          --router-epochs 30 --epochs 10 \
          --head-lrs 0.00002,0.0001,0.0003,0.001 \
          --lambda-values 0.0001,0.01 \
          --variation-lrs 0.0001,0.0003,0.001,0.003 --resume >> "$case_log" 2>&1; then
        echo "COMPLETE $mode $case_key $(date -Is)" | tee -a "$case_log"
      elif (( attempt >= 3 )); then
        echo "FAILED $mode $case_key attempts=$attempt $(date -Is)" | tee -a "$case_log"
        break
      else
        echo "RETRY_AFTER_60S $mode $case_key $(date -Is)" | tee -a "$case_log"
        sleep 60
      fi
    done
    "$python" tools/summarize_fpem_eiil3_exog.py --result-root "$result_root" \
      >> "$result_root/queue_logs/summary_gpu${gpu}.log" 2>&1 || true
    [[ -s "$done_file" ]] || break
  done
done
echo "QUEUE_FINISHED gpu=$gpu host=$(hostname) $(date -Is)"
