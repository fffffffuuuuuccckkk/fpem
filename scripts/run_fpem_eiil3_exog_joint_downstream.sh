#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index}"
case_key="${2:?case key}"
pilot=/tmp/fpem_exog_joint_pilot
repo=/data/OuXiaoyu/Time-Series-Library-FPEM
destination="$repo/results/fpem_eiil3_exog_joint_20261117/server3_pilot"
python=/data/OuXiaoyu/miniconda3/envs/tslib/bin/python
source="$pilot/results/rediscover/$case_key/exog_search.json"
while [[ ! -s "$source" ]]; do
  if ! pgrep -af 'python tools/run_fpem_eiil3_exog.py' | grep -Fq -- "--case-key $case_key "; then
    echo "STAGE_FAILED_OR_STOPPED $case_key $(date -Is)"
    exit 1
  fi
  echo "WAIT_STAGE $case_key $(date -Is)"
  sleep 300
done

cd "$pilot/code"
for branch in head_plus_exog head_only; do
  if [[ "$branch" == head_only ]]; then
    done_file="$pilot/results/head_only/fixed_original/$case_key/exog_complete"
  else
    done_file="$pilot/results/fixed_original/$case_key/exog_complete"
  fi
  [[ -s "$done_file" ]] && continue
  for attempt in 1 2 3; do
    echo "START $case_key $branch attempt=$attempt $(date -Is)"
    if CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
      "$python" tools/run_fpem_eiil3_exog.py \
      --manifest "$pilot/assets/manifest.jsonl" --case-key "$case_key" \
      --output-root "$pilot/results" --gpu 0 --use-exog-embedding \
      --train-forecast-head --downstream-branch "$branch" \
      --q-mode fixed_original --fixed-q-source-root "$pilot/fixed_q_source" \
      --selection-split val --exog-lrs 0.0001,0.0003,0.001 \
      --exog-weight-decays 0 --exog-epochs 10 \
      --exog-batch-size 16 --exog-eval-batch-size 32 \
      --train-stride 8 --val-stride 8 --test-stride 1 \
      --batch-size 16 --router-epochs 30 --epochs 10 \
      --head-lrs 0.00002,0.0001,0.0003,0.001 \
      --lambda-values 0.0001,0.01 \
      --variation-lrs 0.0001,0.0003,0.001,0.003 --resume; then
      break
    fi
    [[ "$attempt" -eq 3 ]] && exit 1
    echo "RETRY_AFTER_60S $case_key $branch $(date -Is)"
    sleep 60
  done
  [[ -s "$done_file" ]] || { echo "MISSING_COMPLETE $done_file"; exit 1; }
done

ssh -o BatchMode=yes OuXiaoyu@211.71.76.25 "mkdir -p '$destination'"
rsync -a "$pilot/results/" "OuXiaoyu@211.71.76.25:$destination/"
echo "MIRRORED $case_key $destination $(date -Is)"
