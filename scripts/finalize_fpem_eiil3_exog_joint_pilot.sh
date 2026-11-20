#!/usr/bin/env bash
set -euo pipefail

pilot=/tmp/fpem_exog_joint_pilot
repo=/data/OuXiaoyu/Time-Series-Library-FPEM
destination="$repo/results/fpem_eiil3_exog_joint_20261117/server3_pilot"
python=/data/OuXiaoyu/miniconda3/envs/tslib/bin/python
cases=(
  itransformer/ETTh1/pred_96/seed_2021
  patchtst/ETTh1/pred_720/seed_2021
  patchtst/Weather/pred_96/seed_2021
  moderntcn/Weather/pred_96/seed_2021
  itransformer/Weather/pred_96/seed_2021
)

while :; do
  missing=0
  for case in "${cases[@]}"; do
    report="$pilot/results/rediscover/$case/exog_search.json"
    if [[ ! -s "$report" ]]; then
      missing=$((missing + 1))
      if ! pgrep -af 'python tools/run_fpem_eiil3_exog.py' | grep -Fq -- "--case-key $case "; then
        echo "FAILED_OR_STOPPED $case $(date -Is)"
        exit 1
      fi
    fi
  done
  [[ "$missing" -eq 0 ]] && break
  echo "WAIT remaining=$missing $(date -Is)"
  sleep 300
done

cd "$pilot/code"
"$python" tools/summarize_fpem_eiil3_exog_joint.py --root "$pilot/results"
ssh -o BatchMode=yes OuXiaoyu@211.71.76.25 "mkdir -p '$destination'"
rsync -a "$pilot/results/" "OuXiaoyu@211.71.76.25:$destination/"
echo "MIRRORED $destination $(date -Is)"
