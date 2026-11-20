#!/usr/bin/env bash
# Two independent, restartable lanes; wait for free GPUs, then stop after grouping.
set -uo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
ROOT="${1:-results/fpem_mechanism_group_discovery}"
PY="${2:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
mkdir -p "${ROOT}/queue_logs"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
echo "$(date -Is) host=$(hostname) commit=$(git rev-parse HEAD)" >> "${ROOT}/queue_logs/queue.log"
wait_free_gpu() {
  local gpu="$1" ready
  while true; do
    ready="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, -v gpu="${gpu}" '$1+0 == gpu && $2+0 < 350 && $3+0 < 6 {print $1; exit}')"
    if [ -n "${ready}" ]; then return 0; fi
    echo "$(date -Is) PENDING gpu=${gpu}" >> "${ROOT}/queue_logs/queue.log"
    sleep 300
  done
}
run_lane() {
  local gpu="$1"; shift
  local failures=0
  for entry in "$@"; do
    read -r backbone dataset horizon <<< "${entry}"
    base="results/fpem_dual_pattern_memory/g1_r3_full_point/${backbone}/${dataset}/pred_${horizon}/base_run_config.json"
    out="${ROOT}/pilots/${backbone}/${dataset}/pred_${horizon}/seed_2022"
    log="${ROOT}/queue_logs/${backbone}_${dataset}_${horizon}.log"
    mkdir -p "${out}"
    if [ -f "${out}/complete" ]; then echo "SKIP ${entry}" >> "${log}"; continue; fi
    if [ ! -f "${base}" ]; then echo "FAIL missing A0 config ${base}" >> "${log}"; failures=$((failures+1)); continue; fi
    wait_free_gpu "${gpu}"
    for attempt in 1 2; do
      echo "$(date -Is) START ${entry} gpu=${gpu} attempt=${attempt}" >> "${log}"
      "${PY}" -u tools/run_fpem_mechanism_group_discovery.py \
        --base_run_config "${base}" --output "${out}" --gpu "${gpu}" --seed 2022 \
        --stride 8 --allow_legacy_reference_signature >> "${log}" 2>&1
      status=$?
      "${PY}" tools/aggregate_fpem_mechanism_groups.py --root "${ROOT}" \
        > "${ROOT}/queue_logs/live_table.txt" 2>&1
      count="$(find "${out}/methods" -name complete -type f 2>/dev/null | wc -l)"
      if [ "${status}" -eq 0 ] && [ "${count}" -eq 12 ]; then
        printf 'complete\n' > "${out}/complete"
        echo "$(date -Is) COMPLETE ${entry} all 12 methods" >> "${log}"
        break
      fi
      echo "$(date -Is) INCOMPLETE ${entry}: ${count}/12; retry after 60s" >> "${log}"
      sleep 60
    done
    if [ ! -f "${out}/complete" ]; then failures=$((failures+1)); fi
  done
  return "${failures}"
}
run_lane "${FPEM_MECH_GPU:-3}" \
  "patchtst ETTh1 96" "patchtst ETTh1 720" \
  "itransformer ETTm1 96" "itransformer ETTm1 720" \
  "moderntcn ETTm2 96" "moderntcn ETTm2 720"
a=$?
"${PY}" tools/aggregate_fpem_mechanism_groups.py --root "${ROOT}" \
  > "${ROOT}/queue_logs/final_table.txt" 2>&1
c=$?
echo "$(date -Is) FINISHED lane=${a} aggregate=${c}" >> "${ROOT}/queue_logs/queue.log"
if [ "${a}" -ne 0 ] || [ "${c}" -ne 0 ]; then exit 1; fi
