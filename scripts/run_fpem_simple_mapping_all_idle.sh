#!/usr/bin/env bash
# Run all 255 combinations for each of six cases; resumable per combination.
set -uo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
ROOT="${1:-results/fpem_simple_mapping_prototype_sweep}"
PY="${2:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
mkdir -p "${ROOT}/queue_logs"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
echo "$(date -Is) host=$(hostname) commit=$(git rev-parse HEAD)" >> "${ROOT}/queue_logs/queue.log"
should_run() {
  local token="$1"
  if [ -z "${FPEM_SIMPLE_SHARD_CASES:-}" ]; then return 0; fi
  case ",${FPEM_SIMPLE_SHARD_CASES}," in
    *",${token},"*) return 0 ;;
    *) return 1 ;;
  esac
}
wait_gpu() {
  local preferred="$1" entry="$2" ready
  while true; do
    ready="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, -v preferred="${preferred}" '$1+0 == preferred && $2+0 < 350 && $3+0 < 6 {print $1; exit}')"
    if [ -n "${ready}" ]; then return 0; fi
    echo "$(date -Is) PENDING ${entry} gpu=${preferred}" >> "${ROOT}/queue_logs/queue.log"
    sleep 300
  done
}
run_lane() {
  local gpu="$1"; shift
  local failures=0
  for entry in "$@"; do
    read -r backbone dataset horizon <<< "${entry}"
    if ! should_run "${backbone}:${dataset}:${horizon}"; then continue; fi
    local base="results/fpem_dual_pattern_memory/g1_r3_full_point/${backbone}/${dataset}/pred_${horizon}/base_run_config.json"
    local out="${ROOT}/pilots/${backbone}/${dataset}/pred_${horizon}/seed_2022"
    local log="${ROOT}/queue_logs/${backbone}_${dataset}_${horizon}.log"
    mkdir -p "${out}"
    if [ -f "${out}/complete" ]; then echo "SKIP ${entry}" >> "${log}"; continue; fi
    if [ ! -f "${base}" ]; then
      echo "$(date -Is) FAIL missing A0 config ${base}" >> "${log}"
      failures=$((failures+1)); continue
    fi
    wait_gpu "${gpu}" "${entry}"
    for attempt in 1 2; do
      echo "$(date -Is) START ${entry} gpu=${gpu} attempt=${attempt}" >> "${log}"
      "${PY}" -u tools/run_fpem_simple_mapping_prototypes.py \
        --base_run_config "${base}" --output "${out}" --gpu "${gpu}" \
        --stride 8 --seed 2022 --allow_legacy_reference_signature >> "${log}" 2>&1
      "${PY}" tools/aggregate_fpem_simple_mapping_prototypes.py --root "${ROOT}" \
        > "${ROOT}/queue_logs/live_table.txt" 2>&1
      count="$(find "${out}/combos" -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l)"
      if [ "${count}" -eq 255 ]; then
        echo "$(date -Is) COMPLETE ${entry} all 255" >> "${log}"
        printf 'complete\n' > "${out}/complete"
        break
      fi
      echo "$(date -Is) INCOMPLETE ${entry} ${count}/255; retry after 60s" >> "${log}"
      sleep 60
    done
    if [ ! -f "${out}/complete" ]; then failures=$((failures+1)); fi
  done
  return "${failures}"
}
run_lane "${FPEM_SIMPLE_GPU_A:-1}" "patchtst ETTh1 720" "itransformer ETTm1 720" "moderntcn ETTm2 720" & lane_a=$!
run_lane "${FPEM_SIMPLE_GPU_B:-3}" "patchtst ETTh1 96" "itransformer ETTm1 96" "moderntcn ETTm2 96" & lane_b=$!
wait "${lane_a}"; a=$?
wait "${lane_b}"; b=$?
"${PY}" tools/aggregate_fpem_simple_mapping_prototypes.py --root "${ROOT}" \
  > "${ROOT}/queue_logs/final_table.txt" 2>&1
c=$?
echo "$(date -Is) FINISHED lanes=${a},${b} aggregate=${c}" >> "${ROOT}/queue_logs/queue.log"
if [ "${a}" -ne 0 ] || [ "${b}" -ne 0 ] || [ "${c}" -ne 0 ]; then exit 1; fi
