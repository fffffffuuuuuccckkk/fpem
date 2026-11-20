#!/usr/bin/env bash
# Usage: bash scripts/run_fpem_mechanism_fusion_queue.sh ROOT GPU backbone dataset horizon [more triples...]
# A configuration's A0 features and F03 head are fitted/cached once; all
# G3/G4/G7 and negative-control trials read the same shared cache.
set -uo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
ROOT="$1"; GPU="$2"; shift 2
PY=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
mkdir -p "${ROOT}/queue_logs"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4
wait_gpu() {
  while true; do
    ready="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F, -v gpu="${GPU}" '$1+0==gpu && $2+0<400 && $3+0<20 {print "yes";exit}')"
    active="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
      grep -E '^[[:space:]]*[0-9]+' || true)"
    if [ "${ready}" = yes ] && [ -z "${active}" ]; then return; fi
    echo "$(date -Is) PENDING GPU ${GPU}" >> "${ROOT}/queue_logs/queue.log"
    sleep 300
  done
}
failure=0
while [ "$#" -ge 3 ]; do
  backbone="$1"; dataset="$2"; horizon="$3"; shift 3
  case_dir="${ROOT}/pilots/${backbone}/${dataset}/pred_${horizon}/seed_2022"
  stage="${ROOT}/stage1_inputs/${backbone}/${dataset}/pred_${horizon}/seed_2022"
  base="results/fpem_dual_pattern_memory/g1_r3_full_point/${backbone}/${dataset}/pred_${horizon}/base_run_config.json"
  paired_base="${ROOT}/base_configs/${backbone}/${dataset}/pred_${horizon}/base_run_config.json"
  if [ -f "${paired_base}" ]; then base="${paired_base}"; fi
  log="${ROOT}/queue_logs/${backbone}_${dataset}_${horizon}.log"
  if [ ! -f "${base}" ] || [ ! -f "${stage}/case_cache.json" ]; then
    echo "MISSING_INPUT ${backbone}/${dataset}-${horizon}" >> "${log}"
    failure=$((failure+1));continue
  fi
  wait_gpu
  for method in G3 G4 G7 Raw Random ShuffleY; do
    name="${method}"
    case "${method}" in
      G3) name=G3_KerHRM_lite;; G4) name=G4_ANM_MM_lite;;
      G7) name=G7_Ridge_Update;; Raw) name=B0_Raw_History;;
      Random) name=B1_Random_Balanced;; ShuffleY) name=B2_ShuffleY_G1;;
    esac
    if [ -f "${case_dir}/methods/${name}/point_complete" ]; then continue; fi
    echo "$(date -Is) START_POINT ${backbone}/${dataset}-${horizon} ${method} GPU=${GPU}" >> "${log}"
    "${PY}" -u tools/run_fpem_mechanism_fusion.py --base_run_config "${base}" \
      --stage1_case "${stage}" --output "${case_dir}" --method "${method}" \
      --gpu "${GPU}" --seed 2022 --allow_legacy_reference_signature >> "${log}" 2>&1
    if [ "$?" -ne 0 ]; then
      echo "$(date -Is) FAILED_POINT ${method}" >> "${log}"
      failure=$((failure+1));break
    fi
    "${PY}" tools/aggregate_fpem_mechanism_fusion.py --root "${ROOT}" \
      > "${ROOT}/queue_logs/live_table.txt" 2>&1
  done
  # Flow only uses the three user-selected groupers. It uses four purged VAL
  # blocks; insufficient independent events produce a reasoned skip.
  for method in G3 G4 G7; do
    case "${method}" in G3) name=G3_KerHRM_lite;; G4) name=G4_ANM_MM_lite;; G7) name=G7_Ridge_Update;; esac
    if [ ! -f "${case_dir}/methods/${name}/point_complete" ]; then continue; fi
    if [ -f "${case_dir}/methods/${name}/flow_complete" ] || \
       [ -f "${case_dir}/methods/${name}/flow_status.json" ]; then continue; fi
    echo "$(date -Is) START_FLOW ${backbone}/${dataset}-${horizon} ${method}" >> "${log}"
    "${PY}" -u tools/run_fpem_mechanism_fusion_flow.py --case_dir "${case_dir}" \
      --method "${method}" --gpu "${GPU}" --seed 2022 --epochs 5 --samples 100 >> "${log}" 2>&1
    if [ "$?" -ne 0 ]; then
      echo "$(date -Is) FAILED_FLOW ${method}; point retained" >> "${log}"
      failure=$((failure+1))
    fi
    "${PY}" tools/aggregate_fpem_mechanism_fusion.py --root "${ROOT}" \
      > "${ROOT}/queue_logs/live_table.txt" 2>&1
  done
  echo "$(date -Is) CASE_END ${backbone}/${dataset}-${horizon}" >> "${log}"
done
"${PY}" tools/aggregate_fpem_mechanism_fusion.py --root "${ROOT}" \
  > "${ROOT}/queue_logs/final_table.txt" 2>&1
echo "$(date -Is) QUEUE_END failures=${failure}" >> "${ROOT}/queue_logs/queue.log"
if [ "${failure}" -ne 0 ]; then exit 1; fi
