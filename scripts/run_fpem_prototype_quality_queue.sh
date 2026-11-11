#!/usr/bin/env bash
# Single restartable screen queue; launches only on an idle local GPU.
set -uo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
ROOT="${1:-results/fpem_prototype_quality_20261107}"
PY="${2:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
mkdir -p "${ROOT}/queue_logs"
failed=0
for CASE in "patchtst ETTh1 720" "itransformer ETTm1 720" "moderntcn ETTm2 720"; do
    read -r BACKBONE DATASET PRED <<< "${CASE}"
    BASE="results/fpem_dual_pattern_memory/g1_r3_full_point/${BACKBONE}/${DATASET}/pred_${PRED}/base_run_config.json"
    OUT="${ROOT}/pilots/${BACKBONE}/${DATASET}/pred_${PRED}"
    LOG="${ROOT}/queue_logs/${BACKBONE}_${DATASET}_${PRED}.log"
    if [ -f "${OUT}/complete" ]; then
        echo "$(date -Is) SKIP complete ${CASE}" >> "${LOG}"
        continue
    fi
    if [ ! -f "${BASE}" ]; then
        echo "$(date -Is) FAIL missing reusable A0 config ${BASE}" >> "${LOG}"
        failed=$((failed+1))
        continue
    fi
    while true; do
        GPU="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
            awk -F, '$2+0 < 350 && $3+0 < 6 {gsub(/ /,"",$1); print $1; exit}')"
        if [ -n "${GPU}" ]; then break; fi
        echo "$(date -Is) PENDING idle GPU for ${CASE}" >> "${LOG}"
        sleep 300
    done
    for attempt in 1 2; do
        echo "$(date -Is) START ${CASE} gpu=${GPU} attempt=${attempt}" >> "${LOG}"
        "${PY}" -u tools/run_fpem_prototype_quality.py \
            --base_run_config "${BASE}" --output "${OUT}" --gpu "${GPU}" \
            --stride 8 --allow_legacy_reference_signature >> "${LOG}" 2>&1
        status=$?
        if [ "${status}" -eq 0 ] && [ -f "${OUT}/complete" ]; then
            echo "$(date -Is) COMPLETE ${CASE}" >> "${LOG}"
            break
        fi
        echo "$(date -Is) FAILED ${CASE} exit=${status}; retry in 60s" >> "${LOG}"
        sleep 60
    done
    if [ ! -f "${OUT}/complete" ]; then failed=$((failed+1)); fi
    "${PY}" -u tools/aggregate_fpem_prototype_quality.py --root "${ROOT}" \
        > "${ROOT}/queue_logs/latest_table.txt" 2>&1
done
"${PY}" -u tools/aggregate_fpem_prototype_quality.py --root "${ROOT}" \
    > "${ROOT}/queue_logs/final_table.txt" 2>&1
echo "$(date -Is) QUEUE_FINISHED failed=${failed}; total table: ${ROOT}/summary.csv and summary.md" \
    >> "${ROOT}/queue_logs/queue_status.log"
exit "${failed}"
