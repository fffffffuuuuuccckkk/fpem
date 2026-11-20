#!/usr/bin/env bash
set -uo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
BACKBONE="${1:?backbone}"
DATASET="${2:?dataset}"
PRED="${3:?pred_len}"
GPU="${4:?gpu}"
PYTHON_BIN="${5:?python path}"
WAIT_IDLE="${6:-0}"
ROOT="results/fpem_direct_invariant_20261102"
CASE="${ROOT}/pilot/${BACKBONE}/${DATASET}/pred_${PRED}"
BASE="results/fpem_dual_pattern_memory/g1_r3_full_point/${BACKBONE}/${DATASET}/pred_${PRED}/base_run_config.json"
mkdir -p "${CASE}"
if [ ! -f "${BASE}" ]; then echo "missing ${BASE}" >&2; exit 2; fi
attempt=0
while [ "${attempt}" -lt 6 ]; do
    if [ "${WAIT_IDLE}" = 1 ]; then
        while true; do
            used="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
            if [ "${used:-999999}" -lt 200 ]; then break; fi
            echo "GPU ${GPU} busy (${used} MiB); waiting 60s" >> "${CASE}/queue.log"
            sleep 60
        done
    fi
    "${PYTHON_BIN}" -u tools/run_fpem_direct_invariant_search.py \
        --base_run_config "${BASE}" --output "${CASE}" --gpu "${GPU}" \
        --train_stride 16 --val_stride 4 --test_stride 1 --epochs 10 \
        --allow_legacy_reference_signature >> "${CASE}/queue.log" 2>&1
    code=$?
    if [ "${code}" -eq 0 ]; then
        echo "complete" > "${CASE}/queue_complete"
        exit 0
    fi
    attempt=$((attempt+1))
    echo "attempt ${attempt} failed with ${code}; retry after 60s" >> "${CASE}/queue.log"
    sleep 60
done
echo "queue exhausted retries" >> "${CASE}/queue.log"
exit 1
