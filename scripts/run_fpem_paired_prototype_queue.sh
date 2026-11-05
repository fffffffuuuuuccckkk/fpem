#!/usr/bin/env bash
# Restartable sequential A -> B/C -> D/E pilot. Waits for an actually idle GPU.
set -uo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
BACKBONE="${1:?backbone}"
DATASET="${2:?dataset}"
PRED="${3:?horizon}"
PY="${4:?python executable}"
PAIRING="${5:-real}"
PREFERRED_GPU="${6:-any}"
OUT="results/fpem_paired_prototype_20261105/${PAIRING}/${BACKBONE}/${DATASET}/pred_${PRED}"
BASE="results/fpem_dual_pattern_memory/g1_r3_full_point/${BACKBONE}/${DATASET}/pred_${PRED}/base_run_config.json"
mkdir -p "${OUT}"
if [ ! -f "${BASE}" ]; then
    echo "missing reusable A0 configuration: ${BASE}" >> "${OUT}/queue.log"
    exit 2
fi
if [ -f "${OUT}/complete" ]; then
    echo "already complete" >> "${OUT}/queue.log"
    exit 0
fi
while true; do
    GPU="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, -v preferred="${PREFERRED_GPU}" '$2+0 < 350 && $3+0 < 6 {gsub(/ /, "", $1); if (preferred == "any" || $1 == preferred) {print $1; exit}}')"
    if [ -n "${GPU}" ]; then
        break
    fi
    echo "$(date -Is) waiting for an idle GPU" >> "${OUT}/queue.log"
    sleep 300
done
echo "$(date -Is) acquired GPU ${GPU}; pairing=${PAIRING}" >> "${OUT}/queue.log"
COMMON=(--base_run_config "${BASE}" --output "${OUT}" --gpu "${GPU}" --seed 2022 \
        --patterns 8 --bank_epochs 10 --head_epochs 10 --flow_epochs 5 \
        --train_stride 8 --val_stride 4 --test_stride 1 --samples 100 \
        --pairing "${PAIRING}" --allow_legacy_reference_signature)
run_stage() {
    label="${1}"; shift
    attempt=0
    while [ "${attempt}" -lt 2 ]; do
        attempt=$((attempt+1))
        echo "$(date -Is) ${label} attempt=${attempt}" >> "${OUT}/queue.log"
        "${PY}" -u tools/run_fpem_paired_prototype.py "${COMMON[@]}" "$@" >> "${OUT}/queue.log" 2>&1
        code=$?
        if [ "${code}" -eq 0 ]; then return 0; fi
        echo "$(date -Is) ${label} exit=${code}; retry in 60 seconds" >> "${OUT}/queue.log"
        sleep 60
    done
    echo "$(date -Is) ${label} failed" >> "${OUT}/queue.log"
    return 1
}
if [ ! -f "${OUT}/A_bank/complete" ]; then
    run_stage A --stop_after A || exit 1
fi
run_stage full --stop_after E || exit 1
if [ -f "${OUT}/complete" ]; then
    echo "$(date -Is) all stages complete" >> "${OUT}/queue.log"
elif [ -f "${OUT}/D_memory/insufficient_support.json" ]; then
    echo "$(date -Is) point pilot complete; probability skipped for audited support shortage" >> "${OUT}/queue.log"
else
    echo "$(date -Is) incomplete without declared support shortage" >> "${OUT}/queue.log"
    exit 1
fi
