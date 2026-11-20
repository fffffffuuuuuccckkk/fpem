#!/usr/bin/env bash
# Read-only checkpoint diagnostics; wait for an idle GPU and retry once after a transient failure.
set -uo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
EXPERIMENT="${1:?existing experiment root}"
OUTPUT="${2:?diagnostic output root}"
PREFERRED="${3:-any}"
PY="${4:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
mkdir -p "${OUTPUT}"
if [ -f "${OUTPUT}/complete" ]; then
    echo "$(date -Is) already complete" >> "${OUTPUT}/queue.log"
    exit 0
fi
for method in F03 F03_cons F03_cons_inv; do
    if [ ! -f "${EXPERIMENT}/B_F03/${method}/head.pt" ]; then
        echo "$(date -Is) missing ${method} checkpoint" >> "${OUTPUT}/queue.log"
        exit 2
    fi
done
while true; do
    GPU="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
        awk -F, -v preferred="${PREFERRED}" '$2+0 < 350 && $3+0 < 6 {gsub(/ /,"",$1); if (preferred == "any" || $1 == preferred) {print $1; exit}}')"
    if [ -n "${GPU}" ]; then break; fi
    echo "$(date -Is) pending idle GPU" >> "${OUTPUT}/queue.log"
    sleep 300
done
for attempt in 1 2; do
    echo "$(date -Is) START gpu=${GPU} attempt=${attempt}" >> "${OUTPUT}/queue.log"
    "${PY}" -u tools/diagnose_fpem_paired_f03.py --experiment "${EXPERIMENT}" \
        --output "${OUTPUT}" --gpu "${GPU}" >> "${OUTPUT}/queue.log" 2>&1
    status=$?
    if [ "${status}" -eq 0 ] && [ -f "${OUTPUT}/complete" ]; then
        echo "$(date -Is) complete" >> "${OUTPUT}/queue.log"
        exit 0
    fi
    echo "$(date -Is) exit=${status}; retry after 60s" >> "${OUTPUT}/queue.log"
    sleep 60
done
exit 1
