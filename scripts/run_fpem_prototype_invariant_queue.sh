#!/usr/bin/env bash
# Full staged pilot, restartable after interruption; no hourly Codex monitor.
set -uo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
BACKBONE="${1:?backbone}"
DATASET="${2:?dataset}"
PRED="${3:?horizon}"
GPU="${4:?gpu}"
PY="${5:?python}"
ROOT="results/fpem_prototype_guided_invariant_20261103/${BACKBONE}/${DATASET}/pred_${PRED}"
BASE="results/fpem_dual_pattern_memory/g1_r3_full_point/${BACKBONE}/${DATASET}/pred_${PRED}/base_run_config.json"
if [ ! -f "${BASE}" ]; then
    BASE="results/fpem_dual_pattern_memory/g1_r3_extended_test_search/${BACKBONE}/${DATASET}/pred_${PRED}/base_run_config.json"
fi
mkdir -p "${ROOT}"
if [ ! -f "${BASE}" ]; then echo "missing A0 run config: ${BASE}" >> "${ROOT}/queue.log"; exit 2; fi
execute_stage() {
    stage="${1}"; shift
    if [ -f "${ROOT}/${stage}_queue_complete" ]; then return 0; fi
    attempts=0
    while [ "${attempts}" -lt 6 ]; do
        echo "stage=${stage} attempt=$((attempts+1))" >> "${ROOT}/queue.log"
        "${PY}" -u "$@" >> "${ROOT}/queue.log" 2>&1
        code=$?
        if [ "${code}" -eq 0 ]; then
            touch "${ROOT}/${stage}_queue_complete"
            return 0
        fi
        attempts=$((attempts+1))
        echo "stage=${stage} exit=${code}, waiting 60 seconds" >> "${ROOT}/queue.log"
        sleep 60
    done
    echo "stage=${stage} failed after six attempts" >> "${ROOT}/queue.log"
    return 1
}
COMMON=(--base_run_config "${BASE}" --gpu "${GPU}" --allow_legacy_reference_signature)
EXTRA=()
if [ -n "${METHODS_OVERRIDE:-}" ]; then EXTRA=(--methods "${METHODS_OVERRIDE}"); fi
execute_stage shared tools/run_fpem_prototype_invariant_search.py "${COMMON[@]}" \
    --output "${ROOT}/shared" --train_stride 16 --val_stride 4 --test_stride 1 --epochs 10 "${EXTRA[@]}" || exit 1
execute_stage finetune tools/run_fpem_prototype_invariant_finetune.py "${COMMON[@]}" \
    --shared_root "${ROOT}/shared" --output "${ROOT}/finetune" \
    --train_stride 16 --val_stride 4 --test_stride 1 --epochs 10 || exit 1
execute_stage forward tools/run_fpem_prototype_invariant_forward.py "${COMMON[@]}" \
    --shared_root "${ROOT}/shared" --output "${ROOT}/forward" \
    --train_stride 16 --val_stride 4 --test_stride 1 --epochs 10 || exit 1
execute_stage fusion tools/run_fpem_prototype_invariant_fusion.py "${COMMON[@]}" \
    --shared_root "${ROOT}/shared" --oof_root "${ROOT}/forward" \
    --output "${ROOT}/fusion" --train_stride 16 --val_stride 4 --test_stride 1 || exit 1
execute_stage analysis tools/analyze_fpem_prototype_invariant.py --case_root "${ROOT}" || exit 1
touch "${ROOT}/all_stages_complete"
echo "all stages complete" >> "${ROOT}/queue.log"
