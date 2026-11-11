#!/usr/bin/env bash
# One backbone per server; pilot first, then conditional 7x4 expansion.
set -uo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
BACKBONE="${1:?backbone}"
PILOT_DATASET="${2:?pilot dataset}"
GPU="${3:?gpu}"
PY="${4:?python}"
GPU2="${5:-${GPU}}"
ROOT="results/fpem_prototype_guided_invariant_20261103"
mkdir -p "${ROOT}/${BACKBONE}"
if [ "${GPU2}" = "${GPU}" ]; then
    for H in 720 96; do
        bash scripts/run_fpem_prototype_invariant_queue.sh "${BACKBONE}" "${PILOT_DATASET}" "${H}" "${GPU}" "${PY}" || exit 1
    done
else
    bash scripts/run_fpem_prototype_invariant_queue.sh "${BACKBONE}" "${PILOT_DATASET}" 720 "${GPU}" "${PY}" & PID720=$!
    bash scripts/run_fpem_prototype_invariant_queue.sh "${BACKBONE}" "${PILOT_DATASET}" 96 "${GPU2}" "${PY}" & PID96=$!
    wait "${PID720}"; CODE720=$?
    wait "${PID96}"; CODE96=$?
    if [ "${CODE720}" -ne 0 ] || [ "${CODE96}" -ne 0 ]; then exit 1; fi
fi
"${PY}" tools/decide_fpem_prototype_expansion.py --root "${ROOT}" \
    --backbone "${BACKBONE}" --dataset "${PILOT_DATASET}" >> "${ROOT}/${BACKBONE}/matrix.log" 2>&1 || exit 1
EXPAND="$("${PY}" -c 'import json,sys;print(int(json.load(open(sys.argv[1]))["expand"]))' "${ROOT}/${BACKBONE}/expansion_decision.json")"
if [ "${EXPAND}" != 1 ]; then
    echo "Pilot lacks independent prototype and forward evidence; 84-case expansion intentionally stopped" >> "${ROOT}/${BACKBONE}/matrix.log"
    exit 0
fi
SELECTED="$("${PY}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_method"])' "${ROOT}/${BACKBONE}/expansion_decision.json")"
SELECTED_H="$("${PY}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_horizon"])' "${ROOT}/${BACKBONE}/expansion_decision.json")"
# J5: repeat the VAL-selected prototype method under two further seeds, with
# the same embargoed chronological TRAIN forward-OOF check as the pilot.
for SEED in 2022 2023; do
    CASE="${ROOT}/${BACKBONE}/${PILOT_DATASET}/pred_${SELECTED_H}"
    SEED_ROOT="${CASE}/seed_${SEED}"
    mkdir -p "${SEED_ROOT}"
    if [ ! -f "${SEED_ROOT}/forward/complete" ]; then
        BASE="results/fpem_dual_pattern_memory/g1_r3_full_point/${BACKBONE}/${PILOT_DATASET}/pred_${SELECTED_H}/base_run_config.json"
        if [ ! -f "${BASE}" ]; then BASE="results/fpem_dual_pattern_memory/g1_r3_extended_test_search/${BACKBONE}/${PILOT_DATASET}/pred_${SELECTED_H}/base_run_config.json"; fi
        tries=0
        until [ "${tries}" -ge 6 ]; do
            "${PY}" -u tools/run_fpem_prototype_invariant_search.py --base_run_config "${BASE}" \
                --output "${SEED_ROOT}/shared" --gpu "${GPU}" --seed "${SEED}" \
                --train_stride 16 --val_stride 4 --test_stride 1 --epochs 10 \
                --methods "A0,p01,${SELECTED}" --allow_legacy_reference_signature \
                >> "${SEED_ROOT}/queue.log" 2>&1 && \
            "${PY}" -u tools/run_fpem_prototype_invariant_forward.py --base_run_config "${BASE}" \
                --shared_root "${SEED_ROOT}/shared" --output "${SEED_ROOT}/forward" \
                --gpu "${GPU}" --seed "${SEED}" --train_stride 16 --val_stride 4 \
                --test_stride 1 --epochs 10 --allow_legacy_reference_signature \
                >> "${SEED_ROOT}/queue.log" 2>&1 && break
            tries=$((tries+1)); sleep 60
        done
        if [ ! -f "${SEED_ROOT}/forward/complete" ]; then exit 1; fi
    fi
done
"${PY}" tools/analyze_fpem_prototype_invariant.py --case_root "${ROOT}/${BACKBONE}/${PILOT_DATASET}/pred_${SELECTED_H}" \
    >> "${ROOT}/${BACKBONE}/matrix.log" 2>&1 || exit 1
export METHODS_OVERRIDE="A0,linear,head_init,p01,p02,${SELECTED},map_consensus"
run_expansion_lane() {
    local LANE="$1" LANE_GPU="$2" INDEX=0 DATASET H CASE BASE ALT
    for DATASET in ETTh1 ETTh2 ETTm1 ETTm2 ExchangeRate Weather Electricity; do
      for H in 96 192 336 720; do
        if [ "$((INDEX%2))" -ne "${LANE}" ]; then INDEX=$((INDEX+1)); continue; fi
        INDEX=$((INDEX+1))
        CASE="${ROOT}/${BACKBONE}/${DATASET}/pred_${H}"
        if [ -f "${CASE}/all_stages_complete" ]; then continue; fi
        BASE="results/fpem_dual_pattern_memory/g1_r3_full_point/${BACKBONE}/${DATASET}/pred_${H}/base_run_config.json"
        ALT="results/fpem_dual_pattern_memory/g1_r3_extended_test_search/${BACKBONE}/${DATASET}/pred_${H}/base_run_config.json"
        if [ ! -f "${BASE}" ] && [ ! -f "${ALT}" ]; then
            echo "missing existing A0 config: ${BACKBONE}/${DATASET}/${H}" >> "${ROOT}/${BACKBONE}/matrix.log"
            continue
        fi
        bash scripts/run_fpem_prototype_invariant_queue.sh "${BACKBONE}" "${DATASET}" "${H}" "${LANE_GPU}" "${PY}" || return 1
      done
    done
}
if [ "${GPU2}" = "${GPU}" ]; then
    run_expansion_lane 0 "${GPU}" || exit 1
    run_expansion_lane 1 "${GPU}" || exit 1
else
    run_expansion_lane 0 "${GPU}" & PID0=$!
    run_expansion_lane 1 "${GPU2}" & PID1=$!
    wait "${PID0}"; CODE0=$?
    wait "${PID1}"; CODE1=$?
    if [ "${CODE0}" -ne 0 ] || [ "${CODE1}" -ne 0 ]; then exit 1; fi
fi
echo "assigned backbone matrix complete (subject to logged missing A0 configs)" >> "${ROOT}/${BACKBONE}/matrix.log"
