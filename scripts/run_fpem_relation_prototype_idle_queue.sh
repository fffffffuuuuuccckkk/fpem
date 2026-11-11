#!/usr/bin/env bash
# Two restartable local GPU lanes, one final auto-aggregated table.
set -uo pipefail
cd /data/OuXiaoyu/Time-Series-Library-FPEM || exit 2
ROOT="${1:-results/fpem_relation_operator_prototype_pilot}"
PY="${2:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
mkdir -p "${ROOT}/queue_logs"
echo "$(date -Is) host=$(hostname) commit=$(git rev-parse HEAD)" >> "${ROOT}/queue_logs/queue.log"
# Cross-server SSH is optional; no credentials are embedded or requested.
for host in 211.71.76.187 211.71.76.44; do
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "OuXiaoyu@${host}" true 2>/dev/null; then
        echo "$(date -Is) reachable ${host}; local A0/data/code affinity keeps this pilot on server1" >> "${ROOT}/queue_logs/queue.log"
    else
        echo "$(date -Is) no unattended SSH to ${host}; local GPU lanes remain available" >> "${ROOT}/queue_logs/queue.log"
    fi
done
run_lane() {
    preferred="${1}"; shift
    lane_fail=0
    for entry in "$@"; do
        read -r backbone dataset horizon <<< "${entry}"
        base="results/fpem_dual_pattern_memory/g1_r3_full_point/${backbone}/${dataset}/pred_${horizon}/base_run_config.json"
        out="${ROOT}/pilots/${backbone}/${dataset}/pred_${horizon}/seed_2022"
        log="${ROOT}/queue_logs/${backbone}_${dataset}_${horizon}.log"
        mkdir -p "${out}"
        if [ -f "${out}/complete" ]; then
            echo "$(date -Is) SKIP complete ${entry}" >> "${log}"
            continue
        fi
        if [ ! -f "${base}" ]; then
            "${PY}" -c 'import json,sys;open(sys.argv[1],"w").write(json.dumps({"reason":"missing reusable base_run_config"}))' "${out}/failed.json"
            echo "$(date -Is) FAIL missing A0 config ${base}" >> "${log}"
            lane_fail=$((lane_fail+1))
            continue
        fi
        while true; do
            ready="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
                awk -F, -v preferred="${preferred}" '$1+0 == preferred && $2+0 < 350 && $3+0 < 6 {print $1; exit}')"
            if [ -n "${ready}" ]; then break; fi
            echo "$(date -Is) PENDING idle gpu=${preferred}" >> "${log}"
            sleep 300
        done
        success=0
        for attempt in 1 2; do
            echo "$(date -Is) START ${entry} gpu=${preferred} attempt=${attempt}" >> "${log}"
            "${PY}" -u tools/run_fpem_relation_prototype_diagnostics.py \
                --base_run_config "${base}" --output "${out}" --gpu "${preferred}" \
                --stride 8 --seed 2022 --allow_legacy_reference_signature >> "${log}" 2>&1
            status=$?
            if [ "${status}" -eq 0 ] && [ -f "${out}/complete" ]; then
                success=1
                echo "$(date -Is) COMPLETE ${entry}" >> "${log}"
                break
            fi
            echo "$(date -Is) exit=${status}; retry after 60s" >> "${log}"
            sleep 60
        done
        if [ "${success}" -ne 1 ]; then
            "${PY}" -c 'import json,sys;open(sys.argv[1],"w").write(json.dumps({"reason":"two attempts failed; see case log", "log":sys.argv[2]}))' "${out}/failed.json" "${log}"
            lane_fail=$((lane_fail+1))
        fi
    done
    return "${lane_fail}"
}
run_lane 1 "patchtst ETTh1 720" "itransformer ETTm1 720" "moderntcn ETTm2 720" &
lane_a=$!
run_lane 3 "patchtst ETTh1 96" "itransformer ETTm1 96" "moderntcn ETTm2 96" &
lane_b=$!
wait "${lane_a}"; status_a=$?
wait "${lane_b}"; status_b=$?
"${PY}" -u tools/aggregate_fpem_relation_prototype_diagnostics.py --root "${ROOT}" \
    > "${ROOT}/queue_logs/final_table.txt" 2>&1
aggregate_status=$?
echo "$(date -Is) FINISHED lane_a=${status_a} lane_b=${status_b} aggregate=${aggregate_status}" \
    >> "${ROOT}/queue_logs/queue.log"
if [ "${status_a}" -ne 0 ] || [ "${status_b}" -ne 0 ] || [ "${aggregate_status}" -ne 0 ]; then
    exit 1
fi
