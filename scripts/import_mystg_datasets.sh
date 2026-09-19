#!/usr/bin/env bash
# Run this script on 211.71.72.121. It pushes the selected MySTG exports to
# Time-Series-Library-FPEM and invokes the checked-in converter on 211.71.76.25.
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/data/OuXiaoyu/mystg/datasets}"
DEST_HOST="${DEST_HOST:-OuXiaoyu@211.71.76.25}"
DEST_PROJECT="${DEST_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
DEST_PYTHON="${DEST_PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
REMOTE_STAGE="${REMOTE_STAGE:-${DEST_PROJECT}/dataset/.mystg_source}"

DATASETS=(Electricity ETTh2 ETTm1 ETTm2 ExchangeRate Weather Traffic)
FILES=(meta.json train_data.npy train_timestamps.npy val_data.npy val_timestamps.npy test_data.npy test_timestamps.npy)

echo "[import] source=${SOURCE_ROOT}"
echo "[import] destination=${DEST_HOST}:${DEST_PROJECT}"
ssh -o BatchMode=yes "${DEST_HOST}" "mkdir -p '${REMOTE_STAGE}'"

for dataset in "${DATASETS[@]}"; do
    source_dir="${SOURCE_ROOT}/${dataset}"
    source_files=()
    if [[ ! -d "${source_dir}" ]]; then
        echo "missing source dataset: ${source_dir}" >&2
        exit 1
    fi
    for filename in "${FILES[@]}"; do
        if [[ ! -f "${source_dir}/${filename}" ]]; then
            echo "missing source file: ${source_dir}/${filename}" >&2
            exit 1
        fi
        source_files+=("${source_dir}/${filename}")
    done

    echo "[transfer] ${dataset}"
    ssh -o BatchMode=yes "${DEST_HOST}" "mkdir -p '${REMOTE_STAGE}/${dataset}'"
    rsync -a --partial --human-readable --info=progress2 \
        "${source_files[@]}" \
        "${DEST_HOST}:${REMOTE_STAGE}/${dataset}/"
done

echo "[convert] rebuilding Time-Series-Library CSV files"
ssh -o BatchMode=yes "${DEST_HOST}" \
    "cd '${DEST_PROJECT}' && '${DEST_PYTHON}' tools/convert_mystg_datasets.py \
        --source-root '${REMOTE_STAGE}' \
        --project-root '${DEST_PROJECT}' \
        2>&1 | tee dataset/mystg_import.log"

echo "[done] report: ${DEST_PROJECT}/dataset/mystg_import_manifest.txt"
