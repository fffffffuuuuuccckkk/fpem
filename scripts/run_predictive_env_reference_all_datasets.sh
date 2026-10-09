#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_reference_all_datasets_zip_k3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"
test -s "$ARCHIVE"
test -s dataset/all_datasets/ETT-small/ETTh1.csv
ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"

dataset_config() {
    local dataset="$1"
    case "$dataset" in
        ETTh1)       echo "./dataset/all_datasets/ETT-small ett_hour ETTh1.csv h 32" ;;
        ETTh2)       echo "./dataset/all_datasets/ETT-small ett_hour ETTh2.csv h 32" ;;
        ETTm1)       echo "./dataset/all_datasets/ETT-small ett_minute ETTm1.csv 15min 32" ;;
        ETTm2)       echo "./dataset/all_datasets/ETT-small ett_minute ETTm2.csv 15min 32" ;;
        Electricity) echo "./dataset/all_datasets/electricity custom electricity.csv h 4" ;;
        ExchangeRate) echo "./dataset/all_datasets/exchange_rate custom exchange_rate.csv d 32" ;;
        Weather)     echo "./dataset/all_datasets/weather custom weather.csv 10min 16" ;;
        Traffic)     echo "./dataset/all_datasets/traffic custom traffic.csv h 2" ;;
        *) echo "unknown dataset: $dataset" >&2; return 2 ;;
    esac
}

result_is_valid() {
    local metrics="$1" dataset="$2" experiment="$3" mode="$4"
    local scale="$5" fusion="$6" conditional_gain="$7" isolated="$8"
    local reference="$9"
    [[ -s "$metrics" && -s "$reference" ]] || return 1
    "$PYTHON" - "$metrics" "$dataset" "$experiment" "$mode" "$scale" \
        "$fusion" "$conditional_gain" "$isolated" "$reference" \
        "$ARCHIVE_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(metrics_path, dataset, experiment, mode, scale, fusion,
 conditional_gain, isolated, reference_path, archive_hash) = sys.argv[1:]
row = json.loads(Path(metrics_path).read_text())
digest = hashlib.sha256(Path(reference_path).read_bytes()).hexdigest()
expected = {
    "dataset_name": dataset,
    "dataset_archive_sha256": archive_hash,
    "backbone": "patchtst",
    "experiment": experiment,
    "predictive_env_refactor_mode": mode,
    "fusion_scale_calibration": scale,
    "variant_fusion_mode": fusion,
    "lambda_var_conditional_gain": float(conditional_gain),
    "z_specific_encoder_gradient_isolated": isolated == "true",
    "representation_constraint": "classification",
    "decomposition_type": "complementary_gate",
    "environment_count": 3,
    "seed": 2021,
    "optimization_epochs": 10,
    "stage_epochs": 2,
    "lambda_h_anchor": 0.0,
    "reference_checkpoint_sha256": digest,
    "reference_source": "loaded",
}
if any(row.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
PY
}

run_one_variant() {
    local dataset="$1" gpu="$2" label="$3" experiment="$4" mode="$5"
    local scale="$6" fusion="$7" conditional_gain="$8" isolated="$9"
    local dataset_root="$OUTPUT_ROOT/$dataset"
    local variant_root="$dataset_root/$label"
    local metrics="$variant_root/$experiment/metrics_and_diagnostics.json"
    if result_is_valid "$metrics" "$dataset" "$experiment" "$mode" "$scale" \
        "$fusion" "$conditional_gain" "$isolated" "$dataset_root/shared_reference.pt"; then
        echo "[$dataset] reuse verified $label"
        return
    fi
    read -r data_root data_class data_path freq batch_size <<<"$(dataset_config "$dataset")"
    echo "[$dataset] run $label on GPU $gpu"
    GPU="$gpu" SEED=2021 ENV_NUM=3 EXPERIMENTS="$experiment" \
    BACKBONE=patchtst DATASET_NAME="$dataset" \
    DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
    DATA_ROOT="$data_root" DATA_CLASS="$data_class" \
    DATA_PATH="$data_path" TARGET=OT FREQ="$freq" BATCH_SIZE="$batch_size" \
    NUM_WORKERS="$NUM_WORKERS" \
    REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate VARIANT_FUSION_MODE="$fusion" \
    PREDICTIVE_ENV_REFACTOR_MODE="$mode" FUSION_SCALE_CALIBRATION="$scale" \
    EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 LAMBDA_H_ANCHOR=0.0 \
    LAMBDA_INVPRED=1.0 LAMBDA_VAR_CONDITIONAL_GAIN="$conditional_gain" \
    VAR_CONDITIONAL_MARGIN=0.0 LAMBDA_VAR_PREDICTIVE=0.0 \
    LAMBDA_VAR_UTILITY=0.0 LAMBDA_FUTURE_VAR=0.0 \
    LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 \
    ENVIRONMENT_MATCHING=overlap REQUIRE_REFERENCE_CHECKPOINT=1 \
    REFERENCE_CHECKPOINT="$dataset_root/shared_reference.pt" \
    OUTPUT="$variant_root" bash scripts/run_predictive_env_iv_patchtst.sh
}

run_dataset() (
    local dataset="$1" gpu="$2"
    local dataset_root="$OUTPUT_ROOT/$dataset"
    local reference="$dataset_root/shared_reference.pt"
    mkdir -p "$dataset_root"
    exec > >(tee -a "$dataset_root/run.log") 2>&1
    echo "[$dataset] start $(date --iso-8601=seconds) GPU=$gpu"
    read -r data_root data_class data_path freq batch_size <<<"$(dataset_config "$dataset")"
    if [[ ! -s "$reference" ]]; then
        echo "[$dataset] create shared reference"
        GPU="$gpu" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
        BACKBONE=patchtst DATASET_NAME="$dataset" \
        DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
        DATA_ROOT="$data_root" DATA_CLASS="$data_class" \
        DATA_PATH="$data_path" TARGET=OT FREQ="$freq" BATCH_SIZE="$batch_size" \
        NUM_WORKERS="$NUM_WORKERS" \
        EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 \
        PREPARE_REFERENCE_ONLY=1 REFERENCE_CHECKPOINT="$reference" \
        OUTPUT="$dataset_root/reference_prepare" \
        bash scripts/run_predictive_env_iv_patchtst.sh
    fi
    sha256sum "$reference"
    run_one_variant "$dataset" "$gpu" A0_patchtst A0 current none off 0.0 false
    run_one_variant "$dataset" "$gpu" A_current A2 current none direct_gated 0.2 false
    run_one_variant "$dataset" "$gpu" B_h_reference A2 h_reference none direct_gated 0.2 false
    run_one_variant "$dataset" "$gpu" C_gradient_isolated A2 h_reference_grad_isolated none direct_gated 0.2 true
    run_one_variant "$dataset" "$gpu" D_isolated_rms A2 h_reference_grad_isolated rms direct_gated 0.2 true
    echo "[$dataset] complete $(date --iso-8601=seconds)"
)

run_queue() {
    local gpu="$1"
    shift
    local dataset
    for dataset in "$@"; do
        run_dataset "$dataset" "$gpu"
    done
}

# Balance by approximate (time steps x variables), with one sequential queue
# per GPU so two wide datasets never contend for the same 8 GB device.
run_queue 0 Traffic & p0=$!
run_queue 1 Electricity & p1=$!
run_queue 2 Weather ETTm1 ETTh2 & p2=$!
run_queue 3 ETTm2 ETTh1 ExchangeRate & p3=$!

status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
    wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
    echo "At least one dataset queue failed; inspect per-dataset run.log" >&2
    exit "$status"
fi

"$PYTHON" tools/summarize_predictive_env_all_datasets.py "$OUTPUT_ROOT"
cat "$OUTPUT_ROOT/all_datasets_summary.txt"
