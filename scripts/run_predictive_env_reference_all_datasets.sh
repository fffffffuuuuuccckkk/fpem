#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_reference_all_datasets_k3}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"

dataset_config() {
    local dataset="$1"
    case "$dataset" in
        ETTh1)       echo "./dataset/ETT-small ett_hour ETTh1.csv h 32" ;;
        ETTh2)       echo "./dataset/ETT-small ett_hour ETTh2.csv h 32" ;;
        ETTm1)       echo "./dataset/ETT-small ett_minute ETTm1.csv 15min 32" ;;
        ETTm2)       echo "./dataset/ETT-small ett_minute ETTm2.csv 15min 32" ;;
        Electricity) echo "./dataset/electricity custom electricity.csv h 4" ;;
        ExchangeRate) echo "./dataset/exchange_rate custom exchange_rate.csv d 32" ;;
        Weather)     echo "./dataset/weather custom weather.csv 10min 16" ;;
        Traffic)     echo "./dataset/traffic custom traffic.csv h 2" ;;
        *) echo "unknown dataset: $dataset" >&2; return 2 ;;
    esac
}

run_one_variant() {
    local dataset="$1" gpu="$2" label="$3" mode="$4"
    local dataset_root="$OUTPUT_ROOT/$dataset"
    local variant_root="$dataset_root/$label"
    local metrics="$variant_root/A2/metrics_and_diagnostics.json"
    if [[ -s "$metrics" ]]; then
        echo "[$dataset] skip completed $label"
        return
    fi
    read -r data_root data_class data_path freq batch_size <<<"$(dataset_config "$dataset")"
    echo "[$dataset] run $label on GPU $gpu"
    GPU="$gpu" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
    DATASET_NAME="$dataset" DATA_ROOT="$data_root" DATA_CLASS="$data_class" \
    DATA_PATH="$data_path" TARGET=OT FREQ="$freq" BATCH_SIZE="$batch_size" \
    NUM_WORKERS="$NUM_WORKERS" \
    REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate VARIANT_FUSION_MODE=direct_gated \
    PREDICTIVE_ENV_REFACTOR_MODE="$mode" FUSION_SCALE_CALIBRATION=none \
    EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 LAMBDA_H_ANCHOR=0.0 \
    LAMBDA_INVPRED=1.0 LAMBDA_VAR_CONDITIONAL_GAIN=0.2 \
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
        DATASET_NAME="$dataset" DATA_ROOT="$data_root" DATA_CLASS="$data_class" \
        DATA_PATH="$data_path" TARGET=OT FREQ="$freq" BATCH_SIZE="$batch_size" \
        NUM_WORKERS="$NUM_WORKERS" \
        EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 \
        PREPARE_REFERENCE_ONLY=1 REFERENCE_CHECKPOINT="$reference" \
        OUTPUT="$dataset_root/reference_prepare" \
        bash scripts/run_predictive_env_iv_patchtst.sh
    fi
    sha256sum "$reference"
    run_one_variant "$dataset" "$gpu" A_current current
    run_one_variant "$dataset" "$gpu" B_h_reference h_reference
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
