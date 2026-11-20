#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_cyclenet_itransformer_all_datasets_k3}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"
NUM_WORKERS="${NUM_WORKERS:-0}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"
test -s "$ARCHIVE"
test -s dataset/all_datasets/ETT-small/ETTh1.csv
ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
printf 'dataset_archive=%s\nsha256=%s\n' "$ARCHIVE" "$ARCHIVE_SHA256" \
  > "$OUTPUT_ROOT/dataset_source.txt"

dataset_config() {
    case "$1" in
        ETTh1)        echo "./dataset/all_datasets/ETT-small ett_hour ETTh1.csv h 32 24" ;;
        ETTh2)        echo "./dataset/all_datasets/ETT-small ett_hour ETTh2.csv h 32 24" ;;
        ETTm1)        echo "./dataset/all_datasets/ETT-small ett_minute ETTm1.csv 15min 32 96" ;;
        ETTm2)        echo "./dataset/all_datasets/ETT-small ett_minute ETTm2.csv 15min 32 96" ;;
        Electricity)  echo "./dataset/all_datasets/electricity custom electricity.csv h 4 168" ;;
        ExchangeRate) echo "./dataset/all_datasets/exchange_rate custom exchange_rate.csv d 32 7" ;;
        Weather)      echo "./dataset/all_datasets/weather custom weather.csv 10min 16 144" ;;
        Traffic)      echo "./dataset/all_datasets/traffic custom traffic.csv h 2 168" ;;
        *) echo "unknown dataset: $1" >&2; return 2 ;;
    esac
}

metrics_valid() {
    local metrics="$1" backbone="$2" dataset="$3" experiment="$4" reference="$5"
    [[ -s "$metrics" && -s "$reference" ]] || return 1
    "$PYTHON" - "$metrics" "$backbone" "$dataset" "$experiment" \
      "$reference" "$ARCHIVE_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path

metrics, backbone, dataset, experiment, reference, archive_hash = sys.argv[1:]
row = json.loads(Path(metrics).read_text())
reference_hash = hashlib.sha256(Path(reference).read_bytes()).hexdigest()
expected = {
    "backbone": backbone,
    "dataset_name": dataset,
    "experiment": experiment,
    "dataset_archive_sha256": archive_hash,
    "reference_checkpoint_sha256": reference_hash,
    "reference_source": "loaded",
    "environment_count": 3,
    "seed": 2021,
    "optimization_epochs": 10,
    "stage_epochs": 2,
}
if any(row.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
PY
}

run_variant() {
    local backbone="$1" dataset="$2" gpu="$3" label="$4" experiment="$5"
    local dataset_root="$OUTPUT_ROOT/$backbone/$dataset"
    local variant_root="$dataset_root/$label"
    local reference="$dataset_root/shared_reference.pt"
    local metrics="$variant_root/$experiment/metrics_and_diagnostics.json"
    if metrics_valid "$metrics" "$backbone" "$dataset" "$experiment" "$reference"; then
        echo "[$backbone/$dataset] reuse verified $label"
        return
    fi
    read -r data_root data_class data_path freq batch_size cycle_len \
      <<<"$(dataset_config "$dataset")"
    local fusion="off" conditional_gain="0.0"
    if [[ "$experiment" == "A2" ]]; then
        fusion="direct_gated"
        conditional_gain="0.2"
    fi
    echo "[$backbone/$dataset] run $label on GPU $gpu"
    GPU="$gpu" BACKBONE="$backbone" SEED=2021 ENV_NUM=3 \
    EXPERIMENTS="$experiment" DATASET_NAME="$dataset" \
    DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" DATA_ROOT="$data_root" \
    DATA_CLASS="$data_class" DATA_PATH="$data_path" TARGET=OT FREQ="$freq" \
    CYCLENET_CYCLE_LEN="$cycle_len" BATCH_SIZE="$batch_size" \
    NUM_WORKERS="$NUM_WORKERS" REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate VARIANT_FUSION_MODE="$fusion" \
    PREDICTIVE_ENV_REFACTOR_MODE=current FUSION_SCALE_CALIBRATION=none \
    EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 LAMBDA_H_ANCHOR=0.0 \
    LAMBDA_INVPRED=1.0 LAMBDA_VAR_CONDITIONAL_GAIN="$conditional_gain" \
    VAR_CONDITIONAL_MARGIN=0.0 LAMBDA_VAR_PREDICTIVE=0.0 \
    LAMBDA_VAR_UTILITY=0.0 LAMBDA_FUTURE_VAR=0.0 \
    LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 ENVIRONMENT_MATCHING=overlap \
    REQUIRE_REFERENCE_CHECKPOINT=1 REFERENCE_CHECKPOINT="$reference" \
    OUTPUT="$variant_root" bash scripts/run_predictive_env_iv_patchtst.sh
}

run_pair() (
    local backbone="$1" dataset="$2" gpu="$3"
    local dataset_root="$OUTPUT_ROOT/$backbone/$dataset"
    local reference="$dataset_root/shared_reference.pt"
    mkdir -p "$dataset_root"
    exec > >(tee -a "$dataset_root/run.log") 2>&1
    echo "[$backbone/$dataset] start $(date --iso-8601=seconds) GPU=$gpu archive=$ARCHIVE_SHA256"
    read -r data_root data_class data_path freq batch_size cycle_len \
      <<<"$(dataset_config "$dataset")"
    if [[ ! -s "$reference" ]]; then
        echo "[$backbone/$dataset] create backbone-specific shared reference"
        GPU="$gpu" BACKBONE="$backbone" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
        DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
        DATA_ROOT="$data_root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
        TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle_len" \
        BATCH_SIZE="$batch_size" NUM_WORKERS="$NUM_WORKERS" \
        EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 PREPARE_REFERENCE_ONLY=1 \
        REFERENCE_CHECKPOINT="$reference" OUTPUT="$dataset_root/reference_build" \
        bash scripts/run_predictive_env_iv_patchtst.sh
    fi
    sha256sum "$reference" | tee "$dataset_root/reference_sha256.txt"
    run_variant "$backbone" "$dataset" "$gpu" A0_backbone A0
    run_variant "$backbone" "$dataset" "$gpu" A2_predictive_env A2
    echo "[$backbone/$dataset] complete $(date --iso-8601=seconds)"
)

run_queue() {
    local gpu="$1"; shift
    local item backbone dataset
    for item in "$@"; do
        backbone="${item%%:*}"
        dataset="${item#*:}"
        run_pair "$backbone" "$dataset" "$gpu"
    done
}

# Keep the two high-dimensional inverted-attention datasets on separate GPUs.
run_queue 0 itransformer:Traffic cyclenet:ETTh1 cyclenet:ETTh2 &
pid0=$!
run_queue 1 itransformer:Electricity cyclenet:Traffic &
pid1=$!
run_queue 2 itransformer:Weather itransformer:ETTm1 itransformer:ETTm2 \
  itransformer:ETTh1 itransformer:ETTh2 itransformer:ExchangeRate &
pid2=$!
run_queue 3 cyclenet:Electricity cyclenet:Weather cyclenet:ETTm1 cyclenet:ETTm2 \
  cyclenet:ExchangeRate &
pid3=$!

status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
    wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
    echo "one or more backbone/dataset queues failed" >&2
    exit "$status"
fi

"$PYTHON" tools/summarize_predictive_env_backbones.py \
  --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/backbone_summary.txt"
echo "all CycleNet/iTransformer comparisons complete $(date --iso-8601=seconds)"
