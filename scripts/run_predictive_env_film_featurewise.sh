#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_film_featurewise_patchtst_cyclenet_k3}"
PATCHTST_REF_ROOT="${PATCHTST_REF_ROOT:-results/predictive_env_featurewise_gate_all_backbones_k3/patchtst}"
CYCLENET_REF_ROOT="${CYCLENET_REF_ROOT:-results/predictive_env_cyclenet_itransformer_all_datasets_k3/cyclenet}"
ARCHIVE="${ARCHIVE:-dataset/all_datasets.zip}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LAMBDA_VAR_GAIN="${LAMBDA_VAR_GAIN:-0.1}"
VAR_GAIN_TEMPERATURE="${VAR_GAIN_TEMPERATURE:-0.01}"
FILM_GAMMA_SCALE="${FILM_GAMMA_SCALE:-0.1}"
FILM_BETA_SCALE="${FILM_BETA_SCALE:-0.1}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/logs"
test -s "$ARCHIVE"
ARCHIVE_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
printf 'backbones=patchtst,cyclenet\ndatasets=8\nfusions=direct_gated,film\nfeature_wise=true\nlambda_var_gain=%s\nfilm_gamma_scale=%s\nfilm_beta_scale=%s\narchive_sha256=%s\n' \
  "$LAMBDA_VAR_GAIN" "$FILM_GAMMA_SCALE" "$FILM_BETA_SCALE" "$ARCHIVE_SHA256" \
  > "$OUTPUT_ROOT/protocol.txt"

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

reference_path() {
    local backbone="$1" dataset="$2"
    if [[ "$backbone" == patchtst ]]; then
        echo "$PATCHTST_REF_ROOT/$dataset/shared_reference.pt"
    else
        echo "$CYCLENET_REF_ROOT/$dataset/shared_reference.pt"
    fi
}

result_valid() {
    local metrics="$1" config="$2" backbone="$3" dataset="$4" fusion="$5" reference="$6"
    [[ -s "$metrics" && -s "$config" && -s "$reference" ]] || return 1
    "$PYTHON" - "$metrics" "$config" "$backbone" "$dataset" "$fusion" \
      "$reference" "$ARCHIVE_SHA256" "$LAMBDA_VAR_GAIN" \
      "$FILM_GAMMA_SCALE" "$FILM_BETA_SCALE" <<'PY'
import hashlib, json, sys
from pathlib import Path

(metrics_path, config_path, backbone, dataset, fusion, reference_path,
 archive_hash, gain_weight, gamma_scale, beta_scale) = sys.argv[1:]
row = json.loads(Path(metrics_path).read_text())
config = json.loads(Path(config_path).read_text())
expected = {
    "backbone": backbone,
    "dataset_name": dataset,
    "dataset_archive_sha256": archive_hash,
    "experiment": "A2",
    "representation_constraint": "classification",
    "decomposition_type": "complementary_gate",
    "variant_fusion_mode": fusion,
    "variant_fusion_gate_type": "feature",
    "environment_count": 3,
    "seed": 2021,
    "optimization_epochs": 10,
    "stage_epochs": 2,
    "lambda_var_gain": float(gain_weight),
    "reference_checkpoint_sha256": hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
    "reference_source": "loaded",
}
expected_config = {
    "film_gamma_scale": float(gamma_scale),
    "film_beta_scale": float(beta_scale),
}
if any(row.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
if any(config.get(key) != value for key, value in expected_config.items()):
    raise SystemExit(1)
PY
}

run_one() (
    local backbone="$1" dataset="$2" fusion="$3" gpu="$4"
    local destination="$OUTPUT_ROOT/$backbone/$dataset/$fusion"
    local metrics="$destination/A2/metrics_and_diagnostics.json"
    local config="$destination/run_config.json"
    local reference
    reference="$(reference_path "$backbone" "$dataset")"
    mkdir -p "$destination"
    exec > >(tee -a "$OUTPUT_ROOT/logs/${backbone}_${dataset}_${fusion}.log") 2>&1
    if result_valid "$metrics" "$config" "$backbone" "$dataset" "$fusion" "$reference"; then
        echo "[$backbone/$dataset/$fusion] reuse verified result"
        exit 0
    fi
    test -s "$reference"
    read -r data_root data_class data_path freq batch cycle <<<"$(dataset_config "$dataset")"
    echo "[$backbone/$dataset/$fusion] start GPU=$gpu $(date --iso-8601=seconds)"
    GPU="$gpu" BACKBONE="$backbone" SEED=2021 ENV_NUM=3 EXPERIMENTS=A2 \
    DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
    DATA_ROOT="$data_root" DATA_CLASS="$data_class" DATA_PATH="$data_path" \
    TARGET=OT FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" BATCH_SIZE="$batch" \
    NUM_WORKERS="$NUM_WORKERS" REPRESENTATION_CONSTRAINT=classification \
    DECOMPOSITION_TYPE=complementary_gate VARIANT_FUSION_MODE="$fusion" \
    VARIANT_FUSION_GATE_TYPE=feature FILM_GAMMA_SCALE="$FILM_GAMMA_SCALE" \
    FILM_BETA_SCALE="$FILM_BETA_SCALE" PREDICTIVE_ENV_REFACTOR_MODE=current \
    FUSION_SCALE_CALIBRATION=none EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 \
    LAMBDA_H_ANCHOR=0.0 LAMBDA_INVPRED=1.0 LAMBDA_VAR_GAIN="$LAMBDA_VAR_GAIN" \
    VAR_GAIN_TEMPERATURE="$VAR_GAIN_TEMPERATURE" \
    LAMBDA_VAR_CONDITIONAL_GAIN=0.2 VAR_CONDITIONAL_MARGIN=0.0 \
    LAMBDA_VAR_PREDICTIVE=0.0 LAMBDA_VAR_UTILITY=0.0 LAMBDA_FUTURE_VAR=0.0 \
    LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 ENVIRONMENT_MATCHING=overlap \
    REQUIRE_REFERENCE_CHECKPOINT=1 SAVE_FINAL_CHECKPOINT=1 \
    REFERENCE_CHECKPOINT="$reference" OUTPUT="$destination" \
    bash scripts/run_predictive_env_iv_patchtst.sh
    echo "[$backbone/$dataset/$fusion] complete $(date --iso-8601=seconds)"
)

# Large PatchTST cases start first; CycleNet/ETT jobs naturally backfill as
# workers finish. Each task is still a single-GPU independent experiment.
TASKS=()
for backbone_dataset in \
    patchtst:Traffic patchtst:Electricity patchtst:Weather \
    patchtst:ETTm1 patchtst:ETTm2 patchtst:ETTh1 patchtst:ETTh2 \
    patchtst:ExchangeRate cyclenet:Traffic cyclenet:Electricity \
    cyclenet:Weather cyclenet:ETTm1 cyclenet:ETTm2 cyclenet:ETTh1 \
    cyclenet:ETTh2 cyclenet:ExchangeRate; do
    TASKS+=("$backbone_dataset:direct_gated" "$backbone_dataset:film")
done
SCHEDULER_DIR="$(mktemp -d "$OUTPUT_ROOT/.scheduler.XXXXXX")"
printf '0\n' > "$SCHEDULER_DIR/next"
touch "$SCHEDULER_DIR/lock"
trap 'rm -f "$SCHEDULER_DIR/next" "$SCHEDULER_DIR/lock"; rmdir "$SCHEDULER_DIR" 2>/dev/null || true' EXIT

worker() {
    local gpu="$1" index task backbone dataset fusion
    while true; do
        exec 9>"$SCHEDULER_DIR/lock"
        flock 9
        index="$(<"$SCHEDULER_DIR/next")"
        if (( index >= ${#TASKS[@]} )); then
            flock -u 9; exec 9>&-; break
        fi
        task="${TASKS[$index]}"
        printf '%s\n' "$((index + 1))" > "$SCHEDULER_DIR/next"
        flock -u 9; exec 9>&-
        IFS=: read -r backbone dataset fusion <<<"$task"
        run_one "$backbone" "$dataset" "$fusion" "$gpu"
    done
}

pids=()
for gpu in $GPU_LIST; do worker "$gpu" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "one or more FiLM jobs failed" >&2; exit "$status"; }
"$PYTHON" tools/summarize_predictive_env_film.py --root "$OUTPUT_ROOT"
date --iso-8601=seconds > "$OUTPUT_ROOT/completion.txt"

