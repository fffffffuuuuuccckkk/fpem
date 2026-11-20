#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-results/fpem_no_future_anchor_patchtst_search}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_etth1_environment_shuffle_ablation}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
TSNE_MAX_SAMPLES="${TSNE_MAX_SAMPLES:-1500}"
cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status"

json_value() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

run_case() {
  local pred_len="$1" supervision="$2" gpu="$3"
  local source_case="$SOURCE_ROOT/patchtst/ETTh1/pred_${pred_len}"
  local best="$source_case/best_config.json"
  local output="$OUTPUT_ROOT/pred_${pred_len}/${supervision}"
  local experiment="$output/A2"
  local metrics="$experiment/metrics_and_diagnostics.json"
  local checkpoint="$experiment/trained_checkpoint.pt"
  local visualization="$experiment/tsne_training_supervision"
  [[ -s "$best" ]] || { echo "missing best config: $best" >&2; return 2; }
  local dec_lr env_lr future_lr reference archive_hash
  dec_lr="$(json_value "$best" decomposer_lr)"
  env_lr="$(json_value "$best" environment_classifier_lr)"
  future_lr="$(json_value "$best" future_zvar_lr)"
  reference="$(json_value "$best" reference_checkpoint)"
  archive_hash="$(json_value "$best" dataset_archive_sha256)"
  if [[ ! -s "$metrics" || ! -s "$checkpoint" ]]; then
    echo "train pred_len=$pred_len supervision=$supervision gpu=$gpu dec=$dec_lr env=$env_lr future=$future_lr"
    env \
      GPU="$gpu" BACKBONE=patchtst DATASET_NAME=ETTh1 PRED_LEN="$pred_len" \
      DATA_ROOT=./dataset/all_datasets/ETT-small DATA_CLASS=ett_hour \
      DATA_PATH=ETTh1.csv TARGET=OT FREQ=h CYCLENET_CYCLE_LEN=24 \
      DATASET_ARCHIVE_SHA256="$archive_hash" EXPERIMENTS=A2 \
      EPOCHS=10 WARMUP_EPOCHS=3 BATCH_SIZE=32 NUM_WORKERS=0 \
      ENV_NUM=3 EIIL_STEPS=50 STAGE_EPOCHS=2 LR=1e-4 \
      LR_BACKBONE=1e-4 LR_INV_HEAD=1e-4 LR_DECOMPOSER="$dec_lr" \
      LR_ENV_HEAD="$env_lr" LR_VARIANT="$future_lr" \
      LR_GAMMA_BETA_BASE=1e-4 LR_RELIABILITY=3e-4 \
      RELIABILITY_OBJECTIVE=mse REPRESENTATION_CONSTRAINT=classification \
      DECOMPOSITION_TYPE=complementary_gate GATE_NEAR_ZERO_THRESHOLD=0.1 \
      LAMBDA_GATE_ACTIVITY=0.0001 LAMBDA_COMPLEMENTARY_GATE_ACTIVITY=0.0 \
      LAMBDA_INVPRED=1.0 LAMBDA_FUTURE_VAR=0.0 LAMBDA_FUTURE_H=0.0 \
      LAMBDA_VARIANT_ANCHOR=0.0 LAMBDA_HORIZON_RELIABILITY=0.0 \
      LAMBDA_VAR_PREDICTIVE=0.0 LAMBDA_VAR_UTILITY=0.0 LAMBDA_VAR_GAIN=0.0 \
      LAMBDA_VAR_CONDITIONAL_GAIN=0.0 LAMBDA_H_ANCHOR=0.0 LAMBDA_DOMAIN=0.1 \
      VARIANT_FUSION_MODE=horizon_future_var VARIANT_FUSION_GATE_TYPE=feature \
      HORIZON_RELIABILITY_GATE=on PREDICTIVE_ENV_REFACTOR_MODE=current \
      ENVIRONMENT_MATCHING=overlap ENVIRONMENT_SUPERVISION="$supervision" \
      DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 SAVE_FINAL_CHECKPOINT=1 \
      ENVIRONMENT_QUALITY_DIAGNOSTICS=1 ENVIRONMENT_QUALITY_FINAL_ONLY=1 \
      RANDOM_PARTITION_REPEATS=1000 REFERENCE_CHECKPOINT="$reference" \
      REQUIRE_REFERENCE_CHECKPOINT=1 OUTPUT="$output" \
      bash scripts/run_predictive_env_iv_patchtst.sh
  else
    echo "reuse completed checkpoint: $checkpoint"
  fi
  if [[ ! -s "$visualization/tsne_H_Zinv_Zvar_Zfinal.png" ]]; then
    "$PYTHON" tools/visualize_environment_representations.py \
      --experiment_dir "$experiment" \
      --checkpoint "$checkpoint" \
      --output_dir "$visualization" \
      --dataset "ETTh1-${pred_len}-${supervision}" \
      --gpu "$gpu" \
      --max_samples "$TSNE_MAX_SAMPLES" \
      --batch_size 32 \
      --seed 2021 \
      --label_mode training_supervision
  fi
}

mapfile -t tasks < <(printf '%s\n' \
  '96 predictive' '96 shuffled' \
  '336 predictive' '336 shuffled' \
  '720 predictive' '720 shuffled')
read -r -a gpus <<<"$GPU_LIST"
(( ${#gpus[@]} > 0 )) || { echo "GPU_LIST is empty" >&2; exit 2; }
scheduler="$(mktemp -d "$OUTPUT_ROOT/.scheduler.XXXXXX")"
printf '0\n' > "$scheduler/next"
touch "$scheduler/lock"
trap 'rm -rf "$scheduler"' EXIT

worker() {
  local gpu="$1" index task pred_len supervision
  while true; do
    exec 9>"$scheduler/lock"; flock 9
    index="$(<"$scheduler/next")"
    if (( index >= ${#tasks[@]} )); then
      flock -u 9; exec 9>&-; break
    fi
    task="${tasks[$index]}"
    printf '%s\n' "$((index + 1))" > "$scheduler/next"
    flock -u 9; exec 9>&-
    read -r pred_len supervision <<<"$task"
    if run_case "$pred_len" "$supervision" "$gpu" \
      > >(tee -a "$OUTPUT_ROOT/logs/pred_${pred_len}_${supervision}.log") 2>&1; then
      echo "$pred_len,$supervision" >> "$OUTPUT_ROOT/status/completed.txt"
    else
      echo "$pred_len,$supervision" >> "$OUTPUT_ROOT/status/failed.txt"
    fi
  done
}

pids=()
for gpu in "${gpus[@]}"; do worker "$gpu" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid" || true; done

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for pred_len in (96, 336, 720):
    values = {}
    for supervision in ("predictive", "shuffled"):
        path = root / f"pred_{pred_len}" / supervision / "A2" / "metrics_and_diagnostics.json"
        if not path.is_file():
            raise SystemExit(f"missing metrics: {path}")
        values[supervision] = json.loads(path.read_text())
    predictive = values["predictive"]
    shuffled = values["shuffled"]
    rows.append({
        "pred_len": pred_len,
        "predictive_MSE": predictive["MSE"],
        "shuffled_MSE": shuffled["MSE"],
        "MSE_shuffled_minus_predictive": shuffled["MSE"] - predictive["MSE"],
        "predictive_MAE": predictive["MAE"],
        "shuffled_MAE": shuffled["MAE"],
        "predictive_inv_MSE": predictive["inv_MSE"],
        "shuffled_inv_MSE": shuffled["inv_MSE"],
        "predictive_var_acc": predictive.get("var_acc"),
        "shuffled_var_acc": shuffled.get("var_acc"),
        "predictive_inv_acc": predictive.get("inv_acc"),
        "shuffled_inv_acc": shuffled.get("inv_acc"),
    })
with (root / "environment_shuffle_ablation.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
(root / "environment_shuffle_ablation.txt").write_text(
    "\n".join(
        f"ETTh1-{row['pred_len']}: predictive MSE={row['predictive_MSE']:.6f}, "
        f"shuffled MSE={row['shuffled_MSE']:.6f}, "
        f"shuffled-predictive={row['MSE_shuffled_minus_predictive']:+.6f}"
        for row in rows
    ) + "\n"
)
PY
