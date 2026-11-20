#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
cd "$PROJECT_DIR"
ARCHIVE_SHA256="${ARCHIVE_SHA256:-$(sha256sum "$ARCHIVE" | awk '{print $1}')}"
run_smoke(){
  local backbone="$1" gpu="$2" output="results/_smoke_fpem_final_$1"
  local reference="$output/shared_reference.pt"
  [[ -e "$output" ]] && mv "$output" "${output}.previous.$(date +%Y%m%d_%H%M%S)"
  GPU="$gpu" BACKBONE="$backbone" SEED=2021 ENV_NUM=2 EXPERIMENTS=A2 \
  DATASET_NAME=ETTh1 DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA256" \
  DATA_ROOT=./dataset/all_datasets/ETT-small DATA_CLASS=ett_hour DATA_PATH=ETTh1.csv \
  TARGET=OT FREQ=h CYCLENET_CYCLE_LEN=24 SEQ_LEN=96 PRED_LEN=96 BATCH_SIZE=8 \
  NUM_WORKERS=0 REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
  VARIANT_FUSION_MODE=horizon_future_var VARIANT_FUSION_GATE_TYPE=feature EPOCHS=1 \
  WARMUP_EPOCHS=1 STAGE_EPOCHS=1 EIIL_STEPS=2 MAX_TRAIN_BATCHES=1 MAX_EVAL_BATCHES=1 \
  DIAGNOSTIC_SAMPLES=32 RANDOM_PARTITION_REPEATS=10 REFERENCE_CHECKPOINT="$reference" \
  REQUIRE_REFERENCE_CHECKPOINT=0 DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 LR=1e-4 \
  LR_BACKBONE=1e-4 LR_INV_HEAD=1e-4 LR_DECOMPOSER=1e-4 LR_ENV_HEAD=1e-4 \
  LR_VARIANT=1e-4 LR_GAMMA_BETA_BASE=5e-5 LR_RELIABILITY=3e-4 LAMBDA_FUTURE_H=0.1 \
  FUTURE_PATCH_LEN=16 FUTURE_TEACHER_PATCHES_PER_BATCH=1 FUTURE_TEACHER_EVAL_PATCH_COUNT=1 \
  SAVE_FINAL_CHECKPOINT=0 OUTPUT="$output" bash scripts/run_predictive_env_iv_patchtst.sh \
  > "$output.log" 2>&1
}
run_smoke patchtst 0 & p0=$!
run_smoke itransformer 1 & p1=$!
run_smoke cyclenet 2 & p2=$!
wait "$p0" "$p1" "$p2"
for backbone in patchtst itransformer cyclenet; do
  test -s "results/_smoke_fpem_final_$backbone/A2/metrics_and_diagnostics.json"
  echo "SMOKE_OK=$backbone"
done
