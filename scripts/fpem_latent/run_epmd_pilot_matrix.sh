#!/usr/bin/env bash
# Bounded, resumable EPMD point-forecast pilot. All selection is on VAL.
set -euo pipefail
cd "${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
ROOT="${EPMD_RESULT_ROOT:-/var/tmp/fpem_epmd_pilot_20261016}"
mkdir -p "$ROOT"
ARCHIVE_SHA=3730ee9f111b7761c465d373cfe004756c365b28245c1c7942a590bd54a2bf7c
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Each lane is sequential to avoid simultaneous large Gram solves on one GPU.
run_case() {
  local lane="$1" backbone="$2" dataset="$3" horizon="$4" supervision="$5" alpha="$6"
  local id="${backbone}_${dataset}_${horizon}_A2_epmd_${supervision}_${alpha}"
  local out="$ROOT/$id"
  if [[ -f "$out/run_complete" ]]; then
    echo "SKIP $id"; return 0
  fi
  mkdir -p "$out"
  local reference d_model=512 d_ff=2048 dropout=0.1 freq=h class=ett_hour cycle=24
  local modern_stages=1 modern_ffn=1 modern_head_dropout=0
  if [[ "$backbone" == patchtst ]]; then
    reference="results/fpem_no_future_anchor_patchtst_search/patchtst/ETTh1/pred_${horizon}/shared_reference.pt"
  elif [[ "$backbone" == moderntcn ]]; then
    reference=results/fpem_probabilistic/test_best_backbones_a0_20261013/baseline_references/moderntcn/ETTm2/pred_720.pt
    d_model=64 d_ff=512 dropout=0.3 freq=15min class=ett_minute cycle=96
    modern_stages=3 modern_ffn=8 modern_head_dropout=0.1
  else
    reference=results/fpem_latent_decomposition_search/references/itransformer_ETTm1_720.pt
    freq=15min class=ett_minute cycle=96
  fi
  echo "START $id GPU=$lane $(date -Is)" | tee -a "$ROOT/queue.log"
  if env CUDA_VISIBLE_DEVICES="$lane" GPU=0 OUTPUT="$out" \
      REFERENCE_CHECKPOINT="$reference" REQUIRE_REFERENCE_CHECKPOINT=1 \
      DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA" \
      DATA_ROOT=./dataset/all_datasets/ETT-small \
      BACKBONE="$backbone" DATASET_NAME="$dataset" DATA_CLASS="$class" \
      DATA_PATH="$dataset.csv" FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" \
      PRED_LEN="$horizon" SEQ_LEN=96 D_MODEL="$d_model" D_FF="$d_ff" DROPOUT="$dropout" \
      MODERN_TCN_NUM_STAGES="$modern_stages" MODERN_TCN_FFN_RATIO="$modern_ffn" \
      MODERN_TCN_HEAD_DROPOUT="$modern_head_dropout" \
      EXPERIMENTS=A2 EPOCHS=10 STAGE_EPOCHS=2 BATCH_SIZE=32 NUM_WORKERS=0 \
      EIIL_STEPS=50 ENV_NUM=3 SEED=2021 \
      VARIANT_FUSION_MODE=inv_only REPRESENTATION_CONSTRAINT=classification \
      DECOMPOSITION_TYPE=environment_mapping ENVIRONMENT_SUPERVISION="$supervision" \
      EPMD_ALPHA_MODE="$alpha" EPMD_INIT_ALPHA_LOGIT=-4 \
      SELECT_INVARIANT_ON_VAL=1 DIFFERENTIAL_LR=1 \
      LR_BACKBONE=0.00002 LR_INV_HEAD=0.00005 LR_DECOMPOSER=0.0001 \
      LR_ENV_HEAD=0.0001 LAMBDA_INVPRED=0 LAMBDA_RISK=0.1 LAMBDA_DOMAIN=0.1 \
      bash scripts/run_predictive_env_iv_patchtst.sh > "$out/run.log" 2>&1; then
    touch "$out/run_complete"
    echo "DONE $id $(date -Is)" | tee -a "$ROOT/queue.log"
  else
    echo "FAILED $id $(date -Is)" | tee -a "$ROOT/queue.log"
    tail -30 "$out/run.log" >> "$ROOT/failures.log"
  fi
}

(
  run_case 0 patchtst ETTh1 720 predictive learned
  run_case 0 patchtst ETTh1 720 shuffled learned
  run_case 0 patchtst ETTh1 720 uniform learned
  run_case 0 patchtst ETTh1 720 predictive fixed_one
) &
(
  run_case 1 moderntcn ETTm2 720 predictive learned
  run_case 1 moderntcn ETTm2 720 shuffled learned
  run_case 1 moderntcn ETTm2 720 uniform learned
) &
(
  run_case 2 itransformer ETTm1 720 predictive learned
  run_case 2 itransformer ETTm1 720 shuffled learned
  run_case 2 itransformer ETTm1 720 uniform learned
) &
(
  run_case 3 patchtst ETTh1 96 predictive learned
) &
wait
echo "QUEUE_FINISHED $(date -Is)" | tee -a "$ROOT/queue.log"
