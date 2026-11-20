#!/usr/bin/env bash
# Limited, resumable VAL-selected decomposition pilot.  No TEST-based search.
set -euo pipefail
cd "${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
ROOT="${LATENT_RESULT_ROOT:-results/fpem_latent_decomposition_search/pilot_20261015}"
mkdir -p "$ROOT"
ARCHIVE_SHA=3730ee9f111b7761c465d373cfe004756c365b28245c1c7942a590bd54a2bf7c
CASES=(
  'patchtst|ETTh1|A0|complementary_gate|predictive|0|0|none|0'
  'patchtst|ETTh1|A2|complementary_gate|predictive|0|0|none|0'
  'patchtst|ETTh1|A2|complementary_gate|predictive|0|0|none|1'
  'patchtst|ETTh1|A2|low_rank_subspace|predictive|0|0|none|0'
  'patchtst|ETTh1|A2|low_rank_subspace|predictive|0|0|none|0|2|-4'
  'patchtst|ETTh1|A2|low_rank_subspace|predictive|0|0|none|0|16|-4'
  'patchtst|ETTh1|A2|low_rank_subspace|predictive|0|0|none|0|8|-3'
  'patchtst|ETTh1|A2|low_rank_subspace|predictive|0|0|none|0|8|-5'
  'patchtst|ETTh1|A2|env_guided_subspace|predictive|0.05|0|none|0'
  'patchtst|ETTh1|A2|env_guided_subspace|predictive|0.01|0|none|0'
  'patchtst|ETTh1|A2|env_guided_subspace|predictive|0.1|0|none|0'
  'patchtst|ETTh1|A2|env_guided_subspace|shuffled|0.05|0|none|0'
  'patchtst|ETTh1|A2|env_guided_subspace|predictive|0.05|0.001|none|0'
  'patchtst|ETTh1|A2|reference_guided|predictive|0|0|global|0'
  'patchtst|ETTh1|A2|reference_guided|predictive|0|0|environment|0'
  'patchtst|ETTh1|A2|reference_guided|shuffled|0|0|environment|0'
  'moderntcn|ETTm2|A0|complementary_gate|predictive|0|0|none|0'
  'moderntcn|ETTm2|A2|low_rank_subspace|predictive|0|0|none|0'
  'moderntcn|ETTm2|A2|env_guided_subspace|predictive|0.05|0|none|0'
  'moderntcn|ETTm2|A2|env_guided_subspace|shuffled|0.05|0|none|0'
  'moderntcn|ETTm2|A2|env_guided_subspace|predictive|0.05|0.001|none|0'
  'moderntcn|ETTm2|A2|reference_guided|predictive|0|0|environment|0'
  'itransformer|ETTm1|A0|complementary_gate|predictive|0|0|none|0'
  'itransformer|ETTm1|A2|low_rank_subspace|predictive|0|0|none|0'
  'itransformer|ETTm1|A2|env_guided_subspace|predictive|0.05|0|none|0'
  'itransformer|ETTm1|A2|env_guided_subspace|shuffled|0.05|0|none|0'
  'itransformer|ETTm1|A2|env_guided_subspace|predictive|0.05|0.001|none|0'
  'itransformer|ETTm1|A2|reference_guided|predictive|0|0|environment|0'
)

run_case() {
  local lane="$1" index="$2" backbone dataset experiment method supervision sub orth reference_mode fpinit rank bias
  IFS='|' read -r backbone dataset experiment method supervision sub orth reference_mode fpinit rank bias <<< "${CASES[$index]}"
  rank="${rank:-8}"
  bias="${bias:--4}"
  local id="${backbone}_${dataset}_720_${experiment}_${method}_${supervision}_sub${sub}_orth${orth}_ref${reference_mode}_fp${fpinit}_rank${rank}_bias${bias}"
  local out="$ROOT/$id"
  if [[ -f "$out/run_complete" ]]; then
    echo "SKIP $id"; return 0
  fi
  mkdir -p "$out"
  local reference d_model=512 d_ff=2048 dropout=0.1 freq=h class=ett_hour cycle=24
  local modern_stages=1 modern_ffn=1 modern_head_dropout=0
  if [[ "$backbone" == patchtst ]]; then
    reference=results/fpem_no_future_anchor_patchtst_search/patchtst/ETTh1/pred_720/shared_reference.pt
  elif [[ "$backbone" == moderntcn ]]; then
    reference=results/fpem_probabilistic/test_best_backbones_a0_20261013/baseline_references/moderntcn/ETTm2/pred_720.pt
    d_model=64 d_ff=512 dropout=0.3 freq=15min class=ett_minute cycle=96
    modern_stages=3 modern_ffn=8 modern_head_dropout=0.1
  else
    reference=results/fpem_latent_decomposition_search/references/itransformer_ETTm1_720.pt
    freq=15min class=ett_minute cycle=96
  fi
  local epochs=10
  [[ "$experiment" == A0 ]] && epochs=0
  echo "START $id GPU=$lane $(date -Is)" | tee -a "$ROOT/queue.log"
  if env CUDA_VISIBLE_DEVICES="$lane" GPU=0 OUTPUT="$out" \
      REFERENCE_CHECKPOINT="$reference" REQUIRE_REFERENCE_CHECKPOINT=1 \
      DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA" \
      DATA_ROOT=./dataset/all_datasets/ETT-small \
      BACKBONE="$backbone" DATASET_NAME="$dataset" DATA_CLASS="$class" \
      DATA_PATH="$dataset.csv" FREQ="$freq" CYCLENET_CYCLE_LEN="$cycle" \
      PRED_LEN=720 SEQ_LEN=96 D_MODEL="$d_model" D_FF="$d_ff" DROPOUT="$dropout" \
      MODERN_TCN_NUM_STAGES="$modern_stages" MODERN_TCN_FFN_RATIO="$modern_ffn" \
      MODERN_TCN_HEAD_DROPOUT="$modern_head_dropout" \
      EXPERIMENTS="$experiment" EPOCHS="$epochs" STAGE_EPOCHS=2 \
      BATCH_SIZE=32 NUM_WORKERS=0 EIIL_STEPS=50 ENV_NUM=3 SEED=2021 \
      VARIANT_FUSION_MODE=inv_only REPRESENTATION_CONSTRAINT=classification \
      DECOMPOSITION_TYPE="$method" ENVIRONMENT_SUPERVISION="$supervision" \
      SUBSPACE_RANK="$rank" INIT_GATE_BIAS="$bias" LAMBDA_SUB="$sub" LAMBDA_ORTH="$orth" \
      REFERENCE_MODE="$reference_mode" LAMBDA_REFERENCE_Q=0.1 \
      FUNCTION_PRESERVING_INIT="$fpinit" SELECT_INVARIANT_ON_VAL=1 \
      DIFFERENTIAL_LR=1 LR_BACKBONE=0.00002 LR_INV_HEAD=0.00005 \
      LR_DECOMPOSER=0.0001 LR_SUBSPACE=0.0001 LR_AUXILIARY=0.0001 \
      LR_ENV_HEAD=0.0001 LAMBDA_INVPRED=0 LAMBDA_RISK=0.1 LAMBDA_DOMAIN=0.1 \
      bash scripts/run_predictive_env_iv_patchtst.sh > "$out/run.log" 2>&1; then
    touch "$out/run_complete"
    echo "DONE $id $(date -Is)" | tee -a "$ROOT/queue.log"
  else
    echo "FAILED $id $(date -Is)" | tee -a "$ROOT/queue.log"
    tail -20 "$out/run.log" >> "$ROOT/failures.log"
  fi
}

for lane in 0 1 2 3; do
  (
    for ((index=lane; index<${#CASES[@]}; index+=4)); do
      run_case "$lane" "$index"
    done
  ) &
done
wait
"${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}" \
  tools/summarize_fpem_latent_pilot.py --root "$ROOT"
