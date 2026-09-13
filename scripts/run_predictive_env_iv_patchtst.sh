#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_DIR="${OUTPUT:-results/predictive_env_iv_ETTh1_96_96}"
REFERENCE_PATH="${REFERENCE_CHECKPOINT:-$OUTPUT_DIR/shared_reference.pt}"
EXTRA_ARGS=()
if [[ "${PREPARE_REFERENCE_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--prepare_reference_only)
fi
if [[ "${REQUIRE_REFERENCE_CHECKPOINT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--require_reference_checkpoint)
fi

"$PYTHON" -u tools/run_predictive_env_iv_patchtst.py \
  --root_path "${DATA_ROOT:-./dataset/ETT-small/}" \
  --dataset_name "${DATASET_NAME:-ETTh1}" \
  --data_class "${DATA_CLASS:-ett_hour}" \
  --data_path "${DATA_PATH:-ETTh1.csv}" \
  --target "${TARGET:-OT}" \
  --freq "${FREQ:-h}" \
  --seq_len "${SEQ_LEN:-96}" \
  --pred_len "${PRED_LEN:-96}" \
  --experiments "${EXPERIMENTS:-A0,A1,A2,A3,A4}" \
  --epochs "${EPOCHS:-10}" \
  --warmup_epochs "${WARMUP_EPOCHS:-3}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --env_num "${ENV_NUM:-6}" \
  --eiil_steps "${EIIL_STEPS:-50}" \
  --stage_epochs "${STAGE_EPOCHS:-2}" \
  --representation_constraint "${REPRESENTATION_CONSTRAINT:-contrastive}" \
  --decomposition_type "${DECOMPOSITION_TYPE:-projection}" \
  --gate_near_zero_threshold "${GATE_NEAR_ZERO_THRESHOLD:-0.1}" \
  --lambda_gate_activity "${LAMBDA_GATE_ACTIVITY:-0.0001}" \
  --lambda_complementary_gate_activity "${LAMBDA_COMPLEMENTARY_GATE_ACTIVITY:-0.0}" \
  --lambda_invpred "${LAMBDA_INVPRED:-0.5}" \
  --lambda_future_var "${LAMBDA_FUTURE_VAR:-0.0}" \
  --lambda_var_predictive "${LAMBDA_VAR_PREDICTIVE:-0.0}" \
  --lambda_var_utility "${LAMBDA_VAR_UTILITY:-0.0}" \
  --lambda_var_conditional_gain "${LAMBDA_VAR_CONDITIONAL_GAIN:-0.0}" \
  --var_conditional_margin "${VAR_CONDITIONAL_MARGIN:-0.0}" \
  --lambda_h_anchor "${LAMBDA_H_ANCHOR:-0.0}" \
  --variant_fusion_mode "${VARIANT_FUSION_MODE:-legacy}" \
  --fusion_scale_calibration "${FUSION_SCALE_CALIBRATION:-none}" \
  --predictive_env_refactor_mode "${PREDICTIVE_ENV_REFACTOR_MODE:-current}" \
  --grl_weight "${GRL_WEIGHT:-1.0}" \
  --environment_matching "${ENVIRONMENT_MATCHING:-overlap}" \
  --matching_q_weight "${MATCHING_Q_WEIGHT:-1.0}" \
  --matching_gradient_weight "${MATCHING_GRADIENT_WEIGHT:-1.0}" \
  --seed "${SEED:-2021}" \
  --gpu "${GPU:-0}" \
  --max_train_batches "${MAX_TRAIN_BATCHES:-0}" \
  --max_eval_batches "${MAX_EVAL_BATCHES:-0}" \
  --diagnostic_samples "${DIAGNOSTIC_SAMPLES:-1024}" \
  --reference_checkpoint "$REFERENCE_PATH" \
  --output "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
