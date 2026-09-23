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
if [[ "${ENVIRONMENT_QUALITY_DIAGNOSTICS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--environment_quality_diagnostics)
fi
if [[ "${ENVIRONMENT_QUALITY_FINAL_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--environment_quality_final_only)
fi
if [[ "${ALLOW_LEGACY_REFERENCE_CHECKPOINT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow_legacy_reference_checkpoint)
fi
if [[ "${SAVE_FINAL_CHECKPOINT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_final_checkpoint)
fi
if [[ "${DIFFERENTIAL_LR:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--differential_lr)
fi
if [[ "${ADAPTIVE_VARIANT_LR:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--adaptive_variant_lr)
fi
if [[ "${FIXED_GAMMA_BETA_LR:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--fixed_gamma_beta_lr)
fi
if [[ "${RELIABILITY_ENVIRONMENT_DISAGREEMENT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--reliability_environment_disagreement)
fi

"$PYTHON" -u tools/run_predictive_env_iv_patchtst.py \
  --backbone "${BACKBONE:-patchtst}" \
  --root_path "${DATA_ROOT:-./dataset/ETT-small/}" \
  --dataset_name "${DATASET_NAME:-ETTh1}" \
  --dataset_archive_sha256 "${DATASET_ARCHIVE_SHA256:-}" \
  --data_class "${DATA_CLASS:-ett_hour}" \
  --data_path "${DATA_PATH:-ETTh1.csv}" \
  --target "${TARGET:-OT}" \
  --freq "${FREQ:-h}" \
  --cyclenet_cycle_len "${CYCLENET_CYCLE_LEN:-24}" \
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
  --lr "${LR:-0.0001}" \
  --lr_backbone "${LR_BACKBONE:-0.00002}" \
  --lr_inv_head "${LR_INV_HEAD:-0.00005}" \
  --lr_decomposer "${LR_DECOMPOSER:-0.0001}" \
  --lr_env_head "${LR_ENV_HEAD:-0.0001}" \
  --lr_variant "${LR_VARIANT:-0.0001}" \
  --lr_gamma_beta_base "${LR_GAMMA_BETA_BASE:-0.0001}" \
  --lr_reliability "${LR_RELIABILITY:-0.0003}" \
  --reliability_objective "${RELIABILITY_OBJECTIVE:-mse}" \
  --representation_constraint "${REPRESENTATION_CONSTRAINT:-contrastive}" \
  --decomposition_type "${DECOMPOSITION_TYPE:-projection}" \
  --gate_near_zero_threshold "${GATE_NEAR_ZERO_THRESHOLD:-0.1}" \
  --lambda_gate_activity "${LAMBDA_GATE_ACTIVITY:-0.0001}" \
  --lambda_complementary_gate_activity "${LAMBDA_COMPLEMENTARY_GATE_ACTIVITY:-0.0}" \
  --lambda_invpred "${LAMBDA_INVPRED:-0.5}" \
  --lambda_future_var "${LAMBDA_FUTURE_VAR:-0.0}" \
  --lambda_future_h "${LAMBDA_FUTURE_H:-0.0}" \
  --lambda_variant_anchor "${LAMBDA_VARIANT_ANCHOR:-1.0}" \
  --lambda_horizon_reliability "${LAMBDA_HORIZON_RELIABILITY:-0.0}" \
  --horizon_reliability_temperature "${HORIZON_RELIABILITY_TEMPERATURE:-0.05}" \
  --horizon_reliability_gate "${HORIZON_RELIABILITY_GATE:-on}" \
  --future_var_anchor_count "${FUTURE_VAR_ANCHOR_COUNT:-16}" \
  --future_patch_len "${FUTURE_PATCH_LEN:-16}" \
  --future_teacher_patches_per_batch "${FUTURE_TEACHER_PATCHES_PER_BATCH:-2}" \
  --future_teacher_eval_patch_count "${FUTURE_TEACHER_EVAL_PATCH_COUNT:-3}" \
  --lambda_var_predictive "${LAMBDA_VAR_PREDICTIVE:-0.0}" \
  --lambda_var_utility "${LAMBDA_VAR_UTILITY:-0.0}" \
  --lambda_var_gain "${LAMBDA_VAR_GAIN:-0.0}" \
  --var_gain_temperature "${VAR_GAIN_TEMPERATURE:-0.01}" \
  --lambda_var_conditional_gain "${LAMBDA_VAR_CONDITIONAL_GAIN:-0.0}" \
  --var_conditional_margin "${VAR_CONDITIONAL_MARGIN:-0.0}" \
  --lambda_h_anchor "${LAMBDA_H_ANCHOR:-0.0}" \
  --lambda_domain "${LAMBDA_DOMAIN:-0.1}" \
  --variant_fusion_mode "${VARIANT_FUSION_MODE:-legacy}" \
  --variant_fusion_gate_type "${VARIANT_FUSION_GATE_TYPE:-token}" \
  --film_gamma_scale "${FILM_GAMMA_SCALE:-0.1}" \
  --film_beta_scale "${FILM_BETA_SCALE:-0.1}" \
  --lambda_decay_mono "${LAMBDA_DECAY_MONO:-0.0}" \
  --lambda_decay_far "${LAMBDA_DECAY_FAR:-0.0}" \
  --horizon_decay_bins "${HORIZON_DECAY_BINS:-4}" \
  --y_film_gamma_scale "${Y_FILM_GAMMA_SCALE:-0.1}" \
  --y_film_beta_scale "${Y_FILM_BETA_SCALE:-0.1}" \
  --y_film_decay_bias "${Y_FILM_DECAY_BIAS:--4.0}" \
  --horizon_future_dim "${HORIZON_FUTURE_DIM:-32}" \
  --horizon_future_gamma_scale "${HORIZON_FUTURE_GAMMA_SCALE:-0.1}" \
  --horizon_future_beta_scale "${HORIZON_FUTURE_BETA_SCALE:-0.1}" \
  --horizon_future_chunk_size "${HORIZON_FUTURE_CHUNK_SIZE:-32}" \
  --fusion_scale_calibration "${FUSION_SCALE_CALIBRATION:-none}" \
  --predictive_env_refactor_mode "${PREDICTIVE_ENV_REFACTOR_MODE:-current}" \
  --grl_weight "${GRL_WEIGHT:-1.0}" \
  --environment_matching "${ENVIRONMENT_MATCHING:-overlap}" \
  --environment_supervision "${ENVIRONMENT_SUPERVISION:-predictive}" \
  --matching_q_weight "${MATCHING_Q_WEIGHT:-1.0}" \
  --matching_gradient_weight "${MATCHING_GRADIENT_WEIGHT:-1.0}" \
  --seed "${SEED:-2021}" \
  --gpu "${GPU:-0}" \
  --max_train_batches "${MAX_TRAIN_BATCHES:-0}" \
  --max_eval_batches "${MAX_EVAL_BATCHES:-0}" \
  --diagnostic_samples "${DIAGNOSTIC_SAMPLES:-1024}" \
  --random_partition_repeats "${RANDOM_PARTITION_REPEATS:-1000}" \
  --reference_checkpoint "$REFERENCE_PATH" \
  --output "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
