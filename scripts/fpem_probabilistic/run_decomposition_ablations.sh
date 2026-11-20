#!/usr/bin/env bash
# Resumable no-adversarial-decomposition ablation. Keep the complementary split,
# but remove both Zinv-GRL and Zvar environment classification supervision.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
BASE_ROOT="${BASE_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/unified_lr_search_20261010}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/no_domain_ablations_20261012}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"

cfg() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "$SOURCE_CONFIG" "$1"
}

config_value() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "$1" "$2"
}

load_case() {
  local backbone="$1" dataset="$2" horizon="$3"
  CASE_ROOT="$BASE_ROOT/$backbone/$dataset/pred_$horizon"
  SOURCE_CONFIG="$CASE_ROOT/anchor/run_config.json"
  SELECTED="$CASE_ROOT/best_heads_test_selected.csv"
  BASE_ANCHOR="$CASE_ROOT/anchor/A2/trained_checkpoint.pt"
  for file in "$SOURCE_CONFIG" "$SELECTED" "$BASE_ANCHOR"; do
    [[ -s "$file" ]] || { echo "Missing source: $file" >&2; return 1; }
  done
  BACKBONE="$(cfg backbone)" DATASET_NAME="$(cfg dataset_name)"
  DATA_ROOT="$(cfg root_path)"
  [[ "$DATA_ROOT" == ./* ]] && DATA_ROOT="$DATA_PROJECT/${DATA_ROOT#./}"
  DATA_CLASS="$(cfg data_class)" DATA_PATH="$(cfg data_path)" FREQ="$(cfg freq)"
  CYCLE_LEN="$(cfg cyclenet_cycle_len)" BATCH_SIZE="$(cfg batch_size)"
  D_MODEL="$(cfg d_model)" D_FF="$(cfg d_ff)" N_HEADS="$(cfg n_heads)"
  E_LAYERS="$(cfg e_layers)" DROPOUT="$(cfg dropout)"
  MODERN_TCN_PATCH_SIZE="$(cfg modern_tcn_patch_size)"
  MODERN_TCN_PATCH_STRIDE="$(cfg modern_tcn_patch_stride)"
  MODERN_TCN_NUM_STAGES="$(cfg modern_tcn_num_stages)"
  MODERN_TCN_FFN_RATIO="$(cfg modern_tcn_ffn_ratio)"
  MODERN_TCN_HEAD_DROPOUT="$(cfg modern_tcn_head_dropout)"
  DEC_LR="$(cfg lr_decomposer)" ENV_LR="$(cfg lr_env_head)"
  FUTURE_LR="$(cfg lr_variant)"
  REFERENCE_CHECKPOINT="$(cfg reference_checkpoint)"
  ARCHIVE_SHA="$(cfg dataset_archive_sha256)"
  [[ -s "$REFERENCE_CHECKPOINT" ]] || {
    echo "Missing exact shared reference: $REFERENCE_CHECKPOINT" >&2; return 1;
  }
  [[ "$(cfg environment_supervision)" == predictive ]] || {
    echo "Original anchor is not predictive-supervised: $SOURCE_CONFIG" >&2; return 1;
  }
  [[ "$(cfg prob_loss_weight_mode)" == historical_ema_maturity ]] || {
    echo "Original anchor does not use historical EMA: $SOURCE_CONFIG" >&2; return 1;
  }
}

run_no_domain_anchor() {
  local gpu="$1" target="$2"
  if [[ -s "$target/A2/trained_checkpoint.pt" && -s "$target/A2/metrics_and_diagnostics.json" ]]; then
    [[ "$(config_value "$target/run_config.json" environment_supervision)" == predictive ]]
    [[ "$(config_value "$target/run_config.json" lambda_domain)" == 0.0 ]]
    echo "REUSE no-domain anchor $target"
    return
  fi
  if [[ -e "$target/run_config.json" ]]; then
    echo "Incomplete no-domain anchor; refusing to overwrite: $target" >&2
    return 1
  fi
  mkdir -p "$target"
  env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
    BACKBONE="$BACKBONE" DATASET_NAME="$DATASET_NAME" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA" \
    DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" FREQ="$FREQ" \
    CYCLENET_CYCLE_LEN="$CYCLE_LEN" SEQ_LEN=96 PRED_LEN="$(cfg pred_len)" \
    BATCH_SIZE="$BATCH_SIZE" D_MODEL="$D_MODEL" D_FF="$D_FF" N_HEADS="$N_HEADS" \
    E_LAYERS="$E_LAYERS" DROPOUT="$DROPOUT" \
    MODERN_TCN_PATCH_SIZE="$MODERN_TCN_PATCH_SIZE" MODERN_TCN_PATCH_STRIDE="$MODERN_TCN_PATCH_STRIDE" \
    MODERN_TCN_NUM_STAGES="$MODERN_TCN_NUM_STAGES" MODERN_TCN_FFN_RATIO="$MODERN_TCN_FFN_RATIO" \
    MODERN_TCN_HEAD_DROPOUT="$MODERN_TCN_HEAD_DROPOUT" \
    GPU="$gpu" EXPERIMENTS=A2 EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 ENV_NUM=3 \
    NUM_WORKERS=0 REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
    ENVIRONMENT_SUPERVISION=predictive VARIANT_FUSION_MODE=prob_affine_flow \
    PROB_ABLATION=invariant_only PROB_LOSS_WEIGHT_MODE=historical_ema_maturity \
    PROB_NUM_SAMPLES=2 MAX_PROB_EVAL_BATCHES=1 FUTURE_PATCH_LEN=16 \
    LAMBDA_FUTURE_H=0 LAMBDA_VARIANT_ANCHOR=0 DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 \
    LR=1e-4 LR_BACKBONE=2e-5 LR_INV_HEAD=5e-5 \
    LR_DECOMPOSER="$DEC_LR" LR_ENV_HEAD="$ENV_LR" LR_VARIANT="$FUTURE_LR" \
    LR_GAMMA_BETA_BASE=1e-4 LR_RELIABILITY=3e-4 LAMBDA_DOMAIN=0.0 \
    REFERENCE_CHECKPOINT="$REFERENCE_CHECKPOINT" REQUIRE_REFERENCE_CHECKPOINT=1 \
    SAVE_FINAL_CHECKPOINT=1 OUTPUT="$target" \
    bash "$PROJECT_DIR/scripts/run_predictive_env_iv_patchtst.sh" >"$target/run.log" 2>&1 || {
      echo "No-domain anchor failed: $target/run.log" >&2; return 1;
    }
  [[ -s "$target/A2/trained_checkpoint.pt" && -s "$target/A2/metrics_and_diagnostics.json" ]]
}

run_head_bank() {
  local arm="$1" gpu="$2" anchor="$3" head_lr="$4" target="$5"
  if [[ -s "$target/summary.csv" && -s "$target/shared_encoder.pt" ]]; then
    echo "REUSE complete $arm LR=$head_lr $target"
    return
  fi
  if [[ -e "$target/run_config.json" ]]; then
    echo "Incomplete head bank; refusing to overwrite: $target" >&2
    return 1
  fi
  mkdir -p "$target"
  local source= future_mode=historical_ema_maturity
  source=future_zvar
  env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" PREDICTIVE_ENV_RUNNER=tools/run_fpem_head_comparison.py \
    BACKBONE="$BACKBONE" DATASET_NAME="$DATASET_NAME" DATASET_ARCHIVE_SHA256="$ARCHIVE_SHA" \
    DATA_ROOT="$DATA_ROOT" DATA_CLASS="$DATA_CLASS" DATA_PATH="$DATA_PATH" FREQ="$FREQ" \
    CYCLENET_CYCLE_LEN="$CYCLE_LEN" SEQ_LEN=96 PRED_LEN="$(cfg pred_len)" \
    BATCH_SIZE="$BATCH_SIZE" D_MODEL="$D_MODEL" D_FF="$D_FF" N_HEADS="$N_HEADS" \
    E_LAYERS="$E_LAYERS" DROPOUT="$DROPOUT" \
    MODERN_TCN_PATCH_SIZE="$MODERN_TCN_PATCH_SIZE" MODERN_TCN_PATCH_STRIDE="$MODERN_TCN_PATCH_STRIDE" \
    MODERN_TCN_NUM_STAGES="$MODERN_TCN_NUM_STAGES" MODERN_TCN_FFN_RATIO="$MODERN_TCN_FFN_RATIO" \
    MODERN_TCN_HEAD_DROPOUT="$MODERN_TCN_HEAD_DROPOUT" \
    GPU="$gpu" EXPERIMENTS=A2 EPOCHS=10 WARMUP_EPOCHS=3 STAGE_EPOCHS=2 ENV_NUM=3 \
    NUM_WORKERS=0 REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
    VARIANT_FUSION_MODE=prob_affine_flow PROB_HEAD=all_shared PROB_CONDITION_MODE=full_shape \
    PROB_CONDITION_SOURCE="$source" PROB_HEAD_LR="$head_lr" PROB_LOSS_WEIGHT_MODE="$future_mode" \
    PROB_LOSS_MIN_WEIGHT=0.05 PROB_NUM_SAMPLES=100 PROB_SAMPLE_CHUNK=10 \
    PROB_FLOW_STEPS=12 MAX_PROB_EVAL_BATCHES=0 FUTURE_PATCH_LEN=16 \
    LAMBDA_FUTURE_H=0 LAMBDA_VARIANT_ANCHOR=0 DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 \
    LR=1e-4 LR_BACKBONE=2e-5 LR_INV_HEAD=5e-5 \
    LR_DECOMPOSER="$DEC_LR" LR_ENV_HEAD="$ENV_LR" LR_VARIANT="$FUTURE_LR" \
    LAMBDA_DOMAIN=0.0 FROZEN_ANCHOR_CHECKPOINT="$anchor" \
    OUTPUT="$target" \
    bash "$PROJECT_DIR/scripts/run_predictive_env_iv_patchtst.sh" >"$target/run.log" 2>&1 || {
      echo "Probability head bank failed: $target/run.log" >&2; return 1;
    }
  [[ -s "$target/summary.csv" && -s "$target/shared_encoder.pt" ]]
}

summarize_case() {
  local arm="$1" case_out="$2"
  "$PYTHON" - "$SELECTED" "$case_out" "$arm" <<'PY'
import csv, json, sys
from pathlib import Path
selected, case_out, arm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows=[]
for original in csv.DictReader(selected.open()):
    lr=original['best_head_lr']
    path=case_out/arm/f'head_lr_{lr.replace(".","p")}'/original['head']/'test_metrics.json'
    result=json.loads(path.read_text())
    rows.append(dict(arm=arm, backbone=original['backbone'], dataset=original['dataset'],
                     pred_len=original['pred_len'], head=original['head'],
                     fixed_prior_head_lr=lr, original_ProbTS_CRPS=original['ProbTS_CRPS'],
                     ablation_ProbTS_CRPS=result['ProbTS_CRPS'],
                     original_CRPS=original['CRPS_empirical'],
                     ablation_CRPS=result['CRPS_empirical'],
                     original_inv_MSE=original['inv_MSE'],
                     original_inv_MAE=original['inv_MAE'],
                     ablation_inv_MSE=result['inv_MSE'],
                     ablation_inv_MAE=result['inv_MAE'],
                     original_mean_MSE=original['mean_MSE'],
                     ablation_mean_MSE=result['mean_MSE'],
                     source=str(path)))
path=case_out/f'{arm}_fixed_lr_summary.csv'
with path.open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
print('COMPLETE',path)
PY
}

run_case() {
  local backbone="$1" dataset="$2" horizon="$3" gpu="$4"
  load_case "$backbone" "$dataset" "$horizon" || return 1
  local case_out="$OUTPUT_ROOT/$backbone/$dataset/pred_$horizon"
  mkdir -p "$case_out"
  local anchor lr arm
  anchor="$case_out/no_domain/anchor"
  run_no_domain_anchor "$gpu" "$anchor" || return 1
  local selected_lrs
  selected_lrs="$("$PYTHON" - "$SELECTED" <<'PY'
import csv,sys
with open(sys.argv[1]) as handle:
    print(' '.join(sorted({r['best_head_lr'] for r in csv.DictReader(handle)},key=float)))
PY
)"
  [[ -n "$selected_lrs" ]] || { echo "No recorded head learning rates: $SELECTED" >&2; return 1; }
  for arm in no_domain; do
    if [[ -s "$case_out/${arm}_fixed_lr_summary.csv" ]]; then
      echo "SKIP complete $arm $backbone/$dataset/$horizon"
      continue
    fi
    local chosen_anchor="$anchor/A2/trained_checkpoint.pt"
    for lr in $selected_lrs; do
      run_head_bank "$arm" "$gpu" "$chosen_anchor" "$lr" \
        "$case_out/$arm/head_lr_${lr//./p}" || return 1
    done
    summarize_case "$arm" "$case_out" || return 1
  done
}

queue() {
  mkdir -p "$OUTPUT_ROOT/queue_logs" "$OUTPUT_ROOT/queue_status"
  local scheduler
  scheduler="$(mktemp -d "$OUTPUT_ROOT/.queue.XXXXXX")"
  printf '0\n' >"$scheduler/next"
  touch "$scheduler/lock"
  mapfile -t cases < <(find "$BASE_ROOT" -name best_heads_test_selected.csv -type f | sort)
  echo "QUEUE host=$HOSTNAME cases=${#cases[@]} GPUs=$GPU_LIST root=$OUTPUT_ROOT"
  worker() {
    local gpu="$1" used index path rel backbone dataset horizon
    while true; do
      used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
      if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > 256 )); then sleep 30; continue; fi
      exec 9>"$scheduler/lock"; flock 9
      index="$(<"$scheduler/next")"
      if (( index >= ${#cases[@]} )); then flock -u 9; exec 9>&-; break; fi
      path="${cases[$index]}"
      printf '%s\n' "$((index+1))" >"$scheduler/next"
      flock -u 9; exec 9>&-
      rel="${path#"$BASE_ROOT"/}"
      IFS=/ read -r backbone dataset horizon _ <<<"$rel"
      if run_case "$backbone" "$dataset" "${horizon#pred_}" "$gpu" \
          >"$OUTPUT_ROOT/queue_logs/${backbone}_${dataset}_${horizon}.log" 2>&1; then
        printf '%s\n' "$backbone,$dataset,${horizon#pred_},$gpu" \
          >>"$OUTPUT_ROOT/queue_status/completed.txt"
        echo "DONE $backbone/$dataset/$horizon GPU=$gpu"
      else
        printf '%s\n' "$backbone,$dataset,${horizon#pred_},$gpu" \
          >>"$OUTPUT_ROOT/queue_status/failed.txt"
        echo "FAILED $backbone/$dataset/$horizon GPU=$gpu" >&2
      fi
    done
  }
  local pids=() gpu
  for gpu in $GPU_LIST; do worker "$gpu" & pids+=("$!"); done
  local status=0
  for gpu in "${pids[@]}"; do wait "$gpu" || status=1; done
  return "$status"
}

case "${1:-}" in
  case) run_case "${2:?backbone}" "${3:?dataset}" "${4:?horizon}" "${5:?gpu}" ;;
  queue) queue ;;
  plan)
    find "$BASE_ROOT" -name best_heads_test_selected.csv -type f | sort
    ;;
  *) echo "usage: $0 case BACKBONE DATASET HORIZON GPU | queue | plan" >&2; exit 2 ;;
esac
