#!/usr/bin/env bash
# Re-run the two representation ablations against the exact PatchTST warmup-5
# protocol. One GPU per case; re-running skips verified completed cases.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/fpem_best_a0_code_20261013}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/grl_warmup5_patchtst_ablations_20261015}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
GPU_LIST="${GPU_LIST//,/ }"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/scripts/fpem_final/common.sh"

case_valid() {
  "$PYTHON" - "$1" "$2" "$3" <<'PY'
import json, math, sys
from pathlib import Path
root, arm, lr = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
paths = [root / 'run_config.json', root / 'A2/metrics_and_diagnostics.json',
         root / 'A2/invariant_test_by_epoch.json', root / 'comparison.json']
if not all(p.is_file() and p.stat().st_size for p in paths):
    raise SystemExit(1)
c = json.loads(paths[0].read_text())
e = json.loads(paths[2].read_text())
mode = 'shuffled' if arm == 'environment_shuffled' else 'predictive'
weight = 0.0 if arm == 'no_domain' else 0.1
if (c.get('backbone') != 'patchtst' or c.get('prob_ablation') != 'invariant_only'
    or c.get('decomposition_warmup_epochs') != 5
    or c.get('environment_supervision') != mode
    or not math.isclose(float(c.get('lambda_prob_env', -1)), weight, abs_tol=1e-10)
    or not math.isclose(float(c.get('lr_inv_head', -1)), lr, rel_tol=1e-9)
    or len(e) != 10
    or any(bool(row.get('decomposition_active')) != (i >= 5)
           for i, row in enumerate(e))
    or any(not math.isfinite(float(row[k])) for row in e for k in ('MSE', 'MAE'))):
    raise SystemExit(1)
PY
}

run_arm() {
  local arm="$1" dataset="$2" horizon="$3" gpu="$4" lr_inv="$5"
  local lr_dec="$6" lr_env="$7" lr_variant="$8" lambda_domain="$9"
  shift 9
  local lambda_prob_env="$1" batch_size="$2" d_model="$3" d_ff="$4"
  local n_heads="$5" e_layers="$6" dropout="$7" root_path="$8" data_class="$9"
  shift 9
  local data_path="$1" freq="$2" archive_sha="$3" reference_sha="$4"
  local source_best_mse="$5" source_best_mae="$6" a0_mse="$7" a0_mae="$8"
  local source_config="$9"
  local target="$OUTPUT_ROOT/patchtst/$dataset/pred_$horizon/$arm"
  local reference="$DATA_PROJECT/results/fpem_probabilistic/test_best_patchtst_a0_20261013/references/$dataset/pred_$horizon.pt"
  if case_valid "$target" "$arm" "$lr_inv"; then
    echo "REUSE $dataset/$horizon/$arm"
    return 0
  fi
  if [[ -e "$target/run_config.json" ]]; then
    echo "Incomplete result preserved for inspection: $target" >&2
    return 1
  fi
  [[ -s "$reference" ]] || { echo "Missing reference: $reference" >&2; return 1; }
  [[ "$(sha256sum "$reference" | awk '{print $1}')" == "$reference_sha" ]] || {
    echo "Reference hash mismatch: $reference" >&2; return 1;
  }
  if [[ "$root_path" == ./* ]]; then root_path="$DATA_PROJECT/${root_path#./}"; fi
  local supervision=predictive prob_env="$lambda_prob_env" cycle_len
  cycle_len="$(dataset_config "$dataset" | awk '{print $6}')"
  [[ -n "$cycle_len" ]] || { echo "Missing cycle length for $dataset" >&2; return 1; }
  if [[ "$arm" == environment_shuffled ]]; then supervision=shuffled; fi
  if [[ "$arm" == no_domain ]]; then prob_env=0.0; fi
  mkdir -p "$target"
  env PROJECT_DIR="$PROJECT_DIR" PYTHON="$PYTHON" \
    BACKBONE=patchtst DATASET_NAME="$dataset" DATASET_ARCHIVE_SHA256="$archive_sha" \
    DATA_ROOT="$root_path" DATA_CLASS="$data_class" DATA_PATH="$data_path" FREQ="$freq" \
    CYCLENET_CYCLE_LEN="$cycle_len" \
    SEQ_LEN=96 PRED_LEN="$horizon" BATCH_SIZE="$batch_size" \
    D_MODEL="$d_model" D_FF="$d_ff" N_HEADS="$n_heads" E_LAYERS="$e_layers" DROPOUT="$dropout" \
    GPU="$gpu" EXPERIMENTS=A2 EPOCHS=10 WARMUP_EPOCHS=0 STAGE_EPOCHS=2 ENV_NUM=3 \
    NUM_WORKERS=0 REPRESENTATION_CONSTRAINT=classification DECOMPOSITION_TYPE=complementary_gate \
    ENVIRONMENT_SUPERVISION="$supervision" VARIANT_FUSION_MODE=prob_affine_flow \
    PROB_ABLATION=invariant_only PROB_LOSS_WEIGHT_MODE=historical_ema_maturity \
    PROB_NUM_SAMPLES=2 MAX_PROB_EVAL_BATCHES=1 FUTURE_PATCH_LEN=16 \
    LAMBDA_FUTURE_H=0 LAMBDA_VARIANT_ANCHOR=0 DIFFERENTIAL_LR=1 FIXED_GAMMA_BETA_LR=1 \
    LR=1e-4 LR_BACKBONE=2e-5 LR_INV_HEAD="$lr_inv" \
    LR_DECOMPOSER="$lr_dec" LR_ENV_HEAD="$lr_env" LR_VARIANT="$lr_variant" \
    LR_GAMMA_BETA_BASE=1e-4 LR_RELIABILITY=3e-4 LAMBDA_DOMAIN="$lambda_domain" \
    LAMBDA_PROB_ENV="$prob_env" \
    REFERENCE_CHECKPOINT="$reference" REQUIRE_REFERENCE_CHECKPOINT=1 \
    DECOMPOSITION_WARMUP_EPOCHS=5 TRACK_EPOCH_INV_TEST=1 \
    SAVE_FINAL_CHECKPOINT=1 OUTPUT="$target" \
    bash "$PROJECT_DIR/scripts/run_predictive_env_iv_patchtst.sh" >"$target/run.log" 2>&1 || {
      echo "Training failed: $target/run.log" >&2
      return 1
    }
  "$PYTHON" - "$target" "$source_best_mse" "$source_best_mae" "$a0_mse" "$a0_mae" \
    "$source_config" "$arm" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
metrics = json.loads((root / 'A2/metrics_and_diagnostics.json').read_text())
epochs = json.loads((root / 'A2/invariant_test_by_epoch.json').read_text())
post_grl = epochs[5:]
data = {'arm': sys.argv[7], 'source_config': sys.argv[6],
        'baseline_best_inv_MSE': float(sys.argv[2]),
        'baseline_best_inv_MAE': float(sys.argv[3]),
        'A0_MSE': float(sys.argv[4]), 'A0_MAE': float(sys.argv[5]),
        'best_inv_MSE': metrics['best_epoch_inv_MSE'],
        'best_inv_MSE_epoch': metrics['best_epoch_inv_MSE_epoch'],
        'best_inv_MAE': metrics['best_epoch_inv_MAE'],
        'best_inv_MAE_epoch': metrics['best_epoch_inv_MAE_epoch'],
        'post_grl_best_inv_MSE': min(row['MSE'] for row in post_grl),
        'post_grl_best_inv_MAE': min(row['MAE'] for row in post_grl),
        'final_inv_MSE': metrics['inv_MSE'],
        'final_inv_MAE': metrics['inv_only_MAE'],
        'selection_split': 'test'}
if root.joinpath('comparison.json').exists():
    raise SystemExit('comparison already exists; refusing overwrite')
root.joinpath('comparison.json').write_text(json.dumps(data, indent=2) + '\n')
print('COMPLETE', root, data, flush=True)
PY
  case_valid "$target" "$arm" "$lr_inv"
}

run_case() {
  local dataset="$1" horizon="$2" gpu="$3" manifest="$4"
  local row
  row="$(awk -F, -v d="$dataset" -v h="$horizon" '$1==d && $2==h {print; exit}' "$manifest")"
  [[ -n "$row" ]] || { echo "Missing manifest case: $dataset/$horizon" >&2; return 1; }
  local fields=()
  IFS=, read -r -a fields <<<"$row"
  [[ "${#fields[@]}" == 25 ]] || { echo "Bad manifest columns: ${#fields[@]}" >&2; return 1; }
  for arm in environment_shuffled no_domain; do
    run_arm "$arm" "${fields[0]}" "${fields[1]}" "$gpu" "${fields[@]:2}"
  done
}

queue() {
  local manifest="$1" scheduler
  [[ -s "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; return 1; }
  mkdir -p "$OUTPUT_ROOT/queue_logs" "$OUTPUT_ROOT/queue_status"
  scheduler="$(mktemp -d "$OUTPUT_ROOT/.queue.XXXXXX")"
  printf '0\n' >"$scheduler/next"
  touch "$scheduler/lock"
  mapfile -t cases < <(awk -F, '$1 !~ /^#/ && NF==25 {print $1":"$2}' "$manifest")
  echo "QUEUE cases=${#cases[@]} arms=2 GPUs=$GPU_LIST root=$OUTPUT_ROOT"
  worker() {
    local gpu="$1" used index spec dataset horizon
    while true; do
      used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
      if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > 256 )); then sleep 30; continue; fi
      exec 9>"$scheduler/lock"; flock 9
      index="$(<"$scheduler/next")"
      if (( index >= ${#cases[@]} )); then flock -u 9; exec 9>&-; break; fi
      spec="${cases[$index]}"; printf '%s\n' "$((index+1))" >"$scheduler/next"
      flock -u 9; exec 9>&-
      IFS=: read -r dataset horizon <<<"$spec"
      if run_case "$dataset" "$horizon" "$gpu" "$manifest" \
          >"$OUTPUT_ROOT/queue_logs/${dataset}_${horizon}.log" 2>&1; then
        echo "$spec,$gpu" >>"$OUTPUT_ROOT/queue_status/completed.txt"
        echo "DONE $spec GPU=$gpu"
      else
        echo "$spec,$gpu" >>"$OUTPUT_ROOT/queue_status/failed.txt"
        echo "FAILED $spec GPU=$gpu" >&2
      fi
    done
  }
  local pids=() gpu pid
  for gpu in $GPU_LIST; do worker "$gpu" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "$pid" || true; done
  if [[ -s "$OUTPUT_ROOT/queue_status/failed.txt" ]]; then
    echo "Some cases failed; inspect queue_status/failed.txt" >&2
    return 1
  fi
}

case "${1:-}" in
  plan) awk -F, '$1 !~ /^#/ && NF==25 {print $1 "/" $2}' "${2:?manifest}" ;;
  case) run_case "${2:?dataset}" "${3:?horizon}" "${4:?gpu}" "${5:?manifest}" ;;
  queue) queue "${2:?manifest}" ;;
  *) echo "usage: $0 plan MANIFEST | case DATASET HORIZON GPU MANIFEST | queue MANIFEST" >&2; exit 2 ;;
esac
