#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_final_baselines}"
GPU_LIST="${GPU_LIST:-0 1}"
DRY_RUN="${DRY_RUN:-0}"
METHODS_CSV="${METHODS:-PatchTST,DLinear,Autoformer,TimesNet,iTransformer,FEDformer}"
DATASETS_CSV="${DATASETS:-ETTh1,ETTh2,ETTm1,ETTm2,ExchangeRate,Weather,Electricity,Traffic}"
PREDS_CSV="${PRED_LENGTHS:-96,336,720}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
RUN_LABEL="${RUN_LABEL:-FPEMFinal}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/status"
IFS=, read -ra METHODS_A <<<"$METHODS_CSV"
IFS=, read -ra DATASETS_A <<<"$DATASETS_CSV"
IFS=, read -ra PREDS_A <<<"$PREDS_CSV"
JOBS=()
for m in "${METHODS_A[@]}"; do
  for d in "${DATASETS_A[@]}"; do
    for p in "${PREDS_A[@]}"; do JOBS+=("$m:$d:$p"); done
  done
done

run_job() {
  local spec="$1" gpu="$2" method dataset pred data_root data_class data_path freq batch cycle channels
  IFS=: read -r method dataset pred <<<"$spec"
  read -r data_root data_class data_path freq batch cycle channels <<<"$(dataset_config "$dataset")"
  local tslib_data=custom
  [[ "$dataset" == ETT* ]] && tslib_data="$dataset"
  local out="$OUTPUT_ROOT/$method/$dataset/pred_$pred"
  if [[ -s "$out/metrics.json" && -s "$out/protocol.txt" && -s "$out/run_complete" ]]; then
    echo "reuse $spec"; return
  fi
  if [[ -e "$out" ]]; then mv "$out" "${out}.incomplete.$(date +%Y%m%d_%H%M%S)"; fi
  mkdir -p "$out" "$out/checkpoints"
  local lr=1e-4 d_model=512 d_ff=2048 e_layers=2 n_heads=8
  local -a extra=()
  case "$method" in
    PatchTST) d_model=128; d_ff=256; e_layers=1; n_heads=8 ;;
    DLinear) ;;
    Autoformer) ;;
    TimesNet) d_model=16; d_ff=32; extra+=(--top_k 5 --num_kernels 6) ;;
    iTransformer) d_model=512; d_ff=512; e_layers=3; lr=5e-4 ;;
    FEDformer) ;;
    *) echo "unavailable method=$method" > "$out/unavailable.txt"; return ;;
  esac
  local model_id="${RUN_LABEL}_${dataset}_96_${pred}"
  local des="${RUN_LABEL}_${method}_${dataset}_${pred}"
  local -a cmd=(env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u run.py
    --task_name long_term_forecast --is_training 1 --root_path "$data_root" --data_path "$data_path"
    --model_id "$model_id" --model "$method" --data "$tslib_data" --features M --target OT --freq "$freq"
    --seq_len 96 --label_len 48 --pred_len "$pred" --e_layers "$e_layers" --d_layers 1 --factor 3
    --enc_in "$channels" --dec_in "$channels" --c_out "$channels" --d_model "$d_model" --d_ff "$d_ff"
    --n_heads "$n_heads" --batch_size "$batch" --learning_rate "$lr" --train_epochs "$TRAIN_EPOCHS" --patience 3
    --num_workers 0 --des "$des" --itr 1 --checkpoints "$out/checkpoints" "${extra[@]}")
  echo "dataset=$dataset pred_len=$pred backbone=$method GPU=$gpu output=$out"
  echo "exact command: $(shell_quote_command "${cmd[@]}")"
  [[ "$DRY_RUN" == 1 ]] && return 0
  shell_quote_command "${cmd[@]}" > "$out/exact_command.sh"
  cat > "$out/protocol.txt" <<EOF
protocol=official/TSLib-style fixed baseline configuration
method=$method
dataset=$dataset
pred_len=$pred
seq_len=96
seed=$SEED
learning_rate=$lr
selection_source=fixed_config
train_epochs=$TRAIN_EPOCHS
EOF
  "${cmd[@]}" 2>&1 | tee "$out/run.log"
  "$PYTHON" - "$model_id" "$method" "$des" "$out" <<'PY'
import glob,json,os,sys
from pathlib import Path
import numpy as np
model_id,method,des,out=sys.argv[1:]
paths=glob.glob(f'results/long_term_forecast_{model_id}_{method}_*_{des}_0/metrics.npy')
if not paths: raise SystemExit('baseline metrics.npy not found')
p=Path(max(paths,key=os.path.getmtime)); x=np.load(p)
d={'MAE':float(x[0]),'MSE':float(x[1]),'RMSE':float(x[2]),'MAPE':float(x[3]),'MSPE':float(x[4]),'source':str(p)}
Path(out,'metrics.json').write_text(json.dumps(d,indent=2)+'\n')
Path(out,'metrics.txt').write_text('\n'.join(f'{k}={v}' for k,v in d.items())+'\n')
for name in ('pred.npy','true.npy'):
    q=p.parent/name
    if q.exists(): q.unlink()
PY
  touch "$out/run_complete"
}

if [[ "$DRY_RUN" == 1 ]]; then
  i=0; ga=($GPU_LIST)
  for j in "${JOBS[@]}"; do run_job "$j" "${ga[$((i%${#ga[@]}))]}"; i=$((i+1)); done
  exit 0
fi
SCHED="$(mktemp -d "$OUTPUT_ROOT/.scheduler.XXXX")"
echo 0 > "$SCHED/next"; touch "$SCHED/lock"; trap 'rm -rf "$SCHED"' EXIT
worker(){
  local gpu="$1" i j
  while true; do
    exec 9>"$SCHED/lock"; flock 9; i="$(<"$SCHED/next")"
    if ((i>=${#JOBS[@]})); then flock -u 9; exec 9>&-; break; fi
    j="${JOBS[$i]}"; echo $((i+1)) > "$SCHED/next"; flock -u 9; exec 9>&-
    run_job "$j" "$gpu" && echo "$j" >> "$OUTPUT_ROOT/status/completed.txt" || echo "$j" >> "$OUTPUT_ROOT/status/failed.txt"
  done
}
pids=(); for g in $GPU_LIST; do worker "$g" & pids+=("$!"); done
for p in "${pids[@]}"; do wait "$p" || true; done
"$PYTHON" tools/summarize_baseline_matrix.py --root "$OUTPUT_ROOT" --csv "$OUTPUT_ROOT/baseline_summary.csv" --txt "$OUTPUT_ROOT/baseline_summary.txt"
