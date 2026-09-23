#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
FOIL_DIR="${FOIL_DIR:-/data/OuXiaoyu/FOIL_upstream}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/results/fpem_final_foil}"
GPU="${GPU:-3}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
mkdir -p "$OUTPUT_ROOT" "$FOIL_DIR/dataset"
commit="$(git -C "$FOIL_DIR" rev-parse HEAD)"
for dataset in ETTh1 ETTh2 ETTm1 ETTm2 ExchangeRate Weather Electricity Traffic; do
  for pred in 96 336 720; do
    out="$OUTPUT_ROOT/$dataset/pred_$pred"; mkdir -p "$out"
    if [[ "$dataset" != ExchangeRate || "$pred" != 96 ]]; then
      printf 'status=unavailable\nreason=FOIL upstream commit %s ships only Exchange 24/96 and ILI 4/12 entrypoints\n' "$commit" > "$out/unavailable.txt"
    fi
  done
done
out="$OUTPUT_ROOT/ExchangeRate/pred_96"
cat > "$out/protocol.txt" <<EOF
upstream_commit=$commit
upstream=$FOIL_DIR
dataset=ExchangeRate
pred_len=96
seed=3407
compatibility_patch=GPU selection plus pandas/NumPy API compatibility; no algorithm change
EOF
echo "FOIL upstream commit=$commit dataset=ExchangeRate pred=96 GPU=$GPU output=$out"
[[ "$DRY_RUN" == 1 ]] && exit 0
ln -sfn "$PROJECT_DIR/dataset/all_datasets/exchange_rate" "$FOIL_DIR/dataset/exchange_rate"
sed 's/os.environ\["CUDA_VISIBLE_DEVICES"\] = "7"/os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("FOIL_GPU", "0")/' \
  "$FOIL_DIR/Informer+FOIL/Exchange-Pred96-0.py" > "$out/Exchange-Pred96-0.compat.py"
sed 's/os.environ\["CUDA_VISIBLE_DEVICES"\] = "7"/os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("FOIL_GPU", "0")/' \
  "$FOIL_DIR/Informer+FOIL/Exchange-Pred96-1.py" > "$out/Exchange-Pred96-1.compat.py"
if [[ "$SMOKE" == 1 ]]; then
  sed -i -E 's/args\.train_epochs ?= ?(5|30)/args.train_epochs = 1/g' "$out"/*.compat.py
fi
cat > "$out/compatibility.patch.txt" <<'EOF'
The hardcoded CUDA_VISIBLE_DEVICES=7 assignment is replaced by FOIL_GPU.
Removed pandas/NumPy API aliases are replaced by their direct modern equivalents.
No model, loss, split, normalization, seed, or optimization logic is changed.
EOF
(cd "$FOIL_DIR/Informer+FOIL" && FOIL_GPU="$GPU" PYTHONPATH="$FOIL_DIR/Informer+FOIL" "$PYTHON" "$out/Exchange-Pred96-0.compat.py") 2>&1 | tee "$out/stage0.log"
(cd "$FOIL_DIR/Informer+FOIL" && FOIL_GPU="$GPU" PYTHONPATH="$FOIL_DIR/Informer+FOIL" "$PYTHON" "$out/Exchange-Pred96-1.compat.py") 2>&1 | tee "$out/stage1.log"
touch "$out/run_complete"
"$PYTHON" "$PROJECT_DIR/tools/summarize_foil_matrix.py" --root "$OUTPUT_ROOT"
