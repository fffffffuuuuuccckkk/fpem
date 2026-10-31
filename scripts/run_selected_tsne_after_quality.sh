#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
ROOT="${ROOT:-results/environment_quality_selected_patchtst_k3}"
WAIT_SCREEN="${WAIT_SCREEN:-env_quality_selected3}"
MAX_SAMPLES="${MAX_SAMPLES:-3000}"

cd "$PROJECT_DIR"
printf 'waiting for %s to finish before t-SNE extraction\n' "$WAIT_SCREEN"
while screen -list 2>/dev/null | grep -q "[.]${WAIT_SCREEN}[[:space:]]"; do
    sleep 30
done
printf 'starting representation t-SNE at %s\n' "$(date --iso-8601=seconds)"

run_one() {
    dataset="$1"; gpu="$2"; batch="$3"
    experiment_dir="$ROOT/$dataset/A2"
    test -s "$experiment_dir/trained_checkpoint.pt"
    test -s "$experiment_dir/environment_quality_summary.txt"
    "$PYTHON" -u tools/visualize_environment_representations.py \
      --experiment_dir "$experiment_dir" --dataset "$dataset" --gpu "$gpu" \
      --batch_size "$batch" --max_samples "$MAX_SAMPLES" \
      > "$ROOT/$dataset/tsne.log" 2>&1
}

run_one ETTh2 0 32 & p0=$!
run_one Electricity 1 4 & p1=$!
run_one Traffic 2 2 & p2=$!
status=0
for pid in "$p0" "$p1" "$p2"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || exit "$status"
printf 'completed %s\n' "$(date --iso-8601=seconds)" > "$ROOT/tsne_completion.txt"
