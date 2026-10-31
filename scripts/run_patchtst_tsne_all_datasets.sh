#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-results/predictive_env_featurewise_gate_all_backbones_k3/patchtst}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_patchtst_tsne_all_datasets_k3}"
WAIT_SCREEN="${WAIT_SCREEN:-var_gain_selected3}"
MAX_SAMPLES="${MAX_SAMPLES:-3000}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"
printf 'waiting for %s before final-checkpoint t-SNE extraction\n' "$WAIT_SCREEN"
while screen -list 2>/dev/null | grep -q "[.]${WAIT_SCREEN}[[:space:]]"; do
    sleep 30
done
printf 'starting PatchTST t-SNE at %s\n' "$(date --iso-8601=seconds)"

run_one() {
    dataset="$1"
    gpu="$2"
    batch="$3"
    experiment_dir="$CHECKPOINT_ROOT/$dataset/token_gate/A2"
    checkpoint="$experiment_dir/trained_checkpoint.pt"
    output_dir="$OUTPUT_ROOT/$dataset"
    mkdir -p "$output_dir"
    test -s "$checkpoint"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      "$PYTHON" -u tools/visualize_environment_representations.py \
        --experiment_dir "$experiment_dir" \
        --checkpoint "$checkpoint" \
        --output_dir "$output_dir" \
        --dataset "$dataset" \
        --gpu 0 \
        --batch_size "$batch" \
        --max_samples "$MAX_SAMPLES" \
        --infer_environment \
        > "$output_dir/run.log" 2>&1
}

(
    run_one Traffic 0 2
    run_one ETTh1 0 32
) & queue0=$!
(
    run_one Electricity 1 4
    run_one ETTm2 1 32
) & queue1=$!
(
    run_one Weather 2 8
    run_one ETTm1 2 32
) & queue2=$!
(
    run_one ETTh2 3 32
    run_one ExchangeRate 3 32
) & queue3=$!

status=0
for pid in "$queue0" "$queue1" "$queue2" "$queue3"; do
    wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
    printf 'one or more t-SNE queues failed; inspect per-dataset run.log\n' >&2
    exit "$status"
fi
printf 'completed %s\n' "$(date --iso-8601=seconds)" > "$OUTPUT_ROOT/completion.txt"

