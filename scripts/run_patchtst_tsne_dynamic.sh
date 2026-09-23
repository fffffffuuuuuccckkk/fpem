#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-results/predictive_env_featurewise_gate_all_backbones_k3/patchtst}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/predictive_env_patchtst_tsne_all_datasets_k3}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
MAX_SAMPLES="${MAX_SAMPLES:-3000}"
IDLE_MEMORY_MB="${IDLE_MEMORY_MB:-256}"
IDLE_CONFIRM_SECONDS="${IDLE_CONFIRM_SECONDS:-45}"
IDLE_POLL_SECONDS="${IDLE_POLL_SECONDS:-20}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT"
TASKS=(Traffic Electricity Weather ETTm1 ETTm2 ETTh1 ETTh2 ExchangeRate)
SCHEDULER_DIR="$(mktemp -d "$OUTPUT_ROOT/.tsne_scheduler.XXXXXX")"
printf '0\n' > "$SCHEDULER_DIR/next"
touch "$SCHEDULER_DIR/lock"
trap 'rm -f "$SCHEDULER_DIR/next" "$SCHEDULER_DIR/lock"; rmdir "$SCHEDULER_DIR" 2>/dev/null || true' EXIT

gpu_used_memory() {
    local gpu="$1"
    nvidia-smi --id="$gpu" --query-compute-apps=used_memory \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk '{sum += $1} END {print sum + 0}'
}

wait_for_sustained_idle() {
    local gpu="$1" first second
    while true; do
        first="$(gpu_used_memory "$gpu")"
        if (( first <= IDLE_MEMORY_MB )); then
            sleep "$IDLE_CONFIRM_SECONDS"
            second="$(gpu_used_memory "$gpu")"
            if (( second <= IDLE_MEMORY_MB )); then return; fi
        else
            sleep "$IDLE_POLL_SECONDS"
        fi
    done
}

claim_task() {
    local index
    exec 9>"$SCHEDULER_DIR/lock"
    flock 9
    index="$(<"$SCHEDULER_DIR/next")"
    if (( index >= ${#TASKS[@]} )); then
        flock -u 9
        exec 9>&-
        return 1
    fi
    CLAIMED_TASK="${TASKS[$index]}"
    printf '%s\n' "$((index + 1))" > "$SCHEDULER_DIR/next"
    flock -u 9
    exec 9>&-
}

run_one() {
    local dataset="$1" gpu="$2" batch=32
    case "$dataset" in
        Traffic) batch=2 ;;
        Electricity) batch=4 ;;
        Weather) batch=8 ;;
    esac
    local experiment_dir="$CHECKPOINT_ROOT/$dataset/token_gate/A2"
    local checkpoint="$experiment_dir/trained_checkpoint.pt"
    local output_dir="$OUTPUT_ROOT/$dataset"
    mkdir -p "$output_dir"
    if [[ -s "$output_dir/tsne_H_Zinv_Zvar_Zfinal.png" && -s "$output_dir/tsne_summary.txt" ]]; then
        echo "[$dataset] reuse completed t-SNE"
        return
    fi
    test -s "$checkpoint"
    echo "[$dataset] start final-checkpoint t-SNE on sustained-idle GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      "$PYTHON" -u tools/visualize_environment_representations.py \
        --experiment_dir "$experiment_dir" --checkpoint "$checkpoint" \
        --output_dir "$output_dir" --dataset "$dataset" --gpu 0 \
        --batch_size "$batch" --max_samples "$MAX_SAMPLES" \
        --infer_environment > "$output_dir/run.log" 2>&1
}

worker() {
    local gpu="$1"
    while true; do
        wait_for_sustained_idle "$gpu"
        CLAIMED_TASK=""
        claim_task || break
        run_one "$CLAIMED_TASK" "$gpu"
    done
}

pids=()
for gpu in $GPU_LIST; do worker "$gpu" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || exit "$status"
date --iso-8601=seconds > "$OUTPUT_ROOT/completion.txt"
