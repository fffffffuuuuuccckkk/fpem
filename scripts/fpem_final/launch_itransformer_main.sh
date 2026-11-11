#!/usr/bin/env bash
set -euo pipefail

# Run the PatchTST main matrix with an iTransformer encoder: first coordinate
# search for the invariant representation, then probabilistic heads / CRPS.
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
SERVER_SLOT="${SERVER_SLOT:?set SERVER_SLOT to primary, server2, or server3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_no_future_anchor_itransformer_search_20261012}"
DATA_PROJECT="${DATA_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PROB_OUTPUT_ROOT="${PROB_OUTPUT_ROOT:-$DATA_PROJECT/results/fpem_probabilistic/itransformer_main_20261012}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
case "$SERVER_SLOT" in
  primary) GPU_LIST="${GPU_LIST:-0 1 2 3}" ;;
  server2) GPU_LIST="${GPU_LIST:-0 1 2 3}" ;;
  server3) GPU_LIST="${GPU_LIST:-1 2 3}" ;;
  *) echo "unknown SERVER_SLOT: $SERVER_SLOT" >&2; exit 2 ;;
esac

MANIFEST="scripts/fpem_final/manifests/itransformer_search_${SERVER_SLOT}.csv"
PROB_MANIFEST="scripts/fpem_probabilistic/itransformer_probability_${SERVER_SLOT}.csv"
SCREEN_NAME="fpem_itr_${SERVER_SLOT}${SCREEN_SUFFIX:-}"
cd "$PROJECT_DIR"
test -s "$MANIFEST"
test -s "$PROB_MANIFEST"
test -x "$PYTHON"
command -v screen >/dev/null
mkdir -p "$OUTPUT_ROOT" "$PROB_OUTPUT_ROOT"
export PROJECT_DIR DATA_PROJECT PYTHON MANIFEST GPU_LIST OUTPUT_ROOT SERVER_SLOT PROB_OUTPUT_ROOT
export SKIP_SEARCH="${SKIP_SEARCH:-0}" WAIT_FOR_SEARCH="${WAIT_FOR_SEARCH:-0}" SCREEN_SUFFIX="${SCREEN_SUFFIX:-}"
export WAIT_FOR_SCREEN="${WAIT_FOR_SCREEN:-}"
export RESUME=1 SEARCH_PROFILE=compact
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  DRY_RUN=1 bash scripts/fpem_final/run_fpem_head_lr_search_matrix.sh
  exit 0
fi
if [[ "${RUN_IN_SCREEN:-0}" != 1 ]]; then
  if screen -ls | grep -Eq "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "already running: $SCREEN_NAME"
    exit 0
  fi
  screen -dmS "$SCREEN_NAME" env RUN_IN_SCREEN=1 bash scripts/fpem_final/launch_itransformer_main.sh
  sleep 2
  screen -ls | grep -Eq "[.]${SCREEN_NAME}[[:space:]]"
  echo "started: $SCREEN_NAME; search=$MANIFEST; probability=$PROB_MANIFEST; GPUs=$GPU_LIST"
  exit 0
fi

exec >> "$OUTPUT_ROOT/launcher_${SERVER_SLOT}${SCREEN_SUFFIX}.log" 2>&1
if [[ -n "$WAIT_FOR_SCREEN" ]]; then
  echo "WAIT_FOR_SCREEN $WAIT_FOR_SCREEN"
  while screen -ls | grep -Eq "[.]${WAIT_FOR_SCREEN}[[:space:]]"; do
    sleep 30
  done
fi
if [[ "$SKIP_SEARCH" != 1 ]]; then
  echo "SEARCH_START slot=$SERVER_SLOT GPUs=$GPU_LIST"
  bash scripts/fpem_final/run_fpem_head_lr_search_matrix.sh
fi
if [[ "$WAIT_FOR_SEARCH" == 1 ]]; then
  while IFS=, read -r backbone dataset pred_len; do
    [[ "$backbone" == itransformer ]] || continue
    until [[ -s "$OUTPUT_ROOT/$backbone/$dataset/pred_$pred_len/best_config.json" ]]; do
      sleep 30
    done
  done < "$MANIFEST"
fi
echo "PROBABILITY_START slot=$SERVER_SLOT"
SEARCH_ROOT_ABS="$(realpath "$OUTPUT_ROOT")"
OUTPUT_ROOT="$PROB_OUTPUT_ROOT" FPEM_SEARCH_ROOT="$SEARCH_ROOT_ABS" \
  bash scripts/fpem_probabilistic/run_unified_matrix.sh queue "$PROB_MANIFEST"
echo "COMPLETE slot=$SERVER_SLOT"
