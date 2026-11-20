#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
ROLE="${SERVER_ROLE:?set SERVER_ROLE=primary|server2|server3}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
cd "$PROJECT_DIR"
case "$ROLE" in
  primary) manifest=scripts/fpem_final/manifests/primary_patchtst.csv; gpus="0 1 2" ;;
  server2) manifest=scripts/fpem_final/manifests/server2_patchtst.csv; gpus="0 1" ;;
  server3) manifest=scripts/fpem_final/manifests/server3_patchtst.csv; gpus="0 1" ;;
  *) echo "invalid SERVER_ROLE=$ROLE" >&2; exit 2 ;;
esac
echo "role=$ROLE manifest=$manifest GPUs=$gpus DRY_RUN=$DRY_RUN RESUME=$RESUME"
if [[ "$DRY_RUN" == 1 ]]; then
  MANIFEST="$manifest" GPU_LIST="$gpus" DRY_RUN=1 RESUME="$RESUME" \
    OUTPUT_ROOT="results/fpem_final_patchtst_search/$ROLE" bash scripts/fpem_final/run_fpem_head_lr_search_matrix.sh
  [[ "$ROLE" == primary ]] && DRY_RUN=1 GPU=3 bash scripts/fpem_final/run_foil_upstream_matrix.sh
  [[ "$ROLE" == server2 ]] && GPU_LIST="2 3" DRY_RUN=1 bash scripts/fpem_final/run_classic_baselines_matrix.sh
  [[ "$ROLE" == server3 ]] && MANIFEST=scripts/fpem_final/manifests/backbone_representative.csv \
    GPU_LIST="2 3" DRY_RUN=1 bash scripts/fpem_final/run_fpem_backbone_matrix.sh
  exit 0
fi
screen -dmS "fpem_final_${ROLE}" bash -lc "cd '$PROJECT_DIR' && MANIFEST='$manifest' GPU_LIST='$gpus' RESUME='$RESUME' OUTPUT_ROOT='results/fpem_final_patchtst_search/$ROLE' bash scripts/fpem_final/run_fpem_head_lr_search_matrix.sh > 'results/fpem_final_${ROLE}.log' 2>&1"
if [[ "$ROLE" == primary ]]; then
  screen -dmS fpem_final_foil bash -lc "cd '$PROJECT_DIR' && GPU=3 bash scripts/fpem_final/run_foil_upstream_matrix.sh > results/fpem_final_foil.log 2>&1"
elif [[ "$ROLE" == server2 ]]; then
  screen -dmS fpem_final_baselines bash -lc "cd '$PROJECT_DIR' && GPU_LIST='2 3' bash scripts/fpem_final/run_classic_baselines_matrix.sh > results/fpem_final_baselines.log 2>&1"
elif [[ "$ROLE" == server3 ]]; then
  screen -dmS fpem_final_backbones bash -lc "cd '$PROJECT_DIR' && MANIFEST=scripts/fpem_final/manifests/backbone_representative.csv GPU_LIST='2 3' bash scripts/fpem_final/run_fpem_backbone_matrix.sh > results/fpem_final_backbones.log 2>&1"
fi
screen -ls
