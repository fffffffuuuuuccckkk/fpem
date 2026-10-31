#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
MANIFEST="${MANIFEST:-scripts/fpem_final/manifests/backbone_representative.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fpem_final_backbones}"
SEARCH_PROFILE="${SEARCH_PROFILE:-representative}"
export PROJECT_DIR MANIFEST OUTPUT_ROOT SEARCH_PROFILE
exec bash "$PROJECT_DIR/scripts/fpem_final/run_fpem_head_lr_search_matrix.sh"
