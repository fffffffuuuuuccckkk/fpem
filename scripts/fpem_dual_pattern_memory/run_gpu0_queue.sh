#!/usr/bin/env bash
set -u
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
bash "$ROOT/scripts/fpem_dual_pattern_memory/run_point_queue.sh" patchtst_ETTh1_720 0
patch_status=$?
bash "$ROOT/scripts/fpem_dual_pattern_memory/run_point_queue.sh" moderntcn_ETTm2_720 0
modern_status=$?
if (( patch_status != 0 || modern_status != 0 )); then exit 1; fi
