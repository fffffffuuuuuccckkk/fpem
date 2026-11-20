#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
PYTHON="${PYTHON:-/data/OuXiaoyu/miniconda3/envs/basicts/bin/python}"
cd "$PROJECT_DIR"
"$PYTHON" -m pytest -q tests/test_probabilistic_affine_dynamics.py \
  tests/test_timefilter_predictive_env_backbone.py \
  tests/test_modern_tcn_predictive_env_backbone.py
echo "IAPAD unit/regression tests passed"
