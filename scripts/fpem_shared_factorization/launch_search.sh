#!/usr/bin/env bash
set -euo pipefail

SERVER_INDEX="${1:?usage: bash launch_search.sh SERVER_INDEX(0|1|2)}"
ROOT=/data/OuXiaoyu/Time-Series-Library-FPEM
NAME=fpem_shared_lr_20261019
MANIFEST="$ROOT/scripts/fpem_shared_factorization/manifest.csv"
RESULT_LINK="$ROOT/results/fpem_shared_factorization/pilot_lr_search_20261019"

case "$SERVER_INDEX" in
  0|1)
    PYTHON=/data/OuXiaoyu/miniconda3/envs/basicts/bin/python
    mkdir -p "$RESULT_LINK"
    ;;
  2)
    PYTHON=/data/OuXiaoyu/miniconda3/envs/tslib/bin/python
    # Server 3 has only ~5 GB free on /data; checkpoints live on /var/tmp.
    RESULT_STORAGE=/var/tmp/fpem_shared_factorization_pilot_lr_search_20261019
    mkdir -p "$RESULT_STORAGE" "$(dirname "$RESULT_LINK")"
    if [[ ! -e "$RESULT_LINK" && ! -L "$RESULT_LINK" ]]; then
      ln -s "$RESULT_STORAGE" "$RESULT_LINK"
    fi
    if [[ "$(readlink -f "$RESULT_LINK")" != "$RESULT_STORAGE" ]]; then
      echo "Unexpected result link: $RESULT_LINK" >&2
      exit 2
    fi
    ;;
  *) echo "Invalid server index $SERVER_INDEX" >&2; exit 2 ;;
esac

if screen -ls 2>/dev/null | grep -q "$NAME"; then
  echo "Screen already exists: $NAME" >&2
  exit 2
fi
test -f "$MANIFEST"
cd "$ROOT"
COMMAND="cd '$ROOT' && '$PYTHON' tools/run_fpem_shared_factor_search.py run --manifest '$MANIFEST' --server-index '$SERVER_INDEX' --project-dir '$ROOT' --output-root '$RESULT_LINK' >> '$RESULT_LINK/launcher.log' 2>&1"
screen -dmS "$NAME" bash -lc "$COMMAND"
sleep 2
screen -ls | grep "$NAME"
echo "log: $RESULT_LINK/launcher.log"
echo "results: $RESULT_LINK"
