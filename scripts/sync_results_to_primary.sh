#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
if [[ -z "${SOURCE_ROOT:-}" ]]; then
    # Use the most recently modified current-line result root on each server
    # (p96 on server1, p336 on server2, p720 on server3). Older result trees
    # remain available through an explicit SOURCE_ROOT override.
    default_source="$({
        find "$PROJECT_DIR/results" -maxdepth 1 -type d \
            -name 'predictive_env_future_patch*_patchtst_*_k3' \
            -printf '%T@ %p\n' 2>/dev/null || true
    } | sort -nr | head -n 1 | cut -d' ' -f2-)"
    SOURCE_ROOT="${default_source:-$PROJECT_DIR/results/predictive_env_patchtst_multihorizon_96}"
fi
PRIMARY_HOST="${PRIMARY_HOST:-OuXiaoyu@211.71.76.25}"
PRIMARY_PROJECT="${PRIMARY_PROJECT:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
SERVER_TAG="${SERVER_TAG:-$(hostname -s)}"
DRY_RUN="${DRY_RUN:-0}"

source_real="$(realpath "$SOURCE_ROOT")"
results_real="$(realpath "$PROJECT_DIR/results")"
case "$source_real/" in
    "$results_real/"*) ;;
    *) echo "SOURCE_ROOT must be inside $results_real" >&2; exit 2 ;;
esac
test -d "$source_real"
source_name="$(basename "$source_real")"
# A dedicated namespace guarantees that native primary-server results are
# never overwritten or deleted by this mirror.
destination="$PRIMARY_PROJECT/results/server2_imports/$SERVER_TAG/$source_name"
ssh_options=(-o StrictHostKeyChecking=no)

echo "incremental source:      $source_real/"
echo "primary mirror target:  $PRIMARY_HOST:$destination/"
echo "policy: no delete; only the isolated server2_imports mirror is updated"
echo "override another result root with: SOURCE_ROOT=/absolute/path $0"

rsync_options=(
    --archive
    --update
    --human-readable
    --itemize-changes
    --partial
    --partial-dir=.rsync-partial
    --exclude=.rsync-partial/
    --exclude='*.tmp'
    --rsync-path="mkdir -p '$destination' && rsync"
    --rsh="ssh ${ssh_options[*]}"
)
if [[ "$DRY_RUN" == "1" ]]; then rsync_options+=(--dry-run); fi
rsync "${rsync_options[@]}" "$source_real/" "$PRIMARY_HOST:$destination/"
