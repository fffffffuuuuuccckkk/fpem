#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/OuXiaoyu/Time-Series-Library-FPEM}"
METHOD="${METHOD:?set METHOD=TimesNet or Autoformer}"
PRIOR_SCREEN="${PRIOR_SCREEN:?set PRIOR_SCREEN}"
ROOT="$PROJECT_DIR/results/fpem_classic_baselines_completion_20261013"
cd "$PROJECT_DIR"

while screen -ls 2>/dev/null | grep -q "[.]$PRIOR_SCREEN"; do
  sleep 20
done

# The first minute-series attempts failed before model construction because
# the old runner passed freq=15min. Its completed.txt had false positives.
# Preserve that record and regenerate an evidence-based status list.
if [[ -f "$ROOT/status/completed.txt" ]]; then
  mv "$ROOT/status/completed.txt" "$ROOT/status/completed.before_freq_fix_20261013.txt"
fi
python3 - "$ROOT" "$METHOD" <<'PY'
import json, sys
from pathlib import Path
root, method = Path(sys.argv[1]), sys.argv[2]
complete = []
for path in root.glob(f'{method}/*/pred_*/metrics.json'):
    if not (path.parent / 'run_complete').is_file():
        continue
    try:
        d = json.loads(path.read_text())
        float(d['MSE']); float(d['MAE'])
    except (OSError, ValueError, KeyError):
        continue
    complete.append(f'{method}:{path.parent.parent.name}:{path.parent.name[5:]}')
(root / 'status/completed.txt').write_text(''.join(x + '\n' for x in sorted(complete)))
print(f'validated previous {method} results: {len(complete)}', flush=True)
PY

METHOD="$METHOD" bash scripts/fpem_final/run_remaining_classic_backbones.sh

count="$(find "$ROOT/$METHOD" -type f -name metrics.json | wc -l)"
echo "$METHOD verified metrics: $count/15"
[[ "$count" -eq 15 ]]
touch "$ROOT/status/${METHOD}.verified_complete"
