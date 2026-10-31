"""Read-only inventory of existing A0 references for O3/O4 expansion."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
datasets = {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "ExchangeRate"}
backbones = {"patchtst", "moderntcn", "itransformer"}
horizons = {24, 48, 96, 192, 336, 720}

found = collections.defaultdict(list)
for path in (root / "results").rglob("run_config.json"):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        backbone = str(data.get("backbone", "")).lower()
        dataset = str(data.get("dataset_name", ""))
        horizon = int(data.get("pred_len", -1))
        reference = data.get("reference_checkpoint", "")
        if backbone not in backbones or dataset not in datasets or horizon not in horizons or not reference:
            continue
        reference_path = Path(reference)
        if not reference_path.is_absolute():
            reference_path = root / reference_path
        if not reference_path.is_file():
            continue
        found[(backbone, dataset, horizon)].append((path, reference_path))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        continue

def priority(item):
    path = str(item[0]).lower()
    score = 0
    if "inv_head_lr_search_20261014" in path:
        score += 100
    if "/a0/" in path or "_a0_" in path or "/a0_backbone/" in path:
        score += 80
    if "/anchor/" in path:
        score += 60
    if "server2_imports" in path or "server3_imports" in path:
        score -= 50
    if "smoke" in path:
        score -= 100
    return (-score, len(path), path)


for backbone in sorted(backbones):
    for dataset in sorted(datasets):
        for horizon in sorted(horizons):
            candidates = sorted(found.get((backbone, dataset, horizon), []), key=priority)
            print(f"{backbone},{dataset},{horizon},{len(candidates)},"
                  + (str(candidates[0][0].relative_to(root)) if candidates else ""))
print(f"TOTAL_CASES_WITH_REFERENCE={len(found)}", file=sys.stderr)
