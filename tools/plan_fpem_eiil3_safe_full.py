#!/usr/bin/env python3
"""Create disjoint per-GPU plans from two audited A0 manifests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path):
    return {r["case_key"]: r for r in (json.loads(x) for x in path.read_text().splitlines())}


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server1-manifest", type=Path, required=True)
    parser.add_argument("--server2-manifest", type=Path, required=True)
    parser.add_argument("--server1-run-manifest", type=Path)
    parser.add_argument("--server2-run-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, b = load(args.server1_manifest), load(args.server2_manifest)
    assert set(a) == set(b) and len(a) == 96
    plans = {(host, gpu): [] for host in (1, 2) for gpu in range(4)}
    a0 = {(host, gpu): [] for host in (1, 2) for gpu in range(4)}
    counts = {(host, gpu): 0 for host in (1, 2) for gpu in range(4)}
    missing = []
    for key in sorted(a):
        row_a, row_b = a[key], b[key]
        # Reuse the locally verified A0 on server 1; all remaining existing
        # references are delegated to server 2. Never duplicate a case.
        if row_a["status"] != "MISSING_A0":
            host, source, needs_a0 = 1, args.server1_run_manifest or args.server1_manifest, False
        elif row_b["status"] != "MISSING_A0":
            host, source, needs_a0 = 2, args.server2_run_manifest or args.server2_manifest, False
        else:
            host = 2 if row_a["dataset"] in ("Electricity", "Traffic") else 1
            needs_a0 = True
            root = Path("results/fpem_eiil3_mse_priority_20261112/a0_assets")
            source = root / row_a["backbone"] / row_a["dataset"] / f"pred_{row_a['pred_len']}" / "seed_2021/case_manifest.jsonl"
            missing.append(key)
        dataset = row_a["dataset"]
        choices = (3,) if dataset in ("Electricity", "Traffic") else (0, 1, 2)
        gpu = min(choices, key=lambda g: counts[(host, g)])
        counts[(host, gpu)] += 1
        plans[(host, gpu)].append({"case_key": key, "manifest": str(source)})
        if needs_a0:
            a0[(host, gpu)].append({"backbone": row_a["backbone"], "dataset": dataset,
                                    "horizon": row_a["pred_len"]})
    for (host, gpu), rows in plans.items():
        write_csv(args.output / f"server{host}_gpu{gpu}.csv", ("case_key", "manifest"), rows)
    for (host, gpu), rows in a0.items():
        target = args.output / f"server{host}_a0_gpu{gpu}.csv"
        with target.open("w") as stream:
            for row in rows:
                stream.write(f"{row['backbone']},{row['dataset']},{row['horizon']}\n")
    summary = {"total": sum(map(len, plans.values())),
               "per_gpu": {f"server{h}_gpu{g}": len(v) for (h, g), v in plans.items()},
               "A0_to_prepare": missing}
    (args.output / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"total": summary["total"], "missing_A0": len(missing),
                      "per_gpu": summary["per_gpu"]}))


if __name__ == "__main__":
    main()
