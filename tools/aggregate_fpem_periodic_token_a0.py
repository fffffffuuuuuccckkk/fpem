#!/usr/bin/env python3
"""Aggregate finished paired A0 calendar-token pilots without selecting on TEST."""

import argparse
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    fields = (
        "case", "backbone", "dataset", "horizon", "seed", "periodic_mode",
        "old_a0_test_mse", "old_a0_test_mae", "baseline_val_mse",
        "baseline_test_mse", "baseline_test_mae", "token_initial_val_mse",
        "token_val_mse", "token_test_mse", "token_test_mae", "token_epoch",
        "token_minus_baseline_test_mse", "token_vs_baseline_pct",
        "token_minus_old_a0_test_mse", "status",
    )
    rows = []
    for path in sorted(root.glob("pilot_*/comparison.json")):
        if not (path.parent / "complete").is_file():
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        old = item.get("old_a0", {}).get("test", {})
        base = item["baseline"]
        token = item["token"]
        meta = item["protocol"]
        base_mse = base["test"]["mse"]
        token_mse = token["test"]["mse"]
        rows.append({
            "case": path.parent.name,
            "backbone": meta["backbone"], "dataset": meta["dataset"],
            "horizon": meta["pred_len"], "seed": meta["seed"],
            "periodic_mode": meta["periodic_mode"],
            "old_a0_test_mse": old.get("mse", ""),
            "old_a0_test_mae": old.get("mae", ""),
            "baseline_val_mse": base["val"]["mse"],
            "baseline_test_mse": base_mse,
            "baseline_test_mae": base["test"]["mae"],
            "token_initial_val_mse": token["initial_val"]["mse"],
            "token_val_mse": token["val"]["mse"],
            "token_test_mse": token_mse,
            "token_test_mae": token["test"]["mae"],
            "token_epoch": token["selected_epoch"],
            "token_minus_baseline_test_mse": token_mse - base_mse,
            "token_vs_baseline_pct": 100.0 * (base_mse - token_mse) / base_mse,
            "token_minus_old_a0_test_mse": token_mse - old["mse"] if old else "",
            "status": "complete",
        })
    target = root / "summary.csv"
    temporary = root / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)
    print(json.dumps({"completed": len(rows), "summary": str(target)}))


if __name__ == "__main__":
    main()
