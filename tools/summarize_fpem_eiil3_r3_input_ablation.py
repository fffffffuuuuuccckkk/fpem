#!/usr/bin/env python3
"""Small live CSV for the frozen F03/R3 input ablation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.output_root.glob("**/metrics.json")):
        item = json.loads(path.read_text())
        for mode, name in (("A", "A0+R3"), ("B", "F03+R3"),
                           ("C", "F03+A0+R3"), ("D", "F03+DeltaF+R3")):
            scores = item.get("variants", {}).get(mode)
            if scores is None:
                continue
            selected = (item.get(f"{mode}_search", {}).get("selected") or
                        ({} if mode not in ("A", "B") else None))
            row = {"case": item["case"], "status": item["status"],
                   "mode": mode, "method": name,
                   "a0_test_mse": item["A0"]["test"]["mse"],
                   "f03_test_mse": item["F03"]["test"]["mse"],
                   "val_mse": scores["val"]["mse"],
                   "val_mae": scores["val"]["mae"],
                   "test_mse": scores["test"]["mse"],
                   "test_mae": scores["test"]["mae"],
                   "lr": "" if selected is None else selected.get("lr", ""),
                   "epoch": "" if selected is None else selected.get("best_epoch", ""),
                   "cosine_f03_r3": item["direction_diagnostics"]["cosine_f03_r3"]}
            rows.append(row)
    columns = ("case", "status", "mode", "method", "a0_test_mse",
               "f03_test_mse", "val_mse", "val_mae", "test_mse", "test_mae",
               "lr", "epoch", "cosine_f03_r3")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
