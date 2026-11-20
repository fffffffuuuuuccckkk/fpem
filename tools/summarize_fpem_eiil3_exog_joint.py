#!/usr/bin/env python3
"""Summarize matched Head-only vs Head+exog pilots without mixing metric scales."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for source in sorted(a.root.rglob("exog_search.json")):
        report = json.loads(source.read_text())
        if report.get("training_mode") != "joint_head":
            continue
        head = report["controls"]["head_only"]
        joint = report["controls"]["head_plus_exog"]
        row = {
            "case": str(source.parent.relative_to(a.root)),
            "backbone": report["backbone"],
            "dataset": report["dataset"],
            "horizon": report["horizon"],
            "selection_split": report["selection_split"],
            "A0_val_mse": report.get("epoch0_val", report["epoch0"] if report["selection_split"] == "val" else {}).get("mse", ""),
            "A0_test_mse": report.get("epoch0_test", report["epoch0"] if report["selection_split"] == "test" else {}).get("mse", ""),
            "head_val_mse": head["val"]["mse"],
            "joint_val_mse": joint["val"]["mse"],
            "head_test_mse": head["test"]["mse"],
            "joint_test_mse": joint["test"]["mse"],
            "head_test_mae": head["test"]["mae"],
            "joint_test_mae": joint["test"]["mae"],
            "joint_minus_head_test_mse": joint["test"]["mse"] - head["test"]["mse"],
            "joint_selected_epoch": joint["selected"]["epoch"],
            "head_selected_epoch": head["selected"]["epoch"],
            "joint_selected_lr": joint["selected"]["lr"],
            "head_selected_lr": head["selected"]["lr"],
            "parent_A0_sha256": report["parent_A0_sha256"],
            "joint_checkpoint_sha256": joint["checkpoint_sha256"],
            "head_checkpoint_sha256": head["checkpoint_sha256"],
        }
        rows.append(row)
    destination = a.root / "joint_head_pilot_summary.csv"
    if rows:
        with destination.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"joint pilots: {len(rows)}; summary: {destination}")


if __name__ == "__main__":
    main()
