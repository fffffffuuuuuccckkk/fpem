#!/usr/bin/env python3
"""Rebuild compact per-version CSV from completed Safe-F03 cases."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = []
    for path in root.rglob("metrics.json"):
        if path.parent == root or not (path.parent / "complete").is_file():
            continue
        try:
            value = json.loads(path.read_text())
            versions = value["versions"]
        except (ValueError, KeyError, OSError):
            continue
        for name in ("A0", "F03", "R3", "F03+R3"):
            for split in ("val", "test"):
                score = versions[name][split]
                rows.append({"case": value["case"], "version": name, "split": split,
                             "mse": score["mse"], "mae": score["mae"],
                             "A0_sha256": value["A0_sha256"],
                             "q_sha256": value["q_sha256"],
                             "F03_alpha": value["head_selected"]["alpha"],
                             "F03_lr": value["head_selected"]["lr"],
                             "F03_lambda_inv": value["head_selected"]["lambda_inv"],
                             "R3_lr": "" if name in ("A0", "F03") else
                               value["r3"]["A0_no_F03" if name == "R3" else "Safe_F03"]["selected"]["lr"]})
    rows.sort(key=lambda x: (x["case"], x["version"], x["split"]))
    dest = root / "all_results.csv"
    temp = root / f"all_results.csv.{os.getpid()}.tmp"
    temp.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case", "version", "split", "mse", "mae",
                                                   "A0_sha256", "q_sha256", "F03_alpha",
                                                   "F03_lr", "F03_lambda_inv", "R3_lr"))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, dest)
    print(json.dumps({"completed_cases": len({r["case"] for r in rows}), "rows": len(rows)}))


if __name__ == "__main__":
    main()
