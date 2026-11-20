#!/usr/bin/env python3
"""Rebalance not-yet-started full-TRAIN plan rows over free local GPUs."""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--from-gpus", required=True)
    p.add_argument("--to-gpus", required=True)
    a = p.parse_args()
    rows = []
    for gpu in map(int, a.from_gpus.split(",")):
        with (a.root / "plans" / f"{a.host}_gpu{gpu}.csv").open(newline="") as f:
            rows.extend(csv.DictReader(f))
    keys = [row["case_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate case")
    destinations = list(map(int, a.to_gpus.split(",")))
    for i, gpu in enumerate(destinations):
        path = a.root / "plans" / f"{a.host}_gpu{gpu}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("case_key", "manifest"))
            writer.writeheader()
            writer.writerows(rows[i::len(destinations)])
    print({"cases": len(rows), "gpus": destinations})


if __name__ == "__main__":
    main()
