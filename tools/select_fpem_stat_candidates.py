"""Select at most two frozen statistical methods using VAL MSE only."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_root", required=True)
    args = parser.parse_args()
    rows = []
    for file in Path(args.case_root).glob("frozen_*/result.json"):
        data = json.loads(file.read_text())
        if data["method"].startswith(("t", "c")) and not data["train_backbone"]:
            rows.append((data["val"]["mse"], data["method"], data["groups"]))
    for _, method, groups in sorted(rows)[:2]:
        print(method, groups)


if __name__ == "__main__":
    main()
