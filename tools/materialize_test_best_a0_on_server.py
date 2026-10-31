"""Stage catalog A0 files on another server without altering original results."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--server", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.catalog.open()))
    for row in rows:
        backbone = row["backbone"]
        target = Path(row.get("checkpoint_path") or row.get("reference_path"))
        if not target.is_absolute():
            target = args.project_root / target
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            print("ready", backbone, row["dataset"], row["pred_len"])
            continue
        if target.is_symlink():
            raise ValueError(f"Broken staged symlink: {target}")
        if row["source_server"] != args.server:
            print("copy_required", backbone, row["dataset"], row["pred_len"], target)
            continue
        if backbone == "itransformer":
            parent = (args.project_root / "results" / row["source_result_root"] /
                      "iTransformer" / row["dataset"] / f"pred_{row['pred_len']}" /
                      "checkpoints")
            matches = list(parent.glob("*/checkpoint.pth"))
            if len(matches) != 1:
                raise ValueError(f"Expected one source checkpoint in {parent}: {matches}")
            source = matches[0].resolve()
        elif backbone == "moderntcn":
            source = Path(row["source_reference_path"])
            if not source.is_file():
                raise ValueError(f"Missing source reference: {source}")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        target.symlink_to(source)
        print("linked", backbone, row["dataset"], row["pred_len"])


if __name__ == "__main__":
    main()
