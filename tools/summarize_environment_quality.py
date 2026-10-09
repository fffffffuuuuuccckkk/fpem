#!/usr/bin/env python
"""Collect final environment-quality diagnostics into one readable table."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "dataset",
    "D_within_grad",
    "D_between_grad",
    "gradient_separation",
    "gradient_separation_gain",
    "conflict_ours",
    "random_partition_conflict_mean",
    "random_partition_z_score",
    "random_partition_p_value",
    "min_environment_mass",
    "normalized_assignment_entropy",
    "max_q_mean",
    "max_q_p50",
    "final_stage_ARI",
    "final_stage_NMI",
    "var_acc",
    "inv_acc",
    "env_acc_gap_var_minus_inv",
)


def collect(root):
    rows = []
    for path in sorted(Path(root).rglob("environment_quality_summary.json")):
        row = json.loads(path.read_text())
        row["result_path"] = str(path.parent)
        rows.append(row)
    return rows


def write_outputs(root, output):
    rows = collect(root)
    if not rows:
        raise FileNotFoundError(f"no environment_quality_summary.json below {root}")
    output = Path(output)
    lines = [" ".join(FIELDS)]
    for row in rows:
        values = []
        for field in FIELDS:
            value = row.get(field, float("nan"))
            values.append(str(value) if field == "dataset" else f"{float(value):.9g}")
        lines.append(" ".join(values))
    output.write_text("\n".join(lines) + "\n")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS + ("result_path",))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})
    return rows, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "environment_quality_all_datasets.txt"
    rows, csv_path = write_outputs(args.root, output)
    print(f"wrote {len(rows)} rows to {output} and {csv_path}")


if __name__ == "__main__":
    main()
