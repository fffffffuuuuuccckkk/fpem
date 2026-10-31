"""Aggregate an independently completed FSLD pilot without selecting on TEST."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    root = Path(parser.parse_args().root)
    records = []
    for case in sorted(root.iterdir()):
        if not case.is_dir():
            continue
        direct = case / "direct" / "direct_metrics.csv"
        if direct.exists():
            for row in csv.DictReader(direct.open(encoding="utf-8")):
                if row["split"] in ("val", "test"):
                    records.append({"case": case.name, "phase": "frozen", "method": row["method"],
                                    "rank": row["rank"], "split": row["split"],
                                    "mse": row["mse"], "mae": row["mae"],
                                    "a0_output_mse": row["a0_output_mse"]})
        for result_file in case.glob("*/result.json"):
            result = json.loads(result_file.read_text())
            spectrum = json.loads((case / "direct" / "spectrum.json").read_text())
            actual_rank = (spectrum["numeric_rank"] if result["method"] == "F1" else
                           "" if result["method"] in ("A1", "D0") else result["rank"])
            for split in ("val", "test"):
                records.append({"case": case.name, "phase": "finetune", "method": result["method"],
                                "rank": actual_rank, "split": split,
                                "mse": result[split]["mse"], "mae": result[split]["mae"],
                                "a0_output_mse": result[split]["a0_output_mse"]})
        probe = case / "probe" / "probe_results.json"
        if probe.exists():
            for result in json.loads(probe.read_text())["results"]:
                records.append({"case": case.name, "phase": "probe", "method": result["mode"],
                                "rank": "", "split": "test", "mse": result["test"]["mse"],
                                "mae": result["test"]["mae"], "a0_output_mse": ""})
    if not records:
        return
    with (root / "all_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# FSLD pilot", "", "Rank and checkpoint selection use VAL only; TEST is evaluation-only.", "",
             "This is a one-seed pilot, not a final statistical claim.", ""]
    for case in sorted({row["case"] for row in records}):
        lines.append(f"## {case}")
        for row in records:
            if row["case"] == case and row["split"] == "test":
                lines.append(f"- {row['phase']} {row['method']} rank={row['rank']}: MSE={float(row['mse']):.6f}, MAE={float(row['mae']):.6f}")
        lines.append("")
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
