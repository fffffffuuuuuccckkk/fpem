"""Produce complete single-seed H-statistics pilot table, VAL ranking first."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    root = Path(parser.parse_args().root)
    rows = []
    for file in root.glob("*/**/result.json"):
        result = json.loads(file.read_text())
        case = file.parents[1].name
        baseline = result["method"] in ("a0", "a1", "gate", "layernorm", "groupnorm", "linear", "mlp")
        rows.append({
            "case": case, "method": result["method"], "groups": result["groups"],
            "train_backbone": result["train_backbone"], "baseline": baseline,
            "best_epoch": result["best_epoch"], "val_mse": result["val"]["mse"],
            "val_mae": result["val"]["mae"], "test_mse": result["test"]["mse"],
            "test_mae": result["test"]["mae"],
            "worst_val_block_mse": result["val"].get("worst_block_mse", ""),
            "trainable_parameters": result["trainable_parameters"],
            "training_seconds": result.get("training_seconds", 0),
            "decomposition_ms_per_batch": result.get("decomposition_milliseconds_per_batch", 0),
            "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes", 0),
            "reference_sha256": result["reference_sha256"],
        })
    rows.sort(key=lambda row: (row["case"], float(row["val_mse"])))
    if not rows:
        return
    with (root / "all_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cases = defaultdict(list)
    for row in rows:
        cases[row["case"]].append(row)
    lines = ["# FPEM Hidden Statistical Factorization — single-seed pilot", "",
             "All ranks/methods/checkpoints are selected by VAL; TEST is final assessment only.",
             "No EIIL, GRL, environment labels, A0 output bypass or probability heads are used.",
             "The matched-parameter controls are approximate and parameter counts are recorded.", ""]
    for case, group in sorted(cases.items()):
        lines += [f"## {case}", "", "VAL ordering (lower is better):", ""]
        for item in group:
            lines.append(f"- {item['method']} G={item['groups']} joint={item['train_backbone']}: "
                         f"VAL {float(item['val_mse']):.6f}; TEST {float(item['test_mse']):.6f}/{float(item['test_mae']):.6f}")
        lines.append("")
    lines += ["## Scope", "", "P0/P1 Zinv-only pilot; Zvar γβ/CRPS, 3-seed confirmation, P2/P3 methods and multi-dataset sweep are not included.", ""]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
