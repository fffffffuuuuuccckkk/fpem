"""Read-only inventory of completed ModernTCN A0 references and TEST metrics."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    found = subprocess.check_output(
        ["find", str(args.results_root), "-path", "*moderntcn*",
         "-name", "shared_reference.pt", "-type", "f"], text=True
    ).splitlines()
    rows = []
    for name in found:
        reference = Path(name)
        try:
            payload = torch.load(reference, map_location="cpu", weights_only=False)
            signature = payload["signature"]
        except Exception:
            continue
        if signature.get("backbone") != "moderntcn":
            continue
        metrics_candidates = [
            reference.parent / "A0" / "metrics_and_diagnostics.json",
            reference.parent / "A0" / "A0" / "metrics_and_diagnostics.json",
        ]
        for metrics_path in metrics_candidates:
            if not metrics_path.is_file():
                continue
            metrics = json.loads(metrics_path.read_text())
            mse = metrics.get("MSE")
            mae = metrics.get("MAE")
            if mse is None or mae is None:
                continue
            rows.append(dict(
                dataset=signature["dataset_name"], pred_len=signature["pred_len"],
                test_mse=mse, test_mae=mae, reference_path=str(reference),
                metrics_path=str(metrics_path), warmup_epochs=signature.get("warmup_epochs"),
                d_model=signature.get("d_model"), d_ff=signature.get("d_ff"),
                n_heads=signature.get("n_heads"), e_layers=signature.get("e_layers"),
                dropout=signature.get("dropout"),
                modern_tcn_patch_size=signature.get("modern_tcn_patch_size"),
                modern_tcn_patch_stride=signature.get("modern_tcn_patch_stride"),
                modern_tcn_num_stages=signature.get("modern_tcn_num_stages"),
                modern_tcn_ffn_ratio=signature.get("modern_tcn_ffn_ratio"),
                modern_tcn_head_dropout=signature.get("modern_tcn_head_dropout"),
            ))
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print("references", len(found), "paired_TEST_A0_metrics", len(rows), "output", args.output)
    for row in sorted(rows, key=lambda item: (item["dataset"], int(item["pred_len"]))):
        print(row["dataset"], row["pred_len"], row["test_mse"], row["reference_path"])


if __name__ == "__main__":
    main()
