"""Create pred=24 iTransformer/ModernTCN catalogs after A0 jobs finish."""

import argparse
import csv
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    datasets = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ExchangeRate")
    for backbone in ("itransformer", "moderntcn"):
        rows = []
        for dataset in datasets:
            if backbone == "itransformer":
                source_root = args.output_root/"tslib_24_baselines"/"iTransformer"/dataset/"pred_24"
                metric = source_root/"metrics.json"
                sources = list((source_root/"checkpoints").glob("*/checkpoint.pth"))
                if not (source_root/"run_complete").is_file() or len(sources) != 1:
                    raise ValueError(f"Missing completed classic baseline: {source_root}")
                source = sources[0].resolve()
                target = args.output_root/"baseline_checkpoints"/"itransformer"/dataset/"pred_24"/"checkpoint.pth"
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() and not target.is_symlink():
                    target.symlink_to(source)
                elif target.resolve() != source:
                    raise ValueError(f"Different staged checkpoint: {target}")
                m = json.loads(metric.read_text())
                rows.append(dict(backbone=backbone, dataset=dataset, pred_len=24,
                                 test_mse=m["MSE"], test_mae=m["MAE"],
                                 checkpoint_path=str(target), source_server="server3",
                                 source_result_root=str(source_root), selection_split="test"))
            else:
                source_root = args.output_root/"moderntcn_a0_pred24"/dataset
                metric = source_root/"A0"/"metrics_and_diagnostics.json"
                reference = args.output_root/"references"/"moderntcn"/dataset/"pred_24.pt"
                if not metric.is_file() or not reference.is_file():
                    raise ValueError(f"Missing completed ModernTCN A0: {source_root}")
                m = json.loads(metric.read_text())
                signature = torch.load(reference, map_location="cpu", weights_only=False)["signature"]
                if signature["backbone"] != backbone or signature["pred_len"] != 24:
                    raise ValueError(f"Reference mismatch: {reference}")
                rows.append(dict(backbone=backbone, dataset=dataset, pred_len=24,
                                 test_mse=m["MSE"], test_mae=m["MAE"],
                                 reference_path=str(reference), source_server="server3",
                                 source_metrics_path=str(metric), selection_split="test"))
        catalog = args.output_root/f"selected_{backbone}_a0_catalog.csv"
        with catalog.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(backbone, "catalog", catalog)
        for row in rows:
            print(backbone, row["dataset"], row["test_mse"], row["test_mae"])


if __name__ == "__main__":
    main()
