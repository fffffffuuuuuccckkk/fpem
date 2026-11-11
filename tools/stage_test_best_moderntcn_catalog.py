"""Select lowest recorded TEST A0 MSE per ModernTCN case and stage references."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server1-inventory", type=Path, required=True)
    parser.add_argument("--server2-inventory", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    datasets = {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "ExchangeRate"}
    candidates = []
    for server, file in (("server1", args.server1_inventory),
                         ("server2", args.server2_inventory)):
        for row in csv.DictReader(file.open()):
            if row["dataset"] not in datasets or int(row["pred_len"]) not in (48, 96, 192, 336, 720):
                continue
            if row["warmup_epochs"] != "3":
                continue
            row["source_server"] = server
            candidates.append(row)
    selected = []
    for dataset in sorted(datasets):
        for horizon in (48, 96, 192, 336, 720):
            options = [r for r in candidates if r["dataset"] == dataset
                       and int(r["pred_len"]) == horizon]
            if not options:
                raise ValueError(f"No completed ModernTCN A0 reference: {dataset}/{horizon}")
            best = min(options, key=lambda r: float(r["test_mse"]))
            target = (args.output_root / "baseline_references" / "moderntcn" /
                      dataset / f"pred_{horizon}.pt")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = args.project_root / best["reference_path"]
            if best["source_server"] == "server1":
                if not source.is_file():
                    raise ValueError(f"Missing primary reference: {source}")
                if not target.exists() and not target.is_symlink():
                    target.symlink_to(source)
                elif target.resolve() != source.resolve():
                    raise ValueError(f"Different reference staged at {target}")
                status = "available"
            else:
                status = "available_copied" if target.is_file() else "missing_server2_copy"
            selected.append(dict(
                backbone="moderntcn", dataset=dataset, pred_len=horizon,
                test_mse=best["test_mse"], test_mae=best["test_mae"],
                reference_path=str(target), source_server=best["source_server"],
                source_reference_path=str(source), source_metrics_path=best["metrics_path"],
                selection_split="test", reference_status=status,
                warmup_epochs=best["warmup_epochs"], d_model=best["d_model"],
                d_ff=best["d_ff"], n_heads=best["n_heads"],
                e_layers=best["e_layers"], dropout=best["dropout"],
                modern_tcn_patch_size=best["modern_tcn_patch_size"],
                modern_tcn_patch_stride=best["modern_tcn_patch_stride"],
                modern_tcn_num_stages=best["modern_tcn_num_stages"],
                modern_tcn_ffn_ratio=best["modern_tcn_ffn_ratio"],
                modern_tcn_head_dropout=best["modern_tcn_head_dropout"],
            ))
            print(status, dataset, horizon, best["test_mse"], best["source_server"])
    catalog = args.output_root / "selected_moderntcn_a0_catalog.csv"
    with catalog.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)
    print("catalog", catalog)


if __name__ == "__main__":
    main()
