"""Stage selected iTransformer A0 checkpoints without changing source results."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    datasets = {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "ExchangeRate"}
    rows = [r for r in csv.DictReader(args.baseline_csv.open())
            if r["method"] == "iTransformer" and r["dataset"] in datasets
            and int(r["pred_len"]) in (48, 96, 192, 336, 720)]
    if len(rows) != 25:
        raise ValueError(f"Expected 25 iTransformer classic baselines, got {len(rows)}")
    server_prefix = {
        "server1": Path(),
        "server2": Path("server2_imports/gpu-47"),
        "server3": Path("server3_imports/gpu-41"),
    }
    staged = []
    for row in rows:
        dataset, horizon = row["dataset"], int(row["pred_len"])
        parent = (args.results_root / server_prefix[row["server"]] /
                  row["result_root"] / "iTransformer" / dataset /
                  f"pred_{horizon}" / "checkpoints")
        sources = list(parent.glob("*/checkpoint.pth"))
        target = (args.output_root / "baseline_checkpoints" / "itransformer" /
                  dataset / f"pred_{horizon}" / "checkpoint.pth")
        target.parent.mkdir(parents=True, exist_ok=True)
        if len(sources) == 1:
            source = sources[0].resolve()
            if not target.exists() and not target.is_symlink():
                target.symlink_to(source)
            elif target.resolve() != source:
                raise ValueError(f"Different checkpoint already staged: {target}")
            status = "available"
        elif len(sources) == 0:
            if not target.exists():
                status = "missing"
            else:
                status = "available_copied"
        else:
            raise ValueError(f"Ambiguous checkpoint sources: {parent}")
        staged.append(dict(
            backbone="itransformer", dataset=dataset, pred_len=horizon,
            test_mse=row["MSE"], test_mae=row["MAE"], checkpoint_path=str(target),
            source_server=row["server"], source_result_root=row["result_root"],
            selection_split="test", checkpoint_status=status,
        ))
        print(status, row["server"], dataset, horizon, parent)
    catalog = args.output_root / "selected_itransformer_a0_catalog.csv"
    with catalog.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(staged[0]))
        writer.writeheader()
        writer.writerows(staged)
    print("catalog", catalog)


if __name__ == "__main__":
    main()
