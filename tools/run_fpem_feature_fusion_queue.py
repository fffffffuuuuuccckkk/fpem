"""Resumable O3/O4 F0-F3 pilot, with matched A0 beside every metric row."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")
VARIANTS = (("F1", "F1", False), ("F2", "F2", False),
            ("F3_F1_swap", "F1", True), ("F3_F2_swap", "F2", True))
FIELDS = ("case", "method", "ablation", "a0_mse", "a0_mae", "a0_reference_sha256",
          "val_mse", "val_mae", "test_mse", "test_mae", "test_inv_mse",
          "test_inv_mae", "best_epoch", "initial_zinv_prediction_mse",
          "fusion_to_inv_ratio", "source", "selection_note")


def write_summary(root: Path, output: Path, case: str, worker: int, selected_rows: dict):
    rows = []
    for method in ("O3", "O4"):
        old = selected_rows[(case, method)]
        completed = []
        for label, _, _ in VARIANTS:
            trial = output / case / method / label
            if (trial / "run_complete").is_file() and (trial / "result.json").is_file():
                completed.append((label, trial, json.loads((trial / "result.json").read_text())))
        if not completed:
            continue
        a0 = completed[0][2]["a0_test"]
        digest = completed[0][2]["reference_sha256"]
        if digest != old["reference_sha256"]:
            raise ValueError(f"F0/new A0 reference mismatch: {case} {method}")
        rows.append({"case": case, "method": method, "ablation": "F0",
                     "a0_mse": a0["mse"], "a0_mae": a0["mae"],
                     "a0_reference_sha256": digest,
                     "val_mse": old["val_mse"], "val_mae": old["val_mae"],
                     "test_mse": old["test_mse"], "test_mae": old["test_mae"],
                     "best_epoch": old["best_epoch_selected_by_val"],
                     "source": old["source"],
                     "selection_note": "legacy O3/O4 LR candidate selected on TEST; exploratory"})
        for label, trial, result in completed:
            if result["reference_sha256"] != digest:
                raise ValueError(f"reference changed within case: {case} {method}")
            rows.append({"case": case, "method": method, "ablation": label,
                         "a0_mse": a0["mse"], "a0_mae": a0["mae"],
                         "a0_reference_sha256": digest,
                         "val_mse": result["val"]["mse"], "val_mae": result["val"]["mae"],
                         "test_mse": result["test"]["mse"],
                         "test_mae": result["test"]["mae"],
                         "test_inv_mse": result["test"]["inv_mse"],
                         "test_inv_mae": result["test"]["inv_mae"],
                         "best_epoch": result["best_epoch"],
                         "initial_zinv_prediction_mse": result["initial_zinv_prediction_mse"],
                         "fusion_to_inv_ratio": result["test"]["fusion_to_inv_ratio"],
                         "source": str(trial),
                         "selection_note": "same F0 LR; epoch selected by VAL MSE; TEST final only"})
    target = output / f"worker{worker}_summary.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", default="results/fpem_orthogonal_spectral_full_testlr_20261020/selected_by_test.csv")
    parser.add_argument("--output", default="results/fpem_feature_fusion_pilot_20261021")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    with (root / args.selected).open(newline="", encoding="utf-8") as handle:
        selected = {(r["case"], r["method"]): r for r in csv.DictReader(handle)}
    for position, case in enumerate(CASES):
        if position % args.workers != args.worker:
            continue
        for method in ("O3", "O4"):
            old = selected[(case, method)]
            source = root / old["source"]
            if not (source / "run_complete").is_file():
                raise RuntimeError(f"F0 source incomplete: {source}")
            cli = json.loads((source / "run_config.json").read_text())["cli"]
            for label, fusion, swap in VARIANTS:
                trial = output / case / method / label
                trial.mkdir(parents=True, exist_ok=True)
                if (trial / "run_complete").is_file() and (trial / "result.json").is_file():
                    print(f"SKIP {case} {method} {label}", flush=True)
                    continue
                command = [sys.executable, str(root / "tools/run_fpem_feature_fusion.py"),
                           "--base_run_config", cli["base_run_config"],
                           "--output", str(trial), "--method", method,
                           "--fusion", fusion, "--variation_fraction",
                           str(cli["variation_fraction"]), "--lr_head", str(cli["lr_head"]),
                           "--lr_basis", str(cli["lr_basis"]), "--seed", str(cli["seed"]),
                           "--epochs", str(cli["epochs"]), "--gpu", str(args.gpu),
                           "--spectrum_cache", cli["spectrum_cache"]]
                if swap:
                    command.append("--swap")
                if cli.get("allow_legacy_reference_signature"):
                    command.append("--allow_legacy_reference_signature")
                if cli.get("train_backbone"):
                    command.append("--train_backbone")
                if args.max_batches:
                    command += ["--max_batches", str(args.max_batches)]
                for attempt in (1, 2):
                    print(f"START {case} {method} {label} attempt={attempt}", flush=True)
                    with (trial / "run.log").open("a", encoding="utf-8") as log:
                        log.write("COMMAND " + " ".join(command) + "\n")
                        log.flush()
                        outcome = subprocess.run(command, cwd=root, stdout=log,
                                                 stderr=subprocess.STDOUT)
                    if outcome.returncode == 0 and (trial / "run_complete").is_file():
                        print(f"DONE {case} {method} {label}", flush=True)
                        break
                    print(f"FAILED {case} {method} {label} rc={outcome.returncode}", flush=True)
                    if attempt == 1:
                        time.sleep(60)
                else:
                    (trial / "run_failed").write_text("two attempts failed; see run.log\n")
                write_summary(root, output, case, args.worker, selected)
        write_summary(root, output, case, args.worker, selected)
    (output / f"worker{args.worker}_complete").touch()
    if all((output / f"worker{index}_complete").is_file()
           for index in range(args.workers)):
        combined = []
        for index in range(args.workers):
            with (output / f"worker{index}_summary.csv").open(newline="", encoding="utf-8") as handle:
                combined.extend(csv.DictReader(handle))
        with (output / "all_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(combined)
        (output / "summary.txt").write_text(
            f"Completed rows: {len(combined)}; A0 MSE/MAE and checkpoint SHA are attached "
            "to every row. F0 source was previously selected by TEST; new F1/F2/F3 "
            "reuse its LR and select epoch by VAL only. Treat all comparisons as exploratory.\n",
            encoding="utf-8")
    print("QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
