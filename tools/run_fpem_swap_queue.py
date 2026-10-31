"""Resumable three-case O3/O4 swap-residual pilot; no new A0 training."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


CASES = ("patchtst_ETTh1_720", "itransformer_ETTm1_720", "moderntcn_ETTm2_720")
ABLATIONS = ("E1", "E2", "E3", "E4", "E5", "E6")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", default="results/fpem_orthogonal_spectral_full_testlr_20261020/selected_by_test.csv")
    parser.add_argument("--output", default="results/fpem_swap_residual_pilot_20261021")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=ABLATIONS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    selected = root / args.selected
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    with selected.open(newline="", encoding="utf-8") as handle:
        selected_rows = list(csv.DictReader(handle))
    rows = {(row["case"], row["method"]): row for row in selected_rows}
    report = []
    for position, case in enumerate(CASES):
        if position % args.workers != args.worker:
            continue
        for method in ("O3", "O4"):
            row = rows[(case, method)]
            run = json.loads((root / row["source"] / "run_config.json").read_text())
            old_cli = run["cli"]
            if not (root / row["source"] / "run_complete").exists():
                raise RuntimeError(f"E0 source incomplete: {row['source']}")
            report.append({"case": case, "method": method, "ablation": "E0",
                           "val_mse": row["val_mse"], "test_mse": row["test_mse"],
                           "test_mae": row["test_mae"], "CRPS": "",
                           "source": row["source"]})
            for ablation in args.ablations:
                trial = output / case / method / ablation
                trial.mkdir(parents=True, exist_ok=True)
                if (trial / "run_complete").exists() and (trial / "result.json").exists():
                    print(f"SKIP complete {case} {method} {ablation}", flush=True)
                else:
                    cmd = [sys.executable, str(root / "tools/run_fpem_swap_residual.py"),
                       "--base_run_config", old_cli["base_run_config"],
                       "--output", str(trial), "--method", method,
                       "--ablation", ablation, "--variation_fraction",
                       str(old_cli["variation_fraction"]), "--lr_head",
                       str(old_cli["lr_head"]), "--lr_basis", str(old_cli["lr_basis"]),
                       "--seed", str(old_cli["seed"]), "--epochs", str(old_cli["epochs"]),
                       "--gpu", str(args.gpu), "--spectrum_cache", old_cli["spectrum_cache"]]
                    if old_cli.get("allow_legacy_reference_signature"):
                        cmd.append("--allow_legacy_reference_signature")
                    if old_cli.get("train_backbone"):
                        cmd.append("--train_backbone")
                    if args.max_batches:
                        cmd.extend(("--max_batches", str(args.max_batches)))
                    for attempt in range(1, 4):
                        print(f"START {case} {method} {ablation} attempt={attempt}", flush=True)
                        with (trial / "run.log").open("a", encoding="utf-8") as log:
                            log.write("COMMAND " + " ".join(cmd) + "\n")
                            log.flush()
                            result = subprocess.run(cmd, cwd=root, stdout=log,
                                                    stderr=subprocess.STDOUT)
                        if result.returncode == 0 and (trial / "run_complete").exists():
                            print(f"DONE {case} {method} {ablation}", flush=True)
                            break
                        print(f"FAILED {case} {method} {ablation} rc={result.returncode}", flush=True)
                        if attempt < 3:
                            time.sleep(60)
                    else:
                        (trial / "run_failed").write_text("three attempts failed; see run.log\n")
                if (trial / "run_complete").exists() and (trial / "result.json").exists():
                    result = json.loads((trial / "result.json").read_text())
                    report.append({"case": case, "method": method, "ablation": ablation,
                                   "val_mse": result["val"]["mse"],
                                   "test_mse": result["test"]["mse"],
                                   "test_mae": result["test"]["mae"],
                                   "CRPS": result["test"].get("CRPS", ""),
                                   "source": str(trial)})
                with (output / f"worker{args.worker}_summary.csv").open(
                        "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=(
                        "case", "method", "ablation", "val_mse", "test_mse",
                        "test_mae", "CRPS", "source"))
                    writer.writeheader(); writer.writerows(report)
    print("QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
