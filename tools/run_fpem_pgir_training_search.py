"""Restartable, one-case-per-GPU PGIR training-trick search.

Each checkpoint is selected by validation MSE; test is reported only after
selection. The frozen A0, TRAIN prototype bank and OOF gain labels are reused.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ("moderntcn_ETTm2_720", "patchtst_ETTh1_720", "itransformer_ETTm1_720")


def read_result(directory):
    if not (directory / "run_complete").exists():
        return None
    return json.loads((directory / "result.json").read_text())


def val_mse(result):
    return float(result["val"]["mse"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--selected", default="results/fpem_orthogonal_spectral_full_testlr_20261020/selected_by_test.csv")
    parser.add_argument("--baseline", default="results/fpem_pgir/pilot_20261022")
    parser.add_argument("--previous", default="results/fpem_pgir/tricks_20261023")
    parser.add_argument("--output", default="results/fpem_pgir/training_tricks_20261024")
    parser.add_argument("--max_batches", type=int, default=0, help="smoke only")
    cli = parser.parse_args()
    case = cli.case
    baseline = ROOT / cli.baseline / case
    previous = ROOT / cli.previous / case
    output = ROOT / cli.output
    case_dir = output / case
    case_dir.mkdir(parents=True, exist_ok=True)
    selected = {(r["case"], r["method"]): r for r in
                csv.DictReader((ROOT / cli.selected).open(newline=""))}
    saved = json.loads((ROOT / selected[(case, "O3")]["source"] / "run_config.json").read_text())["cli"]
    if not (baseline / "P2" / "run_complete").exists():
        raise RuntimeError(f"P2 baseline incomplete: {case}")
    if not (baseline / "P4" / "run_complete").exists():
        raise RuntimeError(f"TRAIN OOF prerequisite incomplete: {case}")
    prior_mode = "T5" if case.startswith("itransformer") else "T3"
    prior = read_result(previous / prior_mode)
    if prior is None:
        raise RuntimeError(f"prior {prior_mode} checkpoint missing: {case}")
    prior_config = json.loads((previous / prior_mode / "run_config.json").read_text())["cli"]
    flags = {key: bool(prior_config[key]) for key in
             ("local", "gain_supervision", "boundary", "recent")}
    base = {"lr_feature": 1e-3, "lr_head": 1e-3,
            "lambda_gain": float(prior_config["lambda_gain"]),
            "lambda_replace": float(prior_config["lambda_replace"]),
            "output_bias": 0.1, "epochs": 10,
            "schedule": "constant", "ema_decay": 0.0,
            "normalize_features": False}
    common = [sys.executable, str(ROOT / "tools/run_fpem_pgir_tricks.py"),
              "--base_run_config", saved["base_run_config"],
              "--bank_checkpoint", str(baseline / "prototype_bank.pt"),
              "--core_p4_dir", str(baseline / "P4"),
              "--gain_cache", str(previous / "oof_gain_labels.pt"),
              "--seed", str(saved["seed"]), "--gpu", str(cli.gpu),
              "--mode", prior_mode]
    if saved.get("allow_legacy_reference_signature"):
        common.append("--allow_legacy_reference_signature")
    if cli.max_batches:
        common += ["--max_batches", str(cli.max_batches)]
    for name, active in flags.items():
        if active:
            common.append("--" + name)

    def run(name, config, *, evaluate_checkpoint="", calibrate=False):
        trial = case_dir / name
        trial.mkdir(parents=True, exist_ok=True)
        old = read_result(trial)
        if old is not None:
            print(f"SKIP {case}/{name}", flush=True)
            return old
        command = common + ["--output", str(trial),
                            "--lr_feature", str(config["lr_feature"]),
                            "--lr_head", str(config["lr_head"]),
                            "--lambda_gain", str(config["lambda_gain"]),
                            "--lambda_replace", str(config["lambda_replace"]),
                            "--output_bias", str(config["output_bias"]),
                            "--epochs", str(config["epochs"]),
                            "--schedule", config["schedule"],
                            "--ema_decay", str(config["ema_decay"])]
        if config["normalize_features"]:
            command.append("--normalize_features")
        if evaluate_checkpoint:
            command += ["--evaluate_checkpoint", evaluate_checkpoint]
        if calibrate:
            command.append("--calibrate_threshold")
        (trial / "search_config.json").write_text(json.dumps(config, indent=2) + "\n")
        for attempt in (1, 2):
            print(f"START {case}/{name} attempt={attempt}", flush=True)
            with (trial / "run.log").open("a") as log:
                log.write("COMMAND " + " ".join(command) + "\n")
                log.flush()
                finished = subprocess.run(command, cwd=ROOT, stdout=log,
                                          stderr=subprocess.STDOUT)
            result = read_result(trial)
            if finished.returncode == 0 and result is not None:
                summarize(output, ROOT / cli.baseline, ROOT / cli.previous)
                return result
            print(f"FAILED {case}/{name} rc={finished.returncode}", flush=True)
            if attempt == 1:
                time.sleep(60)
        (trial / "run_failed").write_text("two failures; inspect run.log\n")
        raise RuntimeError(f"failed twice: {case}/{name}")

    candidates = [(val_mse(prior), "previous_" + prior_mode, base.copy())]
    if case.startswith("moderntcn"):
        # First round: six requested combinations, then the remaining 0.2
        # auxiliary weight at the best VAL-selected output-head LR.
        for head in (1e-3, 3e-3):
            for gain in (0.01, 0.05, 0.1):
                cfg = {**base, "lr_feature": 3e-4, "lr_head": head,
                       "lambda_gain": gain}
                name = f"R1_h{head:g}_g{gain:g}"
                candidates.append((val_mse(run(name, cfg)), name, cfg))
        best_head = min(candidates[1:], key=lambda row: row[0])[2]["lr_head"]
        cfg = {**base, "lr_feature": 3e-4, "lr_head": best_head,
               "lambda_gain": 0.2}
        candidates.append((val_mse(run("R1_g0.2", cfg)), "R1_g0.2", cfg))
    elif case.startswith("patchtst"):
        # Second round: coordinate search covers all requested bias and
        # replacement-cost values without a 3x4 Cartesian product.
        for bias in (0.01, 0.1, 0.3):
            cfg = {**base, "output_bias": bias}
            name = f"R2_bias{bias:g}"
            candidates.append((val_mse(run(name, cfg)), name, cfg))
        best_bias = min(candidates[1:], key=lambda row: row[0])[2]["output_bias"]
        for replace in (0., 1e-4, 1e-3, 3e-3):
            cfg = {**base, "output_bias": best_bias,
                   "lambda_replace": replace}
            name = f"R2_replace{replace:g}"
            candidates.append((val_mse(run(name, cfg)), name, cfg))
    # The third round deliberately uses VAL, not TEST, for its parent.
    parent = min(candidates, key=lambda row: row[0])
    (case_dir / "stage3_parent.json").write_text(json.dumps(
        {"name": parent[1], "val_mse": parent[0], "config": parent[2]}, indent=2) + "\n")
    chosen = parent[2]
    stage3 = [
        ("R3_cosine10", {**chosen, "epochs": 10, "schedule": "cosine_warmup"}),
        ("R3_cosine20", {**chosen, "epochs": 20, "schedule": "cosine_warmup"}),
        ("R3_cosine30", {**chosen, "epochs": 30, "schedule": "cosine_warmup"}),
        ("R3_norm20", {**chosen, "epochs": 20, "schedule": "cosine_warmup",
                        "normalize_features": True}),
        ("R3_ema099_20", {**chosen, "epochs": 20,
                            "schedule": "cosine_warmup", "ema_decay": 0.99}),
        ("R3_ema0995_20", {**chosen, "epochs": 20,
                             "schedule": "cosine_warmup", "ema_decay": 0.995}),
    ]
    for name, config in stage3:
        candidates.append((val_mse(run(name, config)), name, config))
    stage3_best = min(candidates, key=lambda row: row[0])
    source = previous / prior_mode / "best.pt" if stage3_best[1].startswith("previous_") else case_dir / stage3_best[1] / "best.pt"
    run("threshold_calibration", stage3_best[2],
        evaluate_checkpoint=str(source), calibrate=True)
    (case_dir / "queue_complete").touch()
    summarize(output, ROOT / cli.baseline, ROOT / cli.previous)
    print(f"COMPLETE {case}", flush=True)


def summarize(output, baseline_root, previous_root):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".summary.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for case in CASES:
            base = read_result(baseline_root / case / "P2")
            if base is None:
                continue
            for mode in ("P2", "T3" if not case.startswith("itransformer") else "T5"):
                data = base if mode == "P2" else read_result(previous_root / case / mode)
                if data is None:
                    continue
                rows.append({"case": case, "trial": "previous_" + mode,
                             "a0_mse": data["test"]["a0_mse"],
                             "val_mse": data["val"]["mse"],
                             "test_mse": data["test"]["mse"],
                             "test_mae": data["test"]["mae"],
                             "replaced_per_series": data["test"].get("mean_replaced_patches_per_series", ""),
                             "fraction_intervened": data["test"].get("fraction_series_intervened", ""),
                             "best_epoch": data["best_epoch"]})
            for trial in sorted((output / case).iterdir()) if (output / case).exists() else []:
                if not trial.is_dir():
                    continue
                data = read_result(trial)
                if data is None:
                    continue
                rows.append({"case": case, "trial": trial.name,
                             "a0_mse": data["test"]["a0_mse"],
                             "val_mse": data["val"]["mse"],
                             "test_mse": data["test"]["mse"],
                             "test_mae": data["test"]["mae"],
                             "replaced_per_series": data["test"].get("mean_replaced_patches_per_series", ""),
                             "fraction_intervened": data["test"].get("fraction_series_intervened", ""),
                             "best_epoch": data["best_epoch"]})
        fields = ("case", "trial", "a0_mse", "val_mse", "test_mse", "test_mae",
                  "replaced_per_series", "fraction_intervened", "best_epoch")
        tmp = output / "all_results.csv.tmp"
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, output / "all_results.csv")
        if all((output / case / "queue_complete").exists() for case in CASES):
            (output / "queue_complete").touch()


if __name__ == "__main__":
    main()
