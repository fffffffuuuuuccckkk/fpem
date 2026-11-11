"""Frozen-A0 F03 robustness: independent shuffle seeds and VAL-selected LR."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path

import torch

from models.predictive_env.direct_invariant_forecaster import DirectInvariantForecaster
from tools.run_fpem_direct_invariant_search import collect, metrics, predict
from tools.run_fpem_f03_lr_search import train_one
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--seeds", default="2022,2023,2024")
    p.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    p.add_argument("--train_stride", type=int, default=16)
    p.add_argument("--val_stride", type=int, default=4)
    p.add_argument("--test_stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    cli = arguments()
    seeds = [int(value) for value in cli.seeds.split(",")]
    lrs = [float(value) for value in cli.lrs.split(",")]
    if len(set(seeds)) != len(seeds) or len(set(lrs)) != len(lrs):
        raise ValueError("duplicate seed or LR")
    out = Path(cli.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    args, model, loaders, device, sha = load_experiment(cli)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("A0 must be frozen")
    datasets = {name: loader.dataset.dataset for name, loader in loaders.items()}
    data = {name: collect(model, datasets[name], args, device,
                          {"train": cli.train_stride, "val": cli.val_stride,
                           "test": cli.test_stride}[name])
            for name in ("train", "val", "test")}
    a0_val = metrics(data["val"], data["val"]["base"])
    a0_test = metrics(data["test"], data["test"]["base"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    selected = []
    trial_rows = []
    token_count, dim = data["train"]["hidden"].shape[-2:]
    for seed in seeds:
        cli.seed = seed
        folder = out / f"seed_{seed}"
        selection_file = folder / "selection.json"
        if (folder / "complete").is_file() and selection_file.is_file():
            choice = json.loads(selection_file.read_text(encoding="utf-8"))
            selected.append(choice)
            trial_rows.extend(choice["trials"])
            continue
        folder.mkdir(parents=True, exist_ok=True)
        candidates = []
        for lr in lrs:
            result = train_one(lr, data, model, args, cli, device, folder / f"lr_{lr:g}")
            candidates.append(result)
            trial_rows.append({"seed": seed, "lr": lr,
                               "best_epoch": result["best_epoch"],
                               "val_mse": result["val"]["mse"],
                               "val_mae": result["val"]["mae"]})
        winner = min(candidates, key=lambda item: item["val"]["mse"])
        net = DirectInvariantForecaster(token_count, dim, args.pred_len,
                                       pretrained_head=model.h_forecast_head).to(device)
        net.load_state_dict(torch.load(winner["state_file"], map_location=device,
                                       weights_only=False))
        test = metrics(data["test"], predict(net, data["test"], device))
        choice = {"seed": seed, "selected_lr": winner["lr"],
                  "best_epoch": winner["best_epoch"], "val": winner["val"],
                  "test": test, "a0_val": a0_val, "a0_test": a0_test,
                  "trials": [row for row in trial_rows if row["seed"] == seed],
                  "checkpoint": winner["state_file"], "reference_sha256": sha,
                  "git_commit": commit, "backbone": args.backbone,
                  "dataset": args.dataset_name, "pred_len": args.pred_len,
                  "selection": "VAL selects LR and epoch; TEST report-only",
                  "model": "A0 frozen, original forecast head trained"}
        write_json(selection_file, choice)
        (folder / "complete").write_text("complete\n", encoding="utf-8")
        selected.append(choice)
        print("SEED COMPLETE", args.dataset_name, args.pred_len, seed,
              winner["lr"], test["mse"], flush=True)
    with (out / "all_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "lr", "best_epoch",
                                                   "val_mse", "val_mae"])
        writer.writeheader()
        writer.writerows(trial_rows)
    test_mse = [item["test"]["mse"] for item in selected]
    summary = {"backbone": args.backbone, "dataset": args.dataset_name,
               "pred_len": args.pred_len, "reference_sha256": sha,
               "git_commit": commit, "seeds": seeds, "lrs": lrs,
               "epochs": cli.epochs, "a0_val": a0_val, "a0_test": a0_test,
               "selected": selected,
               "test_mse_mean": statistics.mean(test_mse),
               "test_mse_std": statistics.stdev(test_mse) if len(test_mse) > 1 else 0.0,
               "test_win_count": sum(value < a0_test["mse"] for value in test_mse),
               "selection": "VAL LR/epoch; TEST exploratory reporting"}
    write_json(out / "summary.json", summary)
    (out / "complete").write_text("complete\n", encoding="utf-8")
    print("F03 ROBUSTNESS COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
