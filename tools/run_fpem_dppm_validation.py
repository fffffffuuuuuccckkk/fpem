"""Fair, frozen-A0 validation of the VAL-selected DPPM capacity.

E1 deliberately preserves the previous C-best prediction definition. In that
implementation the stable bank constructs the variation key; the final
candidate is the variation-bank prediction, not a sum of two forecasts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.dual_pattern_memory import PatternBank
from tools.run_fpem_dppm import BASE, CASES, normalized_record
from tools.run_fpem_dppm_next import (
    ALPHAS, block_shuffle, fit_memory, mse_mae, query_memory,
    chronological_oof, candidate_oof_metrics,
)
from tools.run_fpem_shared_factorization import load_experiment, write_json
from tools.run_predictive_env_iv_patchtst import IndexedDataset
from tools.run_fpem_smrva import collect_split

SOURCE = ROOT / "results/fpem_dual_pattern_memory/next_pilot"
DEFAULT_OUTPUT = "results/fpem_dual_pattern_memory/validation_pilot"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max_batches", type=int, default=0, help="smoke only")
    return parser.parse_args()


def bank_bytes(bank):
    return sum(int(value.numel() * value.element_size()) for value in bank.state().values()
               if isinstance(value, torch.Tensor))


def fit_control(name, train, blocks, settings, seed):
    if name == "E1-0_A0":
        return None
    if name == "E1-2_shuffled_xy":
        shuffled = dict(train)
        shuffled["truth"] = block_shuffle(train["truth"], blocks, seed + 211)
        return fit_memory(torch.ones(len(blocks), dtype=torch.bool), shuffled,
                          blocks, settings, seed)
    if name in ("E1-1_full", "E1-5_stable_only", "E1-6_variation_only"):
        return fit_memory(torch.ones(len(blocks), dtype=torch.bool), train,
                          blocks, settings, seed)
    count = (1 if name == "E1-3_global_linear" else
             int(settings["ms"]) + int(settings["mv"]))
    bank = PatternBank(count, train["truth"].shape[-1], train["key"].shape[-1],
                       ridge=float(settings["ridge"]), seed=seed)
    bank.fit(train["key"], train["truth"], blocks)
    return bank


def candidate(name, record, fitted, settings):
    if name == "E1-0_A0":
        return record["base"]
    if name == "E1-5_stable_only":
        return fitted[0].query(record["key"])["prediction"]
    if isinstance(fitted, tuple):
        return query_memory(record, *fitted, settings)["prediction"]
    return fitted.query(record["key"])["prediction"]


def metrics(record, prediction):
    mse, mae = mse_mae(prediction, record["truth"], record)
    return {"mse": mse, "mae": mae}


def select_alpha(val, name, fitted, settings):
    pred = candidate(name, val, fitted, settings)
    grid = {}
    for alpha in ALPHAS:
        mixed = val["base"] + alpha * (pred - val["base"])
        grid[str(alpha)] = metrics(val, mixed)["mse"]
    selected = min(ALPHAS, key=lambda alpha: grid[str(alpha)])
    return selected, grid


def run_control(name, train, blocks, val, test, settings, seed):
    started = time.perf_counter()
    fitted = fit_control(name, train, blocks, settings, seed)
    fit_seconds = time.perf_counter() - started
    alpha, grid = select_alpha(val, name, fitted, settings)
    result = {"control": name, "seed": seed, "alpha": alpha,
              "fit_seconds": fit_seconds, "alpha_grid_val_mse": grid}
    if fitted is None:
        result["parameters"] = 0
        result["bank_bytes"] = 0
    elif isinstance(fitted, tuple):
        result["parameters"] = sum(int(bank.mapping.numel()) for bank in fitted)
        result["bank_bytes"] = sum(bank_bytes(bank) for bank in fitted)
    else:
        result["parameters"] = int(fitted.mapping.numel())
        result["bank_bytes"] = bank_bytes(fitted)
    for split, record in (("train", train), ("val", val), ("test", test)):
        raw = candidate(name, record, fitted, settings)
        final = record["base"] + alpha * (raw - record["base"])
        result[f"{split}_candidate"] = metrics(record, raw)
        result[f"{split}_final"] = metrics(record, final)
        result[f"{split}_stable_abs_effect"] = 0.0
        result[f"{split}_variation_abs_effect"] = 0.0
        if isinstance(fitted, tuple):
            stable = fitted[0].query(record["key"])["prediction"]
            result[f"{split}_stable_candidate"] = metrics(record, stable)
            result[f"{split}_stable_abs_effect"] = float(
                (alpha * (stable - record["base"])).abs().mean())
            result[f"{split}_variation_abs_effect"] = float(
                (alpha * (raw - stable)).abs().mean())
        elif fitted is not None:
            result[f"{split}_stable_abs_effect"] = float(
                (alpha * (raw - record["base"])).abs().mean())
    result["definition_note"] = (
        "C-best uses stable bank to construct variation key; the final candidate "
        "is variation-bank prediction, not an additive stable+variation forecast."
    )
    return result


def main():
    cli = parse_args()
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    output = ROOT / cli.output / cli.case / f"seed_{cli.seed}"
    output.mkdir(parents=True, exist_ok=True)
    source = SOURCE / cli.case
    chosen = json.loads((source / "stage_c_selected.json").read_text())
    chosen_name = chosen["choice"]
    selected_result = json.loads((source / f"{chosen_name}.json").read_text())
    settings = selected_result["settings"]
    source_config = json.loads((source / "run_config.json").read_text())
    cli.base_run_config = str(BASE / (f"{cli.case}_A0_complementary_gate_predictive_"
                             "sub0_orth0_refnone_fp0_rank8_bias-4/run_config.json"))
    cli.allow_legacy_reference_signature = cli.case.startswith("moderntcn")
    args, model, loaders, device, sha = load_experiment(cli)
    if sha != source_config["reference_sha256"]:
        raise ValueError("A0 checkpoint changed since C-best selection")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 must be frozen")
    prior = ROOT / "results/fpem_dual_pattern_memory/point_pilot" / cli.case
    bundle = torch.load(prior / "dual_pattern_memory.pt", map_location="cpu",
                        weights_only=False)
    if bundle["reference_sha256"] != sha:
        raise ValueError("A0 and prototype normalization checkpoint mismatch")
    cache = torch.load(ROOT / "results/fpem_predictive_memory/pilot_20261024" /
                       cli.case / "phase0_cache.pt", map_location="cpu",
                       weights_only=False)
    if cache["reference_sha256"] != sha:
        raise ValueError("TRAIN cache checkpoint mismatch")
    rows = cache["records"]
    train = normalized_record({"x": rows["x"], "target": rows["target"],
                               "base": rows["a0_pred"]}, bundle["floor"])
    blocks = (torch.arange(len(train["key"])) * 6 // len(train["key"])).clamp_max(5)
    val_set = loaders["val"].dataset.dataset
    val_loader = DataLoader(Subset(IndexedDataset(val_set,
        cycle_len=args.cyclenet_cycle_len), list(range(max(1, len(val_set) // 2)))),
        batch_size=args.batch_size, shuffle=False, num_workers=0)
    val = normalized_record(collect_split(model, val_loader, device, cli.max_batches),
                            bundle["floor"])
    test = normalized_record(collect_split(model, loaders["test"], device,
                                           cli.max_batches), bundle["floor"])
    write_json(output / "run_config.json", {
        "case": cli.case, "seed": cli.seed, "reference_sha256": sha,
        "capacity_choice": chosen_name, "settings": settings,
        "selection": "same first-half VAL alpha grid per control; TEST report only",
        "causality": "TRAIN-only banks; shuffled XY within TRAIN chronological blocks",
        "scope": "E1 point-only. A0 TRAIN predictions may be in-sample.",
    })
    names = ("E1-0_A0", "E1-1_full", "E1-2_shuffled_xy",
             "E1-3_global_linear", "E1-4_single_bank",
             "E1-5_stable_only", "E1-6_variation_only")
    results = {}
    for name in names:
        path = output / f"{name}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["seed"] != cli.seed or result["control"] != name:
                raise ValueError(f"mismatched result: {path}")
        else:
            result = run_control(name, train, blocks, val, test, settings, cli.seed)
            write_json(path, result)
        results[name] = result
        print(json.dumps({"control": name, "val": result["val_final"]["mse"],
                          "test": result["test_final"]["mse"]}), flush=True)
    with (output / "all_results.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ("control", "seed", "alpha", "val_mse", "val_mae",
                   "test_mse", "test_mae", "train_candidate_mse",
                   "val_candidate_mse", "test_candidate_mse", "parameters",
                   "bank_bytes", "fit_seconds", "stable_abs_effect",
                   "variation_abs_effect")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({"control": name, "seed": cli.seed,
                "alpha": result["alpha"], "val_mse": result["val_final"]["mse"],
                "val_mae": result["val_final"]["mae"],
                "test_mse": result["test_final"]["mse"],
                "test_mae": result["test_final"]["mae"],
                "train_candidate_mse": result["train_candidate"]["mse"],
                "val_candidate_mse": result["val_candidate"]["mse"],
                "test_candidate_mse": result["test_candidate"]["mse"],
                "parameters": result["parameters"],
                "bank_bytes": result["bank_bytes"],
                "fit_seconds": result["fit_seconds"],
                "stable_abs_effect": result["test_stable_abs_effect"],
                "variation_abs_effect": result["test_variation_abs_effect"]})
    full = results["E1-1_full"]["val_final"]["mse"]
    reference = ["E1-2_shuffled_xy", "E1-3_global_linear",
                 "E1-4_single_bank", "E1-5_stable_only"]
    passes = all(full < results[name]["val_final"]["mse"] - 1e-8
                 for name in reference)
    write_json(output / "E1_decision.json", {"full_val_mse": full,
        "comparator_val_mse": {name: results[name]["val_final"]["mse"]
                               for name in reference},
        "passes_for_E2": passes,
        "rule": "C-best must beat all simple controls on first-half VAL; TEST cannot trigger E2"})
    (output / "E1_complete").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
