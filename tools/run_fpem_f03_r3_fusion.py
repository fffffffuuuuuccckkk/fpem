"""Compare a fixed VAL-selected F03 anchor with the original R3 fusion."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.predictive_env.direct_invariant_forecaster import DirectInvariantForecaster
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from models.predictive_env.dual_pattern_memory.forecast_fusion import (
    prototype_segment_reliability, variation_forecast)
from models.predictive_env.dual_pattern_memory.prototype_bank import normalized_pairs
from tools.run_fpem_dppm_g1_r3_full import (
    LR, OFFSETS, history_key, metric, residual_ridge, subset)
from tools.run_fpem_shared_factorization_g1r3 import load_experiment, write_json
from tools.run_predictive_env_iv_patchtst import IndexedDataset


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--base_run_config", required=True)
    p.add_argument("--shared_selection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--train_stride", type=int, default=16)
    p.add_argument("--val_stride", type=int, default=4)
    p.add_argument("--test_stride", type=int, default=1)
    p.add_argument("--allow_legacy_reference_signature", action="store_true")
    return p.parse_args()


@torch.no_grad()
def collect(model, head, dataset, args, device, stride):
    wrapped = IndexedDataset(dataset, cycle_len=args.cyclenet_cycle_len)
    loader = DataLoader(Subset(wrapped, range(0, len(dataset), stride)),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    parts = {name: [] for name in ("x", "target", "base", "a0", "ids")}
    for x, y, ids, cycle in loader:
        x, cycle = x.float().to(device), cycle.to(device)
        hidden, mean, std = model.encode(x, cycle)
        base = head(hidden).transpose(1, 2) * std + mean
        a0 = model.frozen_h_head(hidden, cycle).transpose(1, 2) * std + mean
        for name, value in (("x", x), ("target", y[:, -args.pred_len:]),
                            ("base", base), ("a0", a0), ("ids", ids)):
            parts[name].append(value.detach().cpu())
    return {name: torch.cat(values) for name, values in parts.items()}


def record(raw, floor):
    truth, mean, scale = normalized_pairs(raw["x"], raw["target"], floor)
    return {"key": history_key(raw["x"], 64).cpu().float(),
            "truth": truth.cpu().float(),
            "base": ((raw["base"].float() - mean) / scale).transpose(1, 2).cpu(),
            "a0": ((raw["a0"].float() - mean) / scale).transpose(1, 2).cpu(),
            "scale": scale.cpu()}


def predict_full(data, learner, reliability, offset):
    query = learner.query(data["key"], offset)
    empty = data["base"].new_zeros(data["base"].shape[-1], 0)
    full, beta = variation_forecast(data["base"], query["candidate"],
                                    query["weight"], reliability, empty, True, 8)
    return full, query, beta


def main():
    cli = arguments()
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete").is_file():
        print("SKIP complete", output, flush=True)
        return
    selection_file = Path(cli.shared_selection)
    selected = json.loads(selection_file.read_text(encoding="utf-8"))
    if selected["seed"] != cli.seed:
        raise ValueError("R3 seed and F03 head seed must match")
    checkpoint = Path(selected["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    torch.manual_seed(cli.seed)
    torch.set_num_threads(4)
    started = time.monotonic()
    # The A0 checkpoint has its own fixed seed; R3's seed only controls its
    # new prototype parameters and must not rewrite the A0 identity.
    load_cli = argparse.Namespace(**vars(cli))
    load_cli.seed = json.loads(Path(cli.base_run_config).read_text(encoding="utf-8"))["seed"]
    args, model, loaders, device, sha = load_experiment(load_cli)
    torch.manual_seed(cli.seed)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 must remain frozen")
    if (selected["reference_sha256"] != sha or
            selected["backbone"] != args.backbone or
            selected["dataset"] != args.dataset_name or
            selected["pred_len"] != args.pred_len):
        raise ValueError("F03 selection is not paired with this A0 and dataset")
    sample = next(iter(loaders["train"]))
    with torch.no_grad():
        hidden, _, _ = model.encode(sample[0].float().to(device), sample[3].to(device))
    head = DirectInvariantForecaster(hidden.shape[-2], hidden.shape[-1],
                                    args.pred_len, pretrained_head=model.h_forecast_head).to(device)
    head.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    head.eval()
    head.requires_grad_(False)
    datasets = {name: loader.dataset.dataset for name, loader in loaders.items()}
    raw_train = collect(model, head, datasets["train"], args, device, cli.train_stride)
    starts = raw_train["ids"].long()
    first_val_history = len(datasets["train"].data_x) - args.seq_len
    valid = starts + args.seq_len + args.pred_len <= first_val_history
    raw_train = subset(raw_train, valid)
    starts = starts[valid]
    historical_std = raw_train["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (historical_std.median(0).values * .25).clamp_min(.01)
    train = record(raw_train, floor)
    del raw_train
    raw_val = collect(model, head, datasets["val"], args, device, cli.val_stride)
    val_ids = raw_val["ids"].long()
    val = record(raw_val, floor)
    del raw_val
    cut = len(datasets["val"].data_x) // 2
    first_test_target_start = len(datasets["val"].data_x)
    early_mask = val_ids + args.seq_len + args.pred_len <= cut
    select_mask = (val_ids >= cut) & (val_ids + args.seq_len + args.pred_len <= first_test_target_start)
    early, choice_val = subset(val, early_mask), subset(val, select_mask)
    fit_train = train
    protocol = "R3 TRAIN fit; early purged VAL reliability; later purged VAL NULL; TEST report-only"
    if min(len(early["key"]), len(choice_val["key"])) < 8:
        split = int(starts.max()) // 2
        fit_mask = starts + args.seq_len + args.pred_len <= split
        hold_mask = (starts >= split) & (starts + args.seq_len + args.pred_len <= first_val_history)
        select_mask = val_ids + args.seq_len + args.pred_len <= first_test_target_start
        if min(int(fit_mask.sum()), int(hold_mask.sum()), int(select_mask.sum())) < 8:
            raise ValueError("insufficient purged TRAIN/VAL support")
        if int((starts[fit_mask] + args.seq_len + args.pred_len).max()) > int(starts[hold_mask].min()):
            raise AssertionError("TRAIN fit future overlaps holdout history")
        fit_train = subset(train, fit_mask)
        early = subset(train, hold_mask)
        choice_val = subset(val, select_mask)
        fit_starts = starts[fit_mask]
        protocol = "R3 early TRAIN fit; later purged TRAIN reliability; eligible VAL NULL; TEST report-only"
    else:
        if int((val_ids[early_mask] + args.seq_len + args.pred_len).max()) > int(val_ids[select_mask].min()):
            raise AssertionError("early VAL future overlaps selection history")
        fit_starts = starts
    if int((val_ids[select_mask] + args.seq_len + args.pred_len).max()) > first_test_target_start:
        raise AssertionError("selection VAL future overlaps TEST targets")
    head_lr, dict_ratio, basis_ratio, route_ratio = LR[args.backbone]
    optimizer_config = {"head_lr": head_lr, "dictionary_ratio": dict_ratio,
                        "basis_ratio": basis_ratio, "route_ratio": route_ratio,
                        "optimizer": "adam", "schedule": "constant"}
    reference = residual_ridge(fit_train["key"], fit_train["truth"], 100.)
    learner = RarePatternLearner("R3", patterns=8, rank=8, seed=cli.seed,
                                 epochs=10, optimizer_config=optimizer_config)
    learner.fit(fit_train["key"], fit_train["truth"], {
        "reference_map": reference, "starts": fit_starts,
        "span": args.seq_len + args.pred_len, "device": "cpu",
        "validation": (early["key"], early["truth"])})
    early_query = learner.query(early["key"])
    reliability, evidence = prototype_segment_reliability(
        early["base"], early_query["candidate"], early["truth"],
        early_query["weight"], 8)
    choice = None
    for offset in OFFSETS:
        pred, _, _ = predict_full(choice_val, learner, reliability, offset)
        value = metric(choice_val, pred)
        if choice is None or value["mse"] < choice["val_mse"]:
            choice = {"offset": offset, "val_mse": value["mse"], "val_mae": value["mae"]}
    raw_test = collect(model, head, datasets["test"], args, device, cli.test_stride)
    test = record(raw_test, floor)
    del raw_test
    f03_test = metric(test, test["base"])
    if abs(f03_test["mse"] - selected["test"]["mse"]) > 1e-5:
        raise AssertionError("F03 anchor differs from its selected checkpoint result")
    full, query, beta = predict_full(test, learner, reliability, choice["offset"])
    result = {"backbone": args.backbone, "dataset": args.dataset_name,
              "pred_len": args.pred_len, "seed": cli.seed,
              "a0": metric(test, test["a0"]), "f03": f03_test,
              "f03_r3": metric(test, full), "val": choice,
              "f03_val": metric(choice_val, choice_val["base"]),
              "r3_optimizer": optimizer_config,
              "reliability_mean": float(reliability.mean()),
              "actual_effect_mean": float((full - test["base"]).abs().mean()),
              "active_sample_ratio": float((beta.abs().sum((1, 2)) > 0).float().mean()),
              "null_mean": float(query["null_weight"].mean()),
              "reference_sha256": sha,
              "shared_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "shared_selection": str(selection_file),
              "fit_train_samples": len(fit_train["key"]),
              "reliability_samples": len(early["key"]),
              "selection_samples": len(choice_val["key"]),
              "test_samples": len(test["key"]),
              "elapsed_seconds": time.monotonic() - started,
              "protocol": protocol,
              "caveat": "Exploratory: F03 LR/epoch used VAL; R3 reliability/NULL also use purged VAL or TRAIN holdout; A0 catalog TEST selection is historical."}
    torch.save({"rare": learner.export(), "shared_head": head.state_dict(),
                "reliability": reliability, "offset": choice["offset"],
                "reference_sha256": sha}, output / "model.pt")
    write_json(output / "run_config.json", {"cli": vars(cli),
               "shared_selection": selected, "reference_sha256": sha,
               "r3_optimizer": optimizer_config})
    write_json(output / "metrics.json", result)
    write_json(output / "reliability_evidence.json", evidence)
    (output / "complete").write_text("complete\n", encoding="utf-8")
    print("F03+R3 COMPLETE", args.dataset_name, args.pred_len, cli.seed,
          result["f03"], result["f03_r3"], flush=True)


if __name__ == "__main__":
    main()
