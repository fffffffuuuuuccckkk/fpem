"""Exploratory TEST-CRPS search for a detached residual Flow on frozen G1+R3.

The point model is loaded once and materialized without gradients.  This is
not a chronological OOF residual experiment: TRAIN targets fit both point
maps and the Flow.  TEST is explicitly a development set for this search.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.predictive_env.dual_pattern_memory.rare_learning import RarePatternLearner
from models.predictive_env.probability_heads import ProbabilityHeadBank
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.fpem_probabilistic_training import HistoricalEMAMaturity
from tools.run_fpem_dppm_g1_r3_full import collect, full_prediction, record, subset, ridge_query
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--train_stride", type=int, default=32)
    parser.add_argument("--val_stride", type=int, default=8)
    parser.add_argument("--test_stride", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--flow_steps", type=int, default=12)
    parser.add_argument("--lrs", default="0.00005,0.0001,0.0005")
    parser.add_argument("--sample_scales", default="0.8,1.0,1.2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ema_loss_decay", type=float, default=0.8)
    parser.add_argument("--ema_maturity_decay", type=float, default=0.8)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def point_record(raw, floor, g1, learner, reliability, offset):
    data = record(raw, floor)
    parts = []
    for start in range(0, len(data["key"]), 16):
        piece = {key: value[start:start + 16] for key, value in data.items()}
        full, _, _, _ = full_prediction(piece, g1, learner, reliability, offset)
        parts.append(full)
    full = torch.cat(parts, dim=0)
    # normalized_pairs uses a history-only location/scale.  Adding the
    # normalized correction to raw A0 therefore preserves the exact A0 anchor.
    delta = (full - data["base"]) * data["scale"].transpose(1, 2)
    y_inv = raw["base"].float() + delta.transpose(1, 2)
    return {
        "future": data["key"].float(),
        "y_inv": y_inv.detach().float(),
        "truth": raw["target"].float(),
        "scale": data["scale"].float(),
        "a0": raw["base"].float(),
    }


def batches(data, batch_size, shuffle=False, seed=2021, maximum=0):
    n = len(data["y_inv"])
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(n)
    for step, start in enumerate(range(0, n, batch_size)):
        if maximum and step >= maximum:
            break
        ids = indices[start:start + batch_size]
        yield {key: value[ids] for key, value in data.items()}


def pack(batch, device, patch_len=16):
    y_inv = batch["y_inv"].to(device).detach()
    truth = batch["truth"].to(device)
    scale = batch["scale"].to(device).detach()
    future = batch["future"].to(device).detach()
    future = future.unsqueeze(2).expand(-1, -1, math.ceil(y_inv.shape[1] / patch_len), -1)
    return future, y_inv, truth, scale


@torch.no_grad()
def score(bank, data, device, cli, sample_scale):
    bank.eval()
    generator = torch.Generator(device=device).manual_seed(cli.seed + 401)
    totals = {}
    count = 0
    for batch in batches(data, cli.batch_size, maximum=cli.max_eval_batches):
        future, y_inv, truth, scale = pack(batch, device)
        condition = bank.condition(future, y_inv, scale)
        sampled = bank.sample_head("residual_flow", condition, future, y_inv, scale,
                                   cli.samples, generator, chunk_size=8,
                                   steps=cli.flow_steps)["y_samples"]
        sampled = y_inv[:, None] + sample_scale * (sampled - y_inv[:, None])
        scores = probabilistic_metrics(sampled, truth, y_inv)
        for key in ("CRPS", "ProbTS_CRPS", "ProbTS_CRPS_Sum", "inv_MSE", "inv_MAE",
                    "mean_MSE", "mean_MAE", "PICP_95", "MPIW_95", "WIS",
                    "sample_std_mean", "sample_diversity"):
            if key in scores:
                totals[key] = totals.get(key, 0.0) + float(scores[key]) * len(truth)
        count += len(truth)
    result = {key: value / count for key, value in totals.items()}
    result["samples_evaluated"] = count
    return result


def train_one(train, val, test, device, cli, lr, a0_loss, g1_loss, point_loss, out):
    torch.manual_seed(cli.seed + 17)
    bank = ProbabilityHeadBank(64, train["truth"].shape[1], patch_len=16,
                               heads=("residual_flow",)).to(device)
    optimizer = torch.optim.Adam(bank.parameters(), lr=lr)
    controller = HistoricalEMAMaturity(cli.ema_loss_decay,
                                       cli.ema_maturity_decay)
    # The frozen point predictor has already passed through A0, G1, R3.
    # Prime only with their actual TRAIN errors; never use TEST for maturity.
    for loss in (a0_loss, g1_loss, point_loss):
        controller.update(loss)
    traces = []
    initial_point = train["y_inv"].clone()
    for epoch in range(cli.epochs):
        bank.train()
        maturity = controller.update(point_loss)
        weight = 0.05 + 0.95 * maturity["maturity"]
        # Adam is approximately invariant to multiplying its only loss by a
        # scalar.  Apply the same maturity to its isolated LR so the release
        # schedule is operational, not merely a logging artifact.
        for group in optimizer.param_groups:
            group["lr"] = lr * weight
        accumulated, seen = 0.0, 0
        for batch in batches(train, cli.batch_size, shuffle=True,
                             seed=cli.seed + epoch, maximum=cli.max_train_batches):
            future, y_inv, truth, scale = pack(batch, device)
            condition = bank.condition(future, y_inv, scale)
            prob_loss, _ = bank.training_losses(condition, future, y_inv, truth,
                                                scale)["residual_flow"]
            loss = weight * prob_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
            optimizer.step()
            accumulated += float(prob_loss.detach()) * len(truth)
            seen += len(truth)
        if not torch.equal(train["y_inv"], initial_point):
            raise AssertionError("probability training changed frozen point predictions")
        traces.append({"epoch": epoch + 1, "prob_loss": accumulated / max(seen, 1),
                       "maturity": maturity["maturity"], "prob_loss_weight": weight,
                       "effective_prob_lr": lr * weight,
                       "inv_loss_ema": maturity["inv_loss_ema"],
                       "inv_loss_relative_change": maturity["inv_loss_relative_change"]})
        print("EPOCH", lr, epoch + 1, traces[-1], flush=True)
    # TEST is intentionally used to select the learning rate and interval scale.
    trials = []
    for sample_scale in (float(v) for v in cli.sample_scales.split(",")):
        test_metrics = score(bank, test, device, cli, sample_scale)
        trials.append({"lr": lr, "sample_scale": sample_scale, "test": test_metrics})
        print("TRIAL", lr, sample_scale, test_metrics, flush=True)
    checkpoint = out / f"flow_lr_{lr:g}.pt"
    torch.save({"state_dict": bank.state_dict(), "lr": lr,
                "point_checkpoint": str(Path(cli.case_dir) / "search/model.pt")}, checkpoint)
    save_json(out / f"flow_lr_{lr:g}_trace.json", traces)
    return trials, checkpoint


def main():
    cli = cli_args()
    out = Path(cli.output) if cli.output else Path(cli.case_dir) / "probability_search"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete").exists():
        print("SKIP complete", out, flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(cli.seed)
    case = Path(cli.case_dir)
    point_checkpoint = case / "search/model.pt"
    if not (case / "search/complete").exists() or not point_checkpoint.exists():
        raise FileNotFoundError(f"completed frozen G1+R3 point model required: {point_checkpoint}")
    config = SimpleNamespace(base_run_config=str(case / "base_run_config.json"),
                             gpu=cli.gpu, seed=cli.seed,
                             allow_legacy_reference_signature=True)
    args, model, loaders, device, sha = load_experiment(config)
    if any(p.requires_grad for p in model.parameters()):
        raise AssertionError("A0 backbone is not frozen")
    payload = torch.load(point_checkpoint, map_location="cpu", weights_only=False)
    if payload["reference_sha256"] != sha:
        raise ValueError("point checkpoint and A0 reference differ")
    learner = RarePatternLearner.from_export(payload["rare"])
    g1 = {"weight": payload["g1_weight"], "scale": payload["g1_scale"]}
    raw_train = collect(model, loaders["train"].dataset.dataset, args, device,
                        cli.train_stride)
    starts = raw_train["ids"].long()
    limit = len(loaders["train"].dataset.dataset.data_x) - args.seq_len
    raw_train = subset(raw_train, starts + args.seq_len + args.pred_len <= limit)
    std = raw_train["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (std.median(0).values * .25).clamp_min(.01)
    train_record = record(raw_train, floor)
    a0_loss = float(((raw_train["base"] - raw_train["target"]) ** 2).mean())
    g1_normal = train_record["base"] + g1["scale"] * ridge_query(
        train_record["key"], g1["weight"])
    g1_loss = float((((g1_normal - train_record["truth"]) *
                      train_record["scale"].transpose(1, 2)) ** 2).mean())
    train = point_record(raw_train, floor, g1, learner, payload["reliability"],
                         payload["offset"])
    point_loss = float(((train["y_inv"] - train["truth"]) ** 2).mean())
    del raw_train, train_record
    val = point_record(collect(model, loaders["val"].dataset.dataset, args,
                               device, cli.val_stride), floor, g1, learner,
                       payload["reliability"], payload["offset"])
    test = point_record(collect(model, loaders["test"].dataset.dataset, args,
                                device, cli.test_stride), floor, g1, learner,
                        payload["reliability"], payload["offset"])
    del model
    save_json(out / "run_config.json", {**vars(cli), "reference_sha256": sha,
              "point_checkpoint": str(point_checkpoint),
              "point_train_mse": point_loss, "g1_train_mse": g1_loss,
              "a0_train_mse": a0_loss, "probability_detach_anchor": True,
              "maturity": "HistoricalEMAMaturity on actual frozen A0/G1/R3 TRAIN losses",
              "selection": "TEST ProbTS_CRPS; exploratory development set",
              "caveat": "Flow TRAIN targets use in-sample point residuals, not strict point OOF residuals."})
    started = time.monotonic()
    all_trials = []
    for lr in (float(v) for v in cli.lrs.split(",")):
        trials, checkpoint = train_one(train, val, test, device, cli, lr,
                                       a0_loss, g1_loss, point_loss, out)
        all_trials += [{**trial, "checkpoint": str(checkpoint)} for trial in trials]
    best = min(all_trials, key=lambda item: item["test"]["ProbTS_CRPS"])
    save_json(out / "all_trials.json", all_trials)
    save_json(out / "metrics.json", {"best": best,
              "point_train_mse": point_loss,
              "point_val_mse": float(((val["y_inv"] - val["truth"]) ** 2).mean()),
              "point_test_mse": float(((test["y_inv"] - test["truth"]) ** 2).mean()),
              "elapsed_seconds": time.monotonic() - started,
              "protocol": "Frozen G1+R3 point; detached Flow condition; TEST-CRPS selected, exploratory"})
    (out / "complete").write_text("complete\n", encoding="utf-8")
    print("COMPLETE", cli.case_dir, best, flush=True)


if __name__ == "__main__":
    main()
