#!/usr/bin/env python
"""Shared-encoder, joint single-stage benchmark of plug-in FPEM probability heads.

Each minibatch computes the backbone and Future-Zvar once. Every enabled head
receives the same detached Yinv and shared patch-shape condition. The encoder
receives the mean of all head losses, so this protocol is *not* equivalent to
independently retraining the backbone for each head.
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.PatchTST_PredictiveEnvIV import Model
from models.predictive_env import PredictiveConflictEnvironment
from models.predictive_env.affine_target_projection import oracle_affine_diagnostics
from models.predictive_env.probability_heads import HEAD_NAMES, ProbabilityHeadBank
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.fpem_probabilistic_training import (
    calibrate_affine_scale, probability_loss_weight,
)
from tools.run_predictive_env_iv_patchtst import (
    IndexedDataset, build_data, build_training_optimizer, collect,
    gradient_relative_maturity, load_shared_reference_state, make_train_loader,
    model_config, parse_args, prepare_reference, seed_everything,
)


def _scalar_metrics(score):
    return {key: float(value.detach().cpu()) for key, value in score.items()
            if torch.as_tensor(value).ndim == 0}


def train_shared_epoch(model, bank, loader, assignment, optimizer, args,
                       device, epoch):
    model.train()
    bank.train()
    totals = {}
    seen = 0
    for step, batch in enumerate(loader):
        if args.max_train_batches and step >= args.max_train_batches:
            break
        x, y, ids, _, _, cycle, _ = batch
        x = x.float().to(device)
        y = y[:, -args.pred_len:].float().to(device)
        output = model.forward_components(x, cycle.to(device))
        y_inv = output["invariant_prediction"]
        inv_loss = (y_inv - y).square().mean()
        q = torch.as_tensor(assignment[ids.numpy()], device=device).detach()
        env_loss, env_details = model.environment_classification_loss(
            output["z_inv"], output["z_var"], q,
        )
        future = output["probabilistic_condition"]["future"]
        condition = bank.condition(future, y_inv, output["input_scale"])
        head_results = bank.training_losses(
            condition, future, y_inv, y, output["input_scale"]
        )
        head_losses = [item[0] for item in head_results.values()]
        prob_loss = torch.stack(head_losses).mean()
        weight, maturity_logs = probability_loss_weight(
            inv_loss, prob_loss, tuple(model.head_linear.parameters()),
            tuple(bank.parameters()), args, epoch, gradient_relative_maturity,
        )
        loss = inv_loss + args.lambda_prob_env * env_loss + weight * prob_loss
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("non-finite shared-head loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.prob_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(bank.parameters()),
                args.prob_grad_clip,
            )
        optimizer.step()
        record = {"loss": loss, "loss/inv": inv_loss,
                  "loss/env": env_loss, "loss/prob": prob_loss,
                  "prob_loss_weight": weight,
                  "var_env_soft_ce": env_details["var_env_soft_ce"],
                  "inv_env_soft_ce": env_details["inv_env_soft_ce"],
                  **maturity_logs}
        for name, (head_loss, diagnostics) in head_results.items():
            record[f"head/{name}/loss"] = head_loss
            for key, value in diagnostics.items():
                if torch.as_tensor(value).ndim == 0:
                    record[f"head/{name}/{key}"] = value
        count = x.shape[0]
        seen += count
        for key, value in record.items():
            totals[key] = totals.get(key, 0.0) + count * float(
                torch.as_tensor(value).detach().cpu()
            )
    if not seen:
        raise RuntimeError("empty joint probability training epoch")
    return {key: value / seen for key, value in totals.items()}


@torch.no_grad()
def evaluate_heads(model, bank, loader, args, device, root, split):
    model.eval()
    bank.eval()
    sums = {name: {} for name in bank.heads}
    count = 0
    saved = set()
    shuffle_sums = {name: [0.0, 0.0, 0] for name in bank.heads}
    for step, (x, y, _, cycle) in enumerate(loader):
        if args.max_prob_eval_batches and step >= args.max_prob_eval_batches:
            break
        x = x.float().to(device)
        y = y[:, -args.pred_len:].float().to(device)
        output = model.forward_components(x, cycle.to(device))
        y_inv = output["invariant_prediction"]
        scale = output["input_scale"]
        future = output["probabilistic_condition"]["future"]
        condition = bank.condition(future, y_inv, scale)
        shuffled_future = shuffled_condition = None
        if (split == "test" and step < args.prob_shuffled_eval_batches
                and x.shape[0] > 1):
            shuffled_future = future.roll(1, dims=0)
            shuffled_condition = bank.condition(shuffled_future, y_inv, scale)
        oracle = oracle_affine_diagnostics(
            y, y_inv, scale, args.future_patch_len, args.prob_affine_ridge,
        )
        count += x.shape[0]
        for index, name in enumerate(bank.heads):
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + (25007 if split == "test" else 19001)
                                  + step * 1009)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                memory_before = torch.cuda.memory_allocated(device)
            else:
                memory_before = 0
            start = time.perf_counter()
            paths = bank.sample_head(
                name, condition, future, y_inv, scale,
                args.prob_num_samples, generator, args.prob_sample_chunk,
                args.prob_flow_steps,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = max(torch.cuda.max_memory_allocated(device) - memory_before, 0)
            else:
                peak = 0
            elapsed = time.perf_counter() - start
            samples = paths["y_samples"]
            if not bool(torch.isfinite(samples).all()):
                raise FloatingPointError(f"non-finite generated paths: {name}")
            score = probabilistic_metrics(samples, y, y_inv)
            # Exactly the same ensemble and 19-quantile grid as ordinary CRPS.
            sum_score = probabilistic_metrics(
                samples.sum(-1, keepdim=True), y.sum(-1, keepdim=True),
                y_inv.sum(-1, keepdim=True),
            )
            scalar = _scalar_metrics(score)
            scalar["CRPS_Sum"] = float(sum_score["CRPS"])
            scalar["ProbTS_CRPS_Sum"] = float(sum_score["ProbTS_CRPS"])
            scalar["CRPS_Sum_raw_pinball"] = float(
                sum_score["quantile_crps_raw"])
            scalar["sum_target_abs_mean"] = float(sum_score["target_abs_mean"])
            correction = samples - y_inv[:, None]
            jumps = (correction[:, :, 1:] - correction[:, :, :-1]).abs()
            boundaries = torch.arange(args.future_patch_len - 1, args.pred_len - 1,
                                      args.future_patch_len, device=device)
            scalar["Boundary_Jump_Ratio"] = float(
                jumps[:, :, boundaries].mean() / jumps.mean().clamp_min(1e-8)
            ) if len(boundaries) else float("nan")
            scalar["inference_seconds_per_sample"] = elapsed / x.shape[0]
            scalar["peak_head_bytes"] = peak
            scalar.update({key: float(value) for key, value in oracle.items()
                           if torch.as_tensor(value).ndim == 0})
            for key, value in scalar.items():
                sums[name][key] = sums[name].get(key, 0.0) + x.shape[0] * value
            if shuffled_future is not None:
                shuffled_generator = torch.Generator(device=device)
                shuffled_generator.manual_seed(
                    args.seed + 25007 + step * 1009
                )
                counterfactual = bank.sample_head(
                    name, shuffled_condition, shuffled_future, y_inv, scale,
                    args.prob_num_samples, shuffled_generator,
                    args.prob_sample_chunk, args.prob_flow_steps,
                )
                shuffled_crps = probabilistic_metrics(
                    counterfactual["y_samples"], y, y_inv
                )["CRPS"]
                shuffle_sums[name][0] += x.shape[0] * float(score["CRPS"])
                shuffle_sums[name][1] += x.shape[0] * float(shuffled_crps)
                shuffle_sums[name][2] += x.shape[0]
                del counterfactual
            if name not in saved:
                directory = root / name
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    directory / f"{split}_trajectory_examples.npz",
                    target=y[:4].cpu().numpy(), y_inv=y_inv[:4].cpu().numpy(),
                    y_samples=samples[:4, :min(20, args.prob_num_samples)].cpu().numpy(),
                    lower_95=paths["lower_95"][:4].cpu().numpy(),
                    upper_95=paths["upper_95"][:4].cpu().numpy(),
                )
                saved.add(name)
            del paths, samples, score, sum_score, correction, jumps
    if not count:
        raise RuntimeError(f"empty {split} probability evaluation")
    results = {}
    for name, values in sums.items():
        result = {key: value / count for key, value in values.items()}
        result["quantile_CRPS_normalized"] = (
            result["quantile_crps_raw"] / max(result["target_abs_mean"], 1e-8)
        )
        result["normalized_CRPS_Sum"] = (
            result["CRPS_Sum_raw_pinball"] /
            max(result["sum_target_abs_mean"], 1e-8)
        )
        result["generated_mean_minus_inv_MSE"] = (
            result["mean_MSE"] - result["inv_MSE"]
        )
        result["generated_mean_degradation_ratio"] = (
            result["mean_MSE"] / max(result["inv_MSE"], 1e-8)
        )
        result["sample_collapse"] = (
            name != "affine_deterministic"
            and result["sample_std_mean"] < 1e-6
        )
        shuffled_ours, shuffled_wrong, shuffled_count = shuffle_sums[name]
        if shuffled_count:
            result["matched_subset_CRPS"] = shuffled_ours / shuffled_count
            result["shuffled_subset_CRPS"] = shuffled_wrong / shuffled_count
            result["shuffled_minus_matched_CRPS"] = (
                shuffled_wrong - shuffled_ours
            ) / shuffled_count
            result["shuffled_subset_sample_count"] = shuffled_count
        result.update({
            "dataset": args.dataset_name, "pred_len": args.pred_len,
            "prob_head": name, "split": split,
            "num_samples": args.prob_num_samples,
            "parameter_count_head": sum(p.numel() for p in bank.heads[name].parameters()),
            "parameter_count_condition_encoder": sum(
                p.numel() for p in bank.condition_encoder.parameters()),
            "shared_encoder_joint_training": True,
            "probability_head_stop_gradient_at_zvar": True,
            "inference_timing_scope": "head sampling only; shared encoder excluded",
            "independent_gaussian_time_points": name == "residual_gaussian",
            "deterministic_degenerate_distribution": name == "affine_deterministic",
            "CRPS_definition": "empirical ensemble CRPS in target units",
            "normalized_CRPS_definition": "2*mean pinball q=0.05..0.95 / mean abs target",
            "CRPS_Sum_definition": "same CRPS after sum over variables",
            "normalized_CRPS_Sum_definition":
                "same 19-quantile normalized CRPS after sum over variables",
            "ProbTS_CRPS_definition":
                "ProbTS quantiles_num=20: mean over sequences of mean q=0.05..0.95 twice-pinball sum / sequence absolute-target sum",
            "ProbTS_CRPS_Sum_definition":
                "same ProbTS Evaluator definition after summing variables",
        })
        (root / name / f"{split}_metrics.json").write_text(
            json.dumps(result, indent=2, allow_nan=True)
        )
        results[name] = result
    return results


def write_summary(root, results, prefix=""):
    keys = ("prob_head", "inv_MSE", "mean_MSE", "mean_MAE", "CRPS",
            "ProbTS_CRPS", "ProbTS_CRPS_Sum",
            "quantile_CRPS_normalized", "CRPS_Sum", "normalized_CRPS_Sum",
            "PICP_95", "MPIW_95", "WIS", "EnergyScore", "sample_diversity",
            "Boundary_Jump_Ratio", "inference_seconds_per_sample",
            "peak_head_bytes", "parameter_count_head")
    with (root / f"{prefix}summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: result.get(key, "") for key in keys}
                         for result in results.values())
    winner_keys = {
        "lowest_CRPS": "CRPS", "lowest_normalized_CRPS_Sum":
        "normalized_CRPS_Sum", "lowest_mean_MSE": "mean_MSE",
        "lowest_ProbTS_CRPS": "ProbTS_CRPS",
        "lowest_ProbTS_CRPS_Sum": "ProbTS_CRPS_Sum",
        "lowest_inference_time": "inference_seconds_per_sample",
    }
    winners = {label: min(results, key=lambda name: results[name][key])
               for label, key in winner_keys.items()}
    winners["closest_PICP95_to_0.95"] = min(
        results, key=lambda name: abs(results[name]["PICP_95"] - 0.95)
    )
    (root / f"{prefix}summary.txt").write_text(
        "Shared-encoder joint training; heads are not independent backbone fits.\n"
        "A-class uses oracle-projected gamma/beta; B-class directly models "
        "normalized future residual.\n"
        + "\n".join(f"{key}: {value}" for key, value in winners.items()) + "\n"
    )


def main():
    args = parse_args()
    if args.backbone != "patchtst" or args.variant_fusion_mode != "prob_affine_flow":
        raise ValueError("first shared-head benchmark requires PatchTST IAPAD")
    if args.decomposition_type != "complementary_gate":
        raise ValueError("shared-head benchmark requires complementary gate")
    if args.representation_constraint != "classification":
        raise ValueError("shared-head benchmark requires classification constraint")
    if args.stage_epochs <= 0 or args.env_num < 2:
        raise ValueError("invalid stage/environment configuration")
    names = HEAD_NAMES if args.prob_head == "all_shared" else (args.prob_head,)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    protocol = {**vars(args),
                "shared_encoder_joint_training": True,
                "probability_head_stop_gradient_at_zvar": True,
                "probability_loss_combination": "arithmetic mean of enabled head losses",
                "head_checkpoint_protocol":
                    "one shared encoder checkpoint plus one head checkpoint per head"}
    (root / "run_config.json").write_text(json.dumps(protocol, indent=2))
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    datasets, ordered_train, _, test_loader = build_data(args)
    val_loader = DataLoader(
        IndexedDataset(datasets["val"], cycle_len=args.cyclenet_cycle_len),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )
    train_loader = make_train_loader(datasets["train"], args)
    reference_state, reference_hash, _ = prepare_reference(
        args, train_loader, device
    )
    model = Model(model_config(args, "A2")).to(device)
    load_shared_reference_state(model, reference_state)
    model.probability_detach_zvar_input = True
    scale_record = calibrate_affine_scale(model, ordered_train, args, device)
    bank = ProbabilityHeadBank(
        args.d_model, args.pred_len, args.future_patch_len,
        args.prob_condition_mode, names,
    ).to(device)
    bank.set_theta_scale(model.probabilistic_affine_dynamics.theta_scale)
    (root / "affine_scale_calibration.json").write_text(json.dumps(scale_record, indent=2))
    optimizer = build_training_optimizer(model, args)
    optimizer.add_param_group({"params": list(bank.parameters()),
                               "lr": args.prob_head_lr, "name": "prob_heads"})
    manager = PredictiveConflictEnvironment(
        len(train_loader.dataset), args.env_num, args.eiil_steps, args.eiil_lr,
        args.balance_weight, args.entropy_weight, args.seed,
        args.diagnostic_samples, args.environment_matching,
        args.matching_q_weight, args.matching_gradient_weight,
        False, 100,
    )
    history = []
    epoch = stage = 0
    while epoch < args.epochs:
        prediction, target, ids = collect(model, ordered_train, args, device)
        assignment, env_record = manager.update(
            prediction, target, ids, source_split="train"
        )
        env_record["stage"] = stage
        history.append(env_record)
        print("STAGE", stage, "environment", env_record, flush=True)
        for _ in range(min(args.stage_epochs, args.epochs - epoch)):
            record = train_shared_epoch(
                model, bank, train_loader, assignment, optimizer, args,
                device, epoch,
            )
            record.update({"stage": stage, "epoch": epoch + 1})
            history.append(record)
            print("EPOCH", epoch + 1, record, flush=True)
            epoch += 1
        stage += 1
    (root / "training_history.json").write_text(json.dumps(history, indent=2))
    val = evaluate_heads(model, bank, val_loader, args, device, root, "val")
    if args.prob_validation_only:
        write_summary(root, val, prefix="validation_")
        (root / "validation_complete").write_text("complete\n")
        print("VALIDATION_COMPLETE", root, flush=True)
        return
    test = evaluate_heads(model, bank, test_loader, args, device, root, "test")
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "condition_encoder_state": {k: v.detach().cpu()
                                    for k, v in bank.condition_encoder.state_dict().items()},
        "run_config": vars(args), "reference_sha256": reference_hash,
    }, root / "shared_encoder.pt")
    for name, head in bank.heads.items():
        directory = root / name
        torch.save({"state_dict": {k: v.detach().cpu()
                                    for k, v in head.state_dict().items()},
                    "shared_encoder": "../shared_encoder.pt",
                    "prob_head": name}, directory / "head_checkpoint.pt")
        (directory / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    write_summary(root, test)
    print("COMPLETE", root, {name: (r["mean_MSE"], r["CRPS"])
                              for name, r in test.items()}, flush=True)


if __name__ == "__main__":
    main()
