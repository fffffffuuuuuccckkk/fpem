#!/usr/bin/env python3
"""Sequential paired-prototype pilot; never trains or changes an A0 backbone.

TRAIN chronology: A (bank and F03 fit) -> B (transfer/VAL-like selection and
reliability) -> C (final-point downstream holdout residual memory) -> D (Flow
fit). Purged VAL is used for Flow selection and interval calibration. A0 was
already trained on all TRAIN: C is *not* full-pipeline OOF.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.predictive_env.paired_prototype_memory import PairedPrototypeBank
from models.predictive_env.paired_prototype_memory.stability import prototype_evidence
from models.predictive_env.paired_prototype_memory.fusion import stable_consensus, reliable_variation
from models.predictive_env.paired_prototype_memory.uncertainty import fit_residual_memory, memory_features
from models.predictive_env.paired_prototype_memory.visualize import (
    plot_bank, plot_stability, plot_memory, plot_interval,
)
from models.predictive_env.direct_invariant_forecaster import DirectInvariantForecaster
from models.predictive_env.probability_heads import ProbabilityHeadBank
from models.predictive_env.residual_difference.condition import ResidualDifferenceCondition
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.run_fpem_f03_r3_invariance import collect, prepare, raw_metric
from tools.run_fpem_shared_factorization_g1r3 import load_experiment


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_run_config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--patterns", type=int, default=8)
    parser.add_argument("--pairing", choices=("real", "shuffled"), default="real")
    parser.add_argument("--initialization", choices=("coverage", "spread"), default="coverage")
    parser.add_argument("--lambda_cover", type=float, default=0.2)
    parser.add_argument("--lambda_future", type=float, default=1.0)
    parser.add_argument("--lambda_div", type=float, default=0.01)
    parser.add_argument("--bank_epochs", type=int, default=10)
    parser.add_argument("--head_epochs", type=int, default=10)
    parser.add_argument("--flow_epochs", type=int, default=5)
    parser.add_argument("--train_stride", type=int, default=8)
    parser.add_argument("--val_stride", type=int, default=4)
    parser.add_argument("--test_stride", type=int, default=1)
    parser.add_argument("--lambda_cons", type=float, default=0.1)
    parser.add_argument("--lambda_inv", type=float, default=0.01)
    parser.add_argument("--flow_steps", type=int, default=12)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--stop_after", choices=("A", "C", "E"), default="E")
    parser.add_argument("--allow_legacy_reference_signature", action="store_true")
    return parser.parse_args()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def save_visualization(out, name, function, *args):
    try:
        function(*args)
    except Exception as error:
        write_json(out / "visualizations" / f"{name}_error.json",
                   {"error": repr(error), "status": "plot failed; experiment metrics unaffected"})


def select(record, mask):
    return {key: value[mask] for key, value in record.items()}


def normalizer(record, sl=None, device="cpu"):
    part = record if sl is None else {k: v[sl] for k, v in record.items()}
    return {"model_mean": part["mean"].to(device), "model_std": part["std"].to(device),
            "hist_mean": part["hist_mean"].to(device), "hist_scale": part["scale"].to(device)}


def roles(train, val, span):
    ids = train["ids"].long()
    end = int(ids.max()) + span
    cuts = [int(end * value) for value in (.45, .70, .85)]
    groups = {
        "A_bank_head": ids + span <= cuts[0],
        "B_transfer_reliability": (ids >= cuts[0]) & (ids + span <= cuts[1]),
        "C_residual_memory": (ids >= cuts[1]) & (ids + span <= cuts[2]),
        "D_flow_fit": ids >= cuts[2],
    }
    subsets = {name: select(train, mask) for name, mask in groups.items()}
    for name, block in subsets.items():
        if len(block["ids"]) < 12:
            raise ValueError(f"insufficient purged TRAIN support: {name}={len(block['ids'])}")
    sequence = list(subsets.values())
    for earlier, later in zip(sequence, sequence[1:]):
        if int((earlier["ids"] + span).max()) > int(later["ids"].min()):
            raise AssertionError("TRAIN role futures overlap next role histories")
    v = val["ids"].long()
    midpoint = (int(v.min()) + int(v.max())) // 2
    v_select = select(val, v + span <= midpoint)
    v_cal = select(val, v >= midpoint)
    if min(len(v_select["ids"]), len(v_cal["ids"])) < 8:
        raise ValueError("insufficient purged VAL selection/calibration support")
    if int((v_select["ids"] + span).max()) > int(v_cal["ids"].min()):
        raise AssertionError("VAL selection/calibration label overlap")
    audit = {name: {"count": len(block["ids"]), "first_start": int(block["ids"].min()),
                    "last_future_end_exclusive": int((block["ids"] + span).max())}
             for name, block in subsets.items()}
    audit["VAL_select"] = {"count": len(v_select["ids"]), "first_start": int(v_select["ids"].min())}
    audit["VAL_calibrate"] = {"count": len(v_cal["ids"]), "first_start": int(v_cal["ids"].min())}
    audit["residual_status"] = "downstream-out-of-fit TRAIN holdout; A0 was pretrained on TRAIN, so not full-system OOF"
    return subsets, v_select, v_cal, audit


def bank_query(bank, block, device, batch=32):
    bank.eval()
    fields = {name: [] for name in ("weight", "distance", "candidate", "nearest_distance")}
    with torch.no_grad():
        for lo in range(0, len(block["key"]), batch):
            sl = slice(lo, lo + batch)
            result = bank.query(block["key"][sl].to(device),
                                block["hidden"][sl].to(device), normalizer(block, sl, device))
            for name in fields:
                fields[name].append(result[name].cpu())
    return {name: torch.cat(chunks) for name, chunks in fields.items()}


def point_metric(pred_hist, block):
    error = (pred_hist - block["truth_hist_norm"]) * block["scale"].transpose(1, 2)
    return {"mse": float(error.square().mean()), "mae": float(error.abs().mean())}


def fit_bank(model, early, transfer, args, device, out):
    path = out / "A_bank" / "bank.pt"
    if path.exists() and (out / "A_bank" / "complete").exists():
        return PairedPrototypeBank.from_export(torch.load(path, map_location="cpu", weights_only=False)).to(device)
    torch.manual_seed(args.seed)
    tokens, width = early["hidden"].shape[-2:]
    bank = PairedPrototypeBank(args.patterns, tokens, width, early["truth"].shape[1],
                               model.h_forecast_head.weight, model.h_forecast_head.bias).to(device)
    span = args.seq_len + args.pred_len
    training_target = early["truth_hist_norm"].clone()
    if args.pairing == "shuffled":
        # Destroy X->Y pairing within early TRAIN time blocks, not its marginal
        # future distribution or the real B/VAL/TEST evaluation labels.
        generator = torch.Generator().manual_seed(args.seed + 37)
        for block in range(4):
            cut = torch.div(early["ids"] * 4, max(int(early["ids"].max()) + 1, 1),
                            rounding_mode="floor").clamp(0, 3)
            indices = (cut == block).nonzero().flatten()
            if len(indices) > 1:
                permuted = indices[torch.randperm(len(indices), generator=generator)]
                training_target[indices] = early["truth_hist_norm"][permuted]
    initialization = bank.initialize(early["key"], early["hidden"], training_target,
                                     early["ids"], span, strategy=args.initialization)
    prior = bank_query(bank, select(early, torch.arange(len(early["key"])) < min(8, len(early["key"]))), device)
    initial_error = float((prior["candidate"] - early["base_hist_norm"][:len(prior["candidate"]), :, None]).abs().max())
    if initial_error > 1e-4:
        raise AssertionError(f"zero adapter fails A0 preservation: {initial_error}")
    optimizer = torch.optim.Adam(bank.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(args.seed)
    best, trace = None, []
    for epoch in range(args.bank_epochs):
        bank.train()
        total, seen = 0., 0
        stage = "history" if epoch < 2 else "future" if epoch < args.bank_epochs - 2 else "joint"
        for indices in torch.randperm(len(early["key"]), generator=generator).split(min(32, args.batch_size)):
            key, hidden = early["key"][indices].to(device), early["hidden"][indices].to(device)
            target = training_target[indices].to(device)
            normalization = normalizer(early, indices, device)
            loss, terms = bank.losses(key, hidden, target, normalization,
                                      use_future=stage != "history", lambda_cover=args.lambda_cover,
                                      lambda_div=args.lambda_div, lambda_adapter=.01,
                                      lambda_future=args.lambda_future)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite paired prototype loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            seen += len(indices)
        query = bank_query(bank, transfer, device)
        val = point_metric(query["prediction"] if "prediction" in query else
                           (query["weight"][..., None] * query["candidate"]).sum(2), transfer)
        row = {"epoch": epoch + 1, "stage": stage,
               "train_objective": total / max(seen, 1), "transfer": val}
        trace.append(row)
        print("A_BANK", args.backbone, args.dataset_name, args.pred_len,
              epoch + 1, stage, val["mse"], flush=True)
        if stage != "history" and (best is None or val["mse"] < best[0]):
            best = (val["mse"], {k: v.detach().cpu().clone() for k, v in bank.state_dict().items()}, epoch + 1)
    if best is None:
        raise ValueError("bank_epochs must leave a future-training stage")
    bank.load_state_dict(best[1])
    bank.update_metadata(early["key"], early["hidden"], training_target, early["ids"], span)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank.export(), path)
    write_json(path.parent / "metrics.json", {
        "initial_a0_max_abs": initial_error, "initialization": initialization,
        "best_epoch_by_later_TRAIN": best[2], "best_transfer_mse": best[0],
        "transfer_a0": point_metric(transfer["base_hist_norm"], transfer),
        "support_events": bank.event_support.cpu().tolist(),
        "block_support": bank.block_support.cpu().tolist(),
        "medoid_starts": bank.medoid_ids.cpu().tolist(), "trace": trace,
        "r3_loss_used": False, "test_used_for_selection": False,
        "pairing": args.pairing, "initialization": args.initialization})
    (path.parent / "complete").write_text("complete\n")
    return bank


def head_predict(head, block, device, batch=64):
    head.eval()
    pieces = []
    with torch.no_grad():
        for lo in range(0, len(block["hidden"]), batch):
            sl = slice(lo, lo + batch)
            normalized = head(block["hidden"][sl].to(device)).cpu()
            raw = normalized.transpose(1, 2) * block["std"][sl] + block["mean"][sl]
            pieces.append(((raw - block["hist_mean"][sl]) / block["scale"][sl]).transpose(1, 2))
    return torch.cat(pieces)


def fit_head(method, model, bank, early, transfer, stability, args, device, out):
    path = out / "B_F03" / method / "head.pt"
    tokens, width = early["hidden"].shape[-2:]
    head = DirectInvariantForecaster(tokens, width, args.pred_len, 0,
                                     model.h_forecast_head).to(device)
    if path.exists() and (path.parent / "complete").exists():
        head.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        return head
    initial = head_predict(head, transfer, device)
    if float((initial - transfer["base_hist_norm"]).abs().max()) > 1e-4:
        raise AssertionError("F03 does not preserve A0 at initialization")
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(args.seed)
    trace, best = [], None
    for epoch in range(args.head_epochs):
        head.train()
        for indices in torch.randperm(len(early["key"]), generator=generator).split(min(32, args.batch_size)):
            h = early["hidden"][indices].to(device)
            normalized = head(h)
            raw = normalized.transpose(1, 2) * early["std"][indices].to(device) + early["mean"][indices].to(device)
            mse = (raw - early["truth"][indices].to(device)).square().mean()
            penalty = mse.new_zeros(())
            if method != "F03":
                with torch.no_grad():
                    query = bank.query(early["key"][indices].to(device), h,
                                       normalizer(early, indices, device))
                    teacher, mass = stable_consensus(query["candidate"], query["weight"],
                                                      stability, early["base_hist_norm"][indices].to(device))
                    teacher_raw = teacher.transpose(1, 2) * early["scale"][indices].to(device) + early["hist_mean"][indices].to(device)
                    teacher_norm = ((teacher_raw - early["mean"][indices].to(device)) /
                                    early["std"][indices].to(device)).transpose(1, 2)
                valid = (mass > 1e-6).float()
                penalty = args.lambda_cons * ((normalized - teacher_norm.detach()).square() * valid[..., None]).mean()
                if method == "F03_cons_inv":
                    q = query["weight"].detach()
                    loss_by_sample = (normalized - early["target_norm"][indices].to(device)).square().mean(-1)
                    risks = []
                    for j in range(args.patterns):
                        mass_j = q[..., j].sum()
                        risks.append((q[..., j] * loss_by_sample).sum() / mass_j.clamp_min(1e-8))
                    risks = torch.stack(risks)
                    penalty = penalty + args.lambda_inv * (risks - risks.mean()).square().mean()
            loss = mse + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite F03 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
        measured = point_metric(head_predict(head, transfer, device), transfer)
        trace.append({"epoch": epoch + 1, "transfer": measured})
        if best is None or measured["mse"] < best[0]:
            best = (measured["mse"], {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, epoch + 1)
        print("B_F03", method, epoch + 1, measured["mse"], flush=True)
    head.load_state_dict(best[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), path)
    write_json(path.parent / "metrics.json", {"best_transfer_mse": best[0],
        "best_epoch": best[2], "transfer_a0": point_metric(transfer["base_hist_norm"], transfer),
        "trace": trace, "teacher_source": "independent frozen bank" if method != "F03" else None})
    (path.parent / "complete").write_text("complete\n")
    return head


def point_bundle(bank, head, block, reliability, device):
    queried = bank_query(bank, block, device)
    stable = head_predict(head, block, device)
    full, correction = reliable_variation(stable, queried["candidate"], queried["weight"], reliability)
    return {**queried, "stable": stable, "point": full, "correction": correction}


def flow_condition(prob, adapter, block, point, features, device):
    size = math.ceil(point.shape[-1] / 16)
    future = block["key"].to(device).unsqueeze(2).expand(-1, -1, size, -1)
    raw = point.transpose(1, 2).to(device) * block["scale"].to(device) + block["hist_mean"].to(device)
    cond = prob.condition(future, raw, block["scale"].to(device))
    return adapter(cond, features.to(device)), future, raw


@torch.no_grad()
def quantile_memory_baseline(memory, query, point, block, *, global_only,
                             samples=100, seed=2022, batch=32):
    """Piecewise-quantile residual baseline, not a learned Flow or exact CRPS."""
    generator = torch.Generator().manual_seed(seed)
    totals, seen = {}, 0
    for lo in range(0, len(point), batch):
        sl = slice(lo, lo + batch)
        q = query["weight"][sl]
        if global_only:
            current = {**memory, "prototypes": [memory["global"]] * q.shape[-1]}
            q = torch.zeros_like(q)
            q[..., 0] = 1
        else:
            current = memory
        features = memory_features(current, q, query["distance"][sl], block["scale"][sl],
                                   torch.zeros(q.shape[-1]), torch.zeros(q.shape[-1], 8), low_rank=False)
        center, lower, upper = features[..., 0], features[..., 2], features[..., 3]
        u = torch.rand((len(center), samples, 1, 1), generator=generator)
        residual = torch.where(u < .5,
            lower[:, None] + (center-lower)[:, None] * (2*u),
            center[:, None] + (upper-center)[:, None] * (2*u-1))
        scale = block["scale"][sl].transpose(1, 2)[:, None]
        history_mean = block["hist_mean"][sl].transpose(1, 2)[:, None]
        prediction = (point[sl][:, None] + residual) * scale + history_mean
        draw = prediction.permute(0, 1, 3, 2)
        anchor = (point[sl].transpose(1, 2) * block["scale"][sl]
                  + block["hist_mean"][sl])
        truth = block["truth"][sl].float()
        scores = probabilistic_metrics(draw, truth, anchor)
        for name, value in scores.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                totals[name] = totals.get(name, 0.) + float(value) * len(truth)
        seen += len(truth)
    return {name: value / seen for name, value in totals.items()}


@torch.no_grad()
def evaluate_flow(prob, adapter, block, point, features, device, samples, steps, seed,
                  multiplier=1., ratios=False, batch=16):
    prob.eval(); adapter.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    aggregate, seen, ratio_parts = {}, 0, []
    for lo in range(0, len(block["key"]), batch):
        sl = slice(lo, lo + batch)
        chunk = {k: v[sl] for k, v in block.items()}
        condition, future, anchor = flow_condition(prob, adapter, chunk, point[sl], features[sl], device)
        draw = prob.sample_head("residual_flow", condition, future, anchor,
                                chunk["scale"].to(device), samples, generator,
                                chunk_size=8, steps=steps)["y_samples"]
        draw = anchor[:, None] + multiplier * (draw - anchor[:, None])
        truth = chunk["truth"].float().to(device)
        scores = probabilistic_metrics(draw, truth, anchor)
        for name, value in scores.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                aggregate[name] = aggregate.get(name, 0.) + float(value) * len(truth)
        seen += len(truth)
        if ratios:
            lower, upper = torch.quantile(draw, torch.tensor([.025, .975], device=device), dim=1)
            midpoint = (lower + upper) / 2
            ratio_parts.append(((truth - midpoint).abs() / ((upper-lower)/2).clamp_min(1e-5)).flatten().cpu())
    result = {key: value / seen for key, value in aggregate.items()}
    if ratios:
        result["ratio"] = torch.cat(ratio_parts)
    return result


def run_flow(mode, train, val_select, val_cal, test, point, feature, args, device, out):
    path = out / "E_flow" / mode
    if (path / "complete").exists():
        return json.loads((path / "metrics.json").read_text())
    torch.manual_seed(args.seed)
    prob = ProbabilityHeadBank(64, args.pred_len, patch_len=16,
                               heads=("residual_flow",)).to(device)
    adapter = ResidualDifferenceCondition(14, 64, 16).to(device)
    optimizer = torch.optim.Adam(list(prob.parameters()) + list(adapter.parameters()), lr=1e-4)
    best, trace = None, []
    for epoch in range(args.flow_epochs):
        prob.train(); adapter.train()
        generator = torch.Generator().manual_seed(args.seed + epoch)
        for indices in torch.randperm(len(train["key"]), generator=generator).split(16):
            chunk = {k: v[indices] for k, v in train.items()}
            cond, future, anchor = flow_condition(prob, adapter, chunk, point["D"][indices], feature["D"][indices], device)
            loss, _ = prob.training_losses(cond, future, anchor, chunk["truth"].float().to(device),
                                           chunk["scale"].to(device))["residual_flow"]
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite residual Flow loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val = evaluate_flow(prob, adapter, val_select, point["VS"], feature["VS"],
                            device, min(20, args.samples), args.flow_steps, args.seed + 17)
        trace.append({"epoch": epoch + 1, "val": val})
        if best is None or val["CRPS"] < best[0]:
            best = (val["CRPS"], {"prob": {k: v.detach().cpu().clone() for k, v in prob.state_dict().items()},
                                  "adapter": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}}, epoch + 1)
        print("E_FLOW", mode, epoch + 1, val["CRPS"], flush=True)
    prob.load_state_dict(best[1]["prob"]); adapter.load_state_dict(best[1]["adapter"])
    calibration = evaluate_flow(prob, adapter, val_cal, point["VC"], feature["VC"],
                                device, min(32, args.samples), args.flow_steps, args.seed + 18, ratios=True)
    ratio = calibration.pop("ratio")
    factor = float(torch.quantile(ratio, .95).clamp(.1, 10))
    test_raw = evaluate_flow(prob, adapter, test, point["TEST"], feature["TEST"],
                             device, args.samples, args.flow_steps, args.seed + 19)
    test_cal = evaluate_flow(prob, adapter, test, point["TEST"], feature["TEST"],
                             device, args.samples, args.flow_steps, args.seed + 19, multiplier=factor)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(best[1], path / "best.pt")
    result = {"mode": mode, "best_epoch_by_val_crps": best[2], "val": trace[best[2]-1]["val"],
              "calibration_factor": factor, "calibration": calibration,
              "test_uncalibrated": test_raw, "test_calibrated": test_cal,
              "point_anchor_unchanged": True, "train_residual_source": "final frozen F03+R3 point"}
    write_json(path / "metrics.json", result)
    write_json(path / "trace.json", trace)
    if mode == "U4_lowrank_flow" and len(test["key"]):
        with torch.no_grad():
            first = {key: value[:1] for key, value in test.items()}
            condition, future, anchor = flow_condition(
                prob, adapter, first, point["TEST"][:1], feature["TEST"][:1], device)
            generator = torch.Generator(device=device).manual_seed(args.seed + 19)
            draw = prob.sample_head("residual_flow", condition, future, anchor,
                                    first["scale"].to(device), min(args.samples, 100),
                                    generator, chunk_size=8, steps=args.flow_steps)["y_samples"]
        save_visualization(out, "test_interval", plot_interval,
                           first["truth"][0, :, 0], anchor[0, :, 0],
                           draw[0, :, :, 0], out / "visualizations" / "test_interval_example.png")
    (path / "complete").write_text("complete\n")
    return result


def main():
    cli = arguments()
    out = Path(cli.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    saved = json.loads(Path(cli.base_run_config).read_text(encoding="utf-8"))
    loader_cli = argparse.Namespace(**vars(cli))
    loader_cli.seed = saved["seed"]
    args, model, loaders, device, a0_sha = load_experiment(loader_cli)
    # Keep the checkpoint identity fixed while allowing independent new-module
    # seeds and budgets to be recorded in the experiment configuration.
    for name in ("seed", "patterns", "bank_epochs", "head_epochs", "flow_epochs",
                 "lambda_cons", "lambda_inv", "flow_steps", "samples", "pairing",
                 "initialization", "lambda_cover", "lambda_future", "lambda_div"):
        setattr(args, name, getattr(cli, name))
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("A0 must remain frozen")
    data = {}
    for role, stride in (("train", cli.train_stride), ("val", cli.val_stride), ("test", cli.test_stride)):
        dataset = loaders[role].dataset.dataset
        raw = collect(model, dataset, args, device, stride)
        data[role] = raw
    std = data["train"]["x"].float().var(1, unbiased=False).add(1e-5).sqrt()
    floor = (std.median(0).values * .25).clamp_min(.01)
    data = {name: prepare(record, floor) for name, record in data.items()}
    for record in data.values():
        record["truth"] = record["truth"].float()
    span = args.seq_len + args.pred_len
    split, val_select, val_cal, audit = roles(data["train"], data["val"], span)
    write_json(out / "run_config.json", {"cli": vars(cli), "a0_sha256": a0_sha,
        "dataset": args.dataset_name, "backbone": args.backbone,
        "time_roles": audit, "a0_training_scope": "entire original TRAIN checkpoint",
        "full_pipeline_oof": False, "test_selection": False})
    A, B, C, D = (split[name] for name in ("A_bank_head", "B_transfer_reliability",
                                          "C_residual_memory", "D_flow_fit"))
    bank = fit_bank(model, A, B, args, device, out)
    save_visualization(out, "bank", plot_bank, bank, A, out / "visualizations")
    if cli.stop_after == "A":
        return
    bank.eval().requires_grad_(False)
    qB = bank_query(bank, B, device)
    a0B = B["base_hist_norm"]
    stability, _, evidence = prototype_evidence(a0B, qB["candidate"],
                                                 B["truth_hist_norm"], qB["weight"], B["ids"], span)
    bank.stability.copy_(stability.to(bank.stability))
    write_json(out / "B_stability" / "evidence.json", evidence)
    heads = {}
    for method in ("F03", "F03_cons", "F03_cons_inv"):
        heads[method] = fit_head(method, model, bank, A, B, stability,
                                 args, device, out)
    score = {method: point_metric(head_predict(head, B, device), B)["mse"]
             for method, head in heads.items()}
    selected_name = min(score, key=score.get)
    selected_head = heads[selected_name]
    selected_B = head_predict(selected_head, B, device)
    _, reliability, reliability_evidence = prototype_evidence(selected_B, qB["candidate"],
        B["truth_hist_norm"], qB["weight"], B["ids"], span)
    write_json(out / "C_point" / "reliability.json", reliability_evidence)
    save_visualization(out, "bank", plot_bank, bank, A, out / "visualizations")
    save_visualization(out, "stability", plot_stability, stability, reliability,
                       out / "visualizations", evidence)
    blocks = {"C": C, "D": D, "VS": val_select, "VC": val_cal, "TEST": data["test"]}
    bundles = {name: point_bundle(bank, selected_head, block, reliability, device)
               for name, block in blocks.items()}
    point_results = {name: {"a0": point_metric(block["base_hist_norm"], block),
                            "f03": point_metric(bundle["stable"], block),
                            "f03_r3": point_metric(bundle["point"], block),
                            "active_ratio": float((bundle["correction"].abs().mean(-1) > 1e-7).float().mean())}
                     for name, (block, bundle) in ((key, (blocks[key], bundles[key])) for key in blocks)}
    write_json(out / "C_point" / "metrics.json", {"selected_head_on_B": selected_name,
        "head_scores_B": score, "roles": point_results,
        "reliability_source": "later purged TRAIN B; no TEST labels"})
    (out / "C_point" / "complete").write_text("complete\n")
    if cli.stop_after == "C":
        return
    residual_C = C["truth_hist_norm"] - bundles["C"]["point"]
    memory_path = out / "D_memory" / "memory.pt"
    try:
        memory = fit_residual_memory(residual_C, bundles["C"]["weight"], C["ids"], span)
    except ValueError as error:
        write_json(out / "D_memory" / "insufficient_support.json", {"reason": str(error),
            "point_results": point_results, "full_pipeline_oof": False})
        print("D_MEMORY_INSUFFICIENT", str(error), flush=True)
        return
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(memory, memory_path)
    save_visualization(out, "memory", plot_memory, memory, out / "visualizations")
    write_json(memory_path.parent / "audit.json", {"independent_event_count": memory["independent_event_count"],
        "prototype_effective_support": [item["effective_support"] for item in memory["prototypes"]],
        "role": "C only; D Flow targets and TEST labels absent", "full_pipeline_oof": False})
    (memory_path.parent / "complete").write_text("complete\n")
    point = {name: bundles[name]["point"] for name in blocks}
    summary = {
        "U0_global_quantiles": {"test": quantile_memory_baseline(memory, bundles["TEST"],
            point["TEST"], data["test"], global_only=True, samples=cli.samples, seed=cli.seed)},
        "U2_prototype_quantiles": {"test": quantile_memory_baseline(memory, bundles["TEST"],
            point["TEST"], data["test"], global_only=False, samples=cli.samples, seed=cli.seed)},
    }
    for mode in ("U1_global_flow", "U3_prototype_flow", "U4_lowrank_flow"):
        features = {}
        for name in blocks:
            query = bundles[name]
            q = query["weight"] if mode != "U1_global_flow" else torch.zeros_like(query["weight"])
            if mode == "U1_global_flow":
                q[..., 0] = 1
                global_memory = {**memory, "prototypes": [memory["global"]] * args.patterns}
                current = global_memory
            else:
                current = memory
            features[name] = memory_features(current, q, query["distance"], blocks[name]["scale"],
                stability, reliability, low_rank=(mode == "U4_lowrank_flow"))
        summary[mode] = run_flow(mode, D, val_select, val_cal, data["test"], point,
                                 features, args, device, out)
    write_json(out / "summary.json", {"bank": json.loads((out / "A_bank" / "metrics.json").read_text()),
        "selected_head": selected_name, "point": point_results, "flow": summary,
        "oof_caveat": "A0 trained on full TRAIN; C residual is only downstream holdout"})
    (out / "complete").write_text("complete\n")


if __name__ == "__main__":
    main()
