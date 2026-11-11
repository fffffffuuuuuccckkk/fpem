#!/usr/bin/env python3
"""Frozen-model probes for conditional residual information in Z_var.

The forecasting model is never updated.  Current/predicted Z_var probes are
evaluated on TEST; oracle teacher Z_var remains strictly TRAIN-only and is
reported on a chronological held-out TRAIN tail.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.PatchTST_PredictiveEnvIV import Model
from tools.run_predictive_env_iv_patchtst import (
    build_data,
    future_patch_centers,
    model_config,
    seed_everything,
)


class CurrentResidualProbe(nn.Module):
    def __init__(self, dimension, horizon, hidden=0):
        super().__init__()
        self.net = (
            nn.Linear(dimension, horizon)
            if hidden <= 0
            else nn.Sequential(
                nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, horizon)
            )
        )

    def forward(self, representation):
        return self.net(representation).permute(0, 2, 1)


class PatchResidualProbe(nn.Module):
    def __init__(self, dimension, patch_count, patch_len, hidden=0):
        super().__init__()
        make_head = lambda: (
            nn.Linear(dimension, patch_len)
            if hidden <= 0
            else nn.Sequential(
                nn.Linear(dimension, hidden),
                nn.GELU(),
                nn.Linear(hidden, patch_len),
            )
        )
        self.heads = nn.ModuleList([make_head() for _ in range(patch_count)])

    def forward(self, representation, pred_len):
        patches = [
            head(representation[:, :, index])
            for index, head in enumerate(self.heads)
        ]
        return torch.cat(patches, dim=-1)[..., :pred_len].permute(0, 2, 1)

    def forward_patch(self, representation, patch_index, width):
        return self.heads[int(patch_index)](representation)[..., :width].permute(
            0, 2, 1
        )


def metric_accumulator():
    return {key: [] for key in ("residual", "prediction", "cosine", "beneficial")}


def update_metrics(store, residual, prediction):
    residual_flat = residual.flatten(1)
    prediction_flat = prediction.flatten(1)
    residual_mse = residual_flat.square().mean(1)
    prediction_mse = (residual_flat - prediction_flat).square().mean(1)
    cosine = torch.nn.functional.cosine_similarity(
        residual_flat, prediction_flat, dim=1, eps=1e-8
    )
    store["residual"].append(residual_mse.cpu())
    store["prediction"].append(prediction_mse.cpu())
    store["cosine"].append(cosine.cpu())
    store["beneficial"].append((prediction_mse < residual_mse).float().cpu())


def summarize(store):
    values = {key: torch.cat(items) for key, items in store.items() if items}
    if not values:
        return {key: float("nan") for key in (
            "baseline_residual_MSE", "residual_MSE", "explained_residual_ratio",
            "alignment_cosine", "beneficial_ratio", "sample_count",
        )}
    baseline = float(values["residual"].mean())
    prediction = float(values["prediction"].mean())
    return {
        "baseline_residual_MSE": baseline,
        "residual_MSE": prediction,
        "explained_residual_ratio": 1.0 - prediction / max(baseline, 1e-12),
        "alignment_cosine": float(values["cosine"].mean()),
        "beneficial_ratio": float(values["beneficial"].mean()),
        "sample_count": int(values["residual"].numel()),
    }


def split_metrics(residual, prediction, patch_len):
    overall = metric_accumulator()
    update_metrics(overall, residual, prediction)
    patch_count = math.ceil(residual.shape[1] / patch_len)
    groups = torch.tensor_split(torch.arange(patch_count), min(3, patch_count))
    result = {"overall": summarize(overall)}
    for name, indices in zip(("near", "middle", "far"), groups):
        start = int(indices[0]) * patch_len
        end = min((int(indices[-1]) + 1) * patch_len, residual.shape[1])
        store = metric_accumulator()
        update_metrics(store, residual[:, start:end], prediction[:, start:end])
        result[name] = summarize(store)
    return result


def merge_metric_batches(batches):
    if not batches:
        return {}
    merged = {}
    for section in batches[0]:
        weights = [row[section]["sample_count"] for row in batches]
        total = sum(weights)
        merged[section] = {}
        for key in batches[0][section]:
            if key == "sample_count":
                merged[section][key] = total
            else:
                merged[section][key] = sum(
                    row[section][key] * weight
                    for row, weight in zip(batches, weights)
                ) / max(total, 1)
        baseline = merged[section]["baseline_residual_MSE"]
        merged[section]["explained_residual_ratio"] = (
            1.0 - merged[section]["residual_MSE"] / max(baseline, 1e-12)
        )
    return merged


def make_loader(dataset, indices, batch_size, seed, shuffle):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


@torch.no_grad()
def evaluate_regular(model, loader, current_probe, predicted_probe, args, device):
    current_batches, predicted_batches = [], []
    model.eval()
    current_probe.eval()
    predicted_probe.eval()
    for step, (x, y, _, cycle_index) in enumerate(loader):
        if args.probe_max_batches and step >= args.probe_max_batches:
            break
        y = y[:, -args.pred_len:].float().to(device)
        output = model.forward_components(x.float().to(device), cycle_index.to(device))
        residual = y - output["invariant_prediction"]
        current = output["z_var_tokens"].mean(dim=2)
        current_prediction = current_probe(current)
        future_prediction = predicted_probe(
            output["horizon_future_zvar"], args.pred_len
        )
        current_batches.append(
            split_metrics(residual, current_prediction, args.future_patch_len)
        )
        predicted_batches.append(
            split_metrics(residual, future_prediction, args.future_patch_len)
        )
    return merge_metric_batches(current_batches), merge_metric_batches(predicted_batches)


@torch.no_grad()
def evaluate_teacher(
    model, loader, teacher_dataset, teacher_probe, args, device, patch_indices
):
    stores = {
        "overall": metric_accumulator(),
        "near": metric_accumulator(),
        "middle": metric_accumulator(),
        "far": metric_accumulator(),
    }
    anchors = future_patch_centers(args.pred_len, args.future_patch_len)
    model.eval()
    teacher_probe.eval()
    for step, (x, y, sample_id, cycle_index) in enumerate(loader):
        if args.probe_max_batches and step >= args.probe_max_batches:
            break
        del x, cycle_index
        selected = anchors.index_select(0, patch_indices)
        windows, valid, teacher_cycles = teacher_dataset.horizon_teacher_batch(
            sample_id, selected
        )
        if not bool(valid.any()):
            continue
        features = model.horizon_future_variant_targets(
            windows.float().to(device), teacher_cycles.to(device)
        )
        y = y[valid, -args.pred_len:].float().to(device)
        # Re-load the original past inputs as a compact batch from the subset.
        past = torch.stack([
            torch.as_tensor(teacher_dataset.dataset[int(index)][0])
            for index in sample_id[valid].tolist()
        ]).float().to(device)
        past_cycles = sample_id[valid].to(device) % args.cyclenet_cycle_len
        invariant = model.forward_components(past, past_cycles)["invariant_prediction"]
        residual = y - invariant
        region_names = ("near", "middle", "far")
        for local_index, patch_index in enumerate(patch_indices.tolist()):
            start = patch_index * args.future_patch_len
            width = min(args.future_patch_len, args.pred_len - start)
            prediction = teacher_probe.forward_patch(
                features[:, :, local_index], patch_index, width
            )
            residual_patch = residual[:, start : start + width]
            update_metrics(stores["overall"], residual_patch, prediction)
            update_metrics(
                stores[region_names[local_index]], residual_patch, prediction
            )
    result = {name: summarize(store) for name, store in stores.items()}
    result["evaluated_patch_indices"] = patch_indices.tolist()
    result["teacher_scope"] = "TRAIN only"
    return result


def train_probes(model, train_loader, teacher_dataset, probes, args, device):
    current_probe, predicted_probe, teacher_probe = probes
    optimizer = torch.optim.Adam(
        [parameter for probe in probes for parameter in probe.parameters()],
        lr=args.probe_lr,
    )
    anchors = future_patch_centers(args.pred_len, args.future_patch_len)
    model.eval()
    history = []
    for epoch in range(args.probe_epochs):
        for probe in probes:
            probe.train()
        total, count = 0.0, 0
        for step, (x, y, sample_id, cycle_index) in enumerate(train_loader):
            if args.probe_max_batches and step >= args.probe_max_batches:
                break
            y = y[:, -args.pred_len:].float().to(device)
            with torch.no_grad():
                output = model.forward_components(
                    x.float().to(device), cycle_index.to(device)
                )
                residual = y - output["invariant_prediction"]
            current_prediction = current_probe(
                output["z_var_tokens"].mean(dim=2).detach()
            )
            predicted_prediction = predicted_probe(
                output["horizon_future_zvar"].detach(), args.pred_len
            )
            loss_current = (current_prediction - residual).square().mean()
            loss_predicted = (predicted_prediction - residual).square().mean()

            patch_index = (step + epoch) % anchors.numel()
            windows, valid, teacher_cycles = teacher_dataset.horizon_teacher_batch(
                sample_id, anchors[patch_index : patch_index + 1]
            )
            loss_teacher = residual.new_zeros(())
            if bool(valid.any()):
                with torch.no_grad():
                    teacher_feature = model.horizon_future_variant_targets(
                        windows.float().to(device), teacher_cycles.to(device)
                    )[:, :, 0]
                start = patch_index * args.future_patch_len
                width = min(args.future_patch_len, args.pred_len - start)
                teacher_prediction = teacher_probe.forward_patch(
                    teacher_feature, patch_index, width
                )
                loss_teacher = (
                    teacher_prediction
                    - residual[valid.to(device), start : start + width]
                ).square().mean()
            loss = loss_current + loss_predicted + loss_teacher
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        history.append(total / max(count, 1))
        print(f"probe epoch {epoch + 1}: loss={history[-1]:.6f}", flush=True)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--probe_epochs", type=int, default=5)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_hidden", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--max_batches", type=int, default=0)
    cli = parser.parse_args()

    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    config = dict(checkpoint["run_config"])
    config.setdefault("reliability_environment_disagreement", False)
    config["batch_size"] = cli.batch_size
    config["num_workers"] = 0
    args = SimpleNamespace(**config)
    args.probe_max_batches = cli.max_batches
    args.probe_lr = cli.probe_lr
    args.probe_epochs = cli.probe_epochs
    seed_everything(args.seed)
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    datasets, ordered_train, ordered_future_train, test_loader = build_data(args)
    model = Model(model_config(args, "A2")).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False)

    patch_count = math.ceil(args.pred_len / args.future_patch_len)
    probes = (
        CurrentResidualProbe(args.d_model, args.pred_len, cli.probe_hidden).to(device),
        PatchResidualProbe(
            args.d_model, patch_count, args.future_patch_len, cli.probe_hidden
        ).to(device),
        PatchResidualProbe(
            args.d_model, patch_count, args.future_patch_len, cli.probe_hidden
        ).to(device),
    )
    cutoff = int(len(ordered_train.dataset) * cli.train_fraction)
    probe_train = make_loader(
        ordered_train.dataset, range(cutoff), cli.batch_size, args.seed + 7001, True
    )
    heldout_train = make_loader(
        ordered_train.dataset, range(cutoff, len(ordered_train.dataset)),
        cli.batch_size, args.seed + 7002, False,
    )
    history = train_probes(
        model, probe_train, ordered_future_train.dataset, probes, args, device
    )
    current_train, predicted_train = evaluate_regular(
        model, probe_train, probes[0], probes[1], args, device
    )
    current_test, predicted_test = evaluate_regular(
        model, test_loader, probes[0], probes[1], args, device
    )
    representative = torch.tensor(
        sorted(set((0, patch_count // 2, patch_count - 1))), dtype=torch.long
    )
    teacher_train = evaluate_teacher(
        model, probe_train, ordered_future_train.dataset, probes[2], args,
        device, representative,
    )
    teacher_heldout = evaluate_teacher(
        model, heldout_train, ordered_future_train.dataset, probes[2], args,
        device, representative,
    )
    result = {
        "checkpoint": cli.checkpoint,
        "dataset": args.dataset_name,
        "pred_len": args.pred_len,
        "K": args.env_num,
        "probe_type": "linear" if cli.probe_hidden <= 0 else "two_layer_mlp",
        "probe_hidden": cli.probe_hidden,
        "probe_lr": cli.probe_lr,
        "probe_epochs": cli.probe_epochs,
        "forecast_model_frozen": True,
        "teacher_test_access": False,
        "train_history": history,
        "current_zvar": {"train": current_train, "test": current_test},
        "predicted_future_zvar": {
            "train": predicted_train, "test": predicted_test,
        },
        "teacher_future_zvar": {
            "train": teacher_train,
            "heldout_train": teacher_heldout,
            "test": None,
        },
    }
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "probe_metrics.json").write_text(json.dumps(result, indent=2))
    rows = []
    for source, splits in result.items():
        if source not in (
            "current_zvar", "predicted_future_zvar", "teacher_future_zvar"
        ):
            continue
        for split, values in splits.items():
            if values is None:
                continue
            overall = values.get("overall", values)
            rows.append(
                (source, split, overall["residual_MSE"],
                 overall["explained_residual_ratio"],
                 overall["alignment_cosine"], overall["beneficial_ratio"])
            )
    table = [
        "| source | split | residual MSE | explained ratio | cosine | beneficial |",
        "|---|---|---:|---:|---:|---:|",
    ] + [
        f"| {source} | {split} | {mse:.6g} | {explained:.6g} | {cosine:.6g} | {beneficial:.6g} |"
        for source, split, mse, explained, cosine, beneficial in rows
    ]
    (output / "probe_summary.md").write_text(
        "# Frozen Zvar residual probe\n\n" + "\n".join(table)
        + "\n\nTeacher future-Zvar is TRAIN-only; its test entry is intentionally absent.\n"
    )
    (output / "comparison.txt").write_text("\n".join(table) + "\n")


if __name__ == "__main__":
    main()
