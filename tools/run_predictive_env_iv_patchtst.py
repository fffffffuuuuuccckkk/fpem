#!/usr/bin/env python
"""Train the no-Mapping PredictiveEnvIV PatchTST ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Long chronological sweeps repeatedly materialize large tensors while several
# datasets may run in parallel. The default file_descriptor strategy can
# exhaust a conservative `ulimit -n`; file_system keeps worker IPC robust.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_SplitNpy,
)
from models.PatchTST_PredictiveEnvIV import ABLATIONS, Model
from models.domain_iv.losses import contrastive_constraint, representation_diagnostics
from models.predictive_env import (
    classification_diagnostics,
    PredictiveConflictEnvironment,
    environment_risk_consistency,
)
from models.predictive_env.future_var_consistency import future_variant_objective
from models.predictive_env.conditional_variant_predictor import (
    conditional_gain_ranking_loss,
    reliability_weighted_utility_loss,
)


class IndexedDataset(Dataset):
    def __init__(self, dataset, length=None):
        self.dataset = dataset
        self.length = len(dataset) if length is None else int(length)
        if not 0 <= self.length <= len(dataset):
            raise ValueError("IndexedDataset length is outside the base dataset")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x, y, _, _ = self.dataset[index]
        return x, y, index


class FutureTeacherIndexedDataset(IndexedDataset):
    """TRAIN-only samples with an earliest, equal-length future window."""

    def __init__(self, dataset, seq_len, valid_only=False):
        self.seq_len = int(seq_len)
        available = len(dataset.data_x) - 2 * self.seq_len + 1
        self.valid_length = min(len(dataset), max(available, 0))
        if self.valid_length <= 0:
            raise ValueError("TRAIN split is too short for equal past/future windows")
        super().__init__(dataset, self.valid_length if valid_only else len(dataset))

    def __getitem__(self, index):
        x, y, sample_id = super().__getitem__(index)
        valid = index < self.valid_length
        if valid:
            future_start = index + self.seq_len
            future_end = future_start + self.seq_len
            future = self.dataset.data_x[future_start:future_end]
            if len(future) != self.seq_len or future_end > len(self.dataset.data_x):
                raise RuntimeError("Future teacher window crossed the TRAIN boundary")
        else:
            # Preserve every legacy forecasting sample while explicitly masking
            # the tail samples that cannot supply a full TRAIN-only teacher.
            future = self.dataset.data_x[: self.seq_len] * 0
        return x, y, sample_id, future, valid


class GateAccumulator:
    def __init__(self, enabled, near_zero_threshold=0.1):
        self.enabled = bool(enabled)
        self.threshold = float(near_zero_threshold)
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.absolute_total = 0.0
        self.positive = 0
        self.negative = 0
        self.near_zero = 0

    def update(self, gate):
        if not self.enabled:
            return
        gate = gate.detach().float()
        self.count += gate.numel()
        self.total += float(gate.sum())
        self.square_total += float(gate.square().sum())
        self.absolute_total += float(gate.abs().sum())
        self.positive += int((gate > 0).sum())
        self.negative += int((gate < 0).sum())
        self.near_zero += int((gate.abs() < self.threshold).sum())

    def result(self):
        if not self.enabled or self.count == 0:
            return {
                "gate/mean": float("nan"),
                "gate/std": float("nan"),
                "gate/positive_ratio": float("nan"),
                "gate/negative_ratio": float("nan"),
                "gate/abs_mean": float("nan"),
                "gate/near_zero_ratio": float("nan"),
            }
        mean = self.total / self.count
        variance = max(self.square_total / self.count - mean * mean, 0.0)
        return {
            "gate/mean": mean,
            "gate/std": variance**0.5,
            "gate/positive_ratio": self.positive / self.count,
            "gate/negative_ratio": self.negative / self.count,
            "gate/abs_mean": self.absolute_total / self.count,
            "gate/near_zero_ratio": self.near_zero / self.count,
        }


class DecompositionAccumulator:
    def __init__(self):
        self.element_count = 0
        self.reconstruction_absolute_sum = 0.0
        self.sample_count = 0
        self.invariant_energy_sum = 0.0
        self.variant_energy_sum = 0.0

    def update(self, hidden, z_inv, z_var):
        hidden = hidden.detach().float()
        z_inv = z_inv.detach().float()
        z_var = z_var.detach().float()
        self.element_count += hidden.numel()
        self.reconstruction_absolute_sum += float(
            ((z_inv + z_var) - hidden).abs().sum()
        )
        hidden_norm = hidden.flatten(1).norm(dim=1).clamp_min(1e-8)
        self.invariant_energy_sum += float(
            (z_inv.flatten(1).norm(dim=1) / hidden_norm).sum()
        )
        self.variant_energy_sum += float(
            (z_var.flatten(1).norm(dim=1) / hidden_norm).sum()
        )
        self.sample_count += hidden.shape[0]

    def result(self):
        return {
            "reconstruction_error": (
                self.reconstruction_absolute_sum / max(self.element_count, 1)
            ),
            "inv_energy_ratio": (
                self.invariant_energy_sum / max(self.sample_count, 1)
            ),
            "var_energy_ratio": (
                self.variant_energy_sum / max(self.sample_count, 1)
            ),
        }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_config(args, experiment):
    return SimpleNamespace(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        factor=3,
        activation="gelu",
        patch_len=16,
        predictive_env_stride=8,
        predictive_env_ablation=experiment,
        predictive_env_bottleneck=args.bottleneck,
        predictive_env_num=args.env_num,
        predictive_env_grl_weight=args.grl_weight,
        representation_constraint=args.representation_constraint,
        decomposition_type=args.decomposition_type,
        variant_fusion_mode=args.variant_fusion_mode,
        fusion_scale_calibration=args.fusion_scale_calibration,
    )


def make_train_loader(train_dataset, args):
    """Create an experiment-local shuffled loader with an isolated RNG."""
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(args.seed)
    return DataLoader(
        FutureTeacherIndexedDataset(train_dataset, args.seq_len),
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=args.num_workers,
        drop_last=False,
    )


def build_data(args):
    size = [args.seq_len, min(48, args.seq_len // 2), args.pred_len]
    dataset_classes = {
        "ett_hour": Dataset_ETT_hour,
        "ett_minute": Dataset_ETT_minute,
        "custom": Dataset_Custom,
        "split_npy": Dataset_SplitNpy,
    }
    dataset_class = dataset_classes[args.data_class]
    common = dict(
        args=SimpleNamespace(augmentation_ratio=0),
        root_path=args.root_path,
        size=size,
        features="M",
        data_path=args.data_path,
        target=args.target,
        scale=True,
        timeenc=1,
        freq=args.freq,
    )
    # Validation is deliberately not instantiated: environment inference sees
    # complete chronological TRAIN only; TEST is used only for final metrics.
    datasets = {
        split: dataset_class(flag=split, **common) for split in ("train", "test")
    }
    ordered_train = DataLoader(
        IndexedDataset(datasets["train"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test = DataLoader(
        IndexedDataset(datasets["test"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    ordered_future_train = DataLoader(
        FutureTeacherIndexedDataset(datasets["train"], args.seq_len, valid_only=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return datasets, ordered_train, ordered_future_train, test


@torch.no_grad()
def collect(
    model,
    loader,
    args,
    device,
    components=False,
    prediction_source="full",
    h_diagnostics=False,
):
    model.eval()
    predictions, targets, ids, z_inv, z_var = [], [], [], [], []
    inv_logits, var_logits = [], []
    h_predictions, h_norms = [], []
    for x, y, sample_id in loader:
        output = model.forward_components(x.float().to(device))
        if prediction_source == "full":
            predictions.append(output["prediction"].cpu())
        elif prediction_source == "h_reference":
            predictions.append(output["h_prediction"].cpu())
        else:
            raise ValueError(f"Unknown prediction source: {prediction_source}")
        targets.append(y[:, -args.pred_len :].float())
        ids.append(sample_id)
        if h_diagnostics:
            h_predictions.append(output["h_prediction"].cpu())
            h_norms.append(
                output["hidden_tokens"].square().mean((1, 2, 3)).sqrt().cpu()
            )
        if components:
            z_inv.append(output["z_inv"].cpu())
            z_var.append(output["z_var"].cpu())
            batch_inv_logits, batch_var_logits = model.environment_logits(
                output["z_inv"], output["z_var"]
            )
            inv_logits.append(batch_inv_logits.cpu())
            var_logits.append(batch_var_logits.cpu())
    result = [torch.cat(predictions), torch.cat(targets), torch.cat(ids)]
    if components:
        result.extend(
            [
                torch.cat(z_inv),
                torch.cat(z_var),
                torch.cat(inv_logits),
                torch.cat(var_logits),
            ]
        )
    if h_diagnostics:
        result.extend([torch.cat(h_predictions), torch.cat(h_norms)])
    return result


def train_epoch(
    model,
    loader,
    assignment,
    optimizer,
    args,
    device,
    reference_h_predictions=None,
):
    model.train()
    values = []
    variant_predictive_values = []
    utility_values = []
    conditional_gain_loss_values = []
    conditional_pair_values = []
    conditional_shuffle_values = []
    conditional_gain_values = []
    conditional_positive_values = []
    h_anchor_values = []
    future_totals = {
        "future_var_loss": 0.0,
        "future_var_cosine": 0.0,
        "future_var_l2": 0.0,
        "future_var_pred_norm": 0.0,
        "future_var_target_norm": 0.0,
    }
    future_count = 0
    gate_diagnostics = GateAccumulator(
        model.level == "representation"
        and model.decomposition_type in ("signed_gate", "complementary_gate"),
        args.gate_near_zero_threshold,
    )
    for step, batch in enumerate(loader):
        if args.max_train_batches and step >= args.max_train_batches:
            break
        x, y, sample_id, future, future_valid = batch
        x = x.float().to(device)
        y = y[:, -args.pred_len :].float().to(device)
        output = model.forward_components(x)
        isolate_z_gradients = (
            args.predictive_env_refactor_mode == "h_reference_grad_isolated"
        )
        environment_output = (
            model.detached_environment_components(output)
            if isolate_z_gradients and model.level == "representation"
            else output
        )
        gate_diagnostics.update(output["decomposition_gate"])
        sample_loss = (output["prediction"] - y).square().mean((1, 2))
        loss = sample_loss.mean()
        if model.level != "baseline":
            q = torch.as_tensor(
                assignment[sample_id.numpy()], device=device
            ).detach()
            risk_sample_loss = (
                (environment_output["prediction"] - y).square().mean((1, 2))
                if isolate_z_gradients and model.level == "representation"
                else sample_loss
            )
            risk_loss, _ = environment_risk_consistency(risk_sample_loss, q)
            loss = loss + args.lambda_risk * risk_loss
            if model.level == "representation":
                invariant_sample_loss = (
                    output["invariant_prediction"] - y
                ).square().mean((1, 2))
                invariant_prediction = invariant_sample_loss.mean()
                conditional_sample_loss = (
                    environment_output["conditional_variant_prediction"] - y
                ).square().mean((1, 2))
                variant_predictive_loss = conditional_sample_loss.mean()
                utility_full_loss = (
                    risk_sample_loss if isolate_z_gradients else sample_loss
                )
                utility_invariant_loss = (
                    (environment_output["invariant_prediction"] - y)
                    .square()
                    .mean((1, 2))
                    if isolate_z_gradients
                    else invariant_sample_loss
                )
                reliability = environment_output["feature_gate"].flatten(1).mean(dim=1)
                utility_loss = reliability_weighted_utility_loss(
                    reliability, utility_full_loss, utility_invariant_loss
                )
                if args.representation_constraint == "contrastive":
                    domain_loss, _ = contrastive_constraint(
                        environment_output["z_inv"],
                        environment_output["z_var"],
                        q,
                        args.temperature,
                        args.lambda_con_inv,
                    )
                else:
                    domain_loss, _ = model.environment_classification_loss(
                        environment_output["z_inv"],
                        environment_output["z_var"],
                        q,
                    )
                loss = (
                    loss
                    + args.lambda_invpred * invariant_prediction
                    + args.lambda_domain * domain_loss
                    + args.lambda_z
                    * environment_output["feature_gate"].abs().mean()
                )
                # Keep disabled auxiliary losses out of the autograd graph so
                # lambda=0 exactly preserves the previous training trajectory.
                if args.lambda_var_predictive != 0:
                    loss = loss + (
                        args.lambda_var_predictive * variant_predictive_loss
                    )
                if args.lambda_var_utility != 0:
                    loss = loss + args.lambda_var_utility * utility_loss
                variant_predictive_values.append(
                    float(variant_predictive_loss.detach())
                )
                utility_values.append(float(utility_loss.detach()))
                if x.shape[0] > 1:
                    shuffled_sample_loss = (
                        output["conditional_shuffled_prediction"] - y
                        if not isolate_z_gradients
                        else environment_output[
                            "conditional_shuffled_prediction"
                        ] - y
                    ).square().mean((1, 2))
                    conditional_gain = (
                        shuffled_sample_loss - conditional_sample_loss
                    )
                    conditional_gain_loss = conditional_gain_ranking_loss(
                        conditional_sample_loss,
                        shuffled_sample_loss,
                        args.var_conditional_margin,
                    )
                    if args.lambda_var_conditional_gain != 0:
                        loss = loss + (
                            args.lambda_var_conditional_gain
                            * conditional_gain_loss
                        )
                    conditional_gain_loss_values.append(
                        float(conditional_gain_loss.detach())
                    )
                    conditional_pair_values.extend(
                        conditional_sample_loss.detach().cpu().tolist()
                    )
                    conditional_shuffle_values.extend(
                        shuffled_sample_loss.detach().cpu().tolist()
                    )
                    conditional_gain_values.extend(
                        conditional_gain.detach().cpu().tolist()
                    )
                    conditional_positive_values.extend(
                        (conditional_gain.detach() > 0).float().cpu().tolist()
                    )
                if args.lambda_future_var > 0:
                    future_valid = future_valid.to(device=device, dtype=torch.bool)
                    if future_valid.any():
                        future_target = model.future_variant_target(
                            future[future_valid.cpu()].float().to(device)
                        )
                        future_prediction = model.future_var_predictor(
                            environment_output["z_var"][future_valid]
                        )
                        future_metrics = future_variant_objective(
                            future_prediction, future_target
                        )
                        loss = loss + args.lambda_future_var * future_metrics["loss"]
                        valid_count = int(future_valid.sum())
                        future_count += valid_count
                        future_totals["future_var_loss"] += (
                            float(future_metrics["loss"].detach()) * valid_count
                        )
                        for output_key, metric_key in (
                            ("future_var_cosine", "cosine"),
                            ("future_var_l2", "l2"),
                            ("future_var_pred_norm", "prediction_norm"),
                            ("future_var_target_norm", "target_norm"),
                        ):
                            future_totals[output_key] += float(
                                future_metrics[metric_key].detach().sum()
                            )
                if model.decomposition_type == "signed_gate":
                    gate_activity = 1.0 - environment_output[
                        "decomposition_gate"
                    ].abs()
                    loss = loss + args.lambda_gate_activity * gate_activity.mean()
                elif model.decomposition_type == "complementary_gate":
                    gate_activity = 1.0 - environment_output[
                        "decomposition_gate"
                    ].abs()
                    loss = (
                        loss
                        + args.lambda_complementary_gate_activity
                        * gate_activity.mean()
                    )
        if args.lambda_h_anchor != 0:
            if reference_h_predictions is None:
                raise RuntimeError("H anchor requires pretrained reference cache")
            reference_prediction = reference_h_predictions[sample_id].to(device)
            h_anchor_loss = F.mse_loss(
                output["h_prediction"], reference_prediction.detach()
            )
            loss = loss + args.lambda_h_anchor * h_anchor_loss
            h_anchor_values.append(float(h_anchor_loss.detach()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        values.append(float(loss.detach()))
    result = {"loss": float(np.mean(values))}
    result["var_predictive_loss"] = (
        float(np.mean(variant_predictive_values))
        if variant_predictive_values
        else float("nan")
    )
    result["var_utility_loss"] = (
        float(np.mean(utility_values)) if utility_values else float("nan")
    )
    result["var_conditional_gain_loss"] = (
        float(np.mean(conditional_gain_loss_values))
        if conditional_gain_loss_values
        else float("nan")
    )
    result["conditional_pair_MSE"] = (
        float(np.mean(conditional_pair_values))
        if conditional_pair_values
        else float("nan")
    )
    result["conditional_shuffle_MSE"] = (
        float(np.mean(conditional_shuffle_values))
        if conditional_shuffle_values
        else float("nan")
    )
    result["conditional_gain_mean"] = (
        float(np.mean(conditional_gain_values))
        if conditional_gain_values
        else float("nan")
    )
    result["conditional_positive_ratio"] = (
        float(np.mean(conditional_positive_values))
        if conditional_positive_values
        else float("nan")
    )
    result["h_anchor_loss"] = (
        float(np.mean(h_anchor_values)) if h_anchor_values else float("nan")
    )
    for key, total in future_totals.items():
        result[key] = total / future_count if future_count else float("nan")
    result.update(gate_diagnostics.result())
    return result


def spearman_correlation(left, right):
    left = torch.as_tensor(left, dtype=torch.float64).flatten()
    right = torch.as_tensor(right, dtype=torch.float64).flatten()
    finite = torch.isfinite(left) & torch.isfinite(right)
    left, right = left[finite], right[finite]
    if left.numel() < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    left_rank = torch.argsort(torch.argsort(left)).to(torch.float64)
    right_rank = torch.argsort(torch.argsort(right)).to(torch.float64)
    return float(torch.corrcoef(torch.stack((left_rank, right_rank)))[0, 1])


def pearson_correlation(left, right):
    left = torch.as_tensor(left, dtype=torch.float64).flatten()
    right = torch.as_tensor(right, dtype=torch.float64).flatten()
    finite = torch.isfinite(left) & torch.isfinite(right)
    left, right = left[finite], right[finite]
    if left.numel() < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(torch.corrcoef(torch.stack((left, right)))[0, 1])


@torch.no_grad()
def build_pretrained_h_cache(model, ordered_train, args, device):
    _, _, sample_id, h_prediction, h_norm = collect(
        model,
        ordered_train,
        args,
        device,
        prediction_source="h_reference",
        h_diagnostics=True,
    )
    count = len(ordered_train.dataset)
    cached_prediction = torch.empty(
        count, *h_prediction.shape[1:], dtype=h_prediction.dtype
    )
    cached_norm = torch.empty(count, dtype=h_norm.dtype)
    cached_prediction[sample_id] = h_prediction
    cached_norm[sample_id] = h_norm
    return cached_prediction, cached_norm


def h_stability_diagnostics(
    h_prediction,
    target,
    sample_id,
    h_norm,
    reference_prediction,
    reference_norm,
):
    reference_prediction = reference_prediction[sample_id]
    reference_norm = reference_norm[sample_id]
    return {
        "h_prediction_MSE": float((h_prediction - target).square().mean()),
        "h_prediction_MAE": float((h_prediction - target).abs().mean()),
        "h_prediction_drift_from_pretrained": float(
            (h_prediction - reference_prediction).square().mean()
        ),
        "h_prediction_corr_with_pretrained": pearson_correlation(
            h_prediction, reference_prediction
        ),
        "H_norm": float(h_norm.mean()),
        "H_norm_ratio_to_pretrained": float(
            (h_norm / reference_norm.clamp_min(1e-8)).mean()
        ),
    }


@torch.no_grad()
def evaluate_future_var(model, ordered_future_train, args, device):
    """Evaluate the auxiliary task only on valid TRAIN future windows."""
    model.eval()
    cosine_values, l2_values = [], []
    prediction_norms, target_norms, sample_gains = [], [], []
    for step, (x, y, _, future, future_valid) in enumerate(ordered_future_train):
        if args.max_eval_batches and step >= args.max_eval_batches:
            break
        x = x.float().to(device)
        if not bool(future_valid.all()):
            raise RuntimeError("Diagnostic future loader contains an invalid teacher")
        y = y[:, -args.pred_len :].float().to(device)
        output = model.forward_components(x)
        target = model.future_variant_target(future.float().to(device))
        prediction = model.future_var_predictor(output["z_var"])
        metrics = future_variant_objective(prediction, target)
        cosine_values.append(metrics["cosine"].cpu())
        l2_values.append(metrics["l2"].cpu())
        prediction_norms.append(metrics["prediction_norm"].cpu())
        target_norms.append(metrics["target_norm"].cpu())
        invariant_mse = (output["invariant_prediction"] - y).square().mean((1, 2))
        full_mse = (output["prediction"] - y).square().mean((1, 2))
        sample_gains.append((invariant_mse - full_mse).cpu())
    cosine = torch.cat(cosine_values)
    l2 = torch.cat(l2_values)
    prediction_norm = torch.cat(prediction_norms)
    target_norm = torch.cat(target_norms)
    sample_gain = torch.cat(sample_gains)
    return {
        "future_var_loss": float(1.0 - cosine.mean()),
        "future_var_cosine": float(cosine.mean()),
        "future_var_l2": float(l2.mean()),
        "future_var_pred_norm": float(prediction_norm.mean()),
        "future_var_target_norm": float(target_norm.mean()),
        "corr_future_var_similarity_with_sample_gain": spearman_correlation(
            cosine, sample_gain
        ),
        "corr_future_var_similarity_with_sample_gain_type": "Spearman",
        "future_teacher_sample_count": int(cosine.numel()),
        "future_teacher_source": (
            "TRAIN only; earliest seq_len steps from forecast origin; "
            "stop-gradient shared encoder/decomposer; absent from inference"
        ),
    }


@torch.no_grad()
def evaluate(
    model,
    test_loader,
    ordered_train,
    ordered_future_train,
    assignment,
    args,
    device,
):
    model.eval()
    full_errors, invariant_errors, conditional_errors = [], [], []
    gates, sample_reliabilities, feature_ratios = [], [], []
    fusion_rms_ratios = []
    conditional_pair_losses, conditional_shuffle_losses = [], []
    conditional_main_gains = []
    decomposition_gate_diagnostics = GateAccumulator(
        model.level == "representation"
        and model.decomposition_type in ("signed_gate", "complementary_gate"),
        args.gate_near_zero_threshold,
    )
    decomposition_diagnostics = DecompositionAccumulator()
    for step, (x, y, _) in enumerate(test_loader):
        if args.max_eval_batches and step >= args.max_eval_batches:
            break
        x = x.float().to(device)
        y = y[:, -args.pred_len :].float().to(device)
        output = model.forward_components(x)
        decomposition_gate_diagnostics.update(output["decomposition_gate"])
        decomposition_diagnostics.update(
            output["hidden_tokens"],
            output["z_inv_tokens"],
            output["z_var_tokens"],
        )
        full_errors.append((output["prediction"] - y).cpu())
        invariant_errors.append((output["invariant_prediction"] - y).cpu())
        conditional_errors.append(
            (output["conditional_variant_prediction"] - y).cpu()
        )
        if x.shape[0] > 1:
            pair_loss = (
                output["conditional_variant_prediction"] - y
            ).square().mean((1, 2))
            shuffle_loss = (
                output["conditional_shuffled_prediction"] - y
            ).square().mean((1, 2))
            conditional_pair_losses.append(pair_loss.cpu())
            conditional_shuffle_losses.append(shuffle_loss.cpu())
            conditional_main_gains.append(
                (
                    (output["invariant_prediction"] - y).square().mean((1, 2))
                    - (output["prediction"] - y).square().mean((1, 2))
                ).cpu()
            )
        gates.append(output["feature_gate"].reshape(-1).cpu())
        sample_reliabilities.append(
            output["feature_gate"].flatten(1).mean(dim=1).cpu()
        )
        numerator = output["feature_variation"].flatten(1).norm(dim=1)
        denominator = output["z_inv_tokens"].flatten(1).norm(dim=1).clamp_min(1e-8)
        feature_ratios.append((numerator / denominator).cpu())
        hidden_rms = output["hidden_tokens"].square().mean((-2, -1)).sqrt()
        fused_rms = output["final_tokens"].square().mean((-2, -1)).sqrt()
        fusion_rms_ratios.append(
            (fused_rms / hidden_rms.clamp_min(1e-8)).reshape(-1).cpu()
        )
    full_error = torch.cat(full_errors)
    invariant_error = torch.cat(invariant_errors)
    conditional_error = torch.cat(conditional_errors)
    sample_reliability = torch.cat(sample_reliabilities)
    loss_full_per_sample = full_error.square().mean((1, 2))
    loss_inv_per_sample = invariant_error.square().mean((1, 2))
    gain = loss_inv_per_sample - loss_full_per_sample
    conditional_pair_loss = torch.cat(conditional_pair_losses)
    conditional_shuffle_loss = torch.cat(conditional_shuffle_losses)
    conditional_gain = conditional_shuffle_loss - conditional_pair_loss
    conditional_main_gain = torch.cat(conditional_main_gains)
    gate = torch.cat(gates).numpy()
    result = {
        "environment_count": args.env_num,
        "representation_constraint": args.representation_constraint,
        "decomposition_type": args.decomposition_type,
        "variant_fusion_mode": args.variant_fusion_mode,
        "fusion_scale_calibration": args.fusion_scale_calibration,
        "predictive_env_refactor_mode": args.predictive_env_refactor_mode,
        "lambda_invpred": args.lambda_invpred,
        "lambda_future_var": args.lambda_future_var,
        "lambda_var_predictive": args.lambda_var_predictive,
        "lambda_var_utility": args.lambda_var_utility,
        "lambda_var_conditional_gain": args.lambda_var_conditional_gain,
        "var_conditional_margin": args.var_conditional_margin,
        "MSE": float(full_error.square().mean()),
        "MAE": float(full_error.abs().mean()),
        "inv_only_MSE": float(invariant_error.square().mean()),
        "inv_only_MAE": float(invariant_error.abs().mean()),
        "inv_only_minus_full_MSE": float(
            invariant_error.square().mean() - full_error.square().mean()
        ),
        "inv_MSE": float(invariant_error.square().mean()),
        "inv_minus_full": float(gain.mean()),
        "var_predictive_loss": float(conditional_error.square().mean()),
        "conditional_var_MSE": float(conditional_error.square().mean()),
        "conditional_pair_MSE": float(conditional_pair_loss.mean()),
        "conditional_shuffle_MSE": float(conditional_shuffle_loss.mean()),
        "conditional_gain_mean": float(conditional_gain.mean()),
        "conditional_gain_median": float(conditional_gain.median()),
        "conditional_positive_ratio": float(
            (conditional_gain > 0).float().mean()
        ),
        "conditional_gain_p10": float(torch.quantile(conditional_gain, 0.1)),
        "conditional_gain_p50": float(torch.quantile(conditional_gain, 0.5)),
        "conditional_gain_p90": float(torch.quantile(conditional_gain, 0.9)),
        "corr_conditional_gain_main_gain": spearman_correlation(
            conditional_gain, conditional_main_gain
        ),
        "corr_conditional_gain_main_gain_type": "Spearman",
        "mean_gain": float(gain.mean()),
        "median_gain": float(gain.median()),
        "positive_gain_ratio": float((gain > 0).float().mean()),
        "gain_p10": float(torch.quantile(gain, 0.1)),
        "gain_p50": float(torch.quantile(gain, 0.5)),
        "gain_p90": float(torch.quantile(gain, 0.9)),
        "corr_reliability_gain": spearman_correlation(sample_reliability, gain),
        "corr_reliability_gain_type": "Spearman",
        "g_z_mean": float(gate.mean()),
        "g_z_p10": float(np.quantile(gate, 0.1)),
        "g_z_p50": float(np.quantile(gate, 0.5)),
        "g_z_p90": float(np.quantile(gate, 0.9)),
        "feature_variation_to_Zinv": float(torch.cat(feature_ratios).mean()),
        "fusion_rms_ratio_mean": float(torch.cat(fusion_rms_ratios).mean()),
        "fusion_rms_ratio_p10": float(torch.quantile(torch.cat(fusion_rms_ratios), 0.1)),
        "fusion_rms_ratio_p50": float(torch.quantile(torch.cat(fusion_rms_ratios), 0.5)),
        "fusion_rms_ratio_p90": float(torch.quantile(torch.cat(fusion_rms_ratios), 0.9)),
        "mapping_variation": "removed",
    }
    result.update(decomposition_gate_diagnostics.result())
    result.update(decomposition_diagnostics.result())
    result.update(
        {
            "g_mean": result["g_z_mean"],
            "g_p10": result["g_z_p10"],
            "g_p50": result["g_z_p50"],
            "g_p90": result["g_z_p90"],
            "gate_pos": result["gate/positive_ratio"],
            "gate_neg": result["gate/negative_ratio"],
            "gate_abs": result["gate/abs_mean"],
            "inv_energy": result["inv_energy_ratio"],
            "var_energy": result["var_energy_ratio"],
        }
    )
    result.update(evaluate_future_var(model, ordered_future_train, args, device))
    if assignment is not None:
        prediction, target, _, z_inv, z_var, inv_logits, var_logits = collect(
            model, ordered_train, args, device, components=True
        )
        q = torch.as_tensor(assignment)
        sample_loss = (prediction - target).square().mean((1, 2))
        _, risks = environment_risk_consistency(sample_loss, q)
        result["per_env_MSE"] = risks.tolist()
        count = min(len(q), args.diagnostic_samples)
        selected = torch.linspace(0, len(q) - 1, count).long()
        diagnostics = representation_diagnostics(
            z_inv[selected], z_var[selected], q[selected]
        )
        result.update({key: float(value) for key, value in diagnostics.items()})
        result["diagnostic_sample_count"] = count
        if args.representation_constraint == "classification":
            classifier_diagnostics = classification_diagnostics(
                inv_logits, var_logits, q
            )
            result.update(
                {
                    key: float(value)
                    for key, value in classifier_diagnostics.items()
                }
            )
            result["var_acc"] = result["var_env_accuracy_argmax"]
            result["inv_acc"] = result["inv_env_accuracy_argmax"]
        else:
            for key in (
                "var_env_soft_ce",
                "inv_env_soft_ce",
                "var_env_accuracy_argmax",
                "inv_env_accuracy_argmax",
                "var_env_entropy",
                "inv_env_entropy",
            ):
                result[key] = float("nan")
            result["var_acc"] = float("nan")
            result["inv_acc"] = float("nan")
    return result


def write_summary(path, rows):
    lines = [
        "experiment constraint decomposition fusion lambda_invpred lambda_future_var "
        "lambda_var_predictive lambda_var_utility lambda_var_conditional_gain "
        "var_conditional_margin "
        "env_count MSE MAE "
        "inv_MSE improvement conditional_MSE mean_gain median_gain positive_gain "
        "gain_p10 gain_p50 gain_p90 corr_reliability_gain conditional_pair_MSE "
        "conditional_shuffle_MSE conditional_gain conditional_positive "
        "corr_conditional_main_gain g_z "
        "DeltaZ/Zinv corr_inv_env corr_var_env conflict Dwithin_grad "
        "Dbetween_grad separation random_z min_mass final_env_sim_corr "
        "future_var_loss future_var_cosine future_var_l2 future_var_pred_norm "
        "future_var_target_norm future_similarity_gain_corr"
    ]
    for row in rows:
        lines.append(
            "{experiment} {representation_constraint} {decomposition_type} "
            "{variant_fusion_mode} {lambda_invpred} {lambda_future_var} "
            "{lambda_var_predictive} {lambda_var_utility} "
            "{lambda_var_conditional_gain} {var_conditional_margin} "
            "{environment_count} "
            "{MSE:.9f} {MAE:.9f} "
            "{inv_only_MSE:.9f} {inv_only_minus_full_MSE:.9f} "
            "{conditional_var_MSE:.9f} {mean_gain:.9f} {median_gain:.9f} "
            "{positive_gain_ratio:.6f} {gain_p10:.9f} {gain_p50:.9f} "
            "{gain_p90:.9f} {corr_reliability_gain:.6f} "
            "{conditional_pair_MSE:.9f} {conditional_shuffle_MSE:.9f} "
            "{conditional_gain_mean:.9f} {conditional_positive_ratio:.6f} "
            "{corr_conditional_gain_main_gain:.6f} "
            "{g_z_mean:.6f} {feature_variation_to_Zinv:.6f} "
            "{corr_inv_env:.6f} {corr_var_env:.6f} "
            "{gradient_disagreement:.8g} {D_within_grad:.8g} "
            "{D_between_grad:.8g} {gradient_separation:.6f} "
            "{random_partition_z_score:.6f} {min_environment_mass:.6f} "
            "{final_environment_similarity_correlation:.6f} "
            "{future_var_loss:.6f} {future_var_cosine:.6f} "
            "{future_var_l2:.6f} {future_var_pred_norm:.6f} "
            "{future_var_target_norm:.6f} "
            "{corr_future_var_similarity_with_sample_gain:.6f}".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def reference_signature(args):
    names = (
        "dataset_name",
        "data_class",
        "data_path",
        "target",
        "freq",
        "seq_len",
        "pred_len",
        "d_model",
        "d_ff",
        "n_heads",
        "e_layers",
        "dropout",
        "seed",
        "warmup_epochs",
    )
    signature = {name: getattr(args, name) for name in names}
    signature["protocol_version"] = 4
    return signature


def shared_reference_state(model):
    excluded_prefixes = (
        "environment_classification.",
        "decomposer.",
        "feature_variation.",
        "direct_variant_fusion.",
        "future_var_predictor.",
        "variant_conditional_predictor.",
        "h_forecast_head.",
    )
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(excluded_prefixes)
    }


def load_shared_reference_state(model, state):
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing_prefixes = (
        "environment_classification.",
        "decomposer.",
        "feature_variation.",
        "direct_variant_fusion.",
        "future_var_predictor.",
        "variant_conditional_predictor.",
        "h_forecast_head.",
    )
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            f"Invalid shared reference state: missing={invalid_missing}, "
            f"unexpected={unexpected}"
        )
    model.initialize_h_forecast_head()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_reference(args, train_loader, device):
    reference = Model(model_config(args, "A0")).to(device)
    checkpoint_path = Path(args.reference_checkpoint) if args.reference_checkpoint else None
    signature = reference_signature(args)
    if checkpoint_path and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        if payload.get("signature") != signature:
            raise ValueError("Shared reference checkpoint configuration mismatch")
        load_shared_reference_state(reference, payload["state_dict"])
        source = "loaded"
    else:
        if args.require_reference_checkpoint:
            raise FileNotFoundError(
                f"Required shared reference checkpoint is missing: {checkpoint_path}"
            )
        optimizer = torch.optim.Adam(reference.parameters(), lr=args.lr)
        dummy = np.full(
            (len(train_loader.dataset), args.env_num),
            1.0 / args.env_num,
            dtype="float32",
        )
        for epoch in range(args.warmup_epochs):
            value = train_epoch(
                reference, train_loader, dummy, optimizer, args, device
            )
            print("reference warmup", epoch + 1, value, flush=True)
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"signature": signature, "state_dict": shared_reference_state(reference)},
                checkpoint_path,
            )
        source = "created"
    if checkpoint_path is None:
        raise ValueError("--reference_checkpoint is required for reproducible sweeps")
    checkpoint_hash = file_sha256(checkpoint_path)
    print(
        f"shared_reference source={source} path={checkpoint_path} "
        f"sha256={checkpoint_hash}",
        flush=True,
    )
    state = {
        key: value.detach().cpu()
        for key, value in shared_reference_state(reference).items()
    }
    return state, checkpoint_hash, source


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="./dataset/ETT-small/")
    parser.add_argument("--dataset_name", default="ETTh1")
    parser.add_argument(
        "--data_class",
        choices=("ett_hour", "ett_minute", "custom", "split_npy"),
        default="ett_hour",
    )
    parser.add_argument("--data_path", default="ETTh1.csv")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--experiments", default="A0,A1,A2,A3,A4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--env_num", type=int, default=6)
    parser.add_argument("--env_update_interval", type=int, default=2)
    parser.add_argument("--stage_epochs", type=int, default=2)
    parser.add_argument("--eiil_steps", type=int, default=50)
    parser.add_argument("--eiil_lr", type=float, default=0.1)
    parser.add_argument("--balance_weight", type=float, default=10.0)
    parser.add_argument("--entropy_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_risk", type=float, default=0.1)
    parser.add_argument("--lambda_invpred", type=float, default=0.5)
    parser.add_argument("--lambda_future_var", type=float, default=0.0)
    parser.add_argument("--lambda_var_predictive", type=float, default=0.0)
    parser.add_argument("--lambda_var_utility", type=float, default=0.0)
    parser.add_argument("--lambda_var_conditional_gain", type=float, default=0.0)
    parser.add_argument("--var_conditional_margin", type=float, default=0.0)
    parser.add_argument("--lambda_h_anchor", type=float, default=0.0)
    parser.add_argument("--lambda_domain", type=float, default=0.1)
    parser.add_argument("--lambda_z", type=float, default=1e-3)
    parser.add_argument("--lambda_con_inv", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--representation_constraint",
        choices=("contrastive", "classification"),
        default="contrastive",
    )
    parser.add_argument("--grl_weight", type=float, default=1.0)
    parser.add_argument(
        "--decomposition_type",
        choices=("projection", "signed_gate", "complementary_gate"),
        default="projection",
    )
    parser.add_argument("--gate_near_zero_threshold", type=float, default=0.1)
    parser.add_argument("--lambda_gate_activity", type=float, default=1e-4)
    parser.add_argument(
        "--lambda_complementary_gate_activity", type=float, default=0.0
    )
    parser.add_argument(
        "--variant_fusion_mode",
        choices=("legacy", "off", "direct_gated"),
        default="legacy",
    )
    parser.add_argument(
        "--fusion_scale_calibration", choices=("none", "rms"), default="none"
    )
    parser.add_argument(
        "--predictive_env_refactor_mode",
        choices=("current", "h_reference", "h_reference_grad_isolated"),
        default="current",
    )
    parser.add_argument(
        "--environment_matching",
        choices=("overlap", "gradient_aware"),
        default="overlap",
    )
    parser.add_argument("--matching_q_weight", type=float, default=1.0)
    parser.add_argument("--matching_gradient_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--diagnostic_samples", type=int, default=1024)
    parser.add_argument("--reference_checkpoint", default="")
    parser.add_argument("--require_reference_checkpoint", action="store_true")
    parser.add_argument("--prepare_reference_only", action="store_true")
    parser.add_argument("--output", default="results/predictive_env_iv_ETTh1_96_96")
    return parser.parse_args()


def main():
    args = parse_args()
    experiments = [item.strip().upper() for item in args.experiments.split(",")]
    unknown = set(experiments) - set(ABLATIONS)
    if unknown:
        raise ValueError(f"Unsupported experiments after Mapping removal: {unknown}")
    seed_everything(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    datasets, ordered_train, ordered_future_train, test_loader = build_data(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    reference_train_loader = make_train_loader(datasets["train"], args)
    reference_state, reference_hash, reference_source = prepare_reference(
        args, reference_train_loader, device
    )
    reference_record = {
        "reference_checkpoint": args.reference_checkpoint,
        "reference_checkpoint_sha256": reference_hash,
        "reference_source": reference_source,
        "seed": args.seed,
        "signature": reference_signature(args),
    }
    (output / "reference_checkpoint.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in reference_record.items())
        + "\n"
    )
    if args.prepare_reference_only:
        print("reference preparation complete; no experiment was run", flush=True)
        return
    if args.stage_epochs <= 0:
        raise ValueError("--stage_epochs must be positive")
    # Cache the immutable pretrained predictive behavior once. It is indexed by
    # TRAIN sample id and used only for diagnostics and the optional H anchor.
    reference_model = Model(model_config(args, "A0")).to(device)
    load_shared_reference_state(reference_model, reference_state)
    reference_model.requires_grad_(False)
    reference_h_predictions, reference_h_norms = build_pretrained_h_cache(
        reference_model, ordered_train, args, device
    )
    del reference_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rows = []
    for experiment in experiments:
        seed_everything(args.seed)
        # A fresh private Generator guarantees the same sample order for every
        # independent experiment, regardless of model/head initialization RNG.
        train_loader = make_train_loader(datasets["train"], args)
        model = Model(model_config(args, experiment)).to(device)
        load_shared_reference_state(model, reference_state)
        first_batch = next(iter(ordered_train))[0].float().to(device)
        identity = model.identity_diagnostics(first_batch)
        print(experiment, "identity", identity, flush=True)
        manager, assignment, history, training_history = None, None, [], []
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if experiment != "A0":
            manager = PredictiveConflictEnvironment(
                len(train_loader.dataset),
                args.env_num,
                args.eiil_steps,
                args.eiil_lr,
                args.balance_weight,
                args.entropy_weight,
                args.seed,
                args.diagnostic_samples,
                args.environment_matching,
                args.matching_q_weight,
                args.matching_gradient_weight,
            )
        environment_prediction_source = (
            "full"
            if args.predictive_env_refactor_mode == "current"
            else "h_reference"
        )
        epoch, stage = 0, 0
        while epoch < args.epochs:
            if manager is not None:
                (
                    prediction,
                    target,
                    sample_id,
                    h_prediction,
                    h_norm,
                ) = collect(
                    model,
                    ordered_train,
                    args,
                    device,
                    prediction_source=environment_prediction_source,
                    h_diagnostics=True,
                )
                assignment, record = manager.update(
                    prediction, target, sample_id, source_split="train"
                )
                record.update(
                    h_stability_diagnostics(
                        h_prediction,
                        target,
                        sample_id,
                        h_norm,
                        reference_h_predictions,
                        reference_h_norms,
                    )
                )
                record.update(
                    {
                        "stage": stage,
                        "environment_prediction_source": (
                            "H -> frozen pretrained forecast head"
                            if environment_prediction_source == "h_reference"
                            else "current Z-fused prediction (legacy)"
                        ),
                        "env_assignment_similarity_to_previous": record.get(
                            "environment_similarity_correlation", float("nan")
                        ),
                    }
                )
                history.append(record)
                print(experiment, "STAGE", stage, "EIIL", record, flush=True)
            epochs_this_stage = min(args.stage_epochs, args.epochs - epoch)
            # q is detached by the manager and remains fixed for this entire stage.
            for _ in range(epochs_this_stage):
                epoch_record = train_epoch(
                    model,
                    train_loader,
                    assignment,
                    optimizer,
                    args,
                    device,
                    reference_h_predictions,
                )
                epoch_record["stage"] = stage
                epoch_record["epoch"] = epoch + 1
                training_history.append(epoch_record)
                print(
                    experiment,
                    args.decomposition_type,
                    args.representation_constraint,
                    "stage",
                    stage,
                    "epoch",
                    epoch + 1,
                    epoch_record,
                    flush=True,
                )
                epoch += 1
            stage += 1
        result = evaluate(
            model,
            test_loader,
            ordered_train,
            ordered_future_train,
            assignment,
            args,
            device,
        )
        (
            _,
            final_h_target,
            final_h_sample_id,
            final_h_prediction,
            final_h_norm,
        ) = collect(
            model,
            ordered_train,
            args,
            device,
            prediction_source="h_reference",
            h_diagnostics=True,
        )
        final_h_diagnostics = h_stability_diagnostics(
            final_h_prediction,
            final_h_target,
            final_h_sample_id,
            final_h_norm,
            reference_h_predictions,
            reference_h_norms,
        )
        result.update(final_h_diagnostics)
        latest = history[-1] if history else {}
        result.update(
            {
                "experiment": experiment,
                "identity": identity,
                "environment_updates": history,
                "training_gate_history": training_history,
                "gradient_disagreement": latest.get(
                    "gradient_disagreement", float("nan")
                ),
                "environment_source": (
                    "TRAIN X,Y only; validation/test excluded"
                ),
                "environment_prediction_source": (
                    "H -> frozen pretrained forecast head"
                    if environment_prediction_source == "h_reference"
                    else "current Z-fused prediction (legacy)"
                ),
                "z_specific_encoder_gradient_isolated": (
                    args.predictive_env_refactor_mode
                    == "h_reference_grad_isolated"
                ),
                "stage_epochs": args.stage_epochs,
                "optimization_epochs": args.epochs,
                "lambda_h_anchor": args.lambda_h_anchor,
                "h_forecast_head_frozen": all(
                    not parameter.requires_grad
                    for parameter in model.h_forecast_head.parameters()
                ),
                "reference_checkpoint": args.reference_checkpoint,
                "reference_checkpoint_sha256": reference_hash,
                "reference_source": reference_source,
                "seed": args.seed,
                "train_shuffle_seed": args.seed,
                "train_shuffle_rng": "experiment-local torch.Generator",
                "dataset_name": args.dataset_name,
                "data_class": args.data_class,
                "data_path": args.data_path,
                "target": args.target,
                "frequency": args.freq,
            }
        )
        final_diagnostic_keys = (
            "D_within_grad",
            "D_between_grad",
            "gradient_separation",
            "random_partition_z_score",
            "random_partition_conflict_mean",
            "random_partition_conflict_std",
            "min_environment_mass",
            "alignment_overlap_before",
            "alignment_overlap_after",
        )
        for key in final_diagnostic_keys:
            result[key] = latest.get(key, float("nan"))
        for key in (
            "h_prediction_MSE",
            "h_prediction_MAE",
            "h_prediction_drift_from_pretrained",
            "h_prediction_corr_with_pretrained",
            "H_norm",
            "H_norm_ratio_to_pretrained",
            "env_assignment_similarity_to_previous",
        ):
            if key == "env_assignment_similarity_to_previous":
                result[key] = latest.get(key, float("nan"))
            else:
                result.setdefault(key, latest.get(key, float("nan")))
        result["final_environment_similarity_correlation"] = latest.get(
            "environment_similarity_correlation", float("nan")
        )
        result["hungarian_permutation"] = latest.get(
            "hungarian_permutation", []
        )
        for key in ("corr_inv_env", "corr_var_env"):
            result.setdefault(key, float("nan"))
        experiment_dir = output / experiment
        experiment_dir.mkdir(exist_ok=True)
        (experiment_dir / "metrics_and_diagnostics.json").write_text(
            json.dumps(result, indent=2)
        )
        (experiment_dir / "metrics_and_diagnostics.txt").write_text(
            "\n".join(f"{key}: {value}" for key, value in result.items()) + "\n"
        )
        (experiment_dir / "environment_updates.txt").write_text(
            "\n".join(json.dumps(item) for item in history) + "\n"
        )
        rows.append(result)
        write_summary(output / "ablation_summary.txt", rows)
        print(experiment, "TEST", result["MSE"], result["MAE"], flush=True)


if __name__ == "__main__":
    main()
