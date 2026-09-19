#!/usr/bin/env python
"""Train no-Mapping PredictiveEnvIV ablations on multiple backbones."""

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
)
from models.PatchTST_PredictiveEnvIV import ABLATIONS, Model
from models.domain_iv.losses import contrastive_constraint, representation_diagnostics
from models.predictive_env import (
    classification_diagnostics,
    PredictiveConflictEnvironment,
    environment_risk_consistency,
    variant_gain_loss,
)
from models.predictive_env.future_var_consistency import future_variant_objective
from models.predictive_env.conditional_variant_predictor import (
    conditional_gain_ranking_loss,
    reliability_weighted_utility_loss,
)


class IndexedDataset(Dataset):
    def __init__(self, dataset, length=None, global_offset=0, cycle_len=1):
        self.dataset = dataset
        self.length = len(dataset) if length is None else int(length)
        self.global_offset = int(global_offset)
        self.cycle_len = int(cycle_len)
        if not 0 <= self.length <= len(dataset):
            raise ValueError("IndexedDataset length is outside the base dataset")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x, y, _, _ = self.dataset[index]
        cycle_index = (self.global_offset + index) % self.cycle_len
        return x, y, index, cycle_index


class FutureTeacherIndexedDataset(IndexedDataset):
    """TRAIN-only samples with an earliest, equal-length future window."""

    def __init__(
        self,
        dataset,
        seq_len,
        valid_only=False,
        global_offset=0,
        cycle_len=1,
    ):
        self.seq_len = int(seq_len)
        available = len(dataset.data_x) - 2 * self.seq_len + 1
        self.valid_length = min(len(dataset), max(available, 0))
        if self.valid_length <= 0:
            raise ValueError("TRAIN split is too short for equal past/future windows")
        super().__init__(
            dataset,
            self.valid_length if valid_only else len(dataset),
            global_offset,
            cycle_len,
        )

    def __getitem__(self, index):
        x, y, sample_id, cycle_index = super().__getitem__(index)
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
        future_cycle_index = (cycle_index + self.seq_len) % self.cycle_len
        return (
            x,
            y,
            sample_id,
            future,
            valid,
            cycle_index,
            future_cycle_index,
        )

    def horizon_teacher_batch(self, sample_ids, anchor_positions):
        """Build TRAIN-only causal windows ending at selected future anchors.

        For a past window ``[i, i+L)``, the teacher at horizon ``h`` is
        ``[i+h, i+h+L)``.  It therefore contains no observations later than
        the requested horizon and never crosses the chronological TRAIN end.
        Only samples valid for every requested anchor are returned.
        """
        ids = torch.as_tensor(sample_ids, dtype=torch.long).cpu()
        anchors = torch.as_tensor(anchor_positions, dtype=torch.long).cpu()
        if anchors.ndim != 1 or anchors.numel() == 0:
            raise ValueError("anchor_positions must be a non-empty vector")
        if int(anchors.min()) < 1:
            raise ValueError("future anchor positions are one-based")
        train_length = len(self.dataset.data_x)
        valid = ids + int(anchors.max()) + self.seq_len <= train_length
        valid_ids = ids[valid]
        if valid_ids.numel() == 0:
            empty_shape = (0, anchors.numel(), self.seq_len) + tuple(
                self.dataset.data_x.shape[1:]
            )
            return (
                torch.empty(empty_shape),
                valid,
                torch.empty((0, anchors.numel()), dtype=torch.long),
            )
        windows = []
        for sample_id in valid_ids.tolist():
            sample_windows = []
            for anchor in anchors.tolist():
                start = sample_id + anchor
                end = start + self.seq_len
                window = self.dataset.data_x[start:end]
                if len(window) != self.seq_len or end > train_length:
                    raise RuntimeError("Horizon teacher crossed the TRAIN boundary")
                sample_windows.append(torch.as_tensor(window))
            windows.append(torch.stack(sample_windows))
        cycles = (
            valid_ids[:, None] + anchors[None, :] + self.global_offset
        ) % self.cycle_len
        return torch.stack(windows), valid, cycles


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
        if not self.enabled or gate is None:
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


def moment_diagnostics(accumulator, prefix):
    """Expose mean/std/absolute mean without retaining feature tensors."""
    values = accumulator.result()
    return {
        f"{prefix}_mean": values["gate/mean"],
        f"{prefix}_std": values["gate/std"],
        f"{prefix}_abs_mean": values["gate/abs_mean"],
    }


def horizon_effect_objective(full_normalized, invariant_normalized, bins=4):
    """Measure the actual output influence of variation across the horizon."""
    if full_normalized.shape != invariant_normalized.shape:
        raise ValueError("full/invariant normalized predictions must align")
    effect = (full_normalized - invariant_normalized).abs().mean((0, 2))
    bin_count = min(max(int(bins), 1), int(effect.numel()))
    bin_values = torch.stack(
        [chunk.mean() for chunk in torch.tensor_split(effect, bin_count)]
    )
    mono = F.relu(bin_values[1:] - bin_values[:-1]).sum()
    horizon = torch.arange(
        1, effect.numel() + 1, device=effect.device, dtype=effect.dtype
    ) / effect.numel()
    far = (horizon * effect).mean()
    return {
        "var_effect_mean": effect.mean(),
        "var_effect_bins": bin_values,
        "var_effect_far_near_ratio": bin_values[-1]
        / bin_values[0].clamp_min(1e-8),
        "decay_mono_loss": mono,
        "decay_far_loss": far,
    }


def horizon_gain_values(full_prediction, invariant_prediction, target, bins=4):
    """Return invariant-minus-full MSE for relative horizon intervals."""
    full_h = (full_prediction - target).square().mean((0, 2))
    invariant_h = (invariant_prediction - target).square().mean((0, 2))
    gain_h = invariant_h - full_h
    bin_count = min(max(int(bins), 1), int(gain_h.numel()))
    return torch.stack(
        [chunk.mean() for chunk in torch.tensor_split(gain_h, bin_count)]
    )


def future_patch_centers(pred_len, future_patch_len):
    """Return one-based integer centers for prediction-interval patches."""
    pred_len = int(pred_len)
    patch_len = int(future_patch_len)
    if pred_len <= 0 or patch_len <= 0:
        raise ValueError("pred_len and future_patch_len must be positive")
    starts = torch.arange((pred_len + patch_len - 1) // patch_len) * patch_len
    ends = torch.minimum(starts + patch_len, torch.tensor(pred_len))
    return torch.div(starts + ends - 1, 2, rounding_mode="floor") + 1


def future_patch_mean(values, future_patch_len, pred_len=None):
    """Average ``[B,C,H]`` values inside each prediction-interval patch."""
    if values.ndim != 3:
        raise ValueError("future patch values must be [B,C,H]")
    horizon = values.shape[-1] if pred_len is None else int(pred_len)
    if horizon != values.shape[-1]:
        raise ValueError("future patch horizon mismatch")
    patch_len = int(future_patch_len)
    if patch_len <= 0:
        raise ValueError("future_patch_len must be positive")
    return torch.stack(
        [
            values[..., start : min(start + patch_len, horizon)].mean(-1)
            for start in range(0, horizon, patch_len)
        ],
        dim=-1,
    )


def horizon_anchor_positions(pred_len, anchor_count):
    """Legacy uniform anchors retained for old result readers/tests."""
    count = min(max(int(anchor_count), 1), int(pred_len))
    return torch.linspace(1, int(pred_len), count).round().long().unique()


def loss_gradient_rms(loss, parameters):
    """Detached RMS gradient of ``loss`` for a small parameter subset."""
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    square_sum = loss.detach().new_zeros(())
    element_count = 0
    for gradient in gradients:
        if gradient is not None:
            square_sum = square_sum + gradient.detach().square().sum()
            element_count += gradient.numel()
    if element_count == 0:
        return square_sum.detach()
    return (square_sum / element_count).sqrt().detach()


def gradient_relative_maturity(
    invariant_loss,
    full_loss,
    invariant_parameters,
    variant_parameters,
):
    """Current-batch maturity from invariant/variant gradient magnitudes."""
    invariant_norm = loss_gradient_rms(invariant_loss, invariant_parameters)
    variant_norm = loss_gradient_rms(full_loss, variant_parameters)
    maturity = (
        variant_norm / (invariant_norm + variant_norm + 1e-8)
    ).detach()
    return maturity, invariant_norm, variant_norm


def build_training_optimizer(model, args):
    """Build the optional single-stage differential-LR Adam optimizer."""
    if not args.differential_lr:
        return torch.optim.Adam(model.parameters(), lr=args.lr)

    grouped = {
        "backbone": [],
        "inv_head": [],
        "decomposer": [],
        "env_head": [],
        "variant": [],
        "gamma_beta": [],
        "reliability": [],
    }
    backbone_prefixes = (
        "patch_embedding.",
        "encoder.",
        "inverted_embedding.",
        "cycle_queue.",
        "cycle_input_projection.",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("horizon_future_variant.modulation_generator."):
            group = "gamma_beta"
        elif name.startswith("horizon_future_variant.reliability_net."):
            group = "reliability"
        elif name.startswith("head_linear."):
            group = "inv_head"
        elif name.startswith("decomposer."):
            group = "decomposer"
        elif name.startswith("environment_classification."):
            group = "env_head"
        elif name.startswith(backbone_prefixes):
            group = "backbone"
        else:
            group = "variant"
        grouped[group].append(parameter)

    learning_rates = {
        "backbone": args.lr_backbone,
        "inv_head": args.lr_inv_head,
        "decomposer": args.lr_decomposer,
        "env_head": args.lr_env_head,
        "variant": args.lr_variant,
        "gamma_beta": 0.2 * args.lr_gamma_beta_base,
        "reliability": args.lr_reliability,
    }
    parameter_groups = [
        {"params": parameters, "lr": learning_rates[name], "name": name}
        for name, parameters in grouped.items()
        if parameters
    ]
    assigned = sum(len(group["params"]) for group in parameter_groups)
    expected = sum(parameter.requires_grad for parameter in model.parameters())
    if assigned != expected:
        raise RuntimeError(
            f"Differential-LR parameter partition mismatch: {assigned} != {expected}"
        )
    return torch.optim.Adam(parameter_groups)


def update_gamma_beta_learning_rate(optimizer, maturity, args):
    """Apply lr=1e-4*(0.2+0.8*m) without feeding LR back into maturity."""
    target = args.lr_gamma_beta_base * (
        0.2 + 0.8 * float(torch.as_tensor(maturity).detach().cpu())
    )
    for group in optimizer.param_groups:
        if group.get("name") == "gamma_beta":
            group["lr"] = target
    return target


def invariant_anchored_variant_loss(
    invariant_prediction,
    full_prediction,
    target,
    lambda_maturity,
):
    """Invariant anchor plus normalized, maturity-weighted refinement loss."""
    invariant_loss = F.mse_loss(invariant_prediction, target)
    invariant_anchor = invariant_prediction.detach()
    full_loss = F.mse_loss(full_prediction, target)
    full_inv_distance = F.mse_loss(full_prediction, invariant_anchor)
    invariant_error = F.mse_loss(target, invariant_anchor).detach()
    anchor_loss = full_inv_distance / (invariant_error + 1e-8)
    variant_total = full_loss + anchor_loss
    total = invariant_loss + float(lambda_maturity) * variant_total
    return {
        "loss": total,
        "invariant_loss": invariant_loss,
        "full_loss": full_loss,
        "anchor_loss": anchor_loss,
        "variant_total": variant_total,
        "full_inv_distance": full_inv_distance,
        "invariant_error": invariant_error,
    }


def analytic_patch_reliability_target(
    target,
    invariant_prediction,
    raw_prediction,
    future_patch_len,
):
    """Least-squares optimal correction fraction for each sample/patch."""
    if target.shape != invariant_prediction.shape or target.shape != raw_prediction.shape:
        raise ValueError("reliability target tensors must share [B,H,C] shape")
    residual = target - invariant_prediction.detach()
    correction = raw_prediction - invariant_prediction.detach()
    targets = []
    for start in range(0, target.shape[1], int(future_patch_len)):
        end = min(start + int(future_patch_len), target.shape[1])
        residual_patch = residual[:, start:end]
        correction_patch = correction[:, start:end]
        numerator = (residual_patch * correction_patch).sum((1, 2))
        denominator = correction_patch.square().sum((1, 2)) + 1e-8
        targets.append((numerator / denominator).clamp(0.0, 1.0))
    return torch.stack(targets, dim=1).detach()


def expand_patch_reliability(reliability, pred_len, future_patch_len):
    """Expand ``[B,K]`` patch values to broadcastable ``[B,H,1]``."""
    if reliability.ndim != 2:
        raise ValueError("patch reliability must be [B,K]")
    expanded = reliability.repeat_interleave(int(future_patch_len), dim=1)
    return expanded[:, : int(pred_len)].unsqueeze(-1)


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
        variant_fusion_gate_type=getattr(
            args, "variant_fusion_gate_type", "token"
        ),
        film_gamma_scale=args.film_gamma_scale,
        film_beta_scale=args.film_beta_scale,
        y_film_gamma_scale=args.y_film_gamma_scale,
        y_film_beta_scale=args.y_film_beta_scale,
        y_film_decay_bias=args.y_film_decay_bias,
        horizon_future_dim=args.horizon_future_dim,
        horizon_future_gamma_scale=args.horizon_future_gamma_scale,
        horizon_future_beta_scale=args.horizon_future_beta_scale,
        future_patch_len=args.future_patch_len,
        fusion_scale_calibration=args.fusion_scale_calibration,
        predictive_env_backbone=args.backbone,
        enc_in=args.enc_in,
        embed="timeF",
        freq=args.freq,
        cyclenet_cycle_len=args.cyclenet_cycle_len,
    )


def make_train_loader(train_dataset, args):
    """Create an experiment-local shuffled loader with an isolated RNG."""
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(args.seed)
    return DataLoader(
        FutureTeacherIndexedDataset(
            train_dataset,
            args.seq_len,
            global_offset=0,
            cycle_len=args.cyclenet_cycle_len,
        ),
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
    args.enc_in = int(datasets["train"].data_x.shape[-1])
    if args.data_class == "ett_hour":
        test_offset = 12 * 30 * 24 + 4 * 30 * 24 - args.seq_len
    elif args.data_class == "ett_minute":
        test_offset = (12 * 30 * 24 + 4 * 30 * 24) * 4 - args.seq_len
    else:
        # Dataset_Custom uses the last 20% for TEST plus seq_len history.
        csv_path = Path(args.root_path) / args.data_path
        with csv_path.open("rb") as handle:
            row_count = max(sum(1 for _ in handle) - 1, 0)
        test_offset = row_count - int(row_count * 0.2) - args.seq_len
    ordered_train = DataLoader(
        IndexedDataset(
            datasets["train"], cycle_len=args.cyclenet_cycle_len
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test = DataLoader(
        IndexedDataset(
            datasets["test"],
            global_offset=test_offset,
            cycle_len=args.cyclenet_cycle_len,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    ordered_future_train = DataLoader(
        FutureTeacherIndexedDataset(
            datasets["train"],
            args.seq_len,
            valid_only=True,
            cycle_len=args.cyclenet_cycle_len,
        ),
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
    for x, y, sample_id, cycle_index in loader:
        output = model.forward_components(
            x.float().to(device), cycle_index.to(device)
        )
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
    var_gain_values = []
    full_prediction_values = []
    invariant_prediction_values = []
    delta_ratio_values = []
    fusion_mask_count = 0
    fusion_mask_sum = 0.0
    fusion_mask_square_sum = 0.0
    conditional_gain_loss_values = []
    conditional_pair_values = []
    conditional_shuffle_values = []
    conditional_gain_values = []
    conditional_positive_values = []
    h_anchor_values = []
    horizon_effect_values = []
    horizon_decay_values = []
    horizon_future_loss_values = []
    horizon_reliability_loss_values = []
    invariant_loss_values = []
    full_variant_loss_values = []
    anchor_loss_values = []
    variant_total_loss_values = []
    full_inv_distance_values = []
    invariant_error_values = []
    variant_effect_abs_values = []
    maturity_values = []
    invariant_gradient_values = []
    variant_gradient_values = []
    reliability_prediction_values = []
    reliability_target_values = []
    raw_variant_gain_values = []
    gated_variant_gain_values = []
    raw_variant_effect_bin_values = []
    gated_variant_effect_bin_values = []
    learning_rate_values = {
        "backbone": [],
        "inv_head": [],
        "decomposer": [],
        "env_head": [],
        "variant": [],
        "gamma_beta": [],
        "reliability": [],
    }
    horizon_gamma_abs_values = []
    horizon_beta_abs_values = []
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
    film_gamma_diagnostics = GateAccumulator(
        model.variant_fusion_mode in ("film", "film_decay_reg"), 0.0
    )
    film_beta_diagnostics = GateAccumulator(
        model.variant_fusion_mode in ("film", "film_decay_reg"), 0.0
    )
    y_gamma_diagnostics = GateAccumulator(
        model.variant_fusion_mode == "y_film_decay", 0.0
    )
    y_beta_diagnostics = GateAccumulator(
        model.variant_fusion_mode == "y_film_decay", 0.0
    )
    decay_rho_diagnostics = GateAccumulator(
        model.variant_fusion_mode == "y_film_decay", 0.0
    )
    # Teacher-patch sampling must not consume the model/dropout/DataLoader RNG.
    teacher_generator = torch.Generator()
    teacher_epoch = int(getattr(args, "_future_teacher_epoch", 0))
    teacher_generator.manual_seed(args.seed + 104729 + teacher_epoch)
    args._future_teacher_epoch = teacher_epoch + 1
    for step, batch in enumerate(loader):
        if args.max_train_batches and step >= args.max_train_batches:
            break
        (
            x,
            y,
            sample_id,
            future,
            future_valid,
            cycle_index,
            future_cycle_index,
        ) = batch
        x = x.float().to(device)
        cycle_index = cycle_index.to(device)
        y = y[:, -args.pred_len :].float().to(device)
        output = model.forward_components(x, cycle_index)
        isolate_z_gradients = (
            args.predictive_env_refactor_mode == "h_reference_grad_isolated"
        )
        environment_output = (
            model.detached_environment_components(output)
            if isolate_z_gradients and model.level == "representation"
            else output
        )
        gate_diagnostics.update(output["decomposition_gate"])
        film_gamma_diagnostics.update(output["film_gamma"])
        film_beta_diagnostics.update(output["film_beta"])
        y_gamma_diagnostics.update(output["y_film_gamma"])
        y_beta_diagnostics.update(output["y_film_beta"])
        decay_rho_diagnostics.update(output["decay_rho"])
        if output["horizon_decay"] is not None:
            horizon_decay_values.append(
                output["horizon_decay"].detach().mean((0, 1)).cpu()
            )
        sample_loss = (output["prediction"] - y).square().mean((1, 2))
        if model.variant_fusion_mode == "horizon_future_var":
            invariant_sample_for_maturity = (
                output["invariant_prediction"] - y
            ).square().mean((1, 2))
            invariant_loss = invariant_sample_for_maturity.mean()
            full_loss_for_maturity = sample_loss.mean()
            maturity, invariant_gradient, variant_gradient = (
                gradient_relative_maturity(
                    invariant_loss,
                    full_loss_for_maturity,
                    model.head_linear.parameters(),
                    model.horizon_future_variant.modulation_generator[
                        -1
                    ].parameters(),
                )
            )
            lambda_maturity = float(maturity.cpu())
            if args.differential_lr:
                update_gamma_beta_learning_rate(
                    optimizer, maturity, args
                )
            model._last_gradient_maturity = {
                "maturity": lambda_maturity,
                "G_inv_RMS": float(invariant_gradient.cpu()),
                "G_var_RMS": float(variant_gradient.cpu()),
            }
            anchored = invariant_anchored_variant_loss(
                output["invariant_prediction"],
                output["prediction"],
                y,
                lambda_maturity,
            )
            reliability_prediction = output["horizon_reliability"]
            reliability_target = analytic_patch_reliability_target(
                y,
                output["invariant_prediction"],
                output["prediction"],
                args.future_patch_len,
            )
            if reliability_prediction.shape != reliability_target.shape:
                raise RuntimeError(
                    "predicted and analytic patch reliability must align"
                )
            reliability_loss = F.mse_loss(
                reliability_prediction, reliability_target
            )
            invariant_anchor = output["invariant_prediction"].detach()
            raw_correction = output["prediction"] - invariant_anchor
            reliability_horizon = expand_patch_reliability(
                reliability_prediction.detach(),
                args.pred_len,
                args.future_patch_len,
            )
            gated_prediction = (
                invariant_anchor + reliability_horizon * raw_correction.detach()
            )
            gated_sample_loss = (
                gated_prediction - y
            ).square().mean((1, 2))
            raw_variant_gain_values.extend(
                (invariant_sample_for_maturity - sample_loss)
                .detach().cpu().tolist()
            )
            gated_variant_gain_values.extend(
                (invariant_sample_for_maturity - gated_sample_loss)
                .detach().cpu().tolist()
            )
            raw_effect_curve = raw_correction.detach().abs().mean((0, 2))
            gated_effect_curve = (
                reliability_horizon * raw_correction.detach()
            ).abs().mean((0, 2))
            raw_variant_effect_bin_values.append(
                torch.stack(
                    [chunk.mean() for chunk in torch.tensor_split(
                        raw_effect_curve, min(4, raw_effect_curve.numel())
                    )]
                ).cpu()
            )
            gated_variant_effect_bin_values.append(
                torch.stack(
                    [chunk.mean() for chunk in torch.tensor_split(
                        gated_effect_curve, min(4, gated_effect_curve.numel())
                    )]
                ).cpu()
            )
            # Head inputs and analytic targets are detached, hence this term
            # updates the reliability head and no forecasting representation.
            loss = anchored["loss"] + reliability_loss
            full_variant_loss = anchored["full_loss"]
            anchor_loss = anchored["anchor_loss"]
            variant_total_loss = anchored["variant_total"]
            full_inv_distance = anchored["full_inv_distance"]
            invariant_error = anchored["invariant_error"]
            invariant_loss_values.append(float(invariant_loss.detach()))
            full_variant_loss_values.append(
                float(full_variant_loss.detach())
            )
            anchor_loss_values.append(float(anchor_loss.detach()))
            variant_total_loss_values.append(
                float(variant_total_loss.detach())
            )
            full_inv_distance_values.append(
                float(full_inv_distance.detach())
            )
            invariant_error_values.append(float(invariant_error))
            variant_effect_abs_values.append(
                float(
                    (
                        output["prediction"] - invariant_anchor
                    ).detach().abs().mean()
                )
            )
            maturity_values.append(lambda_maturity)
            invariant_gradient_values.append(float(invariant_gradient.cpu()))
            variant_gradient_values.append(float(variant_gradient.cpu()))
            horizon_reliability_loss_values.append(
                float(reliability_loss.detach())
            )
            reliability_prediction_values.append(
                reliability_prediction.detach().cpu()
            )
            reliability_target_values.append(
                reliability_target.cpu()
            )
            horizon_gamma_abs_values.append(
                float(output["horizon_gamma"].detach().abs().mean())
            )
            horizon_beta_abs_values.append(
                float(output["horizon_beta"].detach().abs().mean())
            )
        else:
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
                var_gain_loss = variant_gain_loss(
                    sample_loss,
                    invariant_sample_loss,
                    args.var_gain_temperature,
                )
                full_prediction_values.extend(sample_loss.detach().cpu().tolist())
                invariant_prediction_values.extend(
                    invariant_sample_loss.detach().cpu().tolist()
                )
                if output["feature_gate"] is not None:
                    fusion_mask = output["feature_gate"].detach().float()
                    fusion_mask_count += fusion_mask.numel()
                    fusion_mask_sum += float(fusion_mask.sum())
                    fusion_mask_square_sum += float(fusion_mask.square().sum())
                delta_ratio_values.extend(
                    output["variation_ratio"].detach().cpu().tolist()
                )
                decay_metrics = horizon_effect_objective(
                    output["effect_normalized_prediction"],
                    output["effect_normalized_invariant_prediction"],
                    args.horizon_decay_bins,
                )
                horizon_effect_values.append(
                    {
                        key: (
                            value.detach().cpu()
                            if torch.is_tensor(value)
                            else value
                        )
                        for key, value in decay_metrics.items()
                    }
                )
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
                reliability = (
                    environment_output["feature_gate"]
                    .detach()
                    .flatten(1)
                    .mean(dim=1)
                    if model.variant_fusion_mode == "horizon_future_var"
                    and environment_output["feature_gate"] is not None
                    else environment_output["feature_gate"].flatten(1).mean(dim=1)
                    if environment_output["feature_gate"] is not None
                    else torch.ones_like(utility_full_loss)
                )
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
                # horizon_future_var already uses L_inv as its forecasting
                # basis. Do not add the legacy invariant loss a second time.
                invariant_weight = (
                    0.0
                    if model.variant_fusion_mode == "horizon_future_var"
                    else args.lambda_invpred
                )
                feature_gate_penalty = (
                    loss.new_zeros(())
                    if model.variant_fusion_mode == "horizon_future_var"
                    else
                    environment_output["feature_gate"].abs().mean()
                    if environment_output["feature_gate"] is not None
                    else loss.new_zeros(())
                )
                loss = (
                    loss
                    + invariant_weight * invariant_prediction
                    + args.lambda_domain * domain_loss
                    + args.lambda_z * feature_gate_penalty
                )
                # Keep disabled auxiliary losses out of the autograd graph so
                # lambda=0 exactly preserves the previous training trajectory.
                if args.lambda_var_predictive != 0:
                    loss = loss + (
                        args.lambda_var_predictive * variant_predictive_loss
                    )
                if args.lambda_var_utility != 0:
                    loss = loss + args.lambda_var_utility * utility_loss
                if args.lambda_var_gain != 0:
                    loss = loss + args.lambda_var_gain * var_gain_loss
                if model.variant_fusion_mode == "film_decay_reg":
                    if args.lambda_decay_mono != 0:
                        loss = loss + (
                            args.lambda_decay_mono
                            * decay_metrics["decay_mono_loss"]
                        )
                    if args.lambda_decay_far != 0:
                        loss = loss + (
                            args.lambda_decay_far
                            * decay_metrics["decay_far_loss"]
                        )
                variant_predictive_values.append(
                    float(variant_predictive_loss.detach())
                )
                utility_values.append(float(utility_loss.detach()))
                var_gain_values.append(float(var_gain_loss.detach()))
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
                            future[future_valid.cpu()].float().to(device),
                            future_cycle_index[future_valid.cpu()].to(device),
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
                if model.variant_fusion_mode == "horizon_future_var":
                    if args.lambda_future_h != 0:
                        all_centers = future_patch_centers(
                            args.pred_len, args.future_patch_len
                        )
                        teacher_count = min(
                            args.future_teacher_patches_per_batch,
                            all_centers.numel(),
                        )
                        patch_indices = torch.randperm(
                            all_centers.numel(), generator=teacher_generator
                        )[:teacher_count].sort().values
                        anchors = all_centers.index_select(0, patch_indices)
                        teacher_windows, anchor_valid, teacher_cycles = (
                            loader.dataset.horizon_teacher_batch(
                                sample_id, anchors
                            )
                        )
                        if bool(anchor_valid.any()):
                            teacher_windows = teacher_windows.float().to(device)
                            teacher_cycles = teacher_cycles.to(device)
                            teacher = model.horizon_future_variant_targets(
                                teacher_windows, teacher_cycles
                            )
                            valid_device = anchor_valid.to(device=device)
                            model_centers = (
                                output["horizon_future_anchor_indices"].cpu() + 1
                            )
                            if not torch.equal(model_centers, all_centers):
                                raise RuntimeError(
                                    "model and future-patch centers differ"
                                )
                            prediction = output["horizon_future_zvar"][
                                valid_device
                            ].index_select(2, patch_indices.to(device))
                            future_h_loss = 1.0 - F.cosine_similarity(
                                prediction, teacher.detach(), dim=-1
                            ).mean()
                            loss = loss + args.lambda_future_h * future_h_loss
                            horizon_future_loss_values.append(
                                float(future_h_loss.detach())
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
        for group in optimizer.param_groups:
            group_name = group.get("name")
            if group_name in learning_rate_values:
                learning_rate_values[group_name].append(float(group["lr"]))
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
    result["var_gain_loss"] = (
        float(np.mean(var_gain_values)) if var_gain_values else float("nan")
    )
    if full_prediction_values:
        epoch_full = torch.tensor(full_prediction_values)
        epoch_inv = torch.tensor(invariant_prediction_values)
        epoch_gain = epoch_inv - epoch_full
        mask_mean = (
            fusion_mask_sum / fusion_mask_count
            if fusion_mask_count
            else float("nan")
        )
        mask_variance = (
            max(
                fusion_mask_square_sum / fusion_mask_count
                - mask_mean * mask_mean,
                0.0,
            )
            if fusion_mask_count
            else float("nan")
        )
        result.update(
            {
                "full_MSE": float(epoch_full.mean()),
                "inv_MSE": float(epoch_inv.mean()),
                "mean_gain": float(epoch_gain.mean()),
                "median_gain": float(epoch_gain.median()),
                "positive_gain_ratio": float((epoch_gain > 0).float().mean()),
                "gain_p10": float(torch.quantile(epoch_gain, 0.1)),
                "gain_p50": float(torch.quantile(epoch_gain, 0.5)),
                "gain_p90": float(torch.quantile(epoch_gain, 0.9)),
                "mask_mean": mask_mean,
                "mask_std": mask_variance**0.5,
                "DeltaZ/Zinv": float(np.mean(delta_ratio_values)),
                "YDelta/Yinv": (
                    float(np.mean(delta_ratio_values))
                    if model.variant_fusion_mode in (
                        "y_film_decay", "horizon_future_var"
                    )
                    else float("nan")
                ),
            }
        )
    result["loss/inv"] = (
        float(np.mean(invariant_loss_values))
        if invariant_loss_values
        else float("nan")
    )
    result["loss/full"] = (
        float(np.mean(full_variant_loss_values))
        if full_variant_loss_values
        else float("nan")
    )
    # Backward-compatible alias for the previous maturity implementation.
    result["loss/mod"] = result["loss/full"]
    result["loss/anchor"] = (
        float(np.mean(anchor_loss_values))
        if anchor_loss_values
        else float("nan")
    )
    result["loss/variant_total"] = (
        float(np.mean(variant_total_loss_values))
        if variant_total_loss_values
        else float("nan")
    )
    result["variant/lambda_maturity"] = (
        float(np.mean(maturity_values)) if maturity_values else float("nan")
    )
    result["maturity"] = result["variant/lambda_maturity"]
    result["G_inv_RMS"] = (
        float(np.mean(invariant_gradient_values))
        if invariant_gradient_values
        else float("nan")
    )
    result["G_var_RMS"] = (
        float(np.mean(variant_gradient_values))
        if variant_gradient_values
        else float("nan")
    )
    # Backward-compatible aliases now carry RMS rather than raw L2 norms.
    result["G_inv"] = result["G_inv_RMS"]
    result["G_var"] = result["G_var_RMS"]
    for group_name in (
        "backbone", "inv_head", "decomposer", "env_head", "variant",
        "reliability",
    ):
        values_for_group = learning_rate_values[group_name]
        result[f"lr/{group_name}"] = (
            float(np.mean(values_for_group))
            if values_for_group else float("nan")
        )
    gamma_beta_lrs = learning_rate_values["gamma_beta"]
    result["lr/gamma_beta_mean"] = (
        float(np.mean(gamma_beta_lrs)) if gamma_beta_lrs else float("nan")
    )
    result["lr/gamma_beta_min"] = (
        float(np.min(gamma_beta_lrs)) if gamma_beta_lrs else float("nan")
    )
    result["lr/gamma_beta_max"] = (
        float(np.max(gamma_beta_lrs)) if gamma_beta_lrs else float("nan")
    )
    result["variant/gamma_abs_mean"] = (
        float(np.mean(horizon_gamma_abs_values))
        if horizon_gamma_abs_values
        else float("nan")
    )
    result["variant/beta_abs_mean"] = (
        float(np.mean(horizon_beta_abs_values))
        if horizon_beta_abs_values
        else float("nan")
    )
    result["variant/full_inv_distance"] = (
        float(np.mean(full_inv_distance_values))
        if full_inv_distance_values
        else float("nan")
    )
    result["variant/inv_error"] = (
        float(np.mean(invariant_error_values))
        if invariant_error_values
        else float("nan")
    )
    result["variant/effect_abs_mean"] = (
        float(np.mean(variant_effect_abs_values))
        if variant_effect_abs_values
        else float("nan")
    )
    if reliability_prediction_values:
        reliability_prediction_epoch = torch.cat(
            reliability_prediction_values, dim=0
        )
        reliability_target_epoch = torch.cat(
            reliability_target_values, dim=0
        )
        result["r_pred_mean"] = float(reliability_prediction_epoch.mean())
        result["r_target_mean"] = float(reliability_target_epoch.mean())
        result["corr_r_pred_r_target"] = spearman_correlation(
            reliability_prediction_epoch, reliability_target_epoch
        )
        for index, chunk in enumerate(
            torch.tensor_split(reliability_prediction_epoch, min(
                4, reliability_prediction_epoch.shape[1]
            ), dim=1),
            1,
        ):
            result[f"r_pred_bin{index}"] = float(chunk.mean())
        for index, chunk in enumerate(
            torch.tensor_split(reliability_target_epoch, min(
                4, reliability_target_epoch.shape[1]
            ), dim=1),
            1,
        ):
            result[f"r_target_bin{index}"] = float(chunk.mean())
        result["raw_variant_gain"] = float(
            np.mean(raw_variant_gain_values)
        )
        result["gated_variant_gain"] = float(
            np.mean(gated_variant_gain_values)
        )
        raw_effect_bins = torch.stack(raw_variant_effect_bin_values).mean(0)
        gated_effect_bins = torch.stack(
            gated_variant_effect_bin_values
        ).mean(0)
        for index, value in enumerate(raw_effect_bins, 1):
            result[f"variant_effect_raw_bin{index}"] = float(value)
        for index, value in enumerate(gated_effect_bins, 1):
            result[f"variant_effect_gated_bin{index}"] = float(value)
    else:
        for key in (
            "r_pred_mean", "r_target_mean", "corr_r_pred_r_target",
            "raw_variant_gain", "gated_variant_gain",
        ):
            result[key] = float("nan")
    for index in range(1, 5):
        result.setdefault(f"r_pred_bin{index}", float("nan"))
        result.setdefault(f"r_target_bin{index}", float("nan"))
        result.setdefault(f"variant_effect_raw_bin{index}", float("nan"))
        result.setdefault(f"variant_effect_gated_bin{index}", float("nan"))
    if horizon_effect_values:
        for key in (
            "var_effect_mean",
            "var_effect_far_near_ratio",
            "decay_mono_loss",
            "decay_far_loss",
        ):
            result[key] = float(
                torch.stack([row[key] for row in horizon_effect_values]).mean()
            )
        bin_values = torch.stack(
            [row["var_effect_bins"] for row in horizon_effect_values]
        ).mean(0)
        for index, value in enumerate(bin_values, 1):
            result[f"var_effect_bin{index}"] = float(value)
    if horizon_decay_values:
        decay_curve = torch.stack(horizon_decay_values).mean(0)
        decay_chunks = torch.tensor_split(
            decay_curve, min(4, decay_curve.numel())
        )
        result["decay_d_near"] = float(decay_chunks[0].mean())
        result["decay_d_mid"] = float(
            decay_chunks[len(decay_chunks) // 2].mean()
        )
        result["decay_d_far"] = float(decay_chunks[-1].mean())
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
    result["horizon_future_var_loss"] = (
        float(np.mean(horizon_future_loss_values))
        if horizon_future_loss_values
        else float("nan")
    )
    result["horizon_reliability_loss"] = (
        float(np.mean(horizon_reliability_loss_values))
        if horizon_reliability_loss_values
        else float("nan")
    )
    for key, total in future_totals.items():
        result[key] = total / future_count if future_count else float("nan")
    result.update(gate_diagnostics.result())
    result.update(moment_diagnostics(film_gamma_diagnostics, "gamma"))
    result.update(moment_diagnostics(film_beta_diagnostics, "beta"))
    result.update(moment_diagnostics(y_gamma_diagnostics, "y_gamma"))
    result.update(moment_diagnostics(y_beta_diagnostics, "y_beta"))
    result.update(moment_diagnostics(decay_rho_diagnostics, "decay_rho"))
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
    for step, batch in enumerate(ordered_future_train):
        x, y, _, future, future_valid, cycle_index, future_cycle_index = batch
        if args.max_eval_batches and step >= args.max_eval_batches:
            break
        x = x.float().to(device)
        if not bool(future_valid.all()):
            raise RuntimeError("Diagnostic future loader contains an invalid teacher")
        y = y[:, -args.pred_len :].float().to(device)
        output = model.forward_components(x, cycle_index.to(device))
        target = model.future_variant_target(
            future.float().to(device), future_cycle_index.to(device)
        )
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
def evaluate_horizon_future_var(model, ordered_future_train, args, device):
    """Evaluate representative future-patch teachers on TRAIN only."""
    keys = (
        "horizon_future_var_loss",
        "future_zvar_cosine_mean",
        "future_zvar_cosine_near",
        "future_zvar_cosine_mid",
        "future_zvar_cosine_far",
    )
    if model.variant_fusion_mode != "horizon_future_var":
        return {key: float("nan") for key in keys}
    model.eval()
    all_centers = future_patch_centers(
        args.pred_len, args.future_patch_len
    )
    eval_count = min(args.future_teacher_eval_patch_count, all_centers.numel())
    patch_indices = torch.linspace(
        0, all_centers.numel() - 1, eval_count
    ).round().long().unique()
    anchors = all_centers.index_select(0, patch_indices)
    cosine_sum = torch.zeros(patch_indices.numel(), dtype=torch.float64)
    cosine_count = 0
    valid_samples = 0
    for step, batch in enumerate(ordered_future_train):
        if args.max_eval_batches and step >= args.max_eval_batches:
            break
        x, _, sample_id, _, _, cycle_index, _ = batch
        windows, valid, teacher_cycles = (
            ordered_future_train.dataset.horizon_teacher_batch(
                sample_id, anchors
            )
        )
        if not bool(valid.any()):
            continue
        output = model.forward_components(
            x[valid].float().to(device), cycle_index[valid].to(device)
        )
        target = model.horizon_future_variant_targets(
            windows.float().to(device), teacher_cycles.to(device)
        )
        model_centers = output["horizon_future_anchor_indices"].cpu() + 1
        if not torch.equal(model_centers, all_centers):
            raise RuntimeError("model and diagnostic future-patch centers differ")
        prediction = output["horizon_future_zvar"].index_select(
            2, patch_indices.to(device)
        )
        cosine = F.cosine_similarity(prediction, target, dim=-1)
        cosine_sum += cosine.double().sum((0, 1)).cpu()
        cosine_count += cosine.shape[0] * cosine.shape[1]
        valid_samples += cosine.shape[0]
    if cosine_count == 0:
        return {key: float("nan") for key in keys}
    curve = (cosine_sum / cosine_count).float()
    thirds = torch.tensor_split(curve, min(3, curve.numel()))
    return {
        "future_zvar_cosine_mean": float(curve.mean()),
        "horizon_future_var_loss": float(1.0 - curve.mean()),
        "future_zvar_cosine_near": float(thirds[0].mean()),
        "future_zvar_cosine_mid": float(thirds[len(thirds) // 2].mean()),
        "future_zvar_cosine_far": float(thirds[-1].mean()),
        "future_zvar_anchor_count": int(anchors.numel()),
        "future_patch_count": int(all_centers.numel()),
        "future_patch_len": int(args.future_patch_len),
        "future_zvar_teacher_sample_count": valid_samples,
        "future_zvar_teacher_source": (
            "representative TRAIN-only future-patch centers; stop-gradient "
            "shared encoder/decomposer; absent from validation/test inference"
        ),
    }


@torch.no_grad()
def _evaluate_materialized_legacy(
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
    film_gamma_diagnostics = GateAccumulator(
        model.variant_fusion_mode == "film", 0.0
    )
    film_beta_diagnostics = GateAccumulator(
        model.variant_fusion_mode == "film", 0.0
    )
    for step, (x, y, _, cycle_index) in enumerate(test_loader):
        if args.max_eval_batches and step >= args.max_eval_batches:
            break
        x = x.float().to(device)
        y = y[:, -args.pred_len :].float().to(device)
        output = model.forward_components(x, cycle_index.to(device))
        decomposition_gate_diagnostics.update(output["decomposition_gate"])
        decomposition_diagnostics.update(
            output["hidden_tokens"],
            output["z_inv_tokens"],
            output["z_var_tokens"],
        )
        film_gamma_diagnostics.update(output["film_gamma"])
        film_beta_diagnostics.update(output["film_beta"])
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
        "variant_fusion_gate_type": args.variant_fusion_gate_type,
        "variant_fusion_gate_last_dim": (
            args.d_model
            if (
                args.variant_fusion_mode == "film"
                or args.variant_fusion_gate_type == "feature"
            )
            else 1
        ),
        "film_gamma_scale": args.film_gamma_scale,
        "film_beta_scale": args.film_beta_scale,
        "fusion_scale_calibration": args.fusion_scale_calibration,
        "predictive_env_refactor_mode": args.predictive_env_refactor_mode,
        "lambda_invpred": args.lambda_invpred,
        "lambda_future_var": args.lambda_future_var,
        "lambda_var_predictive": args.lambda_var_predictive,
        "lambda_var_utility": args.lambda_var_utility,
        "lambda_var_gain": args.lambda_var_gain,
        "var_gain_temperature": args.var_gain_temperature,
        "lambda_var_conditional_gain": args.lambda_var_conditional_gain,
        "var_conditional_margin": args.var_conditional_margin,
        "MSE": float(full_error.square().mean()),
        "full_MSE": float(full_error.square().mean()),
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
        "mask_mean": float(gate.mean()),
        "mask_std": float(gate.std()),
        "g_z_p10": float(np.quantile(gate, 0.1)),
        "g_z_p50": float(np.quantile(gate, 0.5)),
        "g_z_p90": float(np.quantile(gate, 0.9)),
        "feature_variation_to_Zinv": float(torch.cat(feature_ratios).mean()),
        "DeltaZ/Zinv": float(torch.cat(feature_ratios).mean()),
        "var_gain_loss": float(
            args.var_gain_temperature
            * F.softplus(
                (loss_full_per_sample - loss_inv_per_sample.detach())
                / args.var_gain_temperature
            ).mean()
        ),
        "fusion_rms_ratio_mean": float(torch.cat(fusion_rms_ratios).mean()),
        "fusion_rms_ratio_p10": float(torch.quantile(torch.cat(fusion_rms_ratios), 0.1)),
        "fusion_rms_ratio_p50": float(torch.quantile(torch.cat(fusion_rms_ratios), 0.5)),
        "fusion_rms_ratio_p90": float(torch.quantile(torch.cat(fusion_rms_ratios), 0.9)),
        "mapping_variation": "removed",
    }
    result.update(decomposition_gate_diagnostics.result())
    result.update(decomposition_diagnostics.result())
    result.update(moment_diagnostics(film_gamma_diagnostics, "gamma"))
    result.update(moment_diagnostics(film_beta_diagnostics, "beta"))
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
            result["env_acc_gap_var_minus_inv"] = (
                result["var_acc"] - result["inv_acc"]
            )
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
            result["env_acc_gap_var_minus_inv"] = float("nan")
    return result


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
    """Memory-bounded final evaluation for feature-wise fusion experiments."""
    model.eval()
    full_square_sum = full_absolute_sum = 0.0
    raw_square_sum = raw_absolute_sum = 0.0
    invariant_square_sum = invariant_absolute_sum = 0.0
    raw_inv_square_sum = raw_inv_absolute_sum = 0.0
    full_inv_square_sum = full_inv_absolute_sum = 0.0
    conditional_square_sum = 0.0
    prediction_element_count = 0
    loss_full_values, loss_inv_values = [], []
    conditional_pair_losses, conditional_shuffle_losses = [], []
    conditional_main_gains, sample_reliabilities = [], []
    variation_ratios, fusion_rms_ratios = [], []
    gate_samples = []
    gate_sample_count = 0
    horizon_effect_rows, horizon_gain_rows = [], []
    decay_curve_sum = None
    decay_curve_count = 0
    horizon_reliability_curve_sum = None
    horizon_change_curve_sum = None
    horizon_curve_count = 0
    horizon_reliability_samples, horizon_point_gain_samples = [], []
    analytic_reliability_predictions, analytic_reliability_targets = [], []
    raw_variant_gains, gated_variant_gains = [], []
    raw_variant_effect_bins, gated_variant_effect_bins = [], []
    horizon_pair_sample_count = 0
    horizon_reliability_bce_sum = 0.0
    horizon_reliability_element_count = 0
    decomposition_gate_diagnostics = GateAccumulator(
        model.level == "representation"
        and model.decomposition_type in ("signed_gate", "complementary_gate"),
        args.gate_near_zero_threshold,
    )
    fusion_gate_diagnostics = GateAccumulator(True, 0.0)
    decomposition_diagnostics = DecompositionAccumulator()
    film_enabled = model.variant_fusion_mode in ("film", "film_decay_reg")
    film_gamma_diagnostics = GateAccumulator(film_enabled, 0.0)
    film_beta_diagnostics = GateAccumulator(film_enabled, 0.0)
    y_enabled = model.variant_fusion_mode == "y_film_decay"
    horizon_future_enabled = model.variant_fusion_mode == "horizon_future_var"
    y_space_enabled = y_enabled or horizon_future_enabled
    y_gamma_diagnostics = GateAccumulator(y_space_enabled, 0.0)
    y_beta_diagnostics = GateAccumulator(y_space_enabled, 0.0)
    decay_rho_diagnostics = GateAccumulator(y_enabled, 0.0)
    for step, (x, y, _, cycle_index) in enumerate(test_loader):
        if args.max_eval_batches and step >= args.max_eval_batches:
            break
        x = x.float().to(device)
        y = y[:, -args.pred_len :].float().to(device)
        output = model.forward_components(x, cycle_index.to(device))
        if horizon_future_enabled:
            gamma_h = output["horizon_gamma"].permute(0, 2, 1)
            beta_h = output["horizon_beta"].permute(0, 2, 1)
            raw_normalized = (
                (1.0 + gamma_h)
                * output["normalized_invariant_prediction"].detach()
                + beta_h
            )
            raw_prediction = (
                raw_normalized * output["input_scale"]
                + output["input_shift"]
            )
        else:
            raw_prediction = output["prediction"]
        full_error = output["prediction"] - y
        raw_error = raw_prediction - y
        invariant_error = output["invariant_prediction"] - y
        full_inv_delta = (
            output["prediction"] - output["invariant_prediction"].detach()
        )
        conditional_error = output["conditional_variant_prediction"] - y
        full_square_sum += float(full_error.square().sum())
        full_absolute_sum += float(full_error.abs().sum())
        raw_square_sum += float(raw_error.square().sum())
        raw_absolute_sum += float(raw_error.abs().sum())
        invariant_square_sum += float(invariant_error.square().sum())
        invariant_absolute_sum += float(invariant_error.abs().sum())
        full_inv_square_sum += float(full_inv_delta.square().sum())
        full_inv_absolute_sum += float(full_inv_delta.abs().sum())
        raw_inv_delta = raw_prediction - output["invariant_prediction"].detach()
        raw_inv_square_sum += float(raw_inv_delta.square().sum())
        raw_inv_absolute_sum += float(raw_inv_delta.abs().sum())
        conditional_square_sum += float(conditional_error.square().sum())
        prediction_element_count += full_error.numel()
        loss_full = full_error.square().mean((1, 2))
        loss_inv = invariant_error.square().mean((1, 2))
        loss_full_values.append(loss_full.cpu())
        loss_inv_values.append(loss_inv.cpu())
        decomposition_gate_diagnostics.update(output["decomposition_gate"])
        decomposition_diagnostics.update(
            output["hidden_tokens"],
            output["z_inv_tokens"],
            output["z_var_tokens"],
        )
        feature_gate = output["feature_gate"]
        fusion_gate_diagnostics.update(feature_gate)
        film_gamma_diagnostics.update(output["film_gamma"])
        film_beta_diagnostics.update(output["film_beta"])
        y_gamma_diagnostics.update(
            output["horizon_gamma"]
            if horizon_future_enabled
            else output["y_film_gamma"]
        )
        y_beta_diagnostics.update(
            output["horizon_beta"]
            if horizon_future_enabled
            else output["y_film_beta"]
        )
        decay_rho_diagnostics.update(output["decay_rho"])
        if feature_gate is not None:
            flat_gate = feature_gate.detach().reshape(-1).cpu()
            remaining = max(args.diagnostic_samples - gate_sample_count, 0)
            if remaining:
                stride = max((flat_gate.numel() + remaining - 1) // remaining, 1)
                selected_gate = flat_gate[::stride][:remaining]
                gate_samples.append(selected_gate)
                gate_sample_count += selected_gate.numel()
            sample_reliabilities.append(
                feature_gate.flatten(1).mean(dim=1).cpu()
            )
        else:
            sample_reliabilities.append(
                torch.full((x.shape[0],), float("nan"))
            )
        variation_ratios.append(output["variation_ratio"].cpu())
        hidden_rms = output["hidden_tokens"].square().mean((-2, -1)).sqrt()
        fused_rms = output["final_tokens"].square().mean((-2, -1)).sqrt()
        fusion_rms_ratios.append(
            (fused_rms / hidden_rms.clamp_min(1e-8)).reshape(-1).cpu()
        )
        horizon_effect_rows.append(
            horizon_effect_objective(
                output["effect_normalized_prediction"],
                output["effect_normalized_invariant_prediction"],
                args.horizon_decay_bins,
            )
        )
        horizon_gain_rows.append(
            horizon_gain_values(
                output["prediction"],
                output["invariant_prediction"],
                y,
                args.horizon_decay_bins,
            ).cpu()
        )
        if output["horizon_reliability"] is not None:
            reliability_h = output["horizon_reliability"].detach()
            change_h = output["horizon_future_zvar_change"].detach()
            reliability_target_h = analytic_patch_reliability_target(
                y,
                output["invariant_prediction"],
                raw_prediction,
                args.future_patch_len,
            )
            horizon_reliability_bce_sum += float(
                F.mse_loss(
                    reliability_h,
                    reliability_target_h,
                    reduction="sum",
                )
            )
            horizon_reliability_element_count += reliability_h.numel()
            change_curve = change_h.sum((0, 1)).cpu()
            horizon_change_curve_sum = (
                change_curve
                if horizon_change_curve_sum is None
                else horizon_change_curve_sum + change_curve
            )
            horizon_curve_count += change_h.shape[0] * change_h.shape[1]
            analytic_reliability_predictions.append(reliability_h.cpu())
            analytic_reliability_targets.append(reliability_target_h.cpu())
            raw_loss_h = raw_error.square().mean((1, 2))
            raw_variant_gains.append((loss_inv - raw_loss_h).cpu())
            gated_variant_gains.append((loss_inv - loss_full).cpu())
            raw_effect_curve = raw_inv_delta.abs().mean((0, 2))
            gated_effect_curve = full_inv_delta.abs().mean((0, 2))
            raw_variant_effect_bins.append(
                torch.stack([
                    chunk.mean() for chunk in torch.tensor_split(
                        raw_effect_curve, min(4, raw_effect_curve.numel())
                    )
                ]).cpu()
            )
            gated_variant_effect_bins.append(
                torch.stack([
                    chunk.mean() for chunk in torch.tensor_split(
                        gated_effect_curve, min(4, gated_effect_curve.numel())
                    )
                ]).cpu()
            )
            remaining_pairs = max(
                args.diagnostic_samples - horizon_pair_sample_count, 0
            )
            if remaining_pairs:
                flat_rel = reliability_h.reshape(-1).cpu()
                flat_gain = reliability_target_h.reshape(-1).cpu()
                stride = max(
                    (flat_rel.numel() + remaining_pairs - 1)
                    // remaining_pairs,
                    1,
                )
                selected_rel = flat_rel[::stride][:remaining_pairs]
                selected_gain = flat_gain[::stride][:remaining_pairs]
                horizon_reliability_samples.append(selected_rel)
                horizon_point_gain_samples.append(selected_gain)
                horizon_pair_sample_count += selected_rel.numel()
        elif horizon_future_enabled:
            change_h = output["horizon_future_zvar_change"].detach()
            change_curve = change_h.sum((0, 1)).cpu()
            horizon_change_curve_sum = (
                change_curve
                if horizon_change_curve_sum is None
                else horizon_change_curve_sum + change_curve
            )
            horizon_curve_count += change_h.shape[0] * change_h.shape[1]
        if output["horizon_decay"] is not None:
            decay = output["horizon_decay"].detach()
            curve = decay.sum((0, 1)).cpu()
            decay_curve_sum = curve if decay_curve_sum is None else decay_curve_sum + curve
            decay_curve_count += decay.shape[0] * decay.shape[1]
        if x.shape[0] > 1:
            pair_loss = conditional_error.square().mean((1, 2))
            shuffle_loss = (
                output["conditional_shuffled_prediction"] - y
            ).square().mean((1, 2))
            conditional_pair_losses.append(pair_loss.cpu())
            conditional_shuffle_losses.append(shuffle_loss.cpu())
            conditional_main_gains.append((loss_inv - loss_full).cpu())

    loss_full_per_sample = torch.cat(loss_full_values)
    loss_inv_per_sample = torch.cat(loss_inv_values)
    gain = loss_inv_per_sample - loss_full_per_sample
    sample_reliability = torch.cat(sample_reliabilities)
    variation_ratio = torch.cat(variation_ratios)
    fusion_rms_ratio = torch.cat(fusion_rms_ratios)
    if conditional_pair_losses:
        conditional_pair_loss = torch.cat(conditional_pair_losses)
        conditional_shuffle_loss = torch.cat(conditional_shuffle_losses)
        conditional_gain = conditional_shuffle_loss - conditional_pair_loss
        conditional_main_gain = torch.cat(conditional_main_gains)
    else:
        conditional_pair_loss = conditional_shuffle_loss = torch.tensor([float("nan")])
        conditional_gain = conditional_main_gain = torch.tensor([float("nan")])
    gate_sample = (
        torch.cat(gate_samples)
        if gate_samples
        else torch.tensor([float("nan")])
    )
    fusion_gate_stats = fusion_gate_diagnostics.result()
    full_mse = full_square_sum / max(prediction_element_count, 1)
    raw_mse = raw_square_sum / max(prediction_element_count, 1)
    inv_mse = invariant_square_sum / max(prediction_element_count, 1)
    full_mae = full_absolute_sum / max(prediction_element_count, 1)
    raw_mae = raw_absolute_sum / max(prediction_element_count, 1)
    inv_mae = invariant_absolute_sum / max(prediction_element_count, 1)
    full_inv_distance = full_inv_square_sum / max(prediction_element_count, 1)
    effect_abs_mean = full_inv_absolute_sum / max(prediction_element_count, 1)
    anchor_loss_value = full_inv_distance / (inv_mse + 1e-8)
    raw_full_inv_distance = raw_inv_square_sum / max(
        prediction_element_count, 1
    )
    raw_effect_abs_mean = raw_inv_absolute_sum / max(
        prediction_element_count, 1
    )
    raw_anchor_loss_value = raw_full_inv_distance / (inv_mse + 1e-8)
    effect_keys = (
        "var_effect_mean",
        "var_effect_far_near_ratio",
        "decay_mono_loss",
        "decay_far_loss",
    )
    effect_summary = {
        key: float(torch.stack([row[key].cpu() for row in horizon_effect_rows]).mean())
        for key in effect_keys
    }
    effect_bins = torch.stack(
        [row["var_effect_bins"].cpu() for row in horizon_effect_rows]
    ).mean(0)
    gain_bins = torch.stack(horizon_gain_rows).mean(0)
    if horizon_future_enabled and horizon_reliability_samples:
        reliability_gain_corr = spearman_correlation(
            torch.cat(horizon_reliability_samples),
            torch.cat(horizon_point_gain_samples),
        )
    else:
        reliability_gain_corr = spearman_correlation(sample_reliability, gain)
    maturity_snapshot = getattr(
        model,
        "_last_gradient_maturity",
        {
            "maturity": float("nan"),
            "G_inv_RMS": float("nan"),
            "G_var_RMS": float("nan"),
        },
    )
    result = {
        "environment_count": args.env_num,
        "representation_constraint": args.representation_constraint,
        "decomposition_type": args.decomposition_type,
        "variant_fusion_mode": args.variant_fusion_mode,
        "variant_fusion_gate_type": args.variant_fusion_gate_type,
        "variant_fusion_gate_last_dim": (
            ((args.pred_len + args.future_patch_len - 1) // args.future_patch_len)
            if horizon_future_enabled
            else args.pred_len
            if y_enabled
            else (
                args.d_model
                if (
                    film_enabled
                    or (
                        args.variant_fusion_mode == "direct_gated"
                        and args.variant_fusion_gate_type == "feature"
                    )
                )
                else 1
            )
        ),
        "film_gamma_scale": args.film_gamma_scale,
        "film_beta_scale": args.film_beta_scale,
        "y_film_gamma_scale": args.y_film_gamma_scale,
        "y_film_beta_scale": args.y_film_beta_scale,
        "y_film_decay_bias": args.y_film_decay_bias,
        "horizon_future_dim": args.horizon_future_dim,
        "horizon_future_gamma_scale": args.horizon_future_gamma_scale,
        "horizon_future_beta_scale": args.horizon_future_beta_scale,
        "future_patch_len": args.future_patch_len,
        "future_patch_count": (
            (args.pred_len + args.future_patch_len - 1)
            // args.future_patch_len
        ),
        # Backward-compatible summary column; now means future patch count.
        "future_var_anchor_count": (
            (args.pred_len + args.future_patch_len - 1)
            // args.future_patch_len
        ),
        "future_teacher_patches_per_batch": (
            args.future_teacher_patches_per_batch
        ),
        "lambda_future_h": args.lambda_future_h,
        "lambda_horizon_reliability": args.lambda_horizon_reliability,
        "horizon_reliability_temperature": args.horizon_reliability_temperature,
        "horizon_reliability_loss": (
            horizon_reliability_bce_sum
            / max(horizon_reliability_element_count, 1)
            if horizon_reliability_element_count
            else float("nan")
        ),
        "variant_training_schedule": (
            "gradient_relative_invariant_anchored"
            if horizon_future_enabled
            else "legacy"
        ),
        "loss/inv": inv_mse if horizon_future_enabled else float("nan"),
        "loss/full": raw_mse if horizon_future_enabled else float("nan"),
        "loss/mod": raw_mse if horizon_future_enabled else float("nan"),
        "loss/anchor": (
            raw_anchor_loss_value if horizon_future_enabled else float("nan")
        ),
        "loss/variant_total": (
            raw_mse + raw_anchor_loss_value
            if horizon_future_enabled
            else float("nan")
        ),
        "variant/lambda_maturity": maturity_snapshot["maturity"],
        "maturity": maturity_snapshot["maturity"],
        "G_inv_RMS": maturity_snapshot["G_inv_RMS"],
        "G_var_RMS": maturity_snapshot["G_var_RMS"],
        "G_inv": maturity_snapshot["G_inv_RMS"],
        "G_var": maturity_snapshot["G_var_RMS"],
        "variant/full_inv_distance": (
            full_inv_distance if horizon_future_enabled else float("nan")
        ),
        "variant/inv_error": (
            inv_mse if horizon_future_enabled else float("nan")
        ),
        "variant/effect_abs_mean": (
            effect_abs_mean if horizon_future_enabled else float("nan")
        ),
        "variant/raw_full_inv_distance": (
            raw_full_inv_distance if horizon_future_enabled else float("nan")
        ),
        "variant/raw_effect_abs_mean": (
            raw_effect_abs_mean if horizon_future_enabled else float("nan")
        ),
        "lambda_decay_mono": args.lambda_decay_mono,
        "lambda_decay_far": args.lambda_decay_far,
        "horizon_decay_bins": args.horizon_decay_bins,
        "fusion_scale_calibration": args.fusion_scale_calibration,
        "predictive_env_refactor_mode": args.predictive_env_refactor_mode,
        "lambda_invpred": args.lambda_invpred,
        "lambda_future_var": args.lambda_future_var,
        "lambda_var_predictive": args.lambda_var_predictive,
        "lambda_var_utility": args.lambda_var_utility,
        "lambda_var_gain": args.lambda_var_gain,
        "var_gain_temperature": args.var_gain_temperature,
        "lambda_var_conditional_gain": args.lambda_var_conditional_gain,
        "var_conditional_margin": args.var_conditional_margin,
        "MSE": full_mse,
        "full_MSE": full_mse,
        "raw_full_MSE": raw_mse,
        "MAE": full_mae,
        "raw_full_MAE": raw_mae,
        "inv_only_MSE": inv_mse,
        "inv_only_MAE": inv_mae,
        "inv_only_minus_full_MSE": inv_mse - full_mse,
        "inv_MSE": inv_mse,
        "inv_minus_full": float(gain.mean()),
        "var_predictive_loss": conditional_square_sum
        / max(prediction_element_count, 1),
        "conditional_var_MSE": conditional_square_sum
        / max(prediction_element_count, 1),
        "conditional_pair_MSE": float(conditional_pair_loss.mean()),
        "conditional_shuffle_MSE": float(conditional_shuffle_loss.mean()),
        "conditional_gain_mean": float(conditional_gain.mean()),
        "conditional_gain_median": float(conditional_gain.median()),
        "conditional_positive_ratio": float((conditional_gain > 0).float().mean()),
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
        "corr_reliability_gain": reliability_gain_corr,
        "corr_reliability_gain_type": "Spearman",
        "g_z_mean": fusion_gate_stats["gate/mean"],
        "mask_mean": fusion_gate_stats["gate/mean"],
        "mask_std": fusion_gate_stats["gate/std"],
        "g_z_p10": float(torch.quantile(gate_sample, 0.1)),
        "g_z_p50": float(torch.quantile(gate_sample, 0.5)),
        "g_z_p90": float(torch.quantile(gate_sample, 0.9)),
        "feature_variation_to_Zinv": float(variation_ratio.mean()),
        "DeltaZ/Zinv": float(variation_ratio.mean()),
        "YDelta/Yinv": (
            float(variation_ratio.mean()) if y_space_enabled else float("nan")
        ),
        "YDelta_to_Yinv": (
            float(variation_ratio.mean()) if y_space_enabled else float("nan")
        ),
        "var_gain_loss": float(
            args.var_gain_temperature
            * F.softplus(
                (loss_full_per_sample - loss_inv_per_sample.detach())
                / args.var_gain_temperature
            ).mean()
        ),
        "fusion_rms_ratio_mean": float(fusion_rms_ratio.mean()),
        "fusion_rms_ratio_p10": float(torch.quantile(fusion_rms_ratio, 0.1)),
        "fusion_rms_ratio_p50": float(torch.quantile(fusion_rms_ratio, 0.5)),
        "fusion_rms_ratio_p90": float(torch.quantile(fusion_rms_ratio, 0.9)),
        "mapping_variation": "removed",
    }
    result.update(effect_summary)
    if analytic_reliability_predictions:
        r_prediction = torch.cat(analytic_reliability_predictions, dim=0)
        r_target = torch.cat(analytic_reliability_targets, dim=0)
        result["r_pred_mean"] = float(r_prediction.mean())
        result["r_target_mean"] = float(r_target.mean())
        result["corr_r_pred_r_target"] = spearman_correlation(
            r_prediction, r_target
        )
        for index, chunk in enumerate(
            torch.tensor_split(r_prediction, min(4, r_prediction.shape[1]), dim=1),
            1,
        ):
            result[f"r_pred_bin{index}"] = float(chunk.mean())
        for index, chunk in enumerate(
            torch.tensor_split(r_target, min(4, r_target.shape[1]), dim=1),
            1,
        ):
            result[f"r_target_bin{index}"] = float(chunk.mean())
        result["raw_variant_gain"] = float(torch.cat(raw_variant_gains).mean())
        result["gated_variant_gain"] = float(
            torch.cat(gated_variant_gains).mean()
        )
        raw_effect_bins = torch.stack(raw_variant_effect_bins).mean(0)
        gated_effect_bins = torch.stack(gated_variant_effect_bins).mean(0)
        for index, value in enumerate(raw_effect_bins, 1):
            result[f"variant_effect_raw_bin{index}"] = float(value)
        for index, value in enumerate(gated_effect_bins, 1):
            result[f"variant_effect_gated_bin{index}"] = float(value)
    else:
        for key in (
            "r_pred_mean", "r_target_mean", "corr_r_pred_r_target",
            "raw_variant_gain", "gated_variant_gain",
        ):
            result[key] = float("nan")
    for index in range(1, 5):
        result.setdefault(f"r_pred_bin{index}", float("nan"))
        result.setdefault(f"r_target_bin{index}", float("nan"))
        result.setdefault(f"variant_effect_raw_bin{index}", float("nan"))
        result.setdefault(f"variant_effect_gated_bin{index}", float("nan"))
    for index, value in enumerate(effect_bins, 1):
        result[f"var_effect_bin{index}"] = float(value)
    for index, value in enumerate(gain_bins, 1):
        result[f"gain_bin{index}"] = float(value)
    for index in range(1, 5):
        result.setdefault(f"var_effect_bin{index}", float("nan"))
        result.setdefault(f"gain_bin{index}", float("nan"))
    result["reliability_mean"] = fusion_gate_stats["gate/mean"]
    result["reliability_p10"] = float(torch.quantile(gate_sample, 0.1))
    result["reliability_p50"] = float(torch.quantile(gate_sample, 0.5))
    result["reliability_p90"] = float(torch.quantile(gate_sample, 0.9))
    if horizon_reliability_curve_sum is not None:
        reliability_curve = (
            horizon_reliability_curve_sum / max(horizon_curve_count, 1)
        )
        reliability_chunks = torch.tensor_split(
            reliability_curve, min(4, reliability_curve.numel())
        )
        for index, value in enumerate(reliability_chunks, 1):
            result[f"reliability_bin{index}"] = float(value.mean())
    if horizon_change_curve_sum is not None:
        change_curve = horizon_change_curve_sum / max(horizon_curve_count, 1)
        change_chunks = torch.tensor_split(
            change_curve, min(4, change_curve.numel())
        )
        for index, value in enumerate(change_chunks, 1):
            result[f"future_zvar_change_from_current_bin{index}"] = float(
                value.mean()
            )
    for index in range(1, 5):
        result.setdefault(f"reliability_bin{index}", float("nan"))
        result.setdefault(
            f"future_zvar_change_from_current_bin{index}", float("nan")
        )
    result.update(decomposition_gate_diagnostics.result())
    result.update(decomposition_diagnostics.result())
    result.update(moment_diagnostics(film_gamma_diagnostics, "gamma"))
    result.update(moment_diagnostics(film_beta_diagnostics, "beta"))
    result.update(moment_diagnostics(y_gamma_diagnostics, "y_gamma"))
    result.update(moment_diagnostics(y_beta_diagnostics, "y_beta"))
    result.update(moment_diagnostics(decay_rho_diagnostics, "decay_rho"))
    result["variant/gamma_abs_mean"] = (
        result["y_gamma_abs_mean"]
        if horizon_future_enabled
        else float("nan")
    )
    result["variant/beta_abs_mean"] = (
        result["y_beta_abs_mean"]
        if horizon_future_enabled
        else float("nan")
    )
    if decay_curve_sum is not None:
        decay_curve = decay_curve_sum / max(decay_curve_count, 1)
        chunks = torch.tensor_split(decay_curve, min(4, decay_curve.numel()))
        result["decay_d_near"] = float(chunks[0].mean())
        result["decay_d_mid"] = float(chunks[len(chunks) // 2].mean())
        result["decay_d_far"] = float(chunks[-1].mean())
    else:
        result["decay_d_near"] = float("nan")
        result["decay_d_mid"] = float("nan")
        result["decay_d_far"] = float("nan")
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
    result.update(
        evaluate_horizon_future_var(
            model, ordered_future_train, args, device
        )
    )
    if assignment is not None:
        q_all = torch.as_tensor(assignment)
        env_loss_sum = torch.zeros(args.env_num)
        env_mass = torch.zeros(args.env_num)
        diagnostic_z_inv, diagnostic_z_var, diagnostic_q = [], [], []
        diagnostic_inv_logits, diagnostic_var_logits = [], []
        selected_count = 0
        for x, y, sample_id, cycle_index in ordered_train:
            output = model.forward_components(
                x.float().to(device), cycle_index.to(device)
            )
            y = y[:, -args.pred_len :].float().to(device)
            sample_loss = (output["prediction"] - y).square().mean((1, 2)).cpu()
            q_batch = q_all[sample_id]
            env_loss_sum += (q_batch * sample_loss[:, None]).sum(0)
            env_mass += q_batch.sum(0)
            remaining = max(args.diagnostic_samples - selected_count, 0)
            if remaining:
                take = min(remaining, x.shape[0])
                z_inv = output["z_inv"][:take].cpu()
                z_var = output["z_var"][:take].cpu()
                inv_logits, var_logits = model.environment_logits(
                    output["z_inv"][:take], output["z_var"][:take]
                )
                diagnostic_z_inv.append(z_inv)
                diagnostic_z_var.append(z_var)
                diagnostic_q.append(q_batch[:take])
                diagnostic_inv_logits.append(inv_logits.cpu())
                diagnostic_var_logits.append(var_logits.cpu())
                selected_count += take
        result["per_env_MSE"] = (env_loss_sum / env_mass.clamp_min(1e-8)).tolist()
        z_inv = torch.cat(diagnostic_z_inv)
        z_var = torch.cat(diagnostic_z_var)
        q = torch.cat(diagnostic_q)
        diagnostics = representation_diagnostics(z_inv, z_var, q)
        result.update({key: float(value) for key, value in diagnostics.items()})
        result["diagnostic_sample_count"] = selected_count
        if args.representation_constraint == "classification":
            classifier_diagnostics = classification_diagnostics(
                torch.cat(diagnostic_inv_logits),
                torch.cat(diagnostic_var_logits),
                q,
            )
            result.update(
                {key: float(value) for key, value in classifier_diagnostics.items()}
            )
            result["var_acc"] = result["var_env_accuracy_argmax"]
            result["inv_acc"] = result["inv_env_accuracy_argmax"]
            result["env_acc_gap_var_minus_inv"] = (
                result["var_acc"] - result["inv_acc"]
            )
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
            result["env_acc_gap_var_minus_inv"] = float("nan")
    return result


def write_summary(path, rows):
    lines = [
        "experiment constraint decomposition fusion lambda_invpred lambda_future_var "
        "lambda_var_predictive lambda_var_utility lambda_var_conditional_gain "
        "var_conditional_margin lambda_var_gain var_gain_temperature "
        "lambda_decay_mono lambda_decay_far decay_bins "
        "lambda_future_h lambda_horizon_reliability future_anchor_count "
        "env_count MSE MAE "
        "inv_MSE improvement conditional_MSE mean_gain median_gain positive_gain "
        "gain_p10 gain_p50 gain_p90 corr_reliability_gain conditional_pair_MSE "
        "conditional_shuffle_MSE conditional_gain conditional_positive "
        "corr_conditional_main_gain g_z "
        "DeltaZ/Zinv corr_inv_env corr_var_env conflict Dwithin_grad "
        "Dbetween_grad separation random_z min_mass final_env_sim_corr "
        "future_var_loss future_var_cosine future_var_l2 future_var_pred_norm "
        "future_var_target_norm future_similarity_gain_corr "
        "var_effect bin1 bin2 bin3 bin4 far_near decay_mono decay_far "
        "y_gamma_abs y_beta_abs rho_mean decay_near decay_mid decay_far "
        "YDelta/Yinv gain_bin1 gain_bin2 gain_bin3 gain_bin4 "
        "future_zvar_cosine future_zvar_near future_zvar_mid future_zvar_far "
        "reliability_mean reliability_p10 reliability_p50 reliability_p90 "
        "reliability_bin1 reliability_bin2 reliability_bin3 reliability_bin4 "
        "future_change_bin1 future_change_bin2 future_change_bin3 future_change_bin4"
    ]
    for row in rows:
        lines.append(
            "{experiment} {representation_constraint} {decomposition_type} "
            "{variant_fusion_mode} {lambda_invpred} {lambda_future_var} "
            "{lambda_var_predictive} {lambda_var_utility} "
            "{lambda_var_conditional_gain} {var_conditional_margin} "
            "{lambda_var_gain} {var_gain_temperature} "
            "{lambda_decay_mono} {lambda_decay_far} {horizon_decay_bins} "
            "{lambda_future_h} {lambda_horizon_reliability} "
            "{future_var_anchor_count} "
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
            "{corr_future_var_similarity_with_sample_gain:.6f} "
            "{var_effect_mean:.6f} {var_effect_bin1:.6f} "
            "{var_effect_bin2:.6f} {var_effect_bin3:.6f} "
            "{var_effect_bin4:.6f} {var_effect_far_near_ratio:.6f} "
            "{decay_mono_loss:.6f} {decay_far_loss:.6f} "
            "{y_gamma_abs_mean:.6f} {y_beta_abs_mean:.6f} "
            "{decay_rho_mean:.6f} {decay_d_near:.6f} "
            "{decay_d_mid:.6f} {decay_d_far:.6f} "
            "{YDelta_to_Yinv:.6f} {gain_bin1:.6f} {gain_bin2:.6f} "
            "{gain_bin3:.6f} {gain_bin4:.6f} "
            "{future_zvar_cosine_mean:.6f} "
            "{future_zvar_cosine_near:.6f} "
            "{future_zvar_cosine_mid:.6f} "
            "{future_zvar_cosine_far:.6f} "
            "{reliability_mean:.6f} {reliability_p10:.6f} "
            "{reliability_p50:.6f} {reliability_p90:.6f} "
            "{reliability_bin1:.6f} {reliability_bin2:.6f} "
            "{reliability_bin3:.6f} {reliability_bin4:.6f} "
            "{future_zvar_change_from_current_bin1:.6f} "
            "{future_zvar_change_from_current_bin2:.6f} "
            "{future_zvar_change_from_current_bin3:.6f} "
            "{future_zvar_change_from_current_bin4:.6f}".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def reference_signature(args):
    names = (
        "backbone",
        "dataset_name",
        "dataset_archive_sha256",
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
        "enc_in",
        "cyclenet_cycle_len",
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
        "film_variant_fusion.",
        "y_film_decay_fusion.",
        "horizon_future_variant.",
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
        "film_variant_fusion.",
        "y_film_decay_fusion.",
        "horizon_future_variant.",
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
        saved_signature = payload.get("signature")
        if saved_signature != signature:
            legacy_only_missing = {
                "backbone",
                "dataset_archive_sha256",
                "enc_in",
                "cyclenet_cycle_len",
            }
            comparable_signature = {
                key: value
                for key, value in signature.items()
                if key not in legacy_only_missing
            }
            legacy_compatible = (
                args.allow_legacy_reference_checkpoint
                and args.backbone == "patchtst"
                and not args.dataset_archive_sha256
                and saved_signature == comparable_signature
            )
            if not legacy_compatible:
                raise ValueError("Shared reference checkpoint configuration mismatch")
            print(
                "accepted explicitly authorized legacy PatchTST reference "
                "after exact legacy-signature validation",
                flush=True,
            )
        load_shared_reference_state(reference, payload["state_dict"])
        source = "loaded" if saved_signature == signature else "loaded_legacy_compatible"
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
    parser.add_argument(
        "--backbone",
        choices=("patchtst", "itransformer", "cyclenet"),
        default="patchtst",
    )
    parser.add_argument("--root_path", default="./dataset/ETT-small/")
    parser.add_argument("--dataset_name", default="ETTh1")
    parser.add_argument("--dataset_archive_sha256", default="")
    parser.add_argument(
        "--data_class",
        choices=("ett_hour", "ett_minute", "custom"),
        default="ett_hour",
    )
    parser.add_argument("--data_path", default="ETTh1.csv")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--cyclenet_cycle_len", type=int, default=24)
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
    parser.add_argument("--differential_lr", action="store_true")
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_inv_head", type=float, default=5e-5)
    parser.add_argument("--lr_decomposer", type=float, default=1e-4)
    parser.add_argument("--lr_env_head", type=float, default=1e-4)
    parser.add_argument("--lr_variant", type=float, default=1e-4)
    parser.add_argument("--lr_gamma_beta_base", type=float, default=1e-4)
    parser.add_argument("--lr_reliability", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_risk", type=float, default=0.1)
    parser.add_argument("--lambda_invpred", type=float, default=0.5)
    parser.add_argument("--lambda_future_var", type=float, default=0.0)
    parser.add_argument("--lambda_future_h", type=float, default=0.0)
    parser.add_argument(
        "--lambda_horizon_reliability",
        "--lambda_r",
        dest="lambda_horizon_reliability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--horizon_reliability_temperature", type=float, default=0.05
    )
    parser.add_argument(
        "--future_var_anchor_count", type=int, default=16,
        help="Deprecated legacy horizon-teacher option; retained for replay.",
    )
    parser.add_argument("--future_patch_len", type=int, default=16)
    parser.add_argument(
        "--future_teacher_patches_per_batch", type=int, default=2
    )
    parser.add_argument(
        "--future_teacher_eval_patch_count", type=int, default=3
    )
    parser.add_argument("--lambda_var_predictive", type=float, default=0.0)
    parser.add_argument("--lambda_var_utility", type=float, default=0.0)
    parser.add_argument("--lambda_var_gain", type=float, default=0.0)
    parser.add_argument("--var_gain_temperature", type=float, default=0.01)
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
        choices=(
            "legacy",
            "off",
            "inv_only",
            "direct_gated",
            "film",
            "film_decay_reg",
            "y_film_decay",
            "horizon_future_var",
        ),
        default="legacy",
    )
    parser.add_argument(
        "--variant_fusion_gate_type",
        choices=("token", "feature"),
        default="token",
        help=(
            "Granularity of the direct Z_var fusion gate: token gives "
            "[B,C,P,1], feature gives [B,C,P,D]."
        ),
    )
    parser.add_argument(
        "--fusion_scale_calibration", choices=("none", "rms"), default="none"
    )
    parser.add_argument("--film_gamma_scale", type=float, default=0.1)
    parser.add_argument("--film_beta_scale", type=float, default=0.1)
    parser.add_argument("--lambda_decay_mono", type=float, default=0.0)
    parser.add_argument("--lambda_decay_far", type=float, default=0.0)
    parser.add_argument("--horizon_decay_bins", type=int, default=4)
    parser.add_argument("--y_film_gamma_scale", type=float, default=0.1)
    parser.add_argument("--y_film_beta_scale", type=float, default=0.1)
    parser.add_argument("--y_film_decay_bias", type=float, default=-4.0)
    parser.add_argument("--horizon_future_dim", type=int, default=32)
    parser.add_argument("--horizon_future_gamma_scale", type=float, default=0.1)
    parser.add_argument("--horizon_future_beta_scale", type=float, default=0.1)
    parser.add_argument("--horizon_future_chunk_size", type=int, default=32)
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
    parser.add_argument("--environment_quality_diagnostics", action="store_true")
    parser.add_argument("--environment_quality_final_only", action="store_true")
    parser.add_argument("--random_partition_repeats", type=int, default=1000)
    parser.add_argument("--reference_checkpoint", default="")
    parser.add_argument("--require_reference_checkpoint", action="store_true")
    parser.add_argument("--allow_legacy_reference_checkpoint", action="store_true")
    parser.add_argument("--prepare_reference_only", action="store_true")
    parser.add_argument("--save_final_checkpoint", action="store_true")
    parser.add_argument("--output", default="results/predictive_env_iv_ETTh1_96_96")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.horizon_decay_bins <= 0:
        raise ValueError("--horizon_decay_bins must be positive")
    if args.lambda_decay_mono < 0 or args.lambda_decay_far < 0:
        raise ValueError("horizon-decay loss weights must be non-negative")
    if args.lambda_future_h < 0 or args.lambda_horizon_reliability < 0:
        raise ValueError("horizon future loss weights must be non-negative")
    if args.lambda_horizon_reliability != 0:
        raise ValueError(
            "reliability gating was removed from horizon_future_var; "
            "set --lambda_horizon_reliability 0"
        )
    if args.horizon_reliability_temperature <= 0:
        raise ValueError("--horizon_reliability_temperature must be positive")
    if args.future_patch_len <= 0:
        raise ValueError("--future_patch_len must be positive")
    if args.future_teacher_patches_per_batch <= 0:
        raise ValueError("--future_teacher_patches_per_batch must be positive")
    if args.future_teacher_eval_patch_count <= 0:
        raise ValueError("--future_teacher_eval_patch_count must be positive")
    if args.horizon_future_dim <= 0:
        raise ValueError("--horizon_future_dim must be positive")
    if args.horizon_future_chunk_size <= 0:
        raise ValueError("--horizon_future_chunk_size must be positive")
    if (
        args.variant_fusion_mode != "horizon_future_var"
        and (
            args.lambda_future_h != 0
            or args.lambda_horizon_reliability != 0
        )
    ):
        raise ValueError(
            "horizon future losses are exclusive to horizon_future_var"
        )
    if (
        args.variant_fusion_mode != "film_decay_reg"
        and (args.lambda_decay_mono != 0 or args.lambda_decay_far != 0)
    ):
        raise ValueError(
            "decay regularization weights are exclusive to film_decay_reg"
        )
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
        first_ordered_batch = next(iter(ordered_train))
        first_batch = first_ordered_batch[0].float().to(device)
        first_cycle_index = first_ordered_batch[3].to(device)
        identity = model.identity_diagnostics(first_batch, first_cycle_index)
        print(experiment, "identity", identity, flush=True)
        manager, assignment, history, training_history = None, None, [], []
        quality_monitor = None
        experiment_dir = output / experiment
        experiment_dir.mkdir(exist_ok=True)
        optimizer = build_training_optimizer(model, args)
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
                args.environment_quality_diagnostics,
                (
                    args.random_partition_repeats
                    if args.environment_quality_diagnostics
                    else 100
                ),
            )
            if args.environment_quality_diagnostics:
                from models.environment_quality import EnvironmentQualityMonitor

                quality_monitor = EnvironmentQualityMonitor(
                    experiment_dir,
                    args.dataset_name,
                    save_all_stages=not args.environment_quality_final_only,
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
                if quality_monitor is not None:
                    quality_monitor.capture(
                        stage, record, manager.last_quality_payload
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
                "backbone": args.backbone,
                "cyclenet_cycle_len": args.cyclenet_cycle_len,
                "dataset_archive_sha256": args.dataset_archive_sha256,
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
                "differential_lr": args.differential_lr,
            }
        )
        if training_history:
            latest_training = training_history[-1]
            for key in (
                "lr/backbone",
                "lr/inv_head",
                "lr/decomposer",
                "lr/env_head",
                "lr/variant",
                "lr/gamma_beta_mean",
                "lr/gamma_beta_min",
                "lr/gamma_beta_max",
                "lr/reliability",
                "maturity",
                "G_inv_RMS",
                "G_var_RMS",
            ):
                result[key] = latest_training.get(key, float("nan"))
        final_diagnostic_keys = (
            "D_within_grad",
            "D_between_grad",
            "gradient_separation",
            "gradient_separation_gain",
            "conflict_ours",
            "random_partition_z_score",
            "random_partition_p_value",
            "random_partition_conflict_mean",
            "random_partition_conflict_std",
            "min_environment_mass",
            "alignment_overlap_before",
            "alignment_overlap_after",
            "normalized_assignment_entropy",
            "max_q_mean",
            "max_q_p10",
            "max_q_p50",
            "max_q_p90",
            "stage_ARI_to_previous",
            "stage_NMI_to_previous",
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
        result.setdefault("env_acc_gap_var_minus_inv", float("nan"))
        if quality_monitor is not None:
            quality_monitor.finalize(result)
        if args.save_final_checkpoint:
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "run_config": vars(args),
                    "metrics": result,
                },
                experiment_dir / "trained_checkpoint.pt",
            )
        for key in ("corr_inv_env", "corr_var_env"):
            result.setdefault(key, float("nan"))
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
