"""Isolated IAPAD training/evaluation on the existing TRAIN-only EIIL stages."""

import json
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.predictive_env.affine_target_projection import (
    oracle_affine_diagnostics, project_affine_targets,
)
from tools.evaluate_fpem_probabilistic import probabilistic_metrics


@torch.no_grad()
def calibrate_affine_scale(model, ordered_train, args, device):
    """Robust affine coordinate scale from spaced TRAIN samples only."""
    dataset = ordered_train.dataset
    count = min(int(args.prob_scale_samples), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64)
    loader = DataLoader(Subset(dataset, indices.tolist()),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    was_training = model.training
    model.eval()
    values = []
    for x, y, _, cycle_index in loader:
        x = x.float().to(device)
        y = y[:, -args.pred_len:].float().to(device)
        output = model.forward_components(x, cycle_index.to(device))
        theta, _ = project_affine_targets(
            y, output["invariant_prediction"], output["input_scale"],
            args.future_patch_len, args.prob_affine_ridge,
        )
        values.append(theta.detach().abs().reshape(-1, 2).cpu())
    absolute = torch.cat(values, dim=0)
    scale = torch.quantile(absolute, 0.75, dim=0).clamp_min(0.01)
    model.probabilistic_affine_dynamics.set_theta_scale(scale)
    model.train(was_training)
    return {"affine_robust_scale_gamma": float(scale[0]),
            "affine_robust_scale_beta": float(scale[1]),
            "scale_source": f"TRAIN only; {count} spaced samples"}


def active_probability_maturity_parameters(dynamics, args, ablation):
    """Only heads reached by the enabled probabilistic losses enter G_prob.

    Future-Zvar remains trainable through those losses, but the maturity probe
    follows the established head-parameter convention and excludes it.
    """
    if ablation == "invariant_only":
        return ()
    modules = []
    if args.lambda_prob_mu or args.lambda_prob_smooth:
        modules.append(dynamics.center_head)
    if args.lambda_prob_fm:
        if ablation == "gaussian_baseline":
            modules.append(dynamics.gaussian_head)
        elif ablation != "deterministic_affine_center":
            modules.append(dynamics.velocity)
    if (ablation == "full" and args.use_stochastic_innovation
            and args.lambda_prob_innovation):
        modules.append(dynamics.innovation)
    return tuple(parameter for module in modules
                 for parameter in module.parameters() if parameter.requires_grad)


def probability_loss_weight(inv_loss, prob_loss, invariant_parameters,
                            probabilistic_parameters, args, epoch_index,
                            maturity_fn):
    """Detached single-stage probability weight, with an exact legacy option."""
    mode = getattr(args, "prob_loss_weight_mode", "fixed_epoch_ramp")
    if mode == "fixed_epoch_ramp":
        epochs = args.prob_ramp_epochs
        ramp = (0.2 + 0.8 * min(1.0, (epoch_index + 1) / epochs)
                if epochs > 0 else 1.0)
        return inv_loss.new_tensor(ramp), {"prob_loss_ramp": ramp}
    if mode != "gradient_relative_maturity":
        raise ValueError(f"unknown probability loss weight mode: {mode}")
    floor = float(args.prob_loss_min_weight)
    if not 0.0 < floor <= 1.0:
        raise ValueError("prob_loss_min_weight must be in (0, 1]")
    if maturity_fn is None:
        raise ValueError("gradient maturity requires the existing maturity function")
    if not probabilistic_parameters or not prob_loss.requires_grad:
        zero = inv_loss.detach().new_zeros(())
        maturity, inv_norm, prob_norm = zero, zero, zero
    else:
        maturity, inv_norm, prob_norm = maturity_fn(
            inv_loss, prob_loss, invariant_parameters,
            probabilistic_parameters,
        )
    maturity = maturity.detach()
    weight = (floor + (1.0 - floor) * maturity).detach()
    return weight, {"G_inv": inv_norm.detach(), "G_prob": prob_norm.detach(),
                    "maturity": maturity}


def train_epoch_prob(model, loader, assignment, optimizer, args, device,
                     epoch_index=0, maturity_fn=None):
    model.train()
    totals = {}
    seen = 0
    selected = args.prob_ablation
    dynamics = model.probabilistic_affine_dynamics
    use_innovation = args.use_stochastic_innovation and selected == "full"
    use_flow = selected not in ("invariant_only", "deterministic_affine_center",
                                "gaussian_baseline")
    prob_parameters = active_probability_maturity_parameters(
        dynamics, args, selected
    )
    for step, batch in enumerate(loader):
        if args.max_train_batches and step >= args.max_train_batches:
            break
        x, y, sample_id, _, _, cycle_index, _ = batch
        x = x.float().to(device)
        y = y[:, -args.pred_len:].float().to(device)
        output = model.forward_components(x, cycle_index.to(device))
        invariant = output["invariant_prediction"]
        inv_loss = (invariant - y).square().mean()
        q = torch.as_tensor(assignment[sample_id.numpy()], device=device).detach()
        env_loss, env_details = model.environment_classification_loss(
            output["z_inv"], output["z_var"], q,
        )
        loss = inv_loss + args.lambda_prob_env * env_loss
        terms = {"loss/inv": inv_loss, "loss/env": env_loss,
                 "var_env_soft_ce": env_details["var_env_soft_ce"],
                 "inv_env_soft_ce": env_details["inv_env_soft_ce"]}
        if selected != "invariant_only":
            aux = dynamics.supervised_losses(
                output["probabilistic_condition"], y, invariant,
                output["input_scale"], use_innovation=use_innovation,
                gaussian_baseline=selected == "gaussian_baseline",
            )
            prob_loss = (
                args.lambda_prob_mu * aux["center"]
                + args.lambda_prob_fm * (
                    aux["gaussian"] if selected == "gaussian_baseline"
                    else aux["flow"] if use_flow else 0.0
                )
                + args.lambda_prob_innovation * aux["innovation"]
                + args.lambda_prob_smooth * aux["smooth"]
            )
            weight, weight_terms = probability_loss_weight(
                inv_loss, prob_loss, tuple(model.head_linear.parameters()),
                prob_parameters, args, epoch_index, maturity_fn,
            )
            loss = loss + weight * prob_loss
            terms["loss/prob"] = prob_loss
            terms["prob_loss_weight"] = weight
            terms.update(weight_terms)
            terms.update({f"loss/{key}": aux[key] for key in
                          ("center", "flow", "gaussian", "innovation", "smooth")})
            terms.update({key: value for key, value in aux.items()
                          if key not in ("center", "flow", "gaussian",
                                         "innovation", "smooth")})
        aux_backbone = output.get("backbone_aux_loss")
        if aux_backbone is not None:
            loss = loss + args.lambda_timefilter_moe * aux_backbone
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.prob_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.prob_grad_clip)
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("non-finite IAPAD training loss")
        optimizer.step()
        terms["loss"] = loss
        terms["theta_scale_gamma"] = dynamics.theta_scale[0]
        terms["theta_scale_beta"] = dynamics.theta_scale[1]
        batch_size = x.shape[0]
        seen += batch_size
        for key, value in terms.items():
            value = float(torch.as_tensor(value).detach().cpu())
            totals[key] = totals.get(key, 0.0) + batch_size * value
    if seen == 0:
        raise RuntimeError("empty probabilistic training epoch")
    return {key: value / seen for key, value in totals.items()}


@torch.no_grad()
def evaluate_probabilistic(model, loader, args, device, output_dir, split_name,
                           ablation=None):
    model.eval()
    selected = ablation or args.prob_ablation
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + (25007 if split_name == "test" else 19001))
    scalar_sums = {}
    total = 0
    coverage_by_horizon = width_by_horizon = None
    records = {"environment": [], "volatility": [], "coverage": [],
               "width": [], "crps": []}
    oracle_sums = {}
    example_saved = False
    gamma_variance = []
    future_norm = []
    innovation_diagnostics = {}
    boundary_jumps = []
    for step, (x, y, _, cycle_index) in enumerate(loader):
        if args.max_prob_eval_batches and step >= args.max_prob_eval_batches:
            break
        x = x.float().to(device)
        y = y[:, -args.pred_len:].float().to(device)
        # Neither forecast call receives y nor a test environment label.
        paths = model.probabilistic_forecast(
            x, cycle_index.to(device), args.prob_num_samples,
            args.prob_flow_steps, selected, generator,
            args.prob_sample_chunk,
        )
        score = probabilistic_metrics(paths["y_samples"], y, paths["y_inv"])
        output = model.forward_components(x, cycle_index.to(device))
        oracle = oracle_affine_diagnostics(
            y, paths["y_inv"], output["input_scale"],
            args.future_patch_len, args.prob_affine_ridge,
        )
        if (selected == "full" and args.use_stochastic_innovation
                and output["probabilistic_condition"] is not None):
            theta_target, center_target = project_affine_targets(
                y, paths["y_inv"], output["input_scale"],
                args.future_patch_len, args.prob_affine_ridge,
            )
            _, innovation_stats = model.probabilistic_affine_dynamics.innovation.nll(
                output["probabilistic_condition"]["future"], y,
                paths["y_inv"], output["input_scale"], theta_target,
                center_target, args.future_patch_len,
            )
            for key, value in innovation_stats.items():
                innovation_diagnostics[key] = (
                    innovation_diagnostics.get(key, 0.0) + x.shape[0] * float(value)
                )
        batch_size = x.shape[0]
        total += batch_size
        for key, value in score.items():
            if torch.as_tensor(value).ndim == 0:
                scalar_sums[key] = scalar_sums.get(key, 0.0) + batch_size * float(value)
        for key, value in oracle.items():
            oracle_sums[key] = oracle_sums.get(key, 0.0) + batch_size * float(value)
        coverage_curve = score["coverage_per_horizon"].cpu() * batch_size
        width_curve = score["width_per_horizon"].cpu() * batch_size
        coverage_by_horizon = (coverage_curve if coverage_by_horizon is None
                               else coverage_by_horizon + coverage_curve)
        width_by_horizon = (width_curve if width_by_horizon is None
                            else width_by_horizon + width_curve)
        _, var_logits = model.environment_logits(output["z_inv"], output["z_var"])
        records["environment"].extend(var_logits.argmax(-1).cpu().tolist())
        records["volatility"].extend(x.std(1, unbiased=False).mean(-1).cpu().tolist())
        for name, key in (("coverage", "coverage_per_sample"),
                          ("width", "width_per_sample"),
                          ("crps", "crps_per_sample")):
            records[name].extend(score[key].cpu().tolist())
        if paths["gamma_samples"] is not None:
            variance_per_sample = paths["gamma_samples"].var(
                dim=1, unbiased=False).mean((1, 2))
            gamma_variance.extend(variance_per_sample.cpu().tolist())
            future_norm.extend(output["probabilistic_condition"]["future"]
                               .norm(dim=-1).mean((1, 2)).cpu().tolist())
        if args.pred_len > args.future_patch_len:
            correction = paths["y_samples"] - paths["y_inv"][:, None]
            jump = (correction[:, :, 1:] - correction[:, :, :-1]).abs()
            boundaries = torch.arange(args.future_patch_len - 1,
                                      args.pred_len - 1, args.future_patch_len,
                                      device=jump.device)
            boundary_jumps.append(float(jump[:, :, boundaries].mean() /
                                        jump.mean().clamp_min(1e-8)))
        if not example_saved:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            count = min(8, batch_size)
            np.savez_compressed(
                output_dir / f"{split_name}_trajectory_examples.npz",
                target=y[:count].cpu().numpy(),
                y_inv=paths["y_inv"][:count].cpu().numpy(),
                y_samples=paths["y_samples"][:count, :min(20, args.prob_num_samples)].cpu().numpy(),
                lower_95=paths["lower_95"][:count].cpu().numpy(),
                upper_95=paths["upper_95"][:count].cpu().numpy(),
            )
            example_saved = True
    if not total:
        raise RuntimeError(f"empty {split_name} probability evaluation")
    result = {key: value / total for key, value in scalar_sums.items()}
    result.update({key: value / total for key, value in oracle_sums.items()})
    # Preserve the aggregate normalization denominator rather than mixing
    # independently normalized batch ratios.
    result["quantile_CRPS_normalized"] = (
        result["quantile_crps_raw"] / max(result["target_abs_mean"], 1e-8)
    )
    result["gamma_sample_variance_mean"] = (
        float(np.mean(gamma_variance)) if gamma_variance else float("nan")
    )
    if len(gamma_variance) > 2 and np.std(gamma_variance) > 1e-10 and np.std(future_norm) > 1e-10:
        result["corr_gamma_variance_future_zvar_norm"] = float(
            np.corrcoef(gamma_variance, future_norm)[0, 1]
        )
    else:
        result["corr_gamma_variance_future_zvar_norm"] = float("nan")
    result["affine_patch_boundary_jump_ratio"] = (
        float(np.mean(boundary_jumps)) if boundary_jumps else float("nan")
    )
    result.update({key: value / total for key, value in innovation_diagnostics.items()})
    result["samples_evaluated"] = total
    result["prob_num_samples"] = args.prob_num_samples
    result["CRPS_definition"] = "empirical ensemble expectation in target units"
    result["quantile_CRPS_definition"] = (
        "2 * mean pinball over levels 0.05..0.95 / mean absolute target"
    )
    result["EnergyScore_definition"] = "Monte Carlo distinct cyclic sample pairs"
    result["ablation"] = selected
    coverage_curve = coverage_by_horizon / total
    width_curve = width_by_horizon / total
    for index, part in enumerate(torch.tensor_split(torch.arange(args.pred_len), 4)):
        result[f"horizon_bin{index + 1}_PICP_95"] = float(coverage_curve[part].mean())
        result[f"horizon_bin{index + 1}_MPIW_95"] = float(width_curve[part].mean())
    for environment in sorted(set(records["environment"])):
        mask = np.asarray(records["environment"]) == environment
        result[f"predicted_environment_{environment}_count"] = int(mask.sum())
        result[f"predicted_environment_{environment}_PICP_95"] = float(
            np.asarray(records["coverage"])[mask].mean())
        result[f"predicted_environment_{environment}_MPIW_95"] = float(
            np.asarray(records["width"])[mask].mean())
    volatility = np.asarray(records["volatility"])
    boundaries = np.quantile(volatility, [0, 1 / 3, 2 / 3, 1])
    for index in range(3):
        mask = ((volatility >= boundaries[index]) &
                (volatility <= boundaries[index + 1] if index == 2
                 else volatility < boundaries[index + 1]))
        if mask.any():
            result[f"volatility_tertile{index + 1}_PICP_95"] = float(
                np.asarray(records["coverage"])[mask].mean())
            result[f"volatility_tertile{index + 1}_MPIW_95"] = float(
                np.asarray(records["width"])[mask].mean())
    result_path = Path(output_dir) / f"{split_name}_probabilistic_metrics.json"
    result_path.write_text(json.dumps(result, indent=2))
    return result


def write_prob_loss_curves(history, destination):
    """Keep a compact, human-readable loss/scale curve beside full JSON logs."""
    columns = [
        "epoch", "stage", "loss", "loss/inv", "loss/env", "loss/center",
        "loss/flow", "loss/gaussian", "loss/innovation", "loss/smooth",
        "loss/prob", "G_inv", "G_prob", "maturity", "prob_loss_weight",
        "prob_loss_ramp", "target_clip_fraction", "theta_scale_gamma",
        "theta_scale_beta", "innovation_predicted_variance",
        "innovation_residual_variance", "innovation_variance_ratio",
    ]
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in columns})
