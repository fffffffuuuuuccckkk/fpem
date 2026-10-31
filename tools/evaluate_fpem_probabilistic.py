"""Sample-based probabilistic scores; never selects the best sampled future.

Empirical CRPS is in target units. The legacy ``quantile_CRPS_normalized``
uses 19 quantiles and global normalization; it is NOT ProbTS's definition.
These values remain available for checkpoint/result compatibility.

``ProbTS_CRPS`` reproduces ProbTS-main/probts/utils/evaluator.py at
the sample level with its experiment config ``quantiles_num=20``:
q=0.05,...,0.95, twice the quantile pinball loss summed over
the forecast horizon and dimensions, divided by that sample's absolute target
sum, then averaged over quantiles and samples. ``ProbTS_CRPS_Sum`` applies the
same operation after summing the target dimensions. Only the 1e-8 denominator
floor differs for all-zero targets, where upstream returns a non-finite value.
"""

import torch


@torch.no_grad()
def probts_quantile_crps(samples, target, quantiles_num=20):
    """Numerically guarded reproduction of ProbTS Evaluator's ``CRPS``.

    Samples are [B,N,L,C], targets [B,L,C]. The return value is a mean of
    sequence-local normalized scores, not a globally normalized pinball loss.
    """
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("samples [B,N,L,C], target [B,L,C] required")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("sample and target shapes do not align")
    if quantiles_num < 2:
        raise ValueError("quantiles_num must be at least 2")
    quantiles = torch.arange(1, quantiles_num, device=samples.device,
                             dtype=samples.dtype) / quantiles_num
    forecast_q = torch.quantile(samples, quantiles, dim=1)
    error = target.unsqueeze(0) - forecast_q
    pinball_twice = 2 * torch.maximum(
        quantiles[:, None, None, None] * error,
        (quantiles[:, None, None, None] - 1) * error,
    )
    sample_abs_sum = target.abs().sum((1, 2)).clamp_min(1e-8)
    per_sample = pinball_twice.sum((2, 3)) / sample_abs_sum[None, :]
    return per_sample.mean(0)


@torch.no_grad()
def probabilistic_metrics(samples, target, y_inv):
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("samples [B,N,L,C], target [B,L,C] required")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("sample and target shapes do not align")
    batch, count, horizon, channels = samples.shape
    if count < 2:
        raise ValueError("at least two samples required for distributional scores")
    mean = samples.mean(1)
    median = samples.median(1).values
    lower = torch.quantile(samples, 0.025, dim=1)
    upper = torch.quantile(samples, 0.975, dim=1)
    sorted_values = samples.sort(dim=1).values
    ranks = (2 * torch.arange(1, count + 1, device=samples.device,
                            dtype=samples.dtype) - count - 1).view(1, count, 1, 1)
    crps_point = (samples - target[:, None]).abs().mean(1) - (
        sorted_values * ranks
    ).sum(1) / (count * count)
    quantiles = torch.arange(0.05, 1.0, 0.05, device=samples.device, dtype=samples.dtype)
    forecasts = torch.quantile(samples, quantiles, dim=1)
    error = target.unsqueeze(0) - forecasts
    pinball = torch.maximum(quantiles[:, None, None, None] * error,
                            (quantiles[:, None, None, None] - 1) * error)
    quantile_crps_raw = 2 * pinball.mean()
    probts_crps = probts_quantile_crps(samples, target)
    denominator = target.abs().mean().clamp_min(1e-8)
    # Weighted interval score with 50%, 80%, 95% central intervals.
    wis_numerator = 0.5 * (median - target).abs()
    for alpha in (0.5, 0.2, 0.05):
        lo = torch.quantile(samples, alpha / 2, dim=1)
        hi = torch.quantile(samples, 1 - alpha / 2, dim=1)
        interval = (hi - lo) + (2 / alpha) * (
            (lo - target).clamp_min(0) + (target - hi).clamp_min(0)
        )
        wis_numerator = wis_numerator + (alpha / 2) * interval
    wis = wis_numerator / 3.5
    # Unbiased Monte Carlo pair estimate, vectorized over all 100 paths.
    # Circular offsets pair distinct independent trajectories without N^2 memory.
    paths = samples.flatten(2)
    truth = target.flatten(1)
    first_energy = (paths - truth[:, None]).norm(dim=-1).mean(1)
    offsets = [offset % count for offset in (1, 17, 37, 53) if offset % count]
    second_energy = torch.stack([
        (paths - paths.roll(offset, dims=1)).norm(dim=-1).mean(1)
        for offset in offsets
    ]).mean(0)
    energy = first_energy - 0.5 * second_energy
    inv_error = (y_inv - target).square().mean((1, 2))
    mean_error = (mean - target).square().mean((1, 2))
    coverage = ((target >= lower) & (target <= upper)).float()
    width = upper - lower
    return {
        "inv_MSE": inv_error.mean(),
        "inv_MAE": (y_inv - target).abs().mean(),
        "mean_MSE": mean_error.mean(),
        "mean_MAE": (mean - target).abs().mean(),
        "median_NMAE": (median - target).abs().mean() / denominator,
        "CRPS_empirical": crps_point.mean(),
        "CRPS": crps_point.mean(),
        "quantile_CRPS_normalized": quantile_crps_raw / denominator,
        "ProbTS_CRPS": probts_crps.mean(),
        "ProbTS_CRPS_per_sample": probts_crps,
        "PICP_95": coverage.mean(),
        "MPIW_95": width.mean(),
        "WIS": wis.mean(),
        "EnergyScore": energy.mean(),
        "sample_std_mean": samples.std(1, unbiased=False).mean(),
        "correction_win_rate": (mean_error < inv_error).float().mean(),
        "sample_diversity": (samples[:, 0] - samples[:, 1]).abs().mean(),
        "max_boundary_jump": (samples[..., 1:, :] - samples[..., :-1, :]).abs().max(),
        "crps_per_sample": crps_point.mean((1, 2)),
        "coverage_per_sample": coverage.mean((1, 2)),
        "width_per_sample": width.mean((1, 2)),
        "coverage_per_horizon": coverage.mean((0, 2)),
        "width_per_horizon": width.mean((0, 2)),
        "quantile_crps_raw": quantile_crps_raw,
        "target_abs_mean": denominator,
    }


def main():
    """Evaluate counterfactual conditions from one fixed trained checkpoint."""
    import argparse
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import sys
    from torch.utils.data import DataLoader

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from models.PatchTST_PredictiveEnvIV import Model
    from tools.run_predictive_env_iv_patchtst import (
        IndexedDataset, build_data, model_config,
    )
    from tools.fpem_probabilistic_training import evaluate_probabilistic

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ablation", choices=(
        "full", "affine_flow", "deterministic_affine_center",
        "invariant_only", "shuffled_zvar", "unconditional_flow",
        "gaussian_baseline",
    ), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_batches", type=int, default=0)
    cli = parser.parse_args()
    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    payload = torch.load(cli.checkpoint, map_location="cpu")
    args = SimpleNamespace(**payload["run_config"])
    args.gpu = cli.gpu
    args.prob_num_samples = cli.num_samples
    args.max_prob_eval_batches = cli.max_batches
    datasets, _, _, test_loader = build_data(args)
    if cli.split == "val":
        loader = DataLoader(
            IndexedDataset(datasets["val"], cycle_len=args.cyclenet_cycle_len),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
    else:
        loader = test_loader
    model = Model(model_config(args, "A2")).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    result = evaluate_probabilistic(
        model, loader, args, device, Path(cli.output), cli.split,
        ablation=cli.ablation,
    )
    print(json.dumps({key: result.get(key) for key in (
        "ablation", "samples_evaluated", "inv_MSE", "mean_MSE",
        "CRPS", "PICP_95", "MPIW_95", "sample_diversity",
    )}, indent=2))


if __name__ == "__main__":
    main()
