#!/usr/bin/env python3
"""Plot a saved TEST FPEM residual-head trajectory with its exact input window.

The saved trajectories contain a deterministic invariant forecast plus sampled
residual corrections.  Their sample mean is an illustrative point forecast, not
the aggregate 100-sample score reported by the experiment.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-path", default="ETTh1.csv")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=48)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--channel", default="MUFL")
    parser.add_argument("--shared-encoder", type=Path)
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_root))
    from data_provider.data_loader import Dataset_ETT_hour

    dataset = Dataset_ETT_hour(
        args=SimpleNamespace(augmentation_ratio=0),
        root_path=str(args.data_root), flag="test",
        size=[args.seq_len, min(48, args.seq_len // 2), args.pred_len],
        features="M", data_path=args.data_path, target="OT",
        scale=True, timeenc=1, freq="h",
    )
    with np.load(args.examples) as saved:
        target = saved["target"].copy()
        y_inv = saved["y_inv"].copy()
        y_samples = saved["y_samples"].copy()
    saved_reference_sample_count = int(y_samples.shape[1])
    index = args.sample_index
    if index < 0 or index >= len(target):
        raise ValueError(f"sample-index must be in [0, {len(target) - 1}]")
    x, y, _, _ = dataset[index]
    y = y[-args.pred_len:]
    if not np.allclose(target[index], y, atol=2e-5, rtol=1e-5):
        raise ValueError("saved TEST trajectory does not match the indexed TEST target")
    reproduced = False
    if args.shared_encoder and args.head_checkpoint and args.run_config:
        import torch
        from models.PatchTST_PredictiveEnvIV import Model
        from models.predictive_env.probability_heads import ProbabilityHeadBank
        from tools.run_predictive_env_iv_patchtst import model_config

        run = SimpleNamespace(**json.loads(args.run_config.read_text()))
        run.enc_in = x.shape[-1]
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        model = Model(model_config(run, "A2")).to(device)
        encoder_payload = torch.load(args.shared_encoder, map_location=device,
                                     weights_only=False)
        model.load_state_dict(encoder_payload["model_state"], strict=True)
        bank = ProbabilityHeadBank(
            run.d_model, run.pred_len, run.future_patch_len,
            run.prob_condition_mode, ("residual_gaussian",),
        ).to(device)
        bank.condition_encoder.load_state_dict(
            encoder_payload["condition_encoder_state"], strict=True)
        bank.set_theta_scale(model.probabilistic_affine_dynamics.theta_scale)
        head_payload = torch.load(args.head_checkpoint, map_location=device,
                                  weights_only=False)
        bank.heads["residual_gaussian"].load_state_dict(head_payload["state_dict"],
                                                        strict=True)
        model.eval()
        bank.eval()
        # Preserve the original first TEST batch shape and RNG stream so the
        # stored first 20 samples can be checked against the live checkpoint.
        batch_size = int(run.batch_size)
        batch_x = np.stack([dataset[i][0] for i in range(batch_size)])
        test_offset = 12 * 30 * 24 + 4 * 30 * 24 - args.seq_len
        cycle = (test_offset + np.arange(batch_size)) % run.cyclenet_cycle_len
        with torch.no_grad():
            output = model.forward_components(
                torch.as_tensor(batch_x, dtype=torch.float32, device=device),
                torch.as_tensor(cycle, dtype=torch.long, device=device),
            )
            live_inv = output["invariant_prediction"]
            future = output["probabilistic_condition"]["future"]
            condition = bank.condition(future, live_inv, output["input_scale"])
            generator = torch.Generator(device=device)
            generator.manual_seed(run.seed + 25007)
            paths = bank.sample_head(
                "residual_gaussian", condition, future, live_inv,
                output["input_scale"], run.prob_num_samples, generator,
                run.prob_sample_chunk, run.prob_flow_steps,
            )
        live_inv = live_inv.cpu().numpy()
        live_samples = paths["y_samples"].cpu().numpy()
        if not np.allclose(live_inv[:len(y_inv)], y_inv, atol=2e-4, rtol=1e-4):
            raise ValueError("checkpoint invariant prediction differs from saved TEST examples")
        if not np.allclose(live_samples[:len(y_samples), :run.prob_sample_chunk],
                           y_samples[:, :run.prob_sample_chunk], atol=2e-4, rtol=1e-4):
            raise ValueError("checkpoint residual samples differ from saved TEST examples")
        y_inv = live_inv
        y_samples = live_samples
        reproduced = True
    names = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    channel = int(args.channel) if args.channel.isdigit() else names.index(args.channel)
    if channel >= x.shape[-1]:
        raise ValueError("channel exceeds number of dataset variables")
    mean = float(dataset.scaler.mean_[channel])
    std = float(dataset.scaler.scale_[channel])
    history = x[:, channel] * std + mean
    truth = target[index, :, channel] * std + mean
    invariant = y_inv[index, :, channel] * std + mean
    residual_samples = (y_samples[index, :, :, channel]
                        - y_inv[index, None, :, channel]) * std
    correction = residual_samples.mean(axis=0)
    corrected = invariant + correction
    oracle_residual = truth - invariant
    lower, upper = np.quantile(residual_samples, [0.05, 0.95], axis=0)
    history_time = np.arange(-args.seq_len, 0)
    future_time = np.arange(args.pred_len)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4),
                             gridspec_kw={"width_ratios": [1.35, 1]},
                             constrained_layout=True)
    ax, delta_ax = axes
    ax.plot(history_time, history, color="#284b63", lw=1.6,
            label="Observed history (receptive field)")
    ax.plot(future_time, truth, color="#171717", lw=1.8,
            label="Future truth")
    ax.plot(future_time, invariant, color="#d07822", lw=1.7,
            label="Invariant forecast")
    ax.fill_between(future_time, invariant + lower, invariant + upper,
                    color="#2d8b67", alpha=0.20,
                    label="Residual-adjusted prediction range (5-95%)")
    ax.axvline(-0.5, color="#888888", lw=1, ls="--")
    ax.set_xlim(-args.seq_len, args.pred_len - 1)
    ax.set_xlabel("Time relative to forecast origin (hours)")
    ax.set_ylabel(f"{names[channel]} (original scale)")
    ax.set_title("Observed history and invariant forecast")
    ax.grid(alpha=0.18)
    ax.legend(loc="best", frameon=False)

    delta_ax.plot(future_time, correction, color="#2d8b67", lw=1.8,
                  label="Learned mean correction")
    delta_ax.plot(future_time, oracle_residual, color="#555555", lw=1.3,
                  ls="--", label="True residual (future truth - invariant)")
    delta_ax.axhline(0, color="#888888", lw=0.8)
    delta_ax.set_xlim(0, args.pred_len - 1)
    delta_ax.set_xlabel("Forecast step (hours after input window)")
    delta_ax.set_ylabel("Correction (original scale)")
    delta_ax.set_title("Residual correction over the forecast horizon")
    delta_ax.grid(alpha=0.18)
    delta_ax.legend(loc="best", frameon=False)
    fig.suptitle(f"ETTh1 | PatchTST-FPEM | TEST sample {index} | {names[channel]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_hour", "history", "future_truth", "y_inv",
                         "learned_correction_mean", "corrected_forecast",
                         "true_residual", "correction_p05", "correction_p95"])
        for t, value in zip(history_time, history):
            writer.writerow([int(t), float(value), "", "", "", "", "", "", ""])
        for t in future_time:
            writer.writerow([int(t), "", float(truth[t]), float(invariant[t]),
                             float(correction[t]), float(corrected[t]),
                             float(oracle_residual[t]), float(lower[t]), float(upper[t])])
    metrics = {
        "dataset": "ETTh1", "backbone": "patchtst", "head": "residual_gaussian",
        "split": "TEST", "sample_index": index, "channel": names[channel],
        "seq_len": args.seq_len, "pred_len": args.pred_len,
        "generated_trajectory_count": int(len(residual_samples)),
        "saved_reference_sample_count": saved_reference_sample_count,
        "invariant_MSE_this_sample_channel": float(np.mean((truth - invariant) ** 2)),
        "corrected_MSE_this_sample_channel": float(np.mean((truth - corrected) ** 2)),
        "reproduced_from_checkpoint": reproduced,
        "note": ("All evaluation samples regenerated from checkpoint and checked against the saved first chunk."
                 if reproduced else "Mean correction uses the saved trajectory subset, not all 100 evaluation samples."),
        "source_examples": str(args.examples),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
