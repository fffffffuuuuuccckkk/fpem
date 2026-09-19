#!/usr/bin/env python
"""t-SNE views of H, Z_inv, Z_var and fused Z using TRAIN environments only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.PatchTST_PredictiveEnvIV import Model
from models.predictive_env import PredictiveConflictEnvironment
from tools.run_predictive_env_iv_patchtst import build_data, model_config


REPRESENTATIONS = (
    ("H", "Backbone H"),
    ("Zinv", "Invariant Z_inv"),
    ("Zvar", "Variant Z_var"),
    ("Zfinal", "Fused final Z"),
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_environment_payload(experiment_dir):
    candidates = list(Path(experiment_dir).glob("environment_gradient_stage_*.npz"))
    if not candidates:
        return None
    def stage(path):
        return int(path.stem.rsplit("_", 1)[-1])
    return max(candidates, key=stage)


def load_checkpoint_context(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**payload["run_config"])
    _, ordered_train, _, _ = build_data(args)
    model = Model(model_config(args, "A2")).to(device)
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    model.eval()
    return model, ordered_train, args


@torch.no_grad()
def collect_train_probe_statistics(model, ordered_train, run_args, device):
    """Collect only EIIL sufficient statistics, never full Traffic predictions."""
    sample_ids, probe_gradients, sample_mse, sample_mae = [], [], [], []
    for x, y, sample_id, cycle_index in ordered_train:
        output = model.forward_components(
            x.float().to(device), cycle_index.to(device)
        )
        prediction = output["prediction"]
        target = y[:, -run_args.pred_len :].float().to(device)
        error = prediction - target
        reduce_dims = tuple(range(1, prediction.ndim))
        probe_gradients.append(
            torch.stack(
                [
                    2.0 * (error * prediction).mean(reduce_dims),
                    2.0 * error.mean(reduce_dims),
                ],
                dim=-1,
            ).cpu()
        )
        sample_mse.append(error.square().flatten(1).mean(1).cpu())
        sample_mae.append(error.abs().flatten(1).mean(1).cpu())
        sample_ids.append(sample_id.cpu())
    sample_ids = torch.cat(sample_ids)
    expected = torch.arange(len(ordered_train.dataset))
    if not torch.equal(sample_ids, expected):
        raise RuntimeError("environment inference requires complete chronological TRAIN IDs")
    return {
        "sample_id": sample_ids,
        "probe_gradients": torch.cat(probe_gradients),
        "sample_mse": torch.cat(sample_mse),
        "sample_mae": torch.cat(sample_mae),
    }


def infer_final_environment(model, ordered_train, run_args, device):
    """Infer one final TRAIN-only EIIL partition from a frozen final checkpoint."""
    statistics = collect_train_probe_statistics(
        model, ordered_train, run_args, device
    )
    probe = statistics["probe_gradients"]
    manager = PredictiveConflictEnvironment(
        sample_count=len(ordered_train.dataset),
        env_num=run_args.env_num,
        steps=run_args.eiil_steps,
        lr=run_args.eiil_lr,
        balance_weight=run_args.balance_weight,
        entropy_weight=run_args.entropy_weight,
        seed=run_args.seed,
        diagnostic_samples=run_args.diagnostic_samples,
        matching=run_args.environment_matching,
        matching_q_weight=run_args.matching_q_weight,
        matching_gradient_weight=run_args.matching_gradient_weight,
        quality_diagnostics=False,
        random_partition_repeats=1,
    )
    logits = manager.logits.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=manager.lr)
    for _ in range(manager.steps):
        q = logits.softmax(-1)
        gradients = manager._environment_gradients(q, probe)
        conflict = gradients.var(0, unbiased=False).sum()
        balance = (q.mean(0) - 1.0 / manager.env_num).square().sum()
        negative_entropy = (q * q.clamp_min(1e-8).log()).sum(1).mean()
        objective = (
            -conflict
            + manager.balance_weight * balance
            + manager.entropy_weight * negative_entropy
        )
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
    q = logits.detach().softmax(-1)
    gradients = manager._environment_gradients(q, probe)
    return {
        "sample_id": statistics["sample_id"].numpy(),
        "g_scale": probe[:, 0].numpy(),
        "g_bias": probe[:, 1].numpy(),
        "q": q.numpy(),
        "hard_env": q.argmax(1).numpy(),
        "sample_mse": statistics["sample_mse"].numpy(),
        "sample_mae": statistics["sample_mae"].numpy(),
        "environment_centroids": gradients.numpy(),
    }


def stratified_selection(hard_env, max_samples):
    """Deterministic, approximately balanced selection without any RNG."""
    hard_env = np.asarray(hard_env, dtype=np.int64)
    sample_count = len(hard_env)
    if max_samples <= 0 or sample_count <= max_samples:
        return np.arange(sample_count, dtype=np.int64)
    environments = np.unique(hard_env)
    quota = max(max_samples // max(len(environments), 1), 1)
    selected = []
    for environment in environments:
        candidates = np.flatnonzero(hard_env == environment)
        count = min(quota, len(candidates))
        positions = np.linspace(0, len(candidates) - 1, count).round().astype(int)
        selected.extend(candidates[positions].tolist())
    selected = np.unique(np.asarray(selected, dtype=np.int64))
    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(sample_count), selected, assume_unique=True)
        count = min(max_samples - len(selected), len(remaining))
        positions = np.linspace(0, len(remaining) - 1, count).round().astype(int)
        selected = np.concatenate((selected, remaining[positions]))
    return np.sort(selected[:max_samples])


@torch.no_grad()
def extract_representations(
    model, ordered_train, selected_sample_ids, device, batch_size
):
    subset = Subset(ordered_train.dataset, selected_sample_ids.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    values = {name: [] for name, _ in REPRESENTATIONS}
    observed_ids = []
    for x, _, sample_id, cycle_index in loader:
        output = model.forward_components(
            x.float().to(device), cycle_index.to(device)
        )
        observed_ids.append(sample_id.numpy())
        values["H"].append(output["hidden_tokens"].mean((1, 2)).cpu().numpy())
        values["Zinv"].append(output["z_inv"].cpu().numpy())
        values["Zvar"].append(output["z_var"].cpu().numpy())
        values["Zfinal"].append(output["final_tokens"].mean((1, 2)).cpu().numpy())
    observed_ids = np.concatenate(observed_ids)
    if not np.array_equal(observed_ids, selected_sample_ids):
        raise RuntimeError("TRAIN sample correspondence changed during extraction")
    return {key: np.concatenate(chunks) for key, chunks in values.items()}


def compute_tsne(representation, seed, perplexity):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    standardized = StandardScaler().fit_transform(representation)
    effective_perplexity = min(float(perplexity), max((len(standardized) - 1) / 3, 1.0))
    coordinates = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=int(seed),
        method="barnes_hut",
        n_jobs=1,
    ).fit_transform(standardized)
    return coordinates, effective_perplexity


def scatter(axis, coordinates, hard_env, q, title):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    for environment in range(q.shape[1]):
        selected = hard_env == environment
        confidence = q[selected].max(1) if selected.any() else np.array([])
        color = np.tile(np.asarray(cmap(environment % 10)), (selected.sum(), 1))
        if len(color):
            color[:, 3] = 0.2 + 0.65 * confidence
        axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=9,
            c=color,
            label=f"env {environment} (n={int(selected.sum())})",
            rasterized=True,
        )
        weights = q[:, environment]
        if weights.sum() > 1e-12:
            center = (coordinates * weights[:, None]).sum(0) / weights.sum()
            axis.scatter(
                center[0], center[1], marker="X", s=100,
                color=cmap(environment % 10), edgecolors="black", linewidths=0.8,
            )
    axis.set_title(title)
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.grid(alpha=0.18)


def render(output_dir, dataset, coordinates, hard_env, q):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, title in REPRESENTATIONS:
        figure, axis = plt.subplots(figsize=(7.2, 6.0))
        scatter(axis, coordinates[name], hard_env, q, f"{dataset}: {title} by environment")
        axis.legend(fontsize=8, loc="best")
        figure.tight_layout()
        figure.savefig(output_dir / f"tsne_{name}_by_environment.png", dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 10.5))
    for axis, (name, title) in zip(axes.flat, REPRESENTATIONS):
        scatter(axis, coordinates[name], hard_env, q, title)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=q.shape[1])
    figure.suptitle(
        f"{dataset}: TRAIN representations by final environment\n"
        "Each panel uses an independently fitted t-SNE; axes are not cross-panel comparable."
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure.savefig(output_dir / "tsne_H_Zinv_Zvar_Zfinal.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--environment_payload", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument(
        "--infer_environment",
        action="store_true",
        help="infer a final TRAIN-only EIIL partition from the frozen checkpoint",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.experiment_dir / "trained_checkpoint.pt"
    output_dir = args.output_dir or args.experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    model, ordered_train, run_args = load_checkpoint_context(checkpoint, device)
    environment_path = args.environment_payload
    if environment_path is None and not args.infer_environment:
        environment_path = latest_environment_payload(args.experiment_dir)
    if args.infer_environment or environment_path is None:
        environment = infer_final_environment(
            model, ordered_train, run_args, device
        )
        environment_path = output_dir / "environment_gradient_final_inferred.npz"
        np.savez_compressed(environment_path, **environment)
        environment_source = "inferred once from frozen final checkpoint on TRAIN"
    else:
        environment = dict(np.load(environment_path))
        environment_source = "saved training-stage TRAIN environment payload"
    all_sample_ids = environment["sample_id"].astype(np.int64)
    selected_positions = stratified_selection(environment["hard_env"], args.max_samples)
    selected_sample_ids = all_sample_ids[selected_positions]
    representations = extract_representations(
        model, ordered_train, selected_sample_ids, device, args.batch_size
    )
    q = environment["q"][selected_positions]
    hard_env = environment["hard_env"][selected_positions].astype(np.int64)
    coordinates, perplexities = {}, {}
    for name, _ in REPRESENTATIONS:
        coordinates[name], perplexities[name] = compute_tsne(
            representations[name], args.seed, args.perplexity
        )
    render(output_dir, args.dataset, coordinates, hard_env, q)
    np.savez_compressed(
        output_dir / "tsne_representations_and_coordinates.npz",
        sample_id=selected_sample_ids,
        q=q,
        hard_env=hard_env,
        **{f"representation_{key}": value for key, value in representations.items()},
        **{f"tsne_{key}": value for key, value in coordinates.items()},
    )
    with (output_dir / "tsne_coordinates.csv").open("w", newline="") as handle:
        fields = ["sample_id", "hard_env", "max_q"]
        for name, _ in REPRESENTATIONS:
            fields.extend((f"{name}_tsne_x", f"{name}_tsne_y"))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample_id in enumerate(selected_sample_ids):
            row = {
                "sample_id": int(sample_id),
                "hard_env": int(hard_env[index]),
                "max_q": float(q[index].max()),
            }
            for name, _ in REPRESENTATIONS:
                row[f"{name}_tsne_x"] = float(coordinates[name][index, 0])
                row[f"{name}_tsne_y"] = float(coordinates[name][index, 1])
            writer.writerow(row)
    summary = {
        "dataset": args.dataset,
        "source_split": "TRAIN only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "environment_payload": str(environment_path),
        "environment_source": environment_source,
        "sample_count": int(len(selected_sample_ids)),
        "selection": "deterministic hard-environment-stratified evenly spaced",
        "seed": args.seed,
        "perplexity": perplexities,
        "backbone": run_args.backbone,
        "representations": [name for name, _ in REPRESENTATIONS],
    }
    (output_dir / "tsne_summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
