"""Scientific diagnostics; every panel identifies TRAIN versus TEST provenance."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def plot_bank(bank, train, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    centers = bank.centers.detach().cpu()
    futures = bank.future_centers.detach().cpu()
    keys = train["key"].reshape(-1, 64).float()
    distance = torch.cdist(centers, keys)
    nearest = distance.argmin(-1)
    matched = keys[nearest]
    for j in range(bank.patterns):
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes[0, 0].plot(centers[j], label="TRAIN prototype history")
        axes[0, 0].plot(matched[j], alpha=.65, label="matched TRAIN history")
        axes[0, 0].legend(fontsize=7)
        axes[0, 1].plot(futures[j], label="TRAIN future median")
        if all(name in train for name in ("hidden", "mean", "std", "hist_mean", "scale")):
            channel_count = train["key"].shape[1]
            row, channel = divmod(int(nearest[j]), channel_count)
            device = bank.centers.device
            normalization = {
                "model_mean": train["mean"][row:row+1, :, channel:channel+1].to(device),
                "model_std": train["std"][row:row+1, :, channel:channel+1].to(device),
                "hist_mean": train["hist_mean"][row:row+1, :, channel:channel+1].to(device),
                "hist_scale": train["scale"][row:row+1, :, channel:channel+1].to(device),
            }
            candidate = bank.query(
                train["key"][row:row+1, channel:channel+1].to(device),
                train["hidden"][row:row+1, channel:channel+1].to(device),
                normalization)["candidate"][0, 0, j].cpu()
            axes[0, 1].plot(candidate, label="TRAIN medoid A0-H mapped candidate")
        axes[0, 1].set_title("TRAIN-only future center; not a TEST oracle")
        axes[0, 1].legend(fontsize=7)
        axes[1, 0].bar(np.arange(4), bank.block_support[j].detach().cpu().numpy())
        axes[1, 0].set_title("Independent event support by TRAIN time block")
        axes[1, 1].bar([0, 1], [float(bank.event_support[j]), float(bank.stability[j])])
        axes[1, 1].set_xticks([0, 1], ["events", "forward stability"])
        fig.tight_layout()
        fig.savefig(output / f"prototype_{j:02d}.png", dpi=120)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, values, title in ((axes[0], centers, "TRAIN history similarity"),
                              (axes[1], futures, "TRAIN future-center similarity")):
        values = torch.nn.functional.normalize(values, dim=-1)
        image = ax.imshow((values @ values.T).numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=.046)
    fig.tight_layout()
    fig.savefig(output / "prototype_similarity.png", dpi=120)
    plt.close(fig)


def plot_stability(stability, reliability, output, evidence=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(np.arange(len(stability)), stability.detach().cpu().numpy())
    axes[0].set_title("Purged later-TRAIN time-forward stability")
    axes[1].imshow(reliability.detach().cpu().numpy(), aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("F03-relative reliable variation by future segment")
    axes[1].set_xlabel("future segment")
    axes[1].set_ylabel("prototype")
    if evidence and evidence.get("status") == "evaluated":
        gains = np.full(tuple(reliability.shape), np.nan)
        for row in evidence["segments"]:
            gains[row["prototype"], row["segment"]] = row["gain"]
        image = axes[2].imshow(gains, aspect="auto", cmap="coolwarm")
        fig.colorbar(image, ax=axes[2], fraction=.046)
    else:
        axes[2].text(.5, .5, "insufficient independent support", ha="center", va="center")
    axes[2].set_title("Purged later-TRAIN gain vs A0 by horizon segment")
    axes[2].set_xlabel("future segment")
    axes[2].set_ylabel("prototype")
    fig.tight_layout()
    fig.savefig(output / "stability_and_reliability.png", dpi=120)
    plt.close(fig)


def plot_memory(memory, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for j, item in enumerate(memory["prototypes"]):
        center, low, high = (item[name].detach().cpu().numpy()
                             for name in ("center", "q025", "q975"))
        fig, ax = plt.subplots(figsize=(11, 3))
        x = np.arange(len(center))
        ax.fill_between(x, low, high, alpha=.25, label="C TRAIN residual 95% empirical range")
        ax.plot(center, label="C TRAIN residual median")
        ax.plot(item["basis"][0].detach().cpu().numpy(), alpha=.6,
                label="low-rank residual trajectory component")
        ax.legend(fontsize=7)
        ax.set_title(f"Prototype {j}: downstream holdout residual; not full-system OOF")
        fig.tight_layout()
        fig.savefig(output / f"residual_memory_{j:02d}.png", dpi=120)
        plt.close(fig)


def plot_interval(truth, point, samples, output):
    """Plot one TEST example for presentation only, never selection."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    truth = truth.detach().cpu().numpy()
    point = point.detach().cpu().numpy()
    samples = samples.detach().cpu().numpy()
    lower = np.quantile(samples, .025, axis=0)
    upper = np.quantile(samples, .975, axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(truth))
    for trajectory in samples[:10]:
        ax.plot(x, trajectory, color="#79a7d3", alpha=.15, linewidth=.7)
    ax.fill_between(x, lower, upper, color="#79a7d3", alpha=.3, label="TEST 95% sampled interval")
    ax.plot(x, truth, color="black", label="TEST truth (evaluation only)")
    ax.plot(x, point, color="#d85d3f", label="frozen F03+reliable variation point")
    ax.legend(fontsize=8)
    ax.set_title("TEST visualization; TEST truth was not used for fitting or calibration")
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)
