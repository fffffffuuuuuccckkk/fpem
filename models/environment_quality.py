"""TRAIN-only environment-quality artifacts for PredictiveEnvIV.

This module is deliberately post-processing only: it consumes detached CPU
arrays emitted by EIIL and owns no model state or global random-number source.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


SCALAR_STAGE_FIELDS = (
    "stage",
    "D_within_grad",
    "D_between_grad",
    "gradient_separation",
    "gradient_separation_gain",
    "conflict_ours",
    "random_partition_conflict_mean",
    "random_partition_conflict_std",
    "random_partition_z_score",
    "random_partition_p_value",
    "min_environment_mass",
    "normalized_assignment_entropy",
    "max_q_mean",
    "max_q_p10",
    "max_q_p50",
    "max_q_p90",
    "environment_similarity_correlation",
    "stage_ARI_to_previous",
    "stage_NMI_to_previous",
    "mean_q_change",
)

VECTOR_STAGE_FIELDS = (
    "environment_mass",
    "per_env_MSE",
    "per_env_MAE",
    "per_env_mean_sample_loss",
    "env_gradient_scale",
    "env_gradient_bias",
)


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class EnvironmentQualityMonitor:
    def __init__(self, output_dir, dataset, save_all_stages=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = str(dataset)
        self.save_all_stages = bool(save_all_stages)
        self.records = []
        self.payloads = []

    @staticmethod
    def _plot_modules():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        return plt, Ellipse

    def capture(self, stage, record, payload):
        if payload is None:
            raise RuntimeError("quality diagnostics were enabled but EIIL emitted no payload")
        copied = dict(record)
        copied["stage"] = int(stage)
        self.records.append(copied)
        self.payloads.append(payload)
        np.savez_compressed(
            self.output_dir / f"environment_gradient_stage_{stage}.npz",
            **payload,
        )
        if self.save_all_stages:
            self._gradient_plot(
                payload,
                self.output_dir / f"environment_gradient_stage_{stage}.png",
                f"{self.dataset}: predictive gradients, stage {stage}",
            )
        self._write_stage_csv()

    def _gradient_plot(self, payload, path, title):
        plt, Ellipse = self._plot_modules()
        x = np.asarray(payload["g_scale"])
        y = np.asarray(payload["g_bias"])
        q = np.asarray(payload["q"])
        hard = np.asarray(payload["hard_env"])
        centroids = np.asarray(payload["environment_centroids"])
        figure, axis = plt.subplots(figsize=(7.2, 6.0))
        cmap = plt.get_cmap("tab10")
        for environment in range(q.shape[1]):
            selected = hard == environment
            axis.scatter(
                x[selected],
                y[selected],
                s=7,
                alpha=0.28,
                color=cmap(environment % 10),
                label=f"env {environment} (n={int(selected.sum())})",
                rasterized=True,
            )
            weights = q[:, environment].astype(np.float64)
            weight_sum = weights.sum()
            if weight_sum <= 1e-12:
                continue
            center = centroids[environment]
            axis.scatter(
                center[0],
                center[1],
                s=110,
                marker="X",
                edgecolors="black",
                linewidths=0.8,
                color=cmap(environment % 10),
                zorder=5,
            )
            centered = np.column_stack((x - center[0], y - center[1]))
            covariance = (centered * weights[:, None]).T @ centered / weight_sum
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            if eigenvalues[0] <= 1e-16:
                continue
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            # sqrt(chi2.ppf(.95, df=2))^2 = 5.991, avoiding scipy in plotting.
            width, height = 2.0 * np.sqrt(5.991 * eigenvalues)
            axis.add_patch(
                Ellipse(
                    center,
                    width,
                    height,
                    angle=angle,
                    facecolor="none",
                    edgecolor=cmap(environment % 10),
                    linewidth=1.8,
                )
            )
        axis.set_xlabel("probe gradient: scale")
        axis.set_ylabel("probe gradient: bias")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)

    def _write_stage_csv(self):
        path = self.output_dir / "environment_quality_stages.csv"
        fields = SCALAR_STAGE_FIELDS + VECTOR_STAGE_FIELDS
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                row = {key: record.get(key, float("nan")) for key in SCALAR_STAGE_FIELDS}
                row.update(
                    {
                        key: json.dumps(record.get(key, []), separators=(",", ":"))
                        for key in VECTOR_STAGE_FIELDS
                    }
                )
                writer.writerow(row)

    def _line_plot(self, fields, labels, path, title, ylabel):
        plt, _ = self._plot_modules()
        stages = [record["stage"] for record in self.records]
        figure, axis = plt.subplots(figsize=(7.2, 4.4))
        for field, label in zip(fields, labels):
            values = [record.get(field, float("nan")) for record in self.records]
            axis.plot(stages, values, marker="o", linewidth=1.8, label=label)
        axis.set_xlabel("stage")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        if len(fields) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)

    def finalize(self, model_metrics):
        if not self.records:
            return None
        final = self.records[-1]
        payload = self.payloads[-1]
        self._gradient_plot(
            payload,
            self.output_dir / "environment_gradient_final.png",
            f"{self.dataset}: final predictive-gradient environments",
        )
        self._line_plot(
            ("gradient_separation",),
            ("D_between / D_within",),
            self.output_dir / "environment_separation.png",
            f"{self.dataset}: environment separation",
            "gradient separation",
        )
        self._line_plot(
            ("stage_ARI_to_previous", "stage_NMI_to_previous"),
            ("ARI", "NMI"),
            self.output_dir / "environment_stability.png",
            f"{self.dataset}: adjacent-stage stability",
            "agreement",
        )
        plt, _ = self._plot_modules()
        null = np.asarray(payload["random_partition_conflicts"])
        ours = float(np.asarray(payload["conflict_ours"]))
        figure, axis = plt.subplots(figsize=(7.2, 4.4))
        axis.hist(null, bins=40, alpha=0.75, color="#4C78A8")
        axis.axvline(ours, color="#E45756", linewidth=2.2, label="ours")
        axis.axvline(null.mean(), color="black", linestyle="--", label="random mean")
        axis.set_xlabel("predictive-gradient conflict")
        axis.set_ylabel("random partitions")
        axis.set_title(f"{self.dataset}: random-partition null")
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "environment_conflict_null.png", dpi=180)
        plt.close(figure)

        final_csv = self.output_dir / "environment_gradient_final.csv"
        q = np.asarray(payload["q"])
        with final_csv.open("w", newline="") as handle:
            fields = ["sample_id", "g_scale", "g_bias", "hard_env", "sample_mse", "sample_mae"]
            fields.extend(f"q_env_{index}" for index in range(q.shape[1]))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(len(q)):
                row = {
                    "sample_id": int(payload["sample_id"][index]),
                    "g_scale": float(payload["g_scale"][index]),
                    "g_bias": float(payload["g_bias"][index]),
                    "hard_env": int(payload["hard_env"][index]),
                    "sample_mse": float(payload["sample_mse"][index]),
                    "sample_mae": float(payload["sample_mae"][index]),
                }
                row.update({f"q_env_{env}": float(q[index, env]) for env in range(q.shape[1])})
                writer.writerow(row)

        var_acc = float(model_metrics.get("var_acc", float("nan")))
        inv_acc = float(model_metrics.get("inv_acc", float("nan")))
        summary = {
            "dataset": self.dataset,
            **{key: final.get(key, float("nan")) for key in SCALAR_STAGE_FIELDS if key != "stage"},
            "final_stage_ARI": final.get("stage_ARI_to_previous", float("nan")),
            "final_stage_NMI": final.get("stage_NMI_to_previous", float("nan")),
            "var_acc": var_acc,
            "inv_acc": inv_acc,
            "env_acc_gap_var_minus_inv": var_acc - inv_acc,
            "train_only": True,
            "random_partition_repeats": final.get("random_partition_repeats"),
            "environment_mass": final.get("environment_mass", []),
            "per_env_MSE": final.get("per_env_MSE", []),
            "per_env_MAE": final.get("per_env_MAE", []),
            "env_gradient_scale": final.get("env_gradient_scale", []),
            "env_gradient_bias": final.get("env_gradient_bias", []),
        }
        (self.output_dir / "environment_quality_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        (self.output_dir / "environment_quality_summary.txt").write_text(
            "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n"
        )
        return summary
