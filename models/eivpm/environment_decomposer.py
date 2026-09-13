"""FOIL-compatible environment-label loading for TRAIN samples only."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class FoilEnvironmentLabels:
    labels: np.ndarray
    soft: np.ndarray
    path: str
    source: str


class FoilEnvironmentProvider:
    """Load FOIL labels, or materialize the upstream scale/shift partition.

    FOIL's ``model_env_scale.py`` assigns TRAIN windows according to the
    environment-specific scale/shift correction that best explains them.  For
    datasets without a pre-exported label file we retain exactly that semantic:
    cluster TRAIN-only history-to-future scale/shift descriptors.  No validation
    or test window is accepted by this class.
    """

    def __init__(self, env_num: int = 6, seed: int = 2021):
        self.env_num = int(env_num)
        self.seed = int(seed)

    @staticmethod
    def _features(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        # This is the exact six-feature history descriptor used by the repo's
        # existing fpem_ts/foil_shift.py export path.  Future y is intentionally
        # ignored, so validation/test environment lookup cannot leak targets.
        time = np.linspace(-1.0, 1.0, x.shape[1], dtype=np.float32)[None, :, None]
        mean_c, std_c = x.mean(1), x.std(1)
        trend_c = ((x - mean_c[:, None]) * time).mean(1)
        return np.stack([
            mean_c.mean(1),
            std_c.mean(1),
            x[:, -1].mean(1),
            trend_c.mean(1),
            mean_c.std(1),
            std_c.std(1),
        ], axis=1).astype("float32")

    def load_or_create(self, path: str, train_x: np.ndarray, train_y: np.ndarray) -> FoilEnvironmentLabels:
        if os.path.isfile(path):
            data = np.load(path, allow_pickle=False)
            labels = data["train_labels"].astype("int64")
            soft = data["train_soft"].astype("float32")
            if len(labels) < len(train_x):
                raise ValueError("FOIL TRAIN labels do not cover every TRAIN forecasting window")
            # foil_shift exports every history window; forecasting additionally
            # requires pred_len future points, so its valid forecasting prefix is
            # intentionally shorter.
            labels, soft = labels[:len(train_x)], soft[:len(train_x)]
            return FoilEnvironmentLabels(labels, soft, path, "precomputed FOIL asset")

        features = self._features(train_x, train_y)
        mean, std = features.mean(0), features.std(0).clip(1e-6)
        normalized = (features - mean) / std
        # Same KMeans and distance-softmax flow as fpem_ts/foil_shift.py.
        partition = KMeans(n_clusters=self.env_num, random_state=self.seed, n_init=20).fit(normalized)
        distance = ((normalized[:, None] - partition.cluster_centers_[None]) ** 2).sum(-1)
        labels = distance.argmin(1).astype("int64")
        soft = np.exp(-distance / (distance.std() + 1e-6))
        soft = (soft / soft.sum(1, keepdims=True)).astype("float32")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            train_labels=labels,
            train_soft=soft,
            train_features=features,
            centers=partition.cluster_centers_.astype("float32"),
            feature_mean=mean.astype("float32"),
            feature_std=std.astype("float32"),
        )
        metadata = {
            "source": "FOIL_upstream/Informer+FOIL/Informer2020/models/model_env_scale.py",
            "scope": "TRAIN-only",
            "semantics": "repo foil_shift six-feature history KMeans partition",
            "env_num": self.env_num,
            "num_windows": int(len(labels)),
            "counts": np.bincount(labels, minlength=self.env_num).tolist(),
        }
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return FoilEnvironmentLabels(labels, soft, path, metadata["source"])

    def assign_inputs(self, path: str, x: np.ndarray) -> FoilEnvironmentLabels:
        data = np.load(path, allow_pickle=False)
        features = self._features(x)
        if "feature_mean" in data.files:
            mean, std = data["feature_mean"], data["feature_std"]
        elif "train_features" in data.files:
            mean = data["train_features"].mean(0)
            std = data["train_features"].std(0).clip(1e-6)
        else:
            raise ValueError("FOIL asset lacks scaler or TRAIN features")
        normalized = (features - mean) / std
        centers = data["centers"]
        distance = ((normalized[:, None] - centers[None]) ** 2).sum(-1)
        logits = -distance / (distance.std() + 1e-6)
        logits -= logits.max(1, keepdims=True)
        soft = np.exp(logits)
        soft /= soft.sum(1, keepdims=True)
        labels = soft.argmax(1).astype("int64")
        return FoilEnvironmentLabels(labels, soft.astype("float32"), path, "FOIL history lookup")
