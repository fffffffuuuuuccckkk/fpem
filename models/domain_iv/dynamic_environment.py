"""Epoch-wise TRAIN-only dynamic environment discovery."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def _safe_corr(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.size < 2 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


class DynamicEnvironmentDiscovery:
    """Update soft q from current Z_var once per epoch.

    Center matching is used only to make EMA and q-change diagnostics meaningful.
    Domain losses consume q @ q.T and are therefore permutation invariant.
    """
    def __init__(self, env_num=6, temperature=.2, ema_beta=.9, seed=2021, diagnostic_samples=1024):
        self.env_num = int(env_num)
        self.temperature = float(temperature)
        self.ema_beta = float(ema_beta)
        self.seed = int(seed)
        self.diagnostic_samples = int(diagnostic_samples)
        self.centers = None
        self.previous_q = None
        self.epoch = 0

    @torch.no_grad()
    def assignment(self, representations):
        rep = F.normalize(torch.as_tensor(representations, dtype=torch.float32), dim=-1)
        centers = F.normalize(torch.as_tensor(self.centers, dtype=torch.float32), dim=-1)
        return ((rep @ centers.t()) / self.temperature).softmax(-1).cpu().numpy()

    @torch.no_grad()
    def update(self, representations, source_split="train"):
        if source_split.lower() != "train":
            raise ValueError("dynamic environments may only be updated from TRAIN")
        rep = F.normalize(torch.as_tensor(representations, dtype=torch.float32), dim=-1).cpu().numpy()
        fitted = KMeans(n_clusters=self.env_num, random_state=self.seed, n_init=10).fit(rep)
        candidate = fitted.cluster_centers_.astype("float32")
        candidate /= np.maximum(np.linalg.norm(candidate, axis=1, keepdims=True), 1e-8)
        if self.centers is not None:
            similarity = self.centers @ candidate.T
            old_ids, new_ids = linear_sum_assignment(-similarity)
            reordered = np.empty_like(candidate)
            reordered[old_ids] = candidate[new_ids]
            candidate = reordered
            centers = self.ema_beta * self.centers + (1.0 - self.ema_beta) * candidate
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
        else:
            centers = candidate
        self.centers = centers.astype("float32")
        q = self.assignment(rep)
        if self.previous_q is None:
            q_change, similarity_correlation = float("nan"), float("nan")
        else:
            q_change = float(np.linalg.norm(q - self.previous_q, axis=1).mean())
            n = min(len(q), self.diagnostic_samples)
            ids = np.linspace(0, len(q) - 1, n).astype("int64")
            old_similarity = self.previous_q[ids] @ self.previous_q[ids].T
            new_similarity = q[ids] @ q[ids].T
            mask = ~np.eye(n, dtype=bool)
            similarity_correlation = _safe_corr(old_similarity[mask], new_similarity[mask])
        self.previous_q = q.copy()
        record = {
            "epoch": self.epoch,
            "mean_q_change": q_change,
            "environment_similarity_epoch_correlation": similarity_correlation,
            "source_split": "TRAIN only",
            "assignment_entropy": float(-(q * np.log(np.maximum(q, 1e-8))).sum(1).mean()),
        }
        self.epoch += 1
        return q, record
