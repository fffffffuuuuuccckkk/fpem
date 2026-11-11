"""Read-only Stage-1 mechanism labels and common low-rank full-future experts.

No method-specific operator from Stage 1 is required. All group methods feed
the same downstream learner, router and prediction-space fusion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


K = 8


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenGrouping:
    def __init__(self, directory, *, a0_sha, dataset, horizon, seed=2022):
        directory = Path(directory)
        self.path = directory / "assignments_train.npz"
        self.meta_path = directory / "prototype_meta.json"
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        expected = {"A0_sha256": a0_sha, "dataset": dataset,
                    "pred_len": int(horizon), "seed": int(seed), "K": K}
        for name, value in expected.items():
            if meta.get(name) != value:
                raise ValueError(f"group artifact {name} mismatch: {meta.get(name)} != {value}")
        with np.load(self.path) as source:
            sample_id = source["sample_id"].astype(str)
            q = source["q_teacher"].astype(np.float32)
            block = source["time_block"].astype(str)
            starts = source["window_start"].astype(np.int64)
            channels = source["channel"].astype(np.int16)
        if q.ndim != 2 or q.shape[1] != K or len(q) != len(sample_id):
            raise ValueError("invalid K=8 soft assignment shape")
        if not np.isfinite(q).all() or np.min(q) < -1e-6 or not np.allclose(q.sum(1), 1, atol=1e-4):
            raise ValueError("invalid soft probabilities")
        if len(set(sample_id)) != len(sample_id):
            raise ValueError("duplicate group sample_id")
        if not set(block).issubset({"A", "B1", "B2"}):
            raise ValueError("teacher labels must be TRAIN A/B1/B2 only")
        self.meta = meta
        self.q, self.block, self.sample_id = q, block, sample_id
        self.starts, self.channels = starts, channels
        self.hash = sha256(self.path)
        self.lookup = {key: i for i, key in enumerate(sample_id)}

    def join(self, dataset, starts, channels):
        identifiers = [f"{dataset}:TRAIN:{int(s)}:{int(c)}" for s, c in zip(starts, channels)]
        positions = np.array([self.lookup.get(key, -1) for key in identifiers], dtype=np.int64)
        if np.sum(positions >= 0) != len(self.lookup):
            raise ValueError("cached TRAIN rows do not cover every Stage-1 teacher label")
        output = {}
        for block in ("A", "B1", "B2"):
            select = (positions >= 0) & (self.block[np.maximum(positions, 0)] == block)
            output[block] = (np.flatnonzero(select), self.q[positions[select]])
        return output


def fit_full_future_experts(x, y, q, *, alpha=100.0, rank=8):
    """Shared Ridge plus per-group low-rank residual operators, all on TRAIN A."""
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    q = np.asarray(q, np.float32)
    if x.ndim != 2 or y.ndim != 2 or q.shape != (len(x), K):
        raise ValueError("expert input shapes")
    shared = Ridge(alpha=alpha).fit(x, y)
    residual = y - shared.predict(x)
    weights, biases, support = [], [], []
    for j in range(K):
        w = q[:, j]
        support.append(float(w.sum()))
        if w.sum() < 1e-6:
            weights.append(np.zeros((x.shape[1], y.shape[1]), np.float32))
            biases.append(np.zeros(y.shape[1], np.float32))
            continue
        local = Ridge(alpha=alpha / max(float(w.mean()), .02)).fit(x, residual, sample_weight=w)
        coefficient = local.coef_.T.astype(np.float64)
        u, s, vt = np.linalg.svd(coefficient, full_matrices=False)
        low_rank = (u[:, :rank] * s[:rank]) @ vt[:rank]
        weights.append(low_rank.astype(np.float32))
        biases.append(np.asarray(local.intercept_, np.float32))
    return {"shared_weight": shared.coef_.T.astype(np.float32),
            "shared_bias": np.asarray(shared.intercept_, np.float32),
            "delta_weight": np.stack(weights), "delta_bias": np.stack(biases),
            "support": support, "alpha": alpha, "rank": rank}


def candidates(experts, x):
    x = np.asarray(x, np.float32)
    shared = x @ experts["shared_weight"] + experts["shared_bias"]
    delta = np.einsum("nd,kdt->nkt", x, experts["delta_weight"])
    return shared[:, None, :] + delta + experts["delta_bias"][None, :, :]


def independent_support(starts, q, span):
    """Conservative event count: one dominant-group vote per full-window span."""
    starts = np.asarray(starts, np.int64)
    q = np.asarray(q)
    support = []
    for j in range(K):
        selected = np.unique(starts[q.argmax(1) == j])
        count, last_end = 0, -1
        for start in np.sort(selected):
            if start >= last_end:
                count += 1
                last_end = int(start) + span
        support.append(count)
    return np.array(support)
