"""TRAIN-only, per-variable historical/future pattern prototypes.

The bank never sees a query's future target. Only ``fit_train`` and
``update_train_epoch`` may read future labels, and both reject non-TRAIN data.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


class PrototypeBank(nn.Module):
    def __init__(self, pred_len: int, k: int = 8, history_size: int = 64,
                 temperature: float = 0.2, beta: float = 0.99, eps: float = 1e-5):
        super().__init__()
        if k < 1 or history_size < 2 or pred_len < 1 or temperature <= 0:
            raise ValueError("invalid prototype-bank dimensions or temperature")
        self.pred_len, self.k, self.history_size = int(pred_len), int(k), int(history_size)
        self.temperature, self.beta, self.eps = float(temperature), float(beta), float(eps)
        self.register_buffer("U", torch.zeros(k, history_size))
        self.register_buffer("V", torch.zeros(k, pred_len))
        self.register_buffer("counts", torch.zeros(k))
        self.register_buffer("fitted", torch.tensor(False))

    def _pairs_from_train(self, dataset):
        # Dataset_ETT_{hour,minute}/Custom only. The final future point of each
        # anchor must be inside the TRAIN split's data_y array.
        if int(getattr(dataset, "set_type", -1)) != 0:
            raise ValueError("prototype fitting/EMA requires a TRAIN dataset")
        if int(dataset.pred_len) != self.pred_len:
            raise ValueError("bank pred_len differs from TRAIN dataset")
        x_all = np.asarray(dataset.data_x, dtype=np.float32)
        y_all = np.asarray(dataset.data_y, dtype=np.float32)
        seq_len = int(dataset.seq_len)
        starts = np.arange(0, len(dataset), seq_len, dtype=np.int64)
        if len(starts) == 0:
            raise ValueError("no complete TRAIN historical/future windows")
        past = np.stack([x_all[s:s + seq_len].T for s in starts])
        future = np.stack([y_all[s + seq_len:s + seq_len + self.pred_len].T
                           for s in starts])
        if future.shape[-1] != self.pred_len:
            raise AssertionError("TRAIN future window crossed split boundary")
        mean = past.mean(-1, keepdims=True)
        std = np.sqrt(((past - mean) ** 2).mean(-1, keepdims=True) + self.eps)
        historical = (past - mean) / std
        # Crucially: future uses the historical mean/std, not its own moments.
        target = (future - mean) / std
        tensor = torch.from_numpy(historical.reshape(-1, seq_len))[:, None]
        u = F.interpolate(tensor, size=self.history_size, mode="linear",
                          align_corners=False)[:, 0].numpy()
        return u.astype(np.float32), target.reshape(-1, self.pred_len).astype(np.float32), len(starts), past.shape[1]

    @torch.no_grad()
    def fit_train(self, dataset, seed: int = 2021):
        u, v, anchors, channels = self._pairs_from_train(dataset)
        if len(u) < self.k:
            raise ValueError(f"need at least {self.k} TRAIN anchor/channel pairs")
        km = MiniBatchKMeans(n_clusters=self.k, random_state=int(seed),
                             batch_size=min(256, len(u)), n_init=3,
                             reassignment_ratio=0.01)
        km.fit(u)
        centers = km.cluster_centers_.astype(np.float32)
        labels = ((u[:, None] - centers[None]) ** 2).mean(-1).argmin(-1)
        counts = np.bincount(labels, minlength=self.k).astype(np.float32)
        # Reinitialize any empty center from the most poorly represented TRAIN
        # pair; no validation or test observation can affect this decision.
        for cluster in np.where(counts == 0)[0]:
            distance = ((u - centers[labels]) ** 2).mean(-1)
            candidate = int(distance.argmax())
            centers[cluster] = u[candidate]
            labels[candidate] = cluster
            counts = np.bincount(labels, minlength=self.k).astype(np.float32)
        future_centers = np.stack([v[labels == j].mean(0) for j in range(self.k)])
        self.U.copy_(torch.from_numpy(centers).to(self.U))
        self.V.copy_(torch.from_numpy(future_centers).to(self.V))
        self.counts.copy_(torch.from_numpy(counts).to(self.counts))
        self.fitted.fill_(True)
        return {"train_anchor_groups": anchors, "train_channels": channels,
                "train_anchor_channel_pairs": int(len(u)),
                "initial_cluster_counts": counts.astype(int).tolist()}

    @torch.no_grad()
    def update_train_epoch(self, dataset):
        if not bool(self.fitted):
            raise RuntimeError("fit TRAIN prototypes first")
        u, v, anchors, channels = self._pairs_from_train(dataset)
        old_u = self.U.detach().cpu().numpy()
        labels = ((u[:, None] - old_u[None]) ** 2).mean(-1).argmin(-1)
        counts = np.bincount(labels, minlength=self.k).astype(np.float32)
        for cluster in range(self.k):
            if counts[cluster]:
                self.U[cluster].lerp_(torch.from_numpy(u[labels == cluster].mean(0)).to(self.U), 1-self.beta)
                self.V[cluster].lerp_(torch.from_numpy(v[labels == cluster].mean(0)).to(self.V), 1-self.beta)
        self.counts.copy_(torch.from_numpy(counts).to(self.counts))
        return {"epoch_cluster_counts": counts.astype(int).tolist(),
                "train_anchor_groups": anchors, "train_channels": channels}

    def query(self, x: torch.Tensor):
        if not bool(self.fitted):
            raise RuntimeError("prototype bank has not been fitted on TRAIN")
        if x.ndim != 3:
            raise ValueError("expected history [B,L,C]")
        past = x.transpose(1, 2)
        mean = past.mean(-1, keepdim=True)
        std = ((past - mean).square().mean(-1, keepdim=True) + self.eps).sqrt()
        normalized = (past - mean) / std
        u = F.interpolate(normalized.reshape(-1, 1, x.shape[1]),
                          size=self.history_size, mode="linear",
                          align_corners=False).reshape(x.shape[0], x.shape[2], self.history_size)
        distance = (u.unsqueeze(-2) - self.U.to(u.dtype)).square().mean(-1)
        weight = torch.softmax(-distance / self.temperature, dim=-1)
        matched_u = torch.einsum("bck,kr->bcr", weight, self.U.to(weight.dtype))
        matched_v = torch.einsum("bck,kh->bch", weight, self.V.to(weight.dtype))
        historical_proto = F.interpolate(matched_u.reshape(-1, 1, self.history_size),
                                         size=x.shape[1], mode="linear",
                                         align_corners=False).reshape_as(past)
        x_proto = (historical_proto * std + mean).transpose(1, 2)
        y_proto = (matched_v * std + mean).transpose(1, 2)
        entropy = -(weight * weight.clamp_min(1e-8).log()).sum(-1)
        residual_energy = (x - x_proto).square().mean(1).sqrt()
        trend = (matched_v[..., -1] - matched_v[..., 0]).abs()
        roughness = (matched_v[..., 1:] - matched_v[..., :-1]).abs().mean(-1) if self.pred_len > 1 else trend*0
        features = torch.stack((distance.min(-1).values, entropy, residual_energy,
                                trend, roughness), dim=-1)
        return {"x_proto": x_proto, "y_proto": y_proto,
                "weight": weight, "distance": distance,
                "features": features,
                "max_weight": weight.max(-1).values, "entropy": entropy,
                "min_distance": distance.min(-1).values,
                "residual_energy": residual_energy}
