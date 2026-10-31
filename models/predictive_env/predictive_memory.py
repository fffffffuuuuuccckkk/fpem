"""TRAIN-only history keys and purged, environment-conditioned prediction values.

Keys depend on observed history only.  For a TRAIN query, values are fitted
without *any* anchor whose complete history/future interval intersects its
chronological query block.  Validation/test use the full TRAIN value bank.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans


def history_keys(x: torch.Tensor, size: int = 64) -> torch.Tensor:
    """[B,L,C] -> [B,C,R]; never use the future's own statistics."""
    series = x.transpose(1, 2)
    mean = series.mean(-1, keepdim=True)
    std = series.var(-1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    normal = (series - mean) / std
    return F.interpolate(normal.reshape(-1, 1, x.shape[1]), size=size,
                         mode="linear", align_corners=False).reshape(
                             x.shape[0], x.shape[2], size)


class PredictiveMemory(nn.Module):
    """Fixed TRAIN history keys; stage-wise fitted response and hidden values."""

    def __init__(self, k: int, history_size: int, pred_len: int, d_model: int,
                 env_num: int, temperature: float = 0.2, topk: int = 3):
        super().__init__()
        self.k, self.history_size, self.pred_len = k, history_size, pred_len
        self.env_num, self.temperature, self.topk = env_num, temperature, min(topk, k)
        self.register_buffer("keys", torch.zeros(k, history_size))
        self.register_buffer("hidden_value", torch.zeros(k, d_model))
        self.register_buffer("response", torch.zeros(k, pred_len))
        self.register_buffer("env_delta", torch.zeros(k, env_num, pred_len))
        self.register_buffer("support", torch.zeros(k))
        self.register_buffer("env_support", torch.zeros(k, env_num))
        self.register_buffer("between", torch.zeros(k))
        self.register_buffer("within", torch.zeros(k))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_keys(self, x: torch.Tensor, seed: int = 2021) -> None:
        """x is TRAIN anchor history only, [A,L,C]."""
        rows = history_keys(x.float(), self.history_size).reshape(-1, self.history_size)
        if rows.shape[0] < self.k:
            raise ValueError("fewer TRAIN anchor-channel pairs than prototypes")
        km = MiniBatchKMeans(n_clusters=self.k, random_state=seed,
                             batch_size=min(256, len(rows)), n_init=3)
        km.fit(rows.cpu().numpy())
        self.keys.copy_(torch.from_numpy(km.cluster_centers_).to(self.keys))

    @torch.no_grad()
    def fit_values(self, x: torch.Tensor, hidden: torch.Tensor,
                   response: torch.Tensor, q: torch.Tensor, starts: torch.Tensor,
                   source_mask: torch.Tensor | None = None,
                   forbidden_interval: tuple[int, int] | None = None) -> dict:
        """Fit *all* response statistics only from allowed TRAIN anchors.

        starts are original TRAIN sliding-window start indices.  The forbidden
        interval is the union of every complete query interval in a fold.
        """
        n, _, c = x.shape
        if source_mask is None:
            source_mask = torch.ones(n, dtype=torch.bool)
        source_mask = source_mask.cpu().bool()
        if forbidden_interval is not None:
            lo, hi = forbidden_interval
            complete_end = starts.cpu().long() + x.shape[1] + self.pred_len
            overlap = (starts.cpu().long() < hi) & (complete_end > lo)
            source_mask &= ~overlap
        if int(source_mask.sum()) < max(self.k, 2):
            raise ValueError("purged memory has insufficient independent TRAIN anchors")
        keys = history_keys(x[source_mask].float(), self.history_size)
        d = (keys[..., None, :] - self.keys[None, None]).square().mean(-1)
        labels = d.argmin(-1).reshape(-1)
        h = hidden[source_mask].float().mean(2).reshape(-1, hidden.shape[-1])
        r = response[source_mask].float().reshape(-1, self.pred_len)
        # q may be indexed by all stride-1 TRAIN samples or supplied directly
        # for these independent anchors by fold-local EIIL inference.
        anchor_q = q if len(q) == n else q[starts]
        qq = anchor_q[source_mask].float()[:, None].expand(-1, c, -1).reshape(-1, self.env_num)
        # Sliding windows inside the same complete-window temporal block do
        # not count as independent recurrence support for one channel.
        selected_starts = starts[source_mask].long()
        block = (selected_starts // (x.shape[1] + self.pred_len))[:, None]
        channel = torch.arange(c)[None]
        independent_id = (block * c + channel).reshape(-1)
        counts = torch.bincount(labels, minlength=self.k).float()
        independent = torch.tensor([
            torch.unique(independent_id[labels == j]).numel()
            for j in range(self.k)]).float()
        self.support.copy_(independent.to(self.support))
        for j in range(self.k):
            mask = labels == j
            if not bool(mask.any()):
                self.hidden_value[j].zero_(); self.response[j].zero_()
                self.env_delta[j].zero_(); self.env_support[j].zero_()
                self.between[j].zero_(); self.within[j].zero_()
                continue
            hh, rr, qe = h[mask], r[mask], qq[mask]
            self.hidden_value[j].copy_(hh.mean(0).to(self.hidden_value))
            shared = rr.mean(0)
            self.response[j].copy_(shared.to(self.response))
            mass = qe.sum(0)
            self.env_support[j].copy_(mass.to(self.env_support))
            centers = qe.T @ rr / mass[:, None].clamp_min(1e-6)
            delta = centers - shared[None]
            self.env_delta[j].copy_(delta.to(self.env_delta))
            self.between[j] = ((mass / mass.sum().clamp_min(1e-6))[:, None]
                               * delta.square()).sum(-1).mean().to(self.between)
            self.within[j] = (qe[:, :, None] *
                              (rr[:, None] - centers[None]).square()).sum() / (
                                  mass.sum().clamp_min(1e-6) * self.pred_len)
        self.fitted.fill_(True)
        return {"source_anchors": int(source_mask.sum()),
                "raw_window_channel_counts": counts.tolist(),
                "source_start_min": int(starts[source_mask].min()),
                "source_end_max": int((starts[source_mask] + x.shape[1] + self.pred_len).max()),
                "forbidden_interval": forbidden_interval,
                "support": self.support.cpu().tolist(),
                "env_support": self.env_support.cpu().tolist(),
                "between": self.between.cpu().tolist(),
                "within": self.within.cpu().tolist()}

    def query(self, x: torch.Tensor, qhat: torch.Tensor | None = None,
              use_mapping: bool = True, shuffle_mapping: bool = False) -> dict:
        if not bool(self.fitted):
            raise RuntimeError("memory values must be fitted from TRAIN first")
        u = history_keys(x.float(), self.history_size)
        distance = (u[..., None, :] - self.keys[None, None]).square().mean(-1)
        nearest, index = torch.topk(distance, self.topk, dim=-1, largest=False)
        local = torch.softmax(-nearest / self.temperature, -1)
        weight = torch.zeros_like(distance).scatter_(-1, index, local)
        # No response statistic can become trusted solely due to a close key.
        count = self.support.to(weight).clamp_min(0)
        coverage = (self.env_support.to(weight) > 1).float().mean(-1)
        stable = count / (count + 16.) * coverage * torch.exp(
            -self.between.to(weight) / (self.within.to(weight) + 1e-4))
        variant = count / (count + 16.) * coverage * (
            self.between.to(weight) / (self.between.to(weight) +
                                       self.within.to(weight) + 1e-4))
        key_index = torch.arange(self.k - 1, -1, -1, device=weight.device)
        response = self.response.to(weight)[key_index] if shuffle_mapping else self.response.to(weight)
        delta = self.env_delta.to(weight)[key_index] if shuffle_mapping else self.env_delta.to(weight)
        c_inv = torch.einsum("bck,k->bc", weight, stable)
        c_var = torch.einsum("bck,k->bc", weight, variant)
        if qhat is None:
            qhat = x.new_full((x.shape[0], self.env_num), 1. / self.env_num)
        q_conf = (qhat.max(-1).values - 1. / self.env_num) / max(1e-6, 1 - 1. / self.env_num)
        c_var = c_var * q_conf[:, None].clamp(0, 1)
        env_response = torch.einsum("bck,be,keh->bch", weight, qhat, delta)
        return {"weight": weight, "distance": distance,
                "proto_hidden": torch.einsum("bck,kd->bcd", weight, self.hidden_value.to(weight)),
                "stable_response": torch.einsum("bck,kh->bch", weight, response),
                "env_response": env_response if use_mapping else torch.zeros_like(env_response),
                "c_inv": c_inv, "c_var": c_var if use_mapping else torch.zeros_like(c_var),
                "q_conf": q_conf,
                "entropy": -(weight * weight.clamp_min(1e-8).log()).sum(-1),
                "min_distance": distance.min(-1).values}
