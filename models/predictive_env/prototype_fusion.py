"""TRAIN-only prototype extraction and zero-start forecast fusion.

P2 is a prediction-aware *metric* fitted from allowed source residuals. It is
an FPEM adaptation, not a reproduction of a retrieval paper. Querying never
uses a future target. P3 performs a coarse-to-fine historical clustering.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans

from models.predictive_env.predictive_memory import PredictiveMemory, history_keys


class PrototypeFusionMemory(PredictiveMemory):
    def __init__(self, k, history_size, pred_len, d_model, env_num,
                 prototype_mode="P0", temperature=.2, topk=3):
        super().__init__(k, history_size, pred_len, d_model, env_num, temperature, topk)
        self.prototype_mode = prototype_mode
        self.register_buffer("key_mean", torch.zeros(history_size))
        self.register_buffer("key_transform", torch.eye(history_size))
        self.register_buffer("coarse_keys", torch.zeros(2, history_size))
        self.register_buffer("fine_group", torch.zeros(k, dtype=torch.long))

    def encode_keys(self, x):
        raw = history_keys(x.float(), self.history_size)
        if self.prototype_mode == "P2":
            return (raw - self.key_mean.to(raw)) @ self.key_transform.to(raw)
        return raw

    @torch.no_grad()
    def fit_key_space(self, x, response, seed=2021):
        """Both arguments must be TRAIN source anchors; P2 sees source labels only."""
        raw = history_keys(x.float(), self.history_size).reshape(-1, self.history_size).cpu()
        if len(raw) < self.k:
            raise ValueError("too few allowed TRAIN anchor-channel pairs")
        if self.prototype_mode == "P2":
            target = response.float().reshape(-1, self.pred_len).cpu()
            xc = raw - raw.mean(0)
            yc = target - target.mean(0)
            cross = xc.T @ yc / max(1, len(raw))
            u, singular, _ = torch.linalg.svd(cross, full_matrices=False)
            rank = min(8, len(singular), self.history_size)
            transform = torch.zeros_like(self.key_transform)
            transform[:, :rank] = u[:, :rank] * singular[:rank].clamp_min(1e-6).sqrt()
            self.key_mean.copy_(raw.mean(0)); self.key_transform.copy_(transform)
        keys = self.encode_keys(x).reshape(-1, self.history_size).cpu().numpy()
        if self.prototype_mode != "P3":
            km = MiniBatchKMeans(n_clusters=self.k, random_state=seed,
                                 batch_size=min(256, len(keys)), n_init=3).fit(keys)
            self.keys.copy_(torch.from_numpy(km.cluster_centers_).to(self.keys))
            return
        coarse = MiniBatchKMeans(n_clusters=2, random_state=seed,
                                 batch_size=min(256, len(keys)), n_init=3).fit(keys)
        self.coarse_keys.copy_(torch.from_numpy(coarse.cluster_centers_).to(self.coarse_keys))
        per = self.k // 2
        if self.k % 2 or per < 2:
            raise ValueError("P3 requires an even K of at least 4")
        centers = []
        groups = []
        for group in range(2):
            rows = keys[coarse.labels_ == group]
            if len(rows) < per:
                raise ValueError("insufficient TRAIN anchors in hierarchical group")
            fine = MiniBatchKMeans(n_clusters=per, random_state=seed + group + 1,
                                   batch_size=min(256, len(rows)), n_init=3).fit(rows)
            centers.extend(fine.cluster_centers_)
            groups.extend([group] * per)
        self.keys.copy_(torch.as_tensor(np.asarray(centers), dtype=self.keys.dtype))
        self.fine_group.copy_(torch.as_tensor(groups, dtype=torch.long))

    @torch.no_grad()
    def fit_values(self, x, hidden, response, q, starts, source_mask=None,
                   forbidden_interval=None):
        # Base fit_values uses global history_keys. Supply encoded keys through
        # a narrow, per-instance override rather than mutate module globals.
        # The implementation below mirrors the base sufficient statistics.
        n, length, channels = x.shape
        mask = torch.ones(n, dtype=torch.bool) if source_mask is None else source_mask.cpu().bool().clone()
        if forbidden_interval is not None:
            lo, hi = forbidden_interval
            ends = starts.cpu().long() + length + self.pred_len
            mask &= (ends <= lo) | (starts.cpu().long() >= hi)
        if int(mask.sum()) < max(self.k, 2):
            raise ValueError("purged memory has insufficient source anchors")
        encoded = self.encode_keys(x[mask]).cpu()
        dist = (encoded[..., None, :] - self.keys.cpu()[None, None]).square().mean(-1)
        if self.prototype_mode == "P3":
            coarse_dist = (encoded[..., None, :] - self.coarse_keys.cpu()[None, None]).square().mean(-1)
            group = coarse_dist.argmin(-1)
            dist = dist.masked_fill(self.fine_group.cpu()[None, None] != group[..., None], float("inf"))
        label = dist.argmin(-1).reshape(-1)
        h = hidden[mask].float().mean(2).reshape(-1, hidden.shape[-1]).cpu()
        r = response[mask].float().reshape(-1, self.pred_len).cpu()
        aq = q if len(q) == n else q[starts]
        qq = aq[mask].float()[:, None].expand(-1, channels, -1).reshape(-1, self.env_num).cpu()
        block = (starts[mask].long() // (length + self.pred_len))[:, None]
        independent = (block * channels + torch.arange(channels)[None]).reshape(-1)
        for j in range(self.k):
            selected = label == j
            self.support[j] = torch.unique(independent[selected]).numel()
            if not bool(selected.any()):
                self.hidden_value[j].zero_(); self.response[j].zero_(); self.env_delta[j].zero_()
                self.env_support[j].zero_(); self.between[j].zero_(); self.within[j].zero_()
                continue
            hh, rr, qe = h[selected], r[selected], qq[selected]
            shared = rr.mean(0); mass = qe.sum(0)
            centers = qe.T @ rr / mass[:, None].clamp_min(1e-6)
            delta = centers - shared[None]
            self.hidden_value[j].copy_(hh.mean(0)); self.response[j].copy_(shared)
            self.env_delta[j].copy_(delta); self.env_support[j].copy_(mass)
            self.between[j] = ((mass / mass.sum().clamp_min(1e-6))[:, None] * delta.square()).sum(-1).mean()
            self.within[j] = (qe[:, :, None] * (rr[:, None] - centers[None]).square()).sum() / (mass.sum().clamp_min(1e-6) * self.pred_len)
        self.fitted.fill_(True)
        return {"source_anchors": int(mask.sum()), "forbidden_interval": forbidden_interval,
                "support": self.support.tolist(), "prototype_mode": self.prototype_mode}

    def query(self, x, qhat=None, use_mapping=True, shuffle_mapping=False):
        if not bool(self.fitted):
            raise RuntimeError("memory values not fitted")
        u = self.encode_keys(x)
        distance = (u[..., None, :] - self.keys.to(u)[None, None]).square().mean(-1)
        if self.prototype_mode == "P3":
            coarse = (u[..., None, :] - self.coarse_keys.to(u)[None, None]).square().mean(-1)
            group = coarse.argmin(-1)
            distance = distance.masked_fill(self.fine_group.to(u.device)[None, None] != group[..., None], 1e6)
        nearest, index = torch.topk(distance, self.topk, dim=-1, largest=False)
        weight = torch.zeros_like(distance).scatter_(-1, index, torch.softmax(-nearest / self.temperature, -1))
        count = self.support.to(weight)
        coverage = (self.env_support.to(weight) > 1).float().mean(-1)
        stable = count / (count + 16.) * coverage * torch.exp(-self.between.to(weight) / (self.within.to(weight) + 1e-4))
        variant = count / (count + 16.) * coverage * self.between.to(weight) / (self.between.to(weight) + self.within.to(weight) + 1e-4)
        if qhat is None:
            qhat = x.new_full((len(x), self.env_num), 1. / self.env_num)
        q_conf = ((qhat.max(-1).values - 1. / self.env_num) / (1. - 1. / self.env_num)).clamp(0, 1)
        shared = self.response.to(weight)
        env = self.env_delta.to(weight)
        if shuffle_mapping:
            shared = shared.flip(0); env = env.flip(0)
        return {"weight": weight, "keys": self.keys.to(weight),
                "hidden_values": self.hidden_value.to(weight),
                "proto_hidden": torch.einsum("bck,kd->bcd", weight, self.hidden_value.to(weight)),
                "stable_response": torch.einsum("bck,kh->bch", weight, shared),
                "env_response": torch.einsum("bck,be,keh->bch", weight, qhat, env) if use_mapping else torch.zeros_like(torch.einsum("bck,kh->bch", weight, shared)),
                "c_inv": torch.einsum("bck,k->bc", weight, stable),
                "c_var": torch.einsum("bck,k->bc", weight, variant) * q_conf[:, None],
                "distance": distance, "entropy": -(weight * weight.clamp_min(1e-8).log()).sum(-1),
                "min_distance": distance.min(-1).values}


class PrototypeFusion(nn.Module):
    """F1/F2 change H; F3 changes only the shared A0 linear operator."""
    def __init__(self, d_model, token_num, pred_len, history_size, env_num,
                 fusion="F1", mapping="M1", rank=4):
        super().__init__()
        self.fusion, self.mapping = fusion, mapping
        self.env_predictor = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, min(d_model, 64)),
                                           nn.GELU(), nn.Linear(min(d_model, 64), env_num))
        self.response_summary = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, d_model), nn.GELU())
        if fusion == "F1":
            self.correct = nn.Sequential(nn.LayerNorm(3*d_model), nn.Linear(3*d_model, d_model),
                                         nn.GELU(), nn.Linear(d_model, d_model))
            nn.init.zeros_(self.correct[-1].weight); nn.init.zeros_(self.correct[-1].bias)
        elif fusion == "F2":
            self.query_proj = nn.Linear(d_model, history_size)
            self.key_proj = nn.Linear(history_size, history_size)
            self.value_proj = nn.Linear(d_model, d_model)
            self.correct = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.correct.weight); nn.init.zeros_(self.correct.bias)
        elif fusion == "F3":
            self.coeff = nn.Sequential(nn.LayerNorm(2*d_model), nn.Linear(2*d_model, d_model),
                                       nn.GELU(), nn.Linear(d_model, rank))
            self.low_a = nn.Parameter(torch.randn(d_model*token_num, rank) * .01)
            self.low_b = nn.Parameter(torch.zeros(rank, pred_len))
        else:
            raise ValueError(fusion)

    def predict_environment(self, hidden):
        return self.env_predictor(hidden.mean((1, 2))).softmax(-1)

    def forward(self, model, hidden, mean, std, cycle, retrieval):
        response = retrieval["stable_response"]
        if self.mapping == "M0":
            response = torch.zeros_like(response)
        elif self.mapping == "M2":
            response = response + retrieval["env_response"]
        features = torch.stack((response.mean(-1), response.std(-1, unbiased=False),
                                response[..., -1]-response[..., 0], retrieval["c_inv"]), -1)
        context = retrieval["proto_hidden"] + self.response_summary(features)
        if self.fusion == "F1":
            ctx = context[:, :, None, :].expand_as(hidden)
            delta = self.correct(torch.cat((hidden, ctx, ctx-hidden), -1))
            normalized = model.frozen_h_head(hidden + delta, cycle)
        elif self.fusion == "F2":
            # Jointly attend over all TRAIN historical keys; response values
            # remain represented by the retrieval-conditioned context.
            q = self.query_proj(hidden.mean(2))
            keys = self.key_proj(retrieval["keys"])
            attention = torch.softmax(torch.einsum("bcr,kr->bck", q, keys) / math.sqrt(keys.shape[-1]), -1)
            value = self.value_proj(retrieval["hidden_values"])
            selected = torch.einsum("bck,kd->bcd", attention, value)
            delta = self.correct(selected + context)[:, :, None, :].expand_as(hidden)
            normalized = model.frozen_h_head(hidden + delta, cycle)
        else:
            flat = model.head_flatten(hidden.permute(0, 1, 3, 2))
            coeff = self.coeff(torch.cat((hidden.mean(2), context), -1))
            delta_y = ((flat @ self.low_a) * coeff) @ self.low_b
            normalized = model.frozen_h_head(hidden, cycle) + delta_y
            delta = delta_y
        y = normalized.permute(0, 2, 1) * std + mean
        return y, {"correction_rms": delta.square().mean().sqrt(),
                   "retrieval_entropy": retrieval["entropy"].mean(),
                   "retrieval_distance": retrieval["min_distance"].mean()}
