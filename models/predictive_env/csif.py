"""Common--specific factors and prediction-preserving H-space fusion.

The factor memory observes TRAIN history only.  ``observe_train_batch`` must
be called exactly once after each optimizer step, and ``finish_train_epoch``
counts every non-overlapping temporal anchor at most once.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CommonSpecificFactors(nn.Module):
    def __init__(self, width: int, mode: str = "memory", reflections: int = 4,
                 activation_threshold: float = 1.0, popularity_threshold: float = 0.5,
                 temperature: float = 0.1, beta: float = 0.99):
        super().__init__()
        if mode not in ("dictionary", "memory") or reflections < 2 or reflections % 2:
            raise ValueError("mode must be dictionary/memory; reflections must be positive even")
        self.width, self.mode = width, mode
        self.activation_threshold = activation_threshold
        self.popularity_threshold = popularity_threshold
        self.temperature, self.beta = temperature, beta
        # Identical reflection pairs multiply to identity; unlike a zero gate,
        # each independently trainable vector still receives a gradient.
        base = F.normalize(torch.randn(reflections // 2, width), dim=-1)
        self.reflection_vectors = nn.Parameter(base.repeat_interleave(2, dim=0))
        self.register_buffer("scale_ema", torch.ones(width))
        self.register_buffer("p_ema", torch.full((width,), 0.5))
        self.register_buffer("scale_initialized", torch.tensor(False))
        self.register_buffer("popularity_initialized", torch.tensor(False))
        self.register_buffer("stat_updates", torch.zeros((), dtype=torch.long))
        self._anchor_activations = {}

    def _vectors(self):
        return F.normalize(self.reflection_vectors.float(), dim=-1, eps=1e-8)

    def _transform(self, value, inverse=False):
        vectors = self._vectors()
        if inverse:
            vectors = vectors.flip(0)
        result = value.float()
        for vector in vectors:
            result = result - 2 * (result * vector).sum(-1, keepdim=True) * vector
        return result.to(value.dtype)

    def basis(self):
        eye = torch.eye(self.width, device=self.reflection_vectors.device)
        return self._transform(eye)

    def popularity(self):
        if self.mode == "dictionary":
            return (torch.arange(self.width, device=self.p_ema.device) < self.width // 2).float()
        return torch.sigmoid((self.p_ema - self.popularity_threshold) / self.temperature)

    def forward(self, hidden):
        if hidden.ndim != 4 or hidden.shape[-1] != self.width:
            raise ValueError("H must be [B,C,P,D]")
        factors = self._transform(hidden)
        pi = self.popularity().to(factors.dtype)
        common = self._transform(factors * pi, inverse=True)
        specific = hidden - common  # exact additive reconstruction in working precision
        support = (factors.square() * pi).sum(-1, keepdim=True) / factors.square().sum(-1, keepdim=True).clamp_min(1e-8)
        diagnostics = {
            "reconstruction_mae": (common + specific - hidden).abs().mean(),
            "common_energy_ratio": common.square().sum() / hidden.square().sum().clamp_min(1e-8),
            "specific_energy_ratio": specific.square().sum() / hidden.square().sum().clamp_min(1e-8),
            "support_mean": support.mean(),
            "effective_common_factors": pi.sum(),
            "factor_assignment_entropy": -(pi * pi.clamp_min(1e-8).log() + (1-pi) * (1-pi).clamp_min(1e-8).log()).mean(),
        }
        return common, specific, support, factors, diagnostics

    @torch.no_grad()
    def observe_train_batch(self, factors, sample_ids, seq_len):
        if not self.training:
            raise RuntimeError("factor memory is TRAIN-only")
        if factors.ndim != 4 or len(sample_ids) != factors.shape[0]:
            raise ValueError("factor/sample shape mismatch")
        # One channel/window contributes at most one activation.  The anchor
        # merge below prevents overlapping stride-one windows voting repeatedly.
        magnitude = factors.float().square().mean(2).add(1e-8).sqrt().detach()
        batch_scale = magnitude.mean((0, 1))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_scale)
            batch_scale /= torch.distributed.get_world_size()
        if not bool(self.scale_initialized):
            self.scale_ema.copy_(batch_scale.clamp_min(1e-6))
            self.scale_initialized.fill_(True)
        else:
            self.scale_ema.lerp_(batch_scale, 1-self.beta)
        active = (magnitude / self.scale_ema.clamp_min(1e-6) > self.activation_threshold)
        # Every (independent temporal block, channel) supplies ONE vote.
        # Prefer the representative non-overlapping window start=anchor*L.
        # A temporary first-seen fallback permits short smoke runs; the full
        # TRAIN epoch always visits the representative.  OR across windows or
        # channels would make nearly every factor popular.
        active = active.cpu()
        for sample_id, matrix in zip(sample_ids.long().tolist(), active):
            anchor = sample_id // seq_len
            representative = sample_id % seq_len == 0
            for channel, row in enumerate(matrix):
                key = (anchor, channel)
                previous = self._anchor_activations.get(key)
                if previous is None or (representative and not previous[0]):
                    self._anchor_activations[key] = (representative, row.clone())

    @torch.no_grad()
    def finish_train_epoch(self):
        entries = self._anchor_activations
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, entries)
            entries = {}
            for portion in gathered:
                for anchor, vote in portion.items():
                    if anchor not in entries or (vote[0] and not entries[anchor][0]):
                        entries[anchor] = vote
        if not entries:
            raise RuntimeError("no TRAIN anchors observed")
        rate = torch.stack([vote[1] for vote in entries.values()]).float().mean(0).to(self.p_ema.device)
        if not bool(self.popularity_initialized):
            self.p_ema.copy_(rate)
            self.popularity_initialized.fill_(True)
        else:
            self.p_ema.lerp_(rate, 1-self.beta)
        groups = len(entries)
        self.stat_updates.add_(1)
        self._anchor_activations.clear()
        return {"independent_anchor_groups": groups, "activation_rate_mean": float(rate.mean())}


class ConsensusLatentFusion(nn.Module):
    def __init__(self, width: int, learnable: bool, alpha_max: float = 0.25):
        super().__init__()
        self.learnable, self.alpha_max = learnable, alpha_max
        if learnable:
            self.gate = nn.Sequential(nn.LayerNorm(2*width+1), nn.Linear(2*width+1, width),
                                      nn.GELU(), nn.Linear(width, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
            self.k_raw = nn.Parameter(torch.tensor(-8.0))

    def forward(self, common, specific, support):
        if self.learnable:
            logit = self.gate(torch.cat((common, specific, support), -1))
            alpha = 0.5 + self.alpha_max * torch.tanh(
                logit + F.softplus(self.k_raw) * (2*support-1))
        else:
            alpha = torch.full_like(support, 0.5)
        fused = 2*alpha*common + 2*(1-alpha)*specific
        return fused, alpha
