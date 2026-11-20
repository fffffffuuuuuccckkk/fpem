"""Prototype-guided *observed-history* retention for a frozen forecaster.

The prototype bank is fitted separately on TRAIN. This module never reads Y.
ORTE inspired the binary counterfactual mask; this is not an ORTE reproduction.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PGIRMask(nn.Module):
    def __init__(self, segment_len: int = 16, hidden: int = 16):
        super().__init__()
        self.segment_len = int(segment_len)
        self.net = nn.Sequential(
            nn.Conv1d(5, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        # Hard mask is exactly all-keep at initialization, but close enough to
        # the decision boundary that useful replacement can be learned.
        nn.init.constant_(self.net[-1].bias, 0.1)
        self.register_buffer("invariant_threshold", torch.tensor(float("nan")))

    def _features(self, x, x_proto, query, bank):
        # x and x_proto: B,L,C. Pad only for segment statistics.
        b, length, channels = x.shape
        nseg = (length + self.segment_len - 1) // self.segment_len
        pad = nseg * self.segment_len - length
        xp = x.transpose(1, 2)
        pp = x_proto.transpose(1, 2)
        if pad:
            xp = F.pad(xp, (0, pad), mode="replicate")
            pp = F.pad(pp, (0, pad), mode="replicate")
        xp = xp.reshape(b, channels, nseg, self.segment_len)
        pp = pp.reshape(b, channels, nseg, self.segment_len)
        scale = x.transpose(1, 2).std(-1, unbiased=False).clamp_min(1e-4)[..., None]
        residual = (xp - pp) / scale[..., None]
        local_mse = residual.square().mean(-1)
        support = (query["weight"] * (bank.counts / bank.counts.sum().clamp_min(1))).sum(-1)
        # Local shape match, weighted by independently anchored TRAIN coverage.
        score = torch.exp(-local_mse) * support[..., None]
        entropy = query["entropy"][..., None].expand_as(score)
        proto_slope = (pp[..., -1] - pp[..., 0]).abs() / scale
        feature = torch.stack((
            residual.mean(-1), residual.abs().mean(-1),
            residual.square().mean(-1).sqrt(), score, 
            proto_slope + 0.05 * entropy), dim=-1)
        return feature, score, residual.square().mean(-1)

    @torch.no_grad()
    def fit_threshold(self, bank, train_dataset, device, batch_size: int = 64):
        if int(getattr(train_dataset, "set_type", -1)) != 0:
            raise ValueError("PGIR invariant threshold must use TRAIN only")
        starts = np.arange(0, len(train_dataset), int(train_dataset.seq_len))
        values = []
        data = np.asarray(train_dataset.data_x, dtype=np.float32)
        for lo in range(0, len(starts), batch_size):
            hist = np.stack([data[s:s+train_dataset.seq_len] for s in starts[lo:lo+batch_size]])
            x = torch.from_numpy(hist).to(device)
            q = bank.query(x)
            _, score, _ = self._features(x, q["x_proto"], q, bank)
            values.append(score.flatten().cpu())
        if not values:
            raise ValueError("empty TRAIN anchors")
        self.invariant_threshold.copy_(torch.cat(values).median().to(self.invariant_threshold))
        return float(self.invariant_threshold)

    def forward(self, x, bank, *, mode="P2", replacement="prototype", generator=None):
        if not torch.isfinite(self.invariant_threshold):
            raise RuntimeError("fit TRAIN-only invariant threshold first")
        with torch.no_grad():
            query = bank.query(x)
            proto = query["x_proto"]
            features, score, residual_energy = self._features(x, proto, query, bank)
            invariant = (score >= self.invariant_threshold).to(x.dtype)
        b, length, channels = x.shape
        nseg = invariant.shape[-1]
        logits = self.net(features.reshape(b*channels, nseg, 5).transpose(1, 2))
        probability = logits[:, 0].reshape(b, channels, nseg).sigmoid()
        hard = (probability >= 0.5).to(x.dtype)
        variation = hard.detach() - probability.detach() + probability
        mask = invariant + (1-invariant)*variation
        if mode in ("A0", "P1"):
            mask = torch.ones_like(mask)
        elif mode == "random":
            # Same number of replaced non-invariant segments per series.
            selected = (mask.detach() < 0.5) & (invariant == 0)
            random_mask = torch.ones_like(mask)
            for i in range(b):
                for c in range(channels):
                    candidates = (invariant[i, c] == 0).nonzero().flatten()
                    count = int(selected[i, c].sum())
                    if count:
                        order = torch.randperm(len(candidates), generator=generator, device="cpu")[:count]
                        random_mask[i, c, candidates[order.to(candidates.device)]] = 0
            mask = random_mask
        time_mask = mask.repeat_interleave(self.segment_len, -1)[..., :length].transpose(1, 2)
        if replacement == "smooth":
            # Matched kernel width, but no TRAIN prototype pattern.
            smooth = F.avg_pool1d(F.pad(x.transpose(1, 2), (self.segment_len//2, self.segment_len//2),
                                        mode="replicate"), self.segment_len+1, stride=1)
            source = smooth[..., :length].transpose(1, 2)
        else:
            source = proto
        x_selected = time_mask*x + (1-time_mask)*source
        diagnostics = {
            "mask": mask, "invariant": invariant, "probability": probability,
            "score": score, "residual_energy": residual_energy,
            "replacement_fraction": (1-mask.detach()).mean(),
            "variation_keep_fraction": ((1-invariant)*hard).sum()/(1-invariant).sum().clamp_min(1),
            "query": query}
        return x_selected, diagnostics
