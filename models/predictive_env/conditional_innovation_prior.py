"""TRAIN-only forecast-error prior, and an optional zero-init Flow condition adapter.

The distribution describes errors from a strict chronological holdout; it is
not an exogenous-event classifier and does not itself generate an interval.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalFutureInnovationPrior(nn.Module):
    def __init__(self, dim: int, pred_len: int, hidden: int = 128):
        super().__init__()
        self.pred_len = pred_len
        self.net = nn.Sequential(nn.LayerNorm(2*dim+2),
                                 nn.Linear(2*dim+2, hidden), nn.GELU(),
                                 nn.Linear(hidden, 3*pred_len))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, common, specific, support, factor_pi):
        if common.shape != specific.shape or common.ndim != 4:
            raise ValueError("common and specific must align [B,C,P,D]")
        b, c = common.shape[:2]
        popularity = factor_pi.float().mean().to(common.device)
        context = torch.cat((common.detach().mean(2), specific.detach().mean(2),
                             support.detach().mean(2),
                             popularity.expand(b,c,1)), -1)
        raw = self.net(context).view(b,c,self.pred_len,3)
        mixture = torch.sigmoid(raw[...,0].clamp(-8,8)).clamp(1e-5,1-1e-5)
        normal = F.softplus(raw[...,1]).clamp_min(1e-3)
        novel = normal + F.softplus(raw[...,2]).clamp_min(1e-3)
        return {"pi_novel": mixture, "sigma_normal": normal,
                "sigma_novel": novel}

    @staticmethod
    def nll(distribution, normalized_residual):
        """Stable Gaussian/Student-t(df=4) mixture NLL in historical-scale units."""
        r = normalized_residual.detach()
        pi = distribution["pi_novel"]
        normal = distribution["sigma_normal"]
        novel = distribution["sigma_novel"]
        gaussian = -0.5*math.log(2*math.pi)-normal.log()-0.5*(r/normal).square()
        df = 4.0
        student = (math.lgamma((df+1)/2)-math.lgamma(df/2)
                   -0.5*math.log(df*math.pi)-novel.log()
                   -(df+1)/2*torch.log1p((r/novel).square()/df))
        stacked = torch.stack((torch.log1p(-pi)+gaussian, pi.log()+student), -1)
        return -torch.logsumexp(stacked,-1).mean()

    @staticmethod
    def features(distribution):
        return torch.stack((distribution["pi_novel"],
                            distribution["sigma_normal"].log(),
                            distribution["sigma_novel"].log()), -1)


class InnovationConditionAdapter(nn.Module):
    """Adds only a learned condition offset; zero initialization is exact."""
    def __init__(self, condition_dim: int, patch_len: int = 16):
        super().__init__()
        self.patch_len = patch_len
        self.projection = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, condition_dim))
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, existing_condition, innovation_features, enabled=True):
        if not enabled:
            return existing_condition
        if existing_condition.ndim != 4 or innovation_features.ndim != 4:
            raise ValueError("condition [B,C,M,D] and innovation [B,C,H,3] required")
        b,c,m,d = existing_condition.shape
        if innovation_features.shape[:2] != (b,c) or innovation_features.shape[-1] != 3:
            raise ValueError("condition/innovation shape mismatch")
        patches = []
        for k in range(m):
            section = innovation_features[:,:,k*self.patch_len:(k+1)*self.patch_len]
            if not section.shape[2]:
                raise ValueError("not enough innovation steps for the condition")
            patches.append(section.mean(2))
        features = torch.stack(patches, 2).detach()
        return existing_condition + self.projection(features)


def strict_chronological_indices(dataset_length, seq_len, pred_len,
                                 core_fraction=0.8):
    """Sample-start indices with disjoint TARGET time indices across split."""
    if not 0 < core_fraction < 1:
        raise ValueError("core_fraction must be in (0,1)")
    total_time = dataset_length + seq_len + pred_len - 1
    cut = int(total_time * core_fraction)
    core = [i for i in range(dataset_length) if i+seq_len+pred_len <= cut]
    calibration = [i for i in range(dataset_length) if i+seq_len >= cut]
    if not core or not calibration:
        raise ValueError("insufficient TRAIN windows for a strict time split")
    if max(i+seq_len+pred_len-1 for i in core) >= min(i+seq_len for i in calibration):
        raise RuntimeError("core and calibration target indices overlap")
    return core, calibration, cut
