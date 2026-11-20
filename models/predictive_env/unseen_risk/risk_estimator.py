"""History-only mapping-transfer risk features and conditional scale."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def risk_features(memory: dict) -> torch.Tensor:
    """Return [B,C,5], independent of query future or TEST environment label."""
    valid = memory["valid"]
    common = valid & memory["is_common"]
    rare = valid & ~memory["is_common"]
    distance = memory["distance"].clamp(0,100)
    count = valid.sum(-1).clamp_min(1)
    mean_future = memory["future"].sum(2) / count[...,None]
    disagreement = ((memory["future"]-mean_future[:,:,None]).square()
                    * valid[...,None]).sum((2,3)) / (count*memory["future"].shape[-1])
    count_common = common.sum(-1).clamp_min(1)
    count_rare = rare.sum(-1).clamp_min(1)
    mu_common = (memory["future"]*common[...,None]).sum(2) / count_common[...,None]
    mu_rare = (memory["future"]*rare[...,None]).sum(2) / count_rare[...,None]
    conflict = (mu_common-mu_rare).square().mean(-1) * common.any(-1)*rare.any(-1)
    novelty = (distance*valid).sum(-1)/count
    support = (memory["support"]*valid).sum(-1)/count
    density = (memory["density"]*valid).sum(-1)/count
    return torch.stack((torch.log1p(novelty), torch.log1p(disagreement),
                        torch.log1p(conflict), 1./(1.+support),
                        1./(1.+density)), -1)


def fit_transfer_regression(features: torch.Tensor, gain: torch.Tensor,
                            ridge: float = 1e-3) -> torch.Tensor:
    """TRAIN OOF feature/realized-gain ridge coefficients, including intercept."""
    x = features.detach().float().reshape(-1,features.shape[-1]).cpu()
    y = gain.detach().float().reshape(-1,1).cpu()
    design = torch.cat((x,torch.ones(len(x),1)), -1)
    regularizer = ridge*torch.eye(design.shape[-1]); regularizer[-1,-1]=0
    return torch.linalg.solve(design.T@design+regularizer,design.T@y).squeeze(-1)


class RiskScaleHead(nn.Module):
    def __init__(self, feature_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim,16),
                                 nn.GELU(),nn.Linear(16,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -1.)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(features)).squeeze(-1).clamp_min(1e-4)

    @staticmethod
    def nll(pred: torch.Tensor, target: torch.Tensor,
            scale: torch.Tensor) -> torch.Tensor:
        err = (target-pred).transpose(1,2)
        return (torch.log(scale[...,None])+
                .5*(err/scale[...,None]).square()).mean()
