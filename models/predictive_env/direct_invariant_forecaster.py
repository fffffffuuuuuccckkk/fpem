"""Direct full-future forecast from an existing backbone representation.

This module never consumes an A0 forecast or an A0 residual. The optional
projection acts on the true feature dimension, not the temporal token axis.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DirectInvariantForecaster(nn.Module):
    def __init__(self, token_count: int, d_model: int, pred_len: int,
                 rank: int = 0, pretrained_head: nn.Linear | None = None):
        super().__init__()
        self.token_count = token_count
        self.d_model = d_model
        self.pred_len = pred_len
        self.rank = rank
        if rank:
            if rank > d_model:
                raise ValueError("projection rank exceeds hidden dimension")
            self.projection = nn.Linear(d_model, rank, bias=False)
            nn.init.orthogonal_(self.projection.weight)
        else:
            self.projection = None
        self.head = nn.Linear(token_count * (rank or d_model), pred_len)
        if pretrained_head is not None:
            if rank or self.head.weight.shape != pretrained_head.weight.shape:
                raise ValueError("A0 head is incompatible with the direct head")
            self.head.load_state_dict(pretrained_head.state_dict())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or hidden.shape[-2:] != (self.token_count, self.d_model):
            raise ValueError(f"expected [B,C,{self.token_count},{self.d_model}]")
        z = hidden if self.projection is None else self.projection(hidden)
        return self.head(z.permute(0, 1, 3, 2).flatten(-2))

    def orthogonality_loss(self) -> torch.Tensor:
        if self.projection is None:
            raise ValueError("orthogonality only applies to a projected model")
        p = self.projection.weight
        return (p @ p.T - torch.eye(p.shape[0], device=p.device, dtype=p.dtype)).square().mean()


def direct_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float,
                 max_rows: int = 1024, seed: int = 2021) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Train-only full-future Ridge; dual form avoids a D*P square matrix.

    A fixed TRAIN subsample limits the cubic solve. The reported sample count
    makes this approximation explicit rather than implying a full-data fit.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x/y must be [rows,features]/[rows,horizon]")
    if x.shape[0] > max_rows:
        idx = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))[:max_rows]
        x, y = x[idx], y[idx]
    x = x.double()
    y = y.double()
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    scale = max(1, x.shape[1])
    if x.shape[1] <= x.shape[0]:
        gram = (xc.T @ xc) / scale
        rhs = (xc.T @ yc) / scale
        w = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], dtype=x.dtype), rhs)
    else:
        gram = (xc @ xc.T) / scale
        alpha = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], dtype=x.dtype), yc)
        w = xc.T @ alpha / scale
    b = ym - xm @ w
    return w.float(), b.float(), len(x)


def group_risk(pred: torch.Tensor, target: torch.Tensor,
               group: torch.Tensor, mode: str, penalty: float) -> torch.Tensor:
    """Direct full-future group objectives; groups are TRAIN-only metadata."""
    sample_error = (pred - target).square().mean((1, 2))
    groups = [sample_error[group == value].mean() for value in torch.unique(group)
              if int((group == value).sum()) >= 2]
    if len(groups) < 2:
        return sample_error.mean()
    risks = torch.stack(groups)
    if mode == "vrex":
        return risks.mean() + penalty * risks.var(unbiased=False)
    if mode == "dro":
        temperature = max(penalty, 1e-6)
        return temperature * (torch.logsumexp(risks / temperature, dim=0)
                              - torch.log(torch.tensor(len(risks), device=risks.device)))
    raise ValueError(mode)
