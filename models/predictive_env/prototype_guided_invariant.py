"""R3-guided shared full-future forecaster; no new local expert network."""
from __future__ import annotations

import torch
from torch import nn

from models.predictive_env.direct_invariant_forecaster import DirectInvariantForecaster


@torch.no_grad()
def prototype_h_centers(hidden: torch.Tensor, q: torch.Tensor):
    """TRAIN-only H-space references from variable-level R3 routing."""
    if hidden.shape[:2] != q.shape[:2]:
        raise ValueError("H/q batch-channel mismatch")
    pooled = hidden.float().mean(2)
    weights = q.detach().float().clamp_min(0)
    mass = weights.sum((0, 1))
    centers = torch.einsum("bcm,bcd->md", weights, pooled) / mass[:, None].clamp_min(1e-8)
    global_center = pooled.mean((0, 1))
    absent = mass < 1e-8
    centers[absent] = global_center
    effective = mass.square() / weights.square().sum((0, 1)).clamp_min(1e-8)
    return global_center, centers, {"mass": mass, "effective": effective}


class PrototypeGuidedSharedHead(nn.Module):
    def __init__(self, tokens: int, width: int, horizon: int, *, rank: int = 0,
                 reference_mode: str = "none", residual_form: bool = False,
                 pretrained_head: nn.Linear | None = None, prototypes: int = 8):
        super().__init__()
        if reference_mode not in {"none", "global", "prototype"}:
            raise ValueError(reference_mode)
        self.reference_mode, self.residual_form = reference_mode, residual_form
        self.register_buffer("global_center", torch.zeros(width))
        self.register_buffer("prototype_centers", torch.zeros(prototypes, width))
        self.reference_adapter = None
        if reference_mode != "none":
            self.reference_adapter = nn.Linear(3 * width, width)
            nn.init.zeros_(self.reference_adapter.weight)
            nn.init.zeros_(self.reference_adapter.bias)
        self.forecaster = DirectInvariantForecaster(
            tokens, width, horizon, rank=rank,
            pretrained_head=pretrained_head if not rank and not residual_form else None)
        if residual_form:
            nn.init.zeros_(self.forecaster.head.weight)
            nn.init.zeros_(self.forecaster.head.bias)

    @torch.no_grad()
    def set_references(self, global_center: torch.Tensor, centers: torch.Tensor):
        self.global_center.copy_(global_center)
        self.prototype_centers.copy_(centers)

    def forward(self, hidden: torch.Tensor, q: torch.Tensor | None = None,
                base_normalized: torch.Tensor | None = None):
        z = hidden
        if self.reference_mode == "global":
            reference = self.global_center[None, None, None, :].expand_as(hidden)
        elif self.reference_mode == "prototype":
            if q is None or q.shape != (*hidden.shape[:2], self.prototype_centers.shape[0]):
                raise ValueError("prototype reference requires variable-level q")
            reference = (q.detach() @ self.prototype_centers).unsqueeze(2).expand_as(hidden)
        else:
            reference = None
        if reference is not None:
            z = hidden + self.reference_adapter(torch.cat((hidden, reference, hidden-reference), -1))
        prediction = self.forecaster(z)
        if self.residual_form:
            if base_normalized is None or base_normalized.shape != prediction.shape:
                raise ValueError("residual control requires frozen A0 normalized forecast")
            return base_normalized.detach() + prediction
        return prediction

    @property
    def head_weight(self):
        return self.forecaster.head.weight


def routing_control(q: torch.Tensor, mode: str, *, seed: int = 2021,
                    history_ids: torch.Tensor | None = None):
    """Controls have identical [B,C,M] shape; NULL mass is preserved where possible."""
    if mode == "real":
        return q.detach()
    b, c, m = q.shape
    mass = q.sum(-1, keepdim=True).detach()
    if mode == "uniform":
        return mass.expand_as(q) / m
    if mode == "shuffled":
        perm = torch.randperm(b, generator=torch.Generator().manual_seed(seed))
        return q.detach()[perm]
    if mode == "random":
        generator = torch.Generator().manual_seed(seed)
        random = torch.rand(q.shape, generator=generator).softmax(-1)
        return random * mass
    if mode == "time":
        if history_ids is None:
            raise ValueError("time grouping requires TRAIN indices")
        edges = torch.quantile(history_ids.float(), torch.linspace(0, 1, m + 1)[1:-1])
        group = torch.bucketize(history_ids.float(), edges)
        return torch.nn.functional.one_hot(group, m).float()[:, None].expand(-1, c, -1) * mass
    raise ValueError(mode)
