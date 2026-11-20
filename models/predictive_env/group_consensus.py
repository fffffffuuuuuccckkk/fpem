"""TRAIN-only group-consensus subspace tracking and invariant/dynamic projection.

Q is a long-term state for EMA modes, not a gradient-updated parameter. The
static mode uses a differentiable QR basis. Validation/test must never call
``update`` or ``initialize_from_training``.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GroupConsensusSubspace(nn.Module):
    def __init__(self, width: int, rank: int, mode: str, *, beta: float = 0.99,
                 refresh_steps: int = 64):
        super().__init__()
        if mode not in ("random", "static", "ema", "robust"):
            raise ValueError(f"invalid subspace mode: {mode}")
        if not 0 < rank < width or not 0 <= beta < 1 or refresh_steps < 1:
            raise ValueError("invalid subspace rank or tracking configuration")
        self.width, self.rank, self.mode = width, rank, mode
        self.beta, self.refresh_steps = beta, refresh_steps
        initial, _ = torch.linalg.qr(torch.randn(width, rank, dtype=torch.float32), mode="reduced")
        if mode == "static":
            self.raw_basis = nn.Parameter(initial)
        else:
            self.register_buffer("q", initial)
        self.register_buffer("second_moment", torch.zeros(width, width, dtype=torch.float32))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(mode in ("random", "static")))

    def basis(self) -> torch.Tensor:
        if self.mode == "static":
            # QR is run in fp32 even under autocast, then cast at the projection site.
            with torch.autocast(device_type=self.raw_basis.device.type, enabled=False):
                return torch.linalg.qr(self.raw_basis.float(), mode="reduced")[0]
        return self.q

    @staticmethod
    def _normalized_rows(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4:
            raise ValueError("hidden must be [B,C,P,D]")
        return F.normalize(hidden.detach().float().flatten(1, 2), dim=-1)

    @staticmethod
    def _batch_second_moment(rows: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        sample_covariance = torch.einsum("btd,bte->bde", rows, rows) / rows.shape[1]
        return torch.einsum("b,bde->de", weights, sample_covariance) / weights.sum().clamp_min(1e-8)

    @torch.no_grad()
    def initialize_from_training(self, hidden_batches):
        if self.mode not in ("ema", "robust"):
            return
        summed = torch.zeros_like(self.second_moment)
        count = 0
        for hidden in hidden_batches:
            rows = self._normalized_rows(hidden).to(summed.device)
            summed += torch.einsum("btd,bte->de", rows, rows) / rows.shape[1]
            count += len(rows)
        if count == 0:
            raise ValueError("empty TRAIN bootstrap")
        self.second_moment.copy_(summed / count)
        self._refresh_basis()
        self.initialized.fill_(True)

    @torch.no_grad()
    def _refresh_basis(self):
        matrix = (self.second_moment + self.second_moment.T) * 0.5
        _, vectors = torch.linalg.eigh(matrix.float())
        # Re-orthogonalize the selected near-degenerate directions; CUDA eigh
        # can otherwise leave a small but consequential Q^T Q error.
        stable, _ = torch.linalg.qr(vectors[:, -self.rank:], mode="reduced")
        self.q.copy_(stable)

    @torch.no_grad()
    def update(self, hidden: torch.Tensor):
        if self.mode not in ("ema", "robust"):
            return
        if not self.training:
            raise RuntimeError("subspace update is TRAIN-only; call eval() for VAL/TEST")
        rows = self._normalized_rows(hidden).to(self.second_moment.device)
        if self.mode == "robust":
            projection = (rows @ self.q) @ self.q.T
            residual = (rows - projection).square().sum(-1).mean(-1).sqrt()
            delta = residual.median().clamp_min(1e-6)
            weights = (delta / residual.clamp_min(1e-6)).clamp(max=1.)
        else:
            weights = torch.ones(len(rows), device=rows.device)
        batch = self._batch_second_moment(rows, weights)
        self.second_moment.lerp_(batch, 1-self.beta)
        self.updates.add_(1)
        if int(self.updates) % self.refresh_steps == 0:
            self._refresh_basis()

    def forward(self, hidden: torch.Tensor):
        if hidden.ndim != 4 or hidden.shape[-1] != self.width:
            raise ValueError("hidden must be [B,C,P,D] with the configured D")
        q = self.basis().to(device=hidden.device, dtype=hidden.dtype)
        invariant = (hidden @ q) @ q.T
        dynamic = hidden - invariant
        shared = invariant.square().sum(-1) / hidden.square().sum(-1).clamp_min(1e-8)
        shared = shared.clamp(0, 1)
        if hidden.shape[2] > 1:
            continuity = F.cosine_similarity(dynamic[:, :, 1:], dynamic[:, :, :-1], dim=-1)
            continuity = torch.cat((torch.zeros_like(continuity[:, :, :1]), continuity), 2)
        else:
            # A singleton token has no temporal neighbour; never reinterpret features as time.
            continuity = shared.new_zeros(shared.shape)
        diagnostics = {
            "reconstruction_mae": (invariant + dynamic - hidden).abs().mean(),
            "orthogonality": (invariant * dynamic).sum(-1).abs().mean(),
            "basis_orthogonality": (q.T @ q - torch.eye(self.rank, device=q.device,
                                                        dtype=q.dtype)).abs().max(),
            "sharedness_mean": shared.mean(),
            "dynamic_energy_ratio": dynamic.square().sum() / hidden.square().sum().clamp_min(1e-8),
        }
        return invariant, dynamic, shared, continuity, diagnostics


class ConsensusAwareFusion(nn.Module):
    """Predict per-sample/channel/horizon use of the dynamic correction."""
    def __init__(self, width: int, horizon: int, mode: str):
        super().__init__()
        if mode not in ("fixed", "ordinary", "consensus"):
            raise ValueError(mode)
        self.mode = mode
        if mode != "fixed":
            self.net = nn.Sequential(nn.LayerNorm(2*width+2),
                                     nn.Linear(2*width+2, width), nn.GELU(),
                                     nn.Linear(width, horizon))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, -2.)

    def forward(self, invariant, dynamic, sharedness, continuity, horizon):
        if self.mode == "fixed":
            return invariant.new_ones((*invariant.shape[:2], horizon)).permute(0, 2, 1)
        pooled = torch.cat((invariant.mean(2), dynamic.mean(2)), dim=-1)
        if self.mode == "ordinary":
            scores = pooled.new_zeros((*pooled.shape[:2], 2))
        else:
            scores = torch.stack((sharedness.mean(2), continuity.mean(2)), dim=-1)
        gate = torch.sigmoid(self.net(torch.cat((pooled, scores), dim=-1)))
        return gate.permute(0, 2, 1)
