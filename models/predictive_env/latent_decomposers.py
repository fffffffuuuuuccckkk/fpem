"""Optional prediction-preserving latent decomposers for PredictiveEnvIV.

All reference statistics are TRAIN-only buffers.  The third return value is
an H-shaped gate proxy, preserving the existing three-value model interface.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LowRankSubspaceDecomposer(nn.Module):
    def __init__(self, d_model, rank=8, bottleneck=64, init_gate_bias=-4.0):
        super().__init__()
        if not 1 <= rank <= d_model:
            raise ValueError("subspace rank must lie in [1, d_model]")
        self.d_model, self.rank = int(d_model), int(rank)
        self.basis = nn.Parameter(torch.randn(d_model, rank) * (d_model ** -0.5))
        self.gate_net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, min(bottleneck, d_model)),
            nn.GELU(), nn.Linear(min(bottleneck, d_model), rank),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, init_gate_bias)

    def orthogonal_basis(self):
        # QR is performed outside autocast to avoid half-precision instability.
        with torch.amp.autocast(device_type=self.basis.device.type, enabled=False):
            q, r = torch.linalg.qr(self.basis.float(), mode="reduced")
            signs = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
            return q * signs.unsqueeze(0)

    def forward(self, hidden):
        basis = self.orthogonal_basis().to(hidden.dtype)
        gate = torch.sigmoid(self.gate_net(hidden))
        z_var = ((hidden @ basis) * gate) @ basis.T
        z_inv = hidden - z_var
        return z_inv, z_var, torch.zeros_like(hidden)

    def diagnostics(self, hidden, z_inv, z_var):
        basis = self.orthogonal_basis()
        eye = torch.eye(self.rank, device=basis.device)
        return {
            "subspace/orthogonal_error": (basis.T @ basis - eye).norm(),
            "subspace/var_energy_ratio": z_var.float().square().mean().sqrt()
            / hidden.float().square().mean().sqrt().clamp_min(1e-8),
            "subspace/reconstruction_error": (z_inv + z_var - hidden).abs().mean(),
        }


class EnvGuidedSubspaceDecomposer(LowRankSubspaceDecomposer):
    def __init__(self, d_model, rank=8, bottleneck=64, init_gate_bias=-4.0):
        super().__init__(d_model, rank, bottleneck, init_gate_bias)
        self.register_buffer("between_scatter", torch.zeros(d_model, d_model))
        self.register_buffer("total_scatter", torch.zeros(d_model, d_model))
        self.register_buffer("stats_ready", torch.tensor(False))

    @torch.no_grad()
    def update_environment_statistics(self, pooled_hidden, assignment, ema=0.0):
        h = pooled_hidden.detach().float()
        q = assignment.detach().float()
        if h.ndim != 2 or q.ndim != 2 or h.shape[0] != q.shape[0]:
            raise ValueError("expected TRAIN pooled hidden [N,D] and soft q [N,K]")
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-8)
        mean = h.mean(0)
        centered = h - mean
        total = centered.T @ centered / h.shape[0]
        mass = q.sum(0).clamp_min(1e-8)
        centers = q.T @ h / mass[:, None]
        offsets = centers - mean
        between = (offsets * (mass / h.shape[0])[:, None]).T @ offsets
        if bool(self.stats_ready):
            between = ema * self.between_scatter + (1 - ema) * between
            total = ema * self.total_scatter + (1 - ema) * total
        self.between_scatter.copy_(between)
        self.total_scatter.copy_(total)
        self.stats_ready.fill_(True)

    def environment_subspace_loss(self):
        if not bool(self.stats_ready):
            raise RuntimeError("TRAIN environment scatter has not been initialized")
        basis = self.orthogonal_basis()
        explained = (basis * (self.between_scatter @ basis)).sum()
        total = (basis * (self.total_scatter @ basis)).sum().clamp_min(1e-8)
        return -explained / total


class ReferenceGuidedDecomposer(nn.Module):
    def __init__(self, d_model, num_environments, bottleneck=64,
                 reference_mode="environment", init_gate_bias=4.0):
        super().__init__()
        if reference_mode not in ("global", "environment", "none"):
            raise ValueError("reference_mode must be global, environment, or none")
        self.reference_mode = reference_mode
        self.register_buffer("global_center", torch.zeros(d_model))
        self.register_buffer("environment_centers", torch.zeros(num_environments, d_model))
        self.register_buffer("stats_ready", torch.tensor(False))
        width = min(d_model, bottleneck)
        self.q_predictor = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, width),
                                         nn.GELU(), nn.Linear(width, num_environments))
        self.gate_net = nn.Sequential(nn.LayerNorm(3 * d_model),
                                      nn.Linear(3 * d_model, width), nn.GELU(),
                                      nn.Linear(width, d_model))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, init_gate_bias)

    @torch.no_grad()
    def update_references(self, pooled_hidden, assignment, ema=0.0):
        h = pooled_hidden.detach().float()
        q = assignment.detach().float()
        q = q / q.sum(-1, keepdim=True).clamp_min(1e-8)
        global_center = h.mean(0)
        centers = q.T @ h / q.sum(0).clamp_min(1e-8)[:, None]
        if bool(self.stats_ready):
            global_center = ema * self.global_center + (1 - ema) * global_center
            centers = ema * self.environment_centers + (1 - ema) * centers
        self.global_center.copy_(global_center)
        self.environment_centers.copy_(centers)
        self.stats_ready.fill_(True)

    def predicted_q(self, hidden):
        return self.q_predictor(hidden.mean((1, 2))).softmax(-1)

    def q_supervision_loss(self, hidden, target_q):
        return -(target_q.detach() * self.q_predictor(hidden.mean((1, 2))).log_softmax(-1)).sum(-1).mean()

    def forward(self, hidden):
        if self.reference_mode == "none":
            reference = torch.zeros_like(hidden)
        elif self.reference_mode == "global":
            reference = self.global_center.to(hidden.dtype).view(1, 1, 1, -1).expand_as(hidden)
        else:
            # Both TRAIN and inference use qhat(H), never a future-dependent q.
            qhat = self.predicted_q(hidden)
            center = qhat @ self.environment_centers.to(qhat.dtype)
            reference = center[:, None, None, :].to(hidden.dtype).expand_as(hidden)
        gate = torch.sigmoid(self.gate_net(torch.cat((hidden, reference, hidden-reference), -1)))
        z_inv = gate * hidden
        return z_inv, hidden - z_inv, gate
