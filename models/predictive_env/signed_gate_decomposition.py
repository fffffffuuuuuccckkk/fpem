"""Signed allocation of one forecasting representation into inv/var parts."""

import torch
from torch import nn


class SignedGateDecomposer(nn.Module):
    """Element-wise signed gate over H, without synthesizing new features."""

    def __init__(self, d_model, bottleneck=64):
        super().__init__()
        hidden = min(int(bottleneck), int(d_model))
        self.gate_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # Small symmetric initialization avoids an all-zero ReLU dead point
        # while imposing no target positive/negative ratio.
        nn.init.normal_(self.gate_net[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, hidden):
        gate = torch.tanh(self.gate_net(hidden))
        invariant_gate = torch.relu(gate)
        variant_gate = torch.relu(-gate)
        z_inv = hidden * invariant_gate
        z_var = hidden * variant_gate
        return z_inv, z_var, gate


class ComplementaryGateDecomposer(SignedGateDecomposer):
    """Information-preserving signed preference with complementary masks."""

    def forward(self, hidden):
        gate = torch.tanh(self.gate_net(hidden))
        invariant_gate = (1.0 + gate) / 2.0
        variant_gate = (1.0 - gate) / 2.0
        z_inv = hidden * invariant_gate
        z_var = hidden * variant_gate
        return z_inv, z_var, gate


@torch.no_grad()
def signed_gate_diagnostics(gate, near_zero_threshold=0.1):
    gate = gate.detach().float()
    return {
        "gate/mean": gate.mean(),
        "gate/std": gate.std(unbiased=False),
        "gate/positive_ratio": (gate > 0).float().mean(),
        "gate/negative_ratio": (gate < 0).float().mean(),
        "gate/abs_mean": gate.abs().mean(),
        "gate/near_zero_ratio": (
            gate.abs() < float(near_zero_threshold)
        ).float().mean(),
    }
