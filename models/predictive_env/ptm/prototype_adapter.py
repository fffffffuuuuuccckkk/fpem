"""Small input-side and future-shape calibration adapters for a frozen A0."""
from __future__ import annotations

import torch
import torch.nn as nn


class PrototypeAdapter(nn.Module):
    def __init__(self, mode: str, eta_max: float = 0.2, gate_max: float = 0.2,
                 hidden_size: int = 16):
        super().__init__()
        if mode not in ("input", "shape"):
            raise ValueError("mode must be input or shape")
        self.mode, self.eta_max, self.gate_max = mode, float(eta_max), float(gate_max)
        self.gate = nn.Sequential(nn.LayerNorm(5), nn.Linear(5, hidden_size),
                                  nn.GELU(), nn.Linear(hidden_size, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def enhance_input(self, x, query, schedule=1.0):
        if self.mode != "input":
            raise ValueError("not an input adapter")
        eta = self.eta_max * float(schedule) * torch.sigmoid(self.gate(query["features"]))
        enhanced = x + eta.transpose(1, 2) * (query["x_proto"] - x)
        return enhanced, eta.squeeze(-1)

    def calibrate_shape(self, y0, query):
        if self.mode != "shape":
            raise ValueError("not a shape-calibration adapter")
        gate = self.gate_max * torch.tanh(self.gate(query["features"]))
        delta = query["y_proto"] - y0
        shape = delta - delta.mean(dim=1, keepdim=True)
        y = y0 + gate.transpose(1, 2) * shape
        return y, gate.squeeze(-1)
