"""Soft environment-aware Pattern transition decomposition."""
from __future__ import annotations

import math
import torch
from torch import nn
from .pattern_bank import _two_component_soft


class EnvironmentMappingBank(nn.Module):
    def __init__(self, pattern_count: int, d_model: int):
        super().__init__()
        self.pattern_count = int(pattern_count)
        self.d_model = int(d_model)
        edge_count = pattern_count * pattern_count
        self.register_buffer("p_inv", torch.full((edge_count,), 0.5))
        self.register_buffer("env_transition", torch.empty(0, edge_count))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, responsibility_batches, environment_batches):
        env_num = max(int(e.max()) for e in environment_batches) + 1
        transitions = torch.zeros(env_num, self.pattern_count, self.pattern_count)
        for resp, env in zip(responsibility_batches, environment_batches):
            # same sample, same variable, t -> t+1 only
            sample = torch.einsum("bcti,bctj->bij", resp[:, :, :-1], resp[:, :, 1:])
            for e in range(env_num):
                transitions[e] += sample[env == e].sum(0).cpu()
        flat = transitions.reshape(env_num, -1)
        across = flat / flat.sum(1, keepdim=True).clamp_min(1e-6)
        per_edge = across / across.sum(0, keepdim=True).clamp_min(1e-6)
        entropy = -(per_edge * per_edge.clamp_min(1e-8).log()).sum(0) / math.log(max(env_num, 2))
        coverage = (across / across.max(0, keepdim=True).values.clamp_min(1e-6)).mean(0)
        support = flat.sum(0)
        score = 0.4 * entropy + 0.4 * coverage + 0.2 * torch.log1p(support) / torch.log1p(support.max()).clamp_min(1e-6)
        self.p_inv.copy_(_two_component_soft(score).to(self.p_inv.device))
        self.env_transition = flat.to(self.p_inv.device)
        self.fitted.fill_(True)
        return {"entropy": entropy, "coverage": coverage, "support": support, "score": score}

    def query(self, responsibilities: torch.Tensor, prototypes: torch.Tensor):
        b, c, t, k = responsibilities.shape
        if not bool(self.fitted) or t < 2:
            z = torch.zeros(b, self.d_model, device=responsibilities.device)
            return z, z, torch.zeros(b, device=responsibilities.device)
        sample_edges = torch.einsum(
            "bcti,bctj->bij", responsibilities[:, :, :-1], responsibilities[:, :, 1:]
        ).reshape(b, -1)
        inv_w = sample_edges * self.p_inv
        var_w = sample_edges * (1.0 - self.p_inv)
        inv_strength = inv_w.sum(1) / sample_edges.sum(1).clamp_min(1e-6)
        destination = prototypes[None].expand(k, -1, -1).reshape(k * k, -1)
        inv_context = (inv_w @ destination) / inv_w.sum(1, keepdim=True).clamp_min(1e-6)
        var_context = (var_w @ destination) / var_w.sum(1, keepdim=True).clamp_min(1e-6)
        return inv_context, var_context, inv_strength
