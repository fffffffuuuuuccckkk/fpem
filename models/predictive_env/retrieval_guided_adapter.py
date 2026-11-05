"""Small optional H-space retrieval adaptation; no new forecasting head."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.predictive_env.signed_gate_decomposition import ComplementaryGateDecomposer


class RetrievalGuidedAdapter(nn.Module):
    def __init__(self, d_model: int, env_num: int, align_max: float = .2,
                 variant_max: float = .2):
        super().__init__()
        self.decomposer = ComplementaryGateDecomposer(d_model)
        self.env_predictor = nn.Sequential(nn.LayerNorm(d_model),
                                           nn.Linear(d_model, min(64, d_model)),
                                           nn.GELU(), nn.Linear(min(64, d_model), env_num))
        # The final projection starts at zero so a fresh adapter cannot invent
        # a large response-dependent H correction before forecasting gradients.
        self.variant_adapter = nn.Sequential(
            nn.LayerNorm(d_model + 4), nn.Linear(d_model + 4, d_model),
            nn.GELU(), nn.Linear(d_model, d_model))
        nn.init.zeros_(self.variant_adapter[-1].weight)
        nn.init.zeros_(self.variant_adapter[-1].bias)
        self.pred_gate = nn.Sequential(nn.LayerNorm(d_model + 4),
                                       nn.Linear(d_model + 4, 1))
        nn.init.zeros_(self.pred_gate[-1].weight)
        nn.init.constant_(self.pred_gate[-1].bias, 8.)
        self.align_logit = nn.Parameter(torch.tensor(-5.))
        self.variant_logit = nn.Parameter(torch.tensor(-5.))
        self.align_max, self.variant_max = align_max, variant_max

    def forward(self, hidden: torch.Tensor, query: dict | None,
                mode: str = "P3", full_residual: bool = False,
                no_align: bool = False, no_variant: bool = False) -> dict:
        z_inv, z_var, signed = self.decomposer(hidden)
        qhat = self.env_predictor(hidden.mean((1, 2))).softmax(-1)
        if mode == "B1":
            return {"h_fused": z_inv, "z_inv": z_inv, "z_var": z_var,
                    "qhat": qhat, "gate": hidden.new_zeros(hidden.shape[:-1]),
                    "align_effect": hidden.new_zeros(()), "variant_effect": hidden.new_zeros(())}
        if query is None:
            return {"h_fused": hidden, "z_inv": z_inv, "z_var": z_var,
                    "qhat": qhat, "gate": hidden.new_ones(hidden.shape[:-1]),
                    "align_effect": hidden.new_zeros(()), "variant_effect": hidden.new_zeros(())}
        proto = query["proto_hidden"][:, :, None, :].expand_as(hidden)
        proto_inv, _, _ = self.decomposer(proto)
        align = (0. if no_align else self.align_max * torch.sigmoid(self.align_logit))
        correction = align * query["c_inv"][..., None, None] * (proto_inv - z_inv)
        inv = z_inv + correction
        if mode == "P1":
            fused, gate = inv, hidden.new_zeros(hidden.shape[:-1])
            delta = hidden.new_zeros(())
        else:
            # Only summary statistics of TRAIN-fitted future responses enter
            # the adapter, never a current query's true future trajectory.
            response = query["env_response"]
            summaries = torch.stack((response.mean(-1), response.std(-1, unbiased=False),
                                     response[..., -1] - response[..., 0],
                                     query["c_var"]), -1)
            side = summaries[:, :, None].expand(*hidden.shape[:-1], 4)
            inputs = torch.cat((z_var, side), -1)
            gate = self.pred_gate(inputs).sigmoid() if not full_residual else torch.ones_like(inputs[..., :1])
            if no_variant:
                gate = gate * 0
            var = z_var * gate
            if mode in ("P3", "P4") and not no_variant:
                delta = (self.variant_max * torch.sigmoid(self.variant_logit)
                         * query["c_var"][..., None, None] * self.variant_adapter(inputs))
                var = var + delta
            else:
                delta = torch.zeros_like(var)
            fused = inv + var
        return {"h_fused": fused, "z_inv": inv, "z_var": z_var, "qhat": qhat,
                "gate": gate.squeeze(-1), "align_effect": correction.abs().mean(),
                "variant_effect": delta.abs().mean() if torch.is_tensor(delta) else delta,
                "reconstruction_error": (z_inv + z_var - hidden).abs().mean(),
                "signed_gate": signed}

    def predict_environment(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.env_predictor(hidden.mean((1, 2))).softmax(-1)
