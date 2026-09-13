import torch
from torch import nn


class VariationMappingHead(nn.Module):
    """Direct operator modulation W_inv + g_m A diag(alpha(Z_var)) B."""
    def __init__(self, feature_dim, pred_len, d_model, rank=8, dropout=0.0):
        super().__init__()
        self.invariant = nn.Linear(feature_dim, pred_len)
        self.coefficients = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, rank), nn.Tanh())
        self.left = nn.Parameter(torch.zeros(pred_len, rank))
        self.right = nn.Parameter(torch.empty(rank, feature_dim))
        nn.init.xavier_uniform_(self.right)
        self.gate = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.dropout = nn.Dropout(dropout)

    def relative_weight_norm(self, alpha):
        gram_a, gram_b = self.left.t() @ self.left, self.right @ self.right.t()
        squared = torch.einsum("br,rs,bs->b", alpha, gram_a * gram_b, alpha).clamp_min(1e-12)
        return squared.sqrt() / self.invariant.weight.norm().clamp_min(1e-8)

    def forward(self, flat_state, pooled_inv, pooled_var, mode="gated"):
        base = self.invariant(flat_state)
        alpha = self.coefficients(pooled_var)
        if mode == "off":
            gate = torch.zeros(flat_state.shape[0], device=flat_state.device)
        elif mode == "forced":
            gate = torch.ones(flat_state.shape[0], device=flat_state.device)
        elif mode == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([pooled_inv, pooled_var], -1))).squeeze(-1)
        else:
            raise ValueError("mapping variation mode must be off, forced, or gated")
        # This is application of a dynamic Linear operator, evaluated without
        # materializing its [B,pred_len,feature_dim] weight tensor.
        low_rank_state = torch.einsum("bcf,rf->bcr", flat_state, self.right)
        operator_update = torch.einsum("bcr,pr->bcp", low_rank_state * alpha[:, None], self.left)
        full = base + gate[:, None, None] * operator_update
        return self.dropout(full), self.dropout(base), gate, operator_update, self.relative_weight_norm(alpha)
