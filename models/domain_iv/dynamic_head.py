import torch
from torch import nn


class DomainDynamicLinearHead(nn.Module):
    """W_U plus an implicit batch-conditioned A diag(z_var) B correction."""
    def __init__(self, feature_dim, pred_len, d_model, rank=8, dropout=0.0):
        super().__init__()
        self.universal = nn.Linear(feature_dim, pred_len)
        self.variant_code = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, rank), nn.Tanh()
        )
        self.left = nn.Parameter(torch.zeros(pred_len, rank))
        self.right = nn.Parameter(torch.empty(rank, feature_dim))
        nn.init.xavier_uniform_(self.right)
        self.gate = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.dropout = nn.Dropout(dropout)

    def relative_weight_norm(self, code):
        gram_a = self.left.t() @ self.left
        gram_b = self.right @ self.right.t()
        # The low-rank update is exactly zero at initialization. A positive
        # numerical floor avoids the undefined sqrt gradient at that point.
        squared = torch.einsum("br,rs,bs->b", code, gram_a * gram_b, code).clamp_min(1e-12)
        return squared.sqrt() / self.universal.weight.norm().clamp_min(1e-8)

    def forward(self, flat_invariant, pooled_invariant, pooled_variant, mode="gated"):
        base = self.universal(flat_invariant)
        code = self.variant_code(pooled_variant)
        if mode == "off":
            gate = torch.zeros(flat_invariant.shape[0], device=flat_invariant.device)
        elif mode == "forced":
            gate = torch.ones(flat_invariant.shape[0], device=flat_invariant.device)
        elif mode == "gated":
            gate = torch.sigmoid(self.gate(torch.cat([pooled_invariant, pooled_variant], -1))).squeeze(-1)
        else:
            raise ValueError("mapping mode must be off, forced, or gated")
        # [B,C,F] -> [B,C,R] -> [B,C,P], never instantiate [B,P,F].
        projected = torch.einsum("bcf,rf->bcr", flat_invariant, self.right)
        update = torch.einsum("bcr,pr->bcp", projected * code[:, None], self.left)
        full = base + gate[:, None, None] * update
        return self.dropout(full), self.dropout(base), gate, update, self.relative_weight_norm(code)
