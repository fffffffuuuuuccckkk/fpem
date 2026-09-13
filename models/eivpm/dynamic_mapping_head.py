import torch
from torch import nn


class DynamicMappingHead(nn.Module):
    """Shared universal head plus memory-efficient sample-conditioned low rank update."""
    def __init__(self, feature_dim: int, pred_len: int, context_dim: int, rank: int = 8, dropout=0.0):
        super().__init__()
        self.feature_dim, self.pred_len, self.rank = feature_dim, pred_len, rank
        self.universal = nn.Linear(feature_dim, pred_len)
        self.mapping_encoder = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, rank), nn.Tanh())
        self.left = nn.Parameter(torch.zeros(pred_len, rank))
        self.right = nn.Parameter(torch.empty(rank, feature_dim))
        nn.init.xavier_uniform_(self.right)
        self.gate = nn.Sequential(
            nn.LayerNorm(context_dim * 2), nn.Linear(context_dim * 2, 1)
        )
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, flat_hidden, inv_context, var_context, variant_mode="off"):
        base = self.universal(flat_hidden)
        z = self.mapping_encoder(var_context)
        if variant_mode == "off":
            gate = torch.zeros(flat_hidden.shape[0], 1, device=flat_hidden.device)
        elif variant_mode == "forced":
            gate = torch.ones(flat_hidden.shape[0], 1, device=flat_hidden.device)
        else:
            gate = torch.sigmoid(self.gate(torch.cat([inv_context, var_context], -1)))
        projected = torch.einsum("bcf,rf->bcr", flat_hidden, self.right)
        update = torch.einsum("bcr,pr->bcp", projected * z[:, None, :], self.left)
        output = base + gate[:, None, :] * update
        return self.dropout(output), gate.squeeze(-1), update

    def relative_update_norm(self, z):
        gram_left = self.left.t() @ self.left
        gram_right = self.right @ self.right.t()
        update_sq = torch.einsum("br,rs,bs->b", z, gram_left * gram_right, z).clamp_min(0.0)
        return update_sq.sqrt() / self.universal.weight.norm().clamp_min(1e-8)
