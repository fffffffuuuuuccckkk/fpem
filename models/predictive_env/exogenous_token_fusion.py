"""Target/exogenous token fusion modules.

``PatchExogenousTokenFusion`` is retained for historical checkpoints.  New
calendar-token experiments use ``GlobalExogenousTokenFusion`` so every target
patch can attend to exogenous tokens that summarize the complete input window.
"""

import torch
from torch import nn


class PatchExogenousTokenFusion(nn.Module):
    """Attend from each target token to itself and aligned exogenous tokens."""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x_tokens, z_tokens):
        if x_tokens.ndim != 4 or z_tokens.ndim != 4:
            raise ValueError("tokens must have shape [B, variables, patches, D]")
        b, c, p, d = x_tokens.shape
        if (z_tokens.shape[0] != b or z_tokens.shape[2:] != (p, d)
                or z_tokens.shape[1] == 0):
            raise ValueError("exogenous tokens must match batch, patches, and width")
        query = x_tokens.permute(0, 2, 1, 3).reshape(b * p, c, d)
        exog = z_tokens.permute(0, 2, 1, 3).reshape(b * p, z_tokens.shape[1], d)
        # Each target attends to its own token plus all exogenous tokens.
        query = query.reshape(b * p * c, 1, d)
        exog = exog.unsqueeze(1).expand(-1, c, -1, -1).reshape(
            b * p * c, z_tokens.shape[1], d
        )
        candidates = torch.cat((query, exog), dim=1)
        update, _ = self.attention(query, candidates, candidates, need_weights=False)
        update = update.reshape(b, p, c, d).permute(0, 2, 1, 3)
        return x_tokens + self.gate * update


class GlobalExogenousTokenFusion(nn.Module):
    """Fuse full-window variable tokens into channel-independent patches.

    The target history and every exogenous feature share one ``Linear(L, D)``
    embedding.  Each target variable is then processed independently together
    with a copy of all exogenous tokens, so this module introduces no direct
    target-to-target interaction.
    """

    def __init__(self, seq_len, d_model, n_heads=4, gate_init=0.0):
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        if self.seq_len < 1 or self.d_model < 1:
            raise ValueError("seq_len and d_model must be positive")
        self.global_embedding = nn.Linear(self.seq_len, self.d_model)
        self.global_exog_attention = nn.MultiheadAttention(
            self.d_model, int(n_heads), batch_first=True
        )
        self.global_norm1 = nn.LayerNorm(self.d_model)
        self.global_ffn = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Linear(4 * self.d_model, self.d_model),
        )
        self.global_norm2 = nn.LayerNorm(self.d_model)
        self.patch_exog_attention = nn.MultiheadAttention(
            self.d_model, int(n_heads), batch_first=True
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x_tokens, normalized, exog):
        if x_tokens.ndim != 4:
            raise ValueError("x_tokens must have shape [B,C,P,D]")
        if normalized.ndim != 3:
            raise ValueError("normalized must have shape [B,L,C]")
        if exog.ndim != 3:
            raise ValueError("exog must have shape [B,L,F]")
        batch, nvars, patches, width = x_tokens.shape
        if width != self.d_model:
            raise ValueError("x_tokens width does not match d_model")
        if normalized.shape != (batch, self.seq_len, nvars):
            raise ValueError("normalized history does not align with x_tokens")
        if exog.shape[0] != batch or exog.shape[1] != self.seq_len:
            raise ValueError("exog history must align with normalized history")
        if exog.shape[2] < 1:
            raise ValueError("at least one exogenous feature is required")
        if not torch.isfinite(exog).all():
            raise ValueError("exog contains NaN or Inf")

        exog = exog.to(device=normalized.device, dtype=normalized.dtype)
        global_x = self.global_embedding(normalized.transpose(1, 2))
        global_z = self.global_embedding(exog.transpose(1, 2))
        expanded_z = global_z.unsqueeze(1).expand(-1, nvars, -1, -1)
        memory = torch.cat((global_x.unsqueeze(2), expanded_z), dim=2)
        memory = memory.reshape(batch * nvars, 1 + exog.shape[2], width)

        attended, _ = self.global_exog_attention(
            memory, memory, memory, need_weights=False
        )
        encoded = self.global_norm1(memory + attended)
        encoded = self.global_norm2(encoded + self.global_ffn(encoded))

        query = x_tokens.reshape(batch * nvars, patches, width)
        update, _ = self.patch_exog_attention(
            query, encoded, encoded, need_weights=False
        )
        fused = query + self.gate * update
        return fused.reshape(batch, nvars, patches, width)
