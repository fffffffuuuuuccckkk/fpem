"""History-only additive calendar embedding for a frozen FPEM A0.

The adapter uses a forward hook at the first token-producing layer.  The
original model and checkpoint are never modified.  At zero initialization its
forecast is bitwise identical to the eval-mode A0 forecast.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ExogenousEmbedding(nn.Module):
    def __init__(self, base: nn.Module, backbone: str, seq_len: int,
                 exog_dim: int, hidden_dim: int):
        super().__init__()
        if backbone not in {"patchtst", "itransformer", "moderntcn"}:
            raise ValueError(f"unsupported backbone: {backbone}")
        if exog_dim <= 0:
            raise ValueError("the dataset has no calendar features")
        self.base = base
        self.backbone = backbone
        self.seq_len = int(seq_len)
        self.exog_dim = int(exog_dim)
        input_dim = seq_len * exog_dim if backbone == "itransformer" else exog_dim
        self.proj = nn.Linear(input_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self._calendar = None
        self._channels = None
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        if backbone == "patchtst":
            module = base.patch_embedding
        elif backbone == "itransformer":
            module = base.inverted_embedding
        else:
            module = base.modern_tcn_backbone.downsample_layers[0]
        self._hook = module.register_forward_hook(self._add_embedding)

    def _patch_calendar(self, length: int, patch: int, stride: int,
                        padding: int) -> torch.Tensor:
        mark = self._calendar.transpose(1, 2)
        if padding:
            mark = F.pad(mark, (0, padding), mode="replicate")
        mark = mark.unfold(-1, patch, stride).mean(-1).transpose(1, 2)
        if mark.shape[1] != length:
            raise RuntimeError(f"calendar/token alignment: {mark.shape[1]} != {length}")
        return mark

    def _add_embedding(self, module, inputs, output):
        if self._calendar is None:
            return output
        batch, channels = self._calendar.shape[0], self._channels
        if self.backbone == "patchtst":
            tokens, nvars = output
            if nvars != channels or tokens.shape[0] != batch * channels:
                raise RuntimeError("PatchTST channel alignment failed")
            mark = self._patch_calendar(tokens.shape[1], module.patch_len,
                                        module.stride, module.stride)
            addition = self.proj(mark).repeat_interleave(channels, dim=0)
            return tokens + addition, nvars
        if self.backbone == "moderntcn":
            if output.shape[0] != batch * channels:
                raise RuntimeError("ModernTCN channel alignment failed")
            stem = self.base.modern_tcn_backbone
            mark = self._patch_calendar(output.shape[-1], stem.patch_size,
                                        stem.patch_stride,
                                        stem.patch_size - stem.patch_stride)
            addition = self.proj(mark).transpose(1, 2).repeat_interleave(channels, dim=0)
            return output + addition
        if output.shape[0] != batch or output.shape[1] < channels:
            raise RuntimeError("iTransformer variable-token alignment failed")
        # The current A0 calls inverted_embedding(normalized, None): calendar
        # tokens are absent, so this does not duplicate an existing time input.
        addition = self.proj(self._calendar.reshape(batch, -1)).unsqueeze(1)
        return torch.cat((output[:, :channels] + addition,
                          output[:, channels:]), dim=1)

    def encode(self, x: torch.Tensor, calendar: torch.Tensor,
               cycle_index: torch.Tensor | None = None):
        if calendar.shape != (x.shape[0], self.seq_len, self.exog_dim):
            raise ValueError(f"calendar shape {tuple(calendar.shape)} is incompatible")
        if self._calendar is not None:
            raise RuntimeError("nested ExogenousEmbedding forward is unsupported")
        self._calendar = calendar
        self._channels = x.shape[-1]
        try:
            return self.base.encode(x, cycle_index)
        finally:
            self._calendar = None
            self._channels = None

    def forward(self, x: torch.Tensor, calendar: torch.Tensor,
                cycle_index: torch.Tensor | None = None):
        hidden, mean, std = self.encode(x, calendar, cycle_index)
        prediction = self.base.frozen_h_head(hidden, cycle_index).transpose(1, 2)
        return prediction * std + mean

