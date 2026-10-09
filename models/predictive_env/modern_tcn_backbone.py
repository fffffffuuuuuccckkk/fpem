"""ModernTCN feature encoder adapted from the upstream long-term forecasting model.

The stem, reparameterizable large/small depthwise convolution, two grouped
FFNs, and optional temporal downsampling follow ``models/ModernTCN.py`` in
``/data/OuXiaoyu/ModernTCN/ModernTCN-Long-term-forecasting``.  Forecasting
and instance normalization are owned by PredictiveEnvIV's shared interface.
"""

import torch
from torch import nn


class LargeSmallDepthwiseConv(nn.Module):
    def __init__(self, channels, large_size, small_size):
        super().__init__()
        if large_size % 2 != 1 or small_size % 2 != 1 or small_size > large_size:
            raise ValueError("ModernTCN kernels must be odd and small <= large")
        self.large = nn.Sequential(
            nn.Conv1d(channels, channels, large_size, padding=large_size // 2,
                      groups=channels, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.small = nn.Sequential(
            nn.Conv1d(channels, channels, small_size, padding=small_size // 2,
                      groups=channels, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return self.large(x) + self.small(x)


class ModernTCNBlock(nn.Module):
    def __init__(self, nvars, d_model, ffn_ratio, large_size, small_size, dropout):
        super().__init__()
        channels = nvars * d_model
        expanded = channels * ffn_ratio
        self.dw = LargeSmallDepthwiseConv(channels, large_size, small_size)
        self.norm = nn.BatchNorm1d(d_model)
        self.ffn1pw1 = nn.Conv1d(channels, expanded, 1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(expanded, channels, 1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(dropout)
        self.ffn1drop2 = nn.Dropout(dropout)
        self.ffn2pw1 = nn.Conv1d(channels, expanded, 1, groups=d_model)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(expanded, channels, 1, groups=d_model)
        self.ffn2drop1 = nn.Dropout(dropout)
        self.ffn2drop2 = nn.Dropout(dropout)

    def forward(self, x):
        batch, nvars, d_model, length = x.shape
        residual = x
        x = self.dw(x.reshape(batch, nvars * d_model, length))
        x = self.norm(x.reshape(batch * nvars, d_model, length))
        x = x.reshape(batch, nvars * d_model, length)
        x = self.ffn1drop2(self.ffn1pw2(
            self.ffn1act(self.ffn1drop1(self.ffn1pw1(x)))
        ))
        x = x.reshape(batch, nvars, d_model, length).permute(0, 2, 1, 3)
        x = self.ffn2drop2(self.ffn2pw2(
            self.ffn2act(self.ffn2drop1(
                self.ffn2pw1(x.reshape(batch, d_model * nvars, length))
            ))
        ))
        return residual + x.reshape(batch, d_model, nvars, length).permute(0, 2, 1, 3)


class ModernTCNBackbone(nn.Module):
    """Return upstream-style features [B,C,D,P] from normalized [B,C,L]."""

    def __init__(self, nvars, d_model=64, patch_size=8, patch_stride=4,
                 num_stages=1, ffn_ratio=1, large_size=51, small_size=5,
                 downsample_ratio=2, dropout=0.3):
        super().__init__()
        if num_stages < 1 or patch_size < patch_stride or downsample_ratio < 1:
            raise ValueError("Invalid ModernTCN patch/stage configuration")
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.downsample_ratio = int(downsample_ratio)
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, d_model, patch_size, stride=patch_stride),
                nn.BatchNorm1d(d_model),
            )
        ])
        self.stages = nn.ModuleList()
        for stage in range(num_stages):
            if stage:
                self.downsample_layers.append(nn.Sequential(
                    nn.BatchNorm1d(d_model),
                    nn.Conv1d(d_model, d_model, downsample_ratio,
                              stride=downsample_ratio),
                ))
            self.stages.append(nn.Sequential(ModernTCNBlock(
                nvars, d_model, ffn_ratio, large_size, small_size, dropout
            )))

    def output_tokens(self, seq_len):
        tokens = (int(seq_len) + self.patch_stride - 1) // self.patch_stride
        for _ in range(1, len(self.stages)):
            tokens = (tokens + self.downsample_ratio - 1) // self.downsample_ratio
        return tokens

    def forward(self, x):
        batch, nvars, length = x.shape
        x = x.unsqueeze(-2)
        for stage, (downsample, blocks) in enumerate(
            zip(self.downsample_layers, self.stages)
        ):
            _, _, d_model, length = x.shape
            x = x.reshape(batch * nvars, d_model, length)
            if stage == 0 and self.patch_size != self.patch_stride:
                pad = self.patch_size - self.patch_stride
                x = torch.cat((x, x[:, :, -1:].repeat(1, 1, pad)), dim=-1)
            elif stage and length % self.downsample_ratio:
                pad = self.downsample_ratio - length % self.downsample_ratio
                x = torch.cat((x, x[:, :, -pad:]), dim=-1)
            x = downsample(x)
            x = blocks(x.reshape(batch, nvars, x.shape[1], x.shape[2]))
        return x
