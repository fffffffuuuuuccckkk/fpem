"""Standalone PatchTST Domain-aware invariant/variant decomposition."""
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from models.domain_iv import DomainRepresentationDecomposer, DomainDynamicLinearHead


class Transpose(nn.Module):
    def __init__(self, *dims): super().__init__(); self.dims = dims
    def forward(self, x): return x.transpose(*self.dims)


EXPERIMENTS = {
    "D0": ("none", "baseline"),
    "D1": ("contrastive", "off"),
    "D2": ("contrastive", "forced"),
    "D3": ("contrastive", "gated"),
    "D4": ("mutual_info", "off"),
    "D5": ("mutual_info", "forced"),
    "D6": ("mutual_info", "gated"),
}


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.pred_len = int(configs.pred_len)
        self.d_model = int(configs.d_model)
        self.experiment = str(getattr(configs, "domain_iv_experiment", "D6")).upper()
        if self.experiment not in EXPERIMENTS:
            raise ValueError("domain_iv_experiment must be D0...D6")
        self.domain_constraint, self.mapping_mode = EXPERIMENTS[self.experiment]
        constraint_override = str(getattr(configs, "domain_constraint", "") or "")
        if constraint_override:
            if constraint_override not in ("none", "contrastive", "mutual_info"):
                raise ValueError("domain_constraint must be none, contrastive, or mutual_info")
            self.domain_constraint = constraint_override
        self.patch_len = int(getattr(configs, "patch_len", patch_len))
        self.stride = int(getattr(configs, "domain_iv_stride", stride))
        self.patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, self.stride, configs.dropout)
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=False), self.d_model, configs.n_heads),
                self.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)
        ], norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2)))
        patch_num = int((configs.seq_len - self.patch_len) / self.stride + 2)
        feature_dim = self.d_model * patch_num
        self.decomposer = DomainRepresentationDecomposer(
            self.d_model, int(getattr(configs, "domain_iv_bottleneck", 64))
        )
        self.head = DomainDynamicLinearHead(
            feature_dim, self.pred_len, self.d_model,
            int(getattr(configs, "domain_iv_rank", 8)), configs.dropout,
        )
        self.environment_decoder = nn.Sequential(
            nn.LayerNorm(self.d_model), nn.Linear(self.d_model, int(getattr(configs, "domain_iv_env_num", 6)))
        )

    def encode(self, x):
        means = x.mean(1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        embedded, n_vars = self.patch_embedding((centered / stdev).permute(0, 2, 1))
        encoded, _ = self.encoder(embedded)
        hidden = encoded.reshape(-1, n_vars, encoded.shape[-2], encoded.shape[-1])
        return hidden, means, stdev

    def forward_components(self, x):
        hidden, means, stdev = self.encode(x)
        if self.experiment == "D0":
            z_inv, z_var = hidden, torch.zeros_like(hidden)
            mode = "off"
        else:
            z_inv, z_var = self.decomposer(hidden)
            mode = self.mapping_mode
        pooled_inv, pooled_var = z_inv.mean((1, 2)), z_var.mean((1, 2))
        flat = z_inv.reshape(z_inv.shape[0], z_inv.shape[1], -1)
        full, inv_only, gate, update, weight_ratio = self.head(flat, pooled_inv, pooled_var, mode)
        full = full.permute(0, 2, 1) * stdev[:, 0, None] + means[:, 0, None]
        inv_only = inv_only.permute(0, 2, 1) * stdev[:, 0, None] + means[:, 0, None]
        return {
            "prediction": full, "invariant_prediction": inv_only,
            "z_inv": pooled_inv, "z_var": pooled_var,
            "q_logits": self.environment_decoder(pooled_var),
            "gate": gate, "mapping_update": update,
            "deltaW_to_WU": weight_ratio,
        }

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forward_components(x_enc)["prediction"][:, -self.pred_len:]

    @torch.no_grad()
    def initial_mapping_difference(self, x):
        was_training = self.training; self.eval()
        hidden, _, _ = self.encode(x)
        z_inv, z_var = self.decomposer(hidden)
        pi, pv = z_inv.mean((1, 2)), z_var.mean((1, 2))
        flat = z_inv.reshape(z_inv.shape[0], z_inv.shape[1], -1)
        full, base, _, _, _ = self.head(flat, pi, pv, "forced")
        value = float((full - base).abs().mean())
        self.train(was_training)
        return value
