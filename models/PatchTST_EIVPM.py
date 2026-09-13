"""PatchTST with an opt-in environment-aware Pattern/Mapping plugin.

The original ``models/PatchTST.py`` is deliberately untouched.  E0 executes the
same PatchEmbedding -> Transformer Encoder -> Flatten Linear computation.
"""
from __future__ import annotations

import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from models.eivpm import EnvironmentPatternBank, EnvironmentMappingBank, PatternAdapter, DynamicMappingHead


class Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


ABLATIONS = {
    "E0": ("off", "off"),
    "E1": ("inv", "off"),
    "E2": ("forced", "off"),
    "E3": ("gated", "off"),
    "E4": ("off", "inv"),
    "E5": ("off", "forced"),
    "E6": ("off", "gated"),
    "E7": ("inv", "inv"),
    "E8": ("forced", "forced"),
    "E9": ("gated", "gated"),
}


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        if configs.task_name not in ("long_term_forecast", "short_term_forecast"):
            raise ValueError("PatchTST_EIVPM currently supports forecasting only")
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.d_model = int(configs.d_model)
        self.patch_len = int(getattr(configs, "patch_len", patch_len))
        self.stride = int(getattr(configs, "eivpm_stride", stride))
        self.ablation = str(getattr(configs, "eivpm_ablation", "E9")).upper()
        if self.ablation not in ABLATIONS:
            raise ValueError("eivpm_ablation must be one of E0...E9")
        padding = self.stride
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, padding, configs.dropout
        )
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    self.d_model,
                    configs.n_heads,
                ),
                self.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            ) for _ in range(configs.e_layers)
        ], norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2)))
        patch_num = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.feature_dim = self.d_model * patch_num
        pattern_count = int(getattr(configs, "eivpm_pattern_count", 32))
        bottleneck = int(getattr(configs, "eivpm_adapter_dim", 32))
        self.pattern_bank = EnvironmentPatternBank(self.d_model, pattern_count)
        self.mapping_bank = EnvironmentMappingBank(pattern_count, self.d_model)
        self.inv_adapter = PatternAdapter(self.d_model, bottleneck)
        self.var_adapter = PatternAdapter(self.d_model, bottleneck)
        self.pattern_gate = nn.Sequential(
            nn.LayerNorm(2 * self.d_model), nn.Linear(2 * self.d_model, 1)
        )
        nn.init.constant_(self.pattern_gate[-1].bias, -2.0)
        self.head = DynamicMappingHead(
            self.feature_dim,
            self.pred_len,
            self.d_model,
            rank=int(getattr(configs, "eivpm_mapping_rank", 8)),
            dropout=configs.dropout,
        )
        self.last_diagnostics = {}

    def encode(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = centered / stdev
        embedded, n_vars = self.patch_embedding(normalized.permute(0, 2, 1))
        encoded, _ = self.encoder(embedded)
        hidden = encoded.reshape(-1, n_vars, encoded.shape[-2], encoded.shape[-1])
        return hidden, means, stdev

    def _plugin(self, hidden):
        pattern_mode, mapping_mode = ABLATIONS[self.ablation]
        inv_context, var_context, responsibility = self.pattern_bank.query(hidden)
        inv_map, var_map, inv_strength = self.mapping_bank.query(
            responsibility, self.pattern_bank.prototypes
        )
        inv_delta = self.inv_adapter(hidden, inv_context) if pattern_mode != "off" else torch.zeros_like(hidden)
        var_delta = self.var_adapter(hidden, var_context) if pattern_mode in ("forced", "gated") else torch.zeros_like(hidden)
        if pattern_mode == "forced":
            pattern_gate = torch.ones(*hidden.shape[:-1], 1, device=hidden.device)
        elif pattern_mode == "gated":
            pattern_gate = torch.sigmoid(self.pattern_gate(torch.cat([hidden, var_context], -1)))
        else:
            pattern_gate = torch.zeros(*hidden.shape[:-1], 1, device=hidden.device)
        adjusted = hidden + inv_delta + pattern_gate * var_delta
        flat = adjusted.reshape(adjusted.shape[0], adjusted.shape[1], -1)
        head_mode = mapping_mode if mapping_mode in ("forced", "gated") else "off"
        prediction, mapping_gate, mapping_update = self.head(flat, inv_map, var_map, head_mode)
        delta_weight_ratio = self.head.relative_update_norm(self.head.mapping_encoder(var_map))
        self.last_diagnostics = {
            "gP": pattern_gate.mean(),
            "gM": mapping_gate.mean(),
            "pattern_correction_ratio": (inv_delta + pattern_gate * var_delta).norm() / hidden.norm().clamp_min(1e-8),
            "mapping_update_ratio": mapping_update.norm() / prediction.norm().clamp_min(1e-8),
            "deltaW_to_WU": delta_weight_ratio.mean(),
            "deltaW_to_WU_values": delta_weight_ratio,
            "invariant_mapping_strength": inv_strength.mean(),
            "invariant_mapping_strength_values": inv_strength,
            "pattern_gate_values": pattern_gate.detach().reshape(-1),
            "mapping_gate_values": mapping_gate.detach().reshape(-1),
        }
        return prediction, inv_strength

    def forecast(self, x_enc):
        hidden, means, stdev = self.encode(x_enc)
        prediction, _ = self._plugin(hidden)
        prediction = prediction.permute(0, 2, 1)
        return prediction * stdev[:, 0].unsqueeze(1) + means[:, 0].unsqueeze(1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc)[:, -self.pred_len:]

    def configure_stage(self, stage: str):
        stage = stage.upper()
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        pattern_mode, mapping_mode = ABLATIONS[self.ablation]
        if stage == "A":
            for module in (self.var_adapter, self.pattern_gate, self.head.mapping_encoder, self.head.gate):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            self.head.left.requires_grad_(False)
            self.head.right.requires_grad_(False)
        elif stage == "B":
            # Universal mapping is the Stage-A invariant anchor.
            for parameter in self.head.universal.parameters():
                parameter.requires_grad_(False)
            if pattern_mode not in ("forced", "gated"):
                for module in (self.var_adapter, self.pattern_gate):
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
            if mapping_mode not in ("forced", "gated"):
                for module in (self.head.mapping_encoder, self.head.gate):
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
                self.head.left.requires_grad_(False)
                self.head.right.requires_grad_(False)
        else:
            raise ValueError("stage must be A or B")

    @torch.no_grad()
    def initial_identity_differences(self, x):
        was_training = self.training
        self.eval()
        hidden, _, _ = self.encode(x)
        inv, var, resp = self.pattern_bank.query(hidden)
        plugin_hidden = hidden + self.inv_adapter(hidden, inv) + self.var_adapter(hidden, var)
        flat = hidden.reshape(hidden.shape[0], hidden.shape[1], -1)
        inv_map, var_map, _ = self.mapping_bank.query(resp, self.pattern_bank.prototypes)
        base = self.head.universal(flat)
        dynamic, _, _ = self.head(flat, inv_map, var_map, "forced")
        result = {
            "initial_pattern_abs_diff": float((plugin_hidden - hidden).abs().mean()),
            "initial_prediction_abs_diff": float((dynamic - base).abs().mean()),
        }
        self.train(was_training)
        return result
