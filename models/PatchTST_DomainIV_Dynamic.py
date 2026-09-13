"""PatchTST DomainIV with state and Linear-operator variation."""
import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from models.domain_iv.decomposition import DomainRepresentationDecomposer
from models.domain_iv.feature_variation import FeatureVariationAdapter
from models.domain_iv.variation_mapping_head import VariationMappingHead


class Transpose(nn.Module):
    def __init__(self, *dims): super().__init__(); self.dims = dims
    def forward(self, x): return x.transpose(*self.dims)


VARIATIONS = {
    "V0": ("off", "off"),
    "V1": ("forced", "off"),
    "V2": ("gated", "off"),
    "V3": ("off", "forced"),
    "V4": ("off", "gated"),
    "V5": ("gated", "gated"),
}


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.pred_len, self.d_model = int(configs.pred_len), int(configs.d_model)
        self.experiment = str(getattr(configs, "domain_iv_variation", "V5")).upper()
        if self.experiment not in VARIATIONS: raise ValueError("domain_iv_variation must be V0...V5")
        self.feature_mode, self.mapping_mode = VARIATIONS[self.experiment]
        patch_len = int(getattr(configs, "patch_len", patch_len)); stride = int(getattr(configs, "domain_iv_stride", stride))
        self.patch_embedding = PatchEmbedding(self.d_model, patch_len, stride, stride, configs.dropout)
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=False), self.d_model, configs.n_heads),
                self.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)
        ], norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2)))
        patch_num = int((configs.seq_len - patch_len) / stride + 2)
        self.decomposer = DomainRepresentationDecomposer(self.d_model, int(getattr(configs, "domain_iv_bottleneck", 64)))
        self.feature_variation = FeatureVariationAdapter(self.d_model, int(getattr(configs, "domain_iv_bottleneck", 64)))
        self.head = VariationMappingHead(
            self.d_model * patch_num, self.pred_len, self.d_model,
            int(getattr(configs, "domain_iv_rank", 8)), configs.dropout,
        )

    def encode(self, x):
        means = x.mean(1, keepdim=True).detach(); centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        embedded, n_vars = self.patch_embedding((centered / stdev).permute(0, 2, 1))
        encoded, _ = self.encoder(embedded)
        return encoded.reshape(-1, n_vars, encoded.shape[-2], encoded.shape[-1]), means, stdev

    def environment_representation(self, x):
        hidden, _, _ = self.encode(x)
        _, z_var = self.decomposer(hidden)
        return z_var.mean((1, 2))

    def forward_components(self, x):
        hidden, means, stdev = self.encode(x)
        z_inv, z_var = self.decomposer(hidden)
        state, feature_gate, feature_applied = self.feature_variation(z_inv, z_var, self.feature_mode)
        pooled_inv, pooled_var = z_inv.mean((1, 2)), z_var.mean((1, 2))
        flat_state = state.reshape(state.shape[0], state.shape[1], -1)
        flat_inv = z_inv.reshape(z_inv.shape[0], z_inv.shape[1], -1)
        full, _, mapping_gate, mapping_update, weight_ratio = self.head(
            flat_state, pooled_inv, pooled_var, self.mapping_mode
        )
        inv_only = self.head.invariant(flat_inv)
        scale, shift = stdev[:, 0, None], means[:, 0, None]
        return {
            "prediction": full.permute(0, 2, 1) * scale + shift,
            "invariant_prediction": inv_only.permute(0, 2, 1) * scale + shift,
            "z_inv": pooled_inv, "z_var": pooled_var,
            "feature_gate": feature_gate, "mapping_gate": mapping_gate,
            "feature_variation": feature_applied, "z_inv_tokens": z_inv,
            "mapping_update": mapping_update, "deltaW_to_Winv": weight_ratio,
        }

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forward_components(x_enc)["prediction"][:, -self.pred_len:]

    @torch.no_grad()
    def identity_diagnostics(self, x):
        was_training = self.training; self.eval()
        hidden, _, _ = self.encode(x); z_inv, z_var = self.decomposer(hidden)
        state, _, _ = self.feature_variation(z_inv, z_var, "forced")
        pi, pv = z_inv.mean((1, 2)), z_var.mean((1, 2))
        flat = z_inv.reshape(z_inv.shape[0], z_inv.shape[1], -1)
        full, base, _, _, _ = self.head(flat, pi, pv, "forced")
        result = {"feature_identity_difference": float((state - z_inv).abs().mean()),
                  "mapping_identity_difference": float((full - base).abs().mean())}
        self.train(was_training); return result
