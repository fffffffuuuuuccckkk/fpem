"""PatchTST plugin for predictive-conflict invariant/variant forecasting.

Mapping variation was intentionally removed.  The remaining plugin studies
dynamic EIIL-style environments, representation decomposition, and optional
feature/state variation only.
"""

import copy

import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer
from models.domain_iv.decomposition import DomainRepresentationDecomposer
from models.domain_iv.feature_variation import FeatureVariationAdapter
from models.predictive_env.classification_constraint import (
    EnvironmentClassificationConstraint,
)
from models.predictive_env.signed_gate_decomposition import (
    ComplementaryGateDecomposer,
    SignedGateDecomposer,
)
from models.predictive_env.direct_variant_fusion import DirectGatedVariantFusion
from models.predictive_env.future_var_consistency import FutureVariantPredictor
from models.predictive_env.conditional_variant_predictor import (
    ConditionalVariantPredictor,
)


class Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, value):
        return value.transpose(*self.dims)


# A5--A7 (Mapping Variation) have been retired from this method.
ABLATIONS = {
    "A0": ("baseline", "off"),
    "A1": ("core", "off"),
    "A2": ("representation", "off"),
    "A3": ("representation", "forced"),
    "A4": ("representation", "gated"),
}


class Model(nn.Module):
    def __init__(self, cfg, patch_len=16, stride=8):
        super().__init__()
        self.pred_len = int(cfg.pred_len)
        self.d_model = int(cfg.d_model)
        self.experiment = str(
            getattr(cfg, "predictive_env_ablation", "A4")
        ).upper()
        if self.experiment not in ABLATIONS:
            raise ValueError(
                f"Unknown ablation {self.experiment}; Mapping ablations A5-A7 "
                f"were removed. Choose one of {tuple(ABLATIONS)}"
            )
        self.level, self.feature_mode = ABLATIONS[self.experiment]
        self.decomposition_type = str(
            getattr(cfg, "decomposition_type", "projection")
        )
        if self.decomposition_type not in (
            "projection",
            "signed_gate",
            "complementary_gate",
        ):
            raise ValueError(
                "decomposition_type must be projection, signed_gate, or "
                "complementary_gate"
            )
        self.representation_constraint = str(
            getattr(cfg, "representation_constraint", "contrastive")
        )
        if self.representation_constraint not in ("contrastive", "classification"):
            raise ValueError(
                "representation_constraint must be contrastive or classification"
            )
        self.variant_fusion_mode = str(
            getattr(cfg, "variant_fusion_mode", "legacy")
        )
        if self.variant_fusion_mode not in ("legacy", "off", "direct_gated"):
            raise ValueError(
                "variant_fusion_mode must be legacy, off, or direct_gated"
            )
        self.fusion_scale_calibration = str(
            getattr(cfg, "fusion_scale_calibration", "none")
        )
        if self.fusion_scale_calibration not in ("none", "rms"):
            raise ValueError("fusion_scale_calibration must be none or rms")
        patch_len = int(getattr(cfg, "patch_len", patch_len))
        stride = int(getattr(cfg, "predictive_env_stride", stride))
        self.patch_embedding = PatchEmbedding(
            self.d_model, patch_len, stride, stride, cfg.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            cfg.factor,
                            attention_dropout=cfg.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        cfg.n_heads,
                    ),
                    self.d_model,
                    cfg.d_ff,
                    dropout=cfg.dropout,
                    activation=cfg.activation,
                )
                for _ in range(cfg.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2)
            ),
        )
        patch_num = int((cfg.seq_len - patch_len) / stride + 2)
        bottleneck = int(getattr(cfg, "predictive_env_bottleneck", 64))
        self.feature_variation = FeatureVariationAdapter(self.d_model, bottleneck)
        self.direct_variant_fusion = DirectGatedVariantFusion(self.d_model)
        self.environment_classification = EnvironmentClassificationConstraint(
            self.d_model,
            int(getattr(cfg, "predictive_env_num", 6)),
            bottleneck,
            float(getattr(cfg, "predictive_env_grl_weight", 1.0)),
        )
        self.head_flatten = nn.Flatten(start_dim=-2)
        self.head_linear = nn.Linear(self.d_model * patch_num, self.pred_len)
        self.head_dropout = nn.Dropout(cfg.dropout)
        # This copy is synchronized once after loading the shared pretrained
        # checkpoint and remains frozen throughout representation learning.
        self.h_forecast_head = copy.deepcopy(self.head_linear)
        self.h_forecast_head.requires_grad_(False)
        # Construct decomposition last so its architecture-dependent parameter
        # count cannot perturb initialization of the shared adapter/classifiers.
        if self.decomposition_type == "projection":
            self.decomposer = DomainRepresentationDecomposer(
                self.d_model, bottleneck
            )
        elif self.decomposition_type == "signed_gate":
            self.decomposer = SignedGateDecomposer(self.d_model, bottleneck)
        else:
            self.decomposer = ComplementaryGateDecomposer(
                self.d_model, bottleneck
            )
        # Constructed after every forecasting/decomposition component so adding
        # this training-only auxiliary head does not perturb their seeded init.
        self.future_var_predictor = FutureVariantPredictor(self.d_model)
        # Isolate auxiliary-head initialization as well: with lambda=0, adding
        # this head must not advance the main training/dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(314159)
            self.variant_conditional_predictor = ConditionalVariantPredictor(
                self.d_model, self.pred_len
            )

    def initialize_h_forecast_head(self):
        self.h_forecast_head.load_state_dict(self.head_linear.state_dict())
        self.h_forecast_head.requires_grad_(False)
        self.h_forecast_head.eval()

    def encode(self, x):
        mean = x.mean(1, keepdim=True).detach()
        centered = x - mean
        std = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        embedded, nvars = self.patch_embedding((centered / std).permute(0, 2, 1))
        encoded, _ = self.encoder(embedded)
        tokens = encoded.reshape(
            -1, nvars, encoded.shape[-2], encoded.shape[-1]
        )
        return tokens, mean, std

    def forecast_head(self, tokens):
        # Match PatchTST exactly: [B,C,P,D] -> [B,C,D,P] before flattening.
        features = self.head_flatten(tokens.permute(0, 1, 3, 2))
        return self.head_dropout(self.head_linear(features))

    def frozen_h_head(self, hidden):
        features = self.head_flatten(hidden.permute(0, 1, 3, 2))
        return self.h_forecast_head(features)

    def decompose(self, hidden):
        if self.level in ("baseline", "core"):
            return hidden, torch.zeros_like(hidden), torch.zeros_like(hidden)
        if self.decomposition_type in ("signed_gate", "complementary_gate"):
            return self.decomposer(hidden)
        z_inv, z_var = self.decomposer(hidden)
        return z_inv, z_var, torch.zeros_like(hidden)

    def detached_decomposition(self, hidden):
        """Environment-supervised decomposition with no Encoder gradient."""
        return self.decompose(hidden.detach())

    def conditional_predictions(self, z_inv, z_var, scale, shift):
        paired, shuffled, permutation = (
            self.variant_conditional_predictor.paired_and_shuffled(z_inv, z_var)
        )
        return (
            paired * scale + shift,
            shuffled * scale + shift,
            permutation,
        )

    def detached_environment_components(self, forward_output):
        z_inv, z_var, gate = self.detached_decomposition(
            forward_output["hidden_tokens"]
        )
        paired, shuffled, permutation = self.conditional_predictions(
            z_inv,
            z_var,
            forward_output["input_scale"],
            forward_output["input_shift"],
        )
        adapted, feature_gate, feature_delta, fusion_scale = self.fuse_tokens(
            z_inv, z_var, forward_output["hidden_tokens"].detach()
        )
        prediction = self.forecast_head(adapted)
        invariant_prediction = self.forecast_head(z_inv)
        return {
            "prediction": (
                prediction.permute(0, 2, 1) * forward_output["input_scale"]
                + forward_output["input_shift"]
            ),
            "invariant_prediction": (
                invariant_prediction.permute(0, 2, 1)
                * forward_output["input_scale"]
                + forward_output["input_shift"]
            ),
            "z_inv": z_inv.mean((1, 2)),
            "z_var": z_var.mean((1, 2)),
            "z_inv_tokens": z_inv,
            "z_var_tokens": z_var,
            "decomposition_gate": gate,
            "conditional_variant_prediction": paired,
            "conditional_shuffled_prediction": shuffled,
            "conditional_permutation": permutation,
            "feature_gate": feature_gate,
            "feature_variation": feature_delta,
            "final_tokens": adapted,
            "fusion_scale": fusion_scale,
        }

    def fuse_tokens(self, z_inv, z_var, hidden):
        if self.variant_fusion_mode == "legacy":
            adapted, feature_gate, feature_delta = self.feature_variation(
                z_inv, z_var, self.feature_mode
            )
        elif self.variant_fusion_mode == "direct_gated":
            adapted, feature_gate, feature_delta = self.direct_variant_fusion(
                z_inv, z_var
            )
        else:
            feature_gate = torch.zeros(
                *z_var.shape[:-1], 1, device=z_var.device, dtype=z_var.dtype
            )
            feature_delta = torch.zeros_like(z_var)
            adapted = z_inv
        z_raw = adapted
        if self.fusion_scale_calibration == "rms":
            hidden_rms = hidden.square().mean((-2, -1), keepdim=True).sqrt().detach()
            raw_rms = z_raw.square().mean((-2, -1), keepdim=True).sqrt().detach()
            fusion_scale = hidden_rms / (raw_rms + 1e-8)
            adapted = z_raw * fusion_scale
        else:
            fusion_scale = torch.ones_like(
                hidden.square().mean((-2, -1), keepdim=True)
            )
        return adapted, feature_gate, feature_delta, fusion_scale

    @torch.no_grad()
    def future_variant_target(self, future_x):
        """Encode a TRAIN-only future window as a stop-gradient teacher.

        Teacher modules are temporarily put in evaluation mode so dropout and
        BatchNorm buffers cannot make the auxiliary branch mutate the model.
        The exact same module instances and current parameters are used.
        """
        modules = (self.patch_embedding, self.encoder, self.decomposer)
        training_states = tuple(module.training for module in modules)
        try:
            for module in modules:
                module.eval()
            hidden, _, _ = self.encode(future_x)
            _, z_var, _ = self.decompose(hidden)
            return z_var.mean((1, 2)).detach()
        finally:
            for module, was_training in zip(modules, training_states):
                module.train(was_training)

    def forward_components(self, x):
        hidden, mean, std = self.encode(x)
        z_inv, z_var, decomposition_gate = self.decompose(hidden)
        adapted, feature_gate, feature_delta, fusion_scale = self.fuse_tokens(
            z_inv, z_var, hidden
        )
        full = self.forecast_head(adapted)
        invariant = self.forecast_head(z_inv)
        h_prediction = self.frozen_h_head(hidden)
        scale, shift = std[:, 0, None], mean[:, 0, None]
        conditional, conditional_shuffled, conditional_permutation = (
            self.conditional_predictions(z_inv, z_var, scale, shift)
        )
        return {
            "prediction": full.permute(0, 2, 1) * scale + shift,
            "invariant_prediction": invariant.permute(0, 2, 1) * scale + shift,
            "h_prediction": h_prediction.permute(0, 2, 1) * scale + shift,
            "conditional_variant_prediction": conditional,
            "conditional_shuffled_prediction": conditional_shuffled,
            "conditional_permutation": conditional_permutation,
            "z_inv": z_inv.mean((1, 2)),
            "z_var": z_var.mean((1, 2)),
            "z_inv_tokens": z_inv,
            "z_var_tokens": z_var,
            "feature_gate": feature_gate,
            "feature_variation": feature_delta,
            "decomposition_gate": decomposition_gate,
            "hidden_tokens": hidden,
            "final_tokens": adapted,
            "fusion_scale": fusion_scale,
            "input_scale": scale,
            "input_shift": shift,
        }

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
    ):
        return self.forward_components(x_enc)["prediction"]

    def environment_classification_loss(self, z_inv, z_var, target_q):
        return self.environment_classification(z_inv, z_var, target_q.detach())

    def environment_logits(self, z_inv, z_var):
        return self.environment_classification.logits(
            z_inv, z_var, reverse_invariant=False
        )

    @torch.no_grad()
    def identity_diagnostics(self, x):
        was_training = self.training
        self.eval()
        hidden, _, _ = self.encode(x)
        if self.decomposition_type in ("signed_gate", "complementary_gate"):
            z_inv, z_var, _ = self.decomposer(hidden)
        else:
            z_inv, z_var = self.decomposer(hidden)
        adapted, _, _ = self.feature_variation(z_inv, z_var, "forced")
        result = {"feature": float((adapted - z_inv).abs().mean())}
        self.train(was_training)
        return result
