"""Backbone-agnostic predictive-conflict invariant/variant forecasting.

Mapping variation was intentionally removed.  The remaining plugin studies
dynamic EIIL-style environments, representation decomposition, and optional
feature/state variation only.  PatchTST remains the legacy default; iTransformer
and the official CycleNet/MLP residual-cycle backbone expose the same token and
forecast-head contract so every PredictiveEnvIV loss is kept identical.
"""

import copy

import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.Embed import DataEmbedding_inverted
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
from models.predictive_env.film_variant_fusion import FilmVariantFusion
from models.predictive_env.y_film_decay_fusion import YFilmDecayFusion
from models.predictive_env.horizon_future_variant import HorizonFutureVariant
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


class RecurrentCycle(nn.Module):
    """Learnable periodic queue used by CycleNet's Residual Cycle Forecasting."""

    def __init__(self, cycle_len, channel_size):
        super().__init__()
        self.cycle_len = int(cycle_len)
        self.data = nn.Parameter(torch.zeros(self.cycle_len, int(channel_size)))

    def forward(self, index, length):
        index = index.to(device=self.data.device, dtype=torch.long).reshape(-1, 1)
        positions = torch.arange(int(length), device=self.data.device).reshape(1, -1)
        return self.data[(index + positions) % self.cycle_len]


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
        self.seq_len = int(cfg.seq_len)
        self.enc_in = int(getattr(cfg, "enc_in", 1))
        self.backbone = str(
            getattr(cfg, "predictive_env_backbone", "patchtst")
        ).lower()
        if self.backbone not in ("patchtst", "itransformer", "cyclenet"):
            raise ValueError(
                "predictive_env_backbone must be patchtst, itransformer, or cyclenet"
            )
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
        if self.variant_fusion_mode not in (
            "legacy",
            "off",
            "inv_only",
            "direct_gated",
            "film",
            "film_decay_reg",
            "y_film_decay",
            "horizon_future_var",
        ):
            raise ValueError(
                "variant_fusion_mode must be legacy, off, inv_only, direct_gated, film, "
                "film_decay_reg, y_film_decay, or horizon_future_var"
            )
        self.variant_fusion_gate_type = str(
            getattr(cfg, "variant_fusion_gate_type", "token")
        )
        if self.variant_fusion_gate_type not in ("token", "feature"):
            raise ValueError(
                "variant_fusion_gate_type must be token or feature"
            )
        self.fusion_scale_calibration = str(
            getattr(cfg, "fusion_scale_calibration", "none")
        )
        if self.fusion_scale_calibration not in ("none", "rms"):
            raise ValueError("fusion_scale_calibration must be none or rms")
        patch_len = int(getattr(cfg, "patch_len", patch_len))
        stride = int(getattr(cfg, "predictive_env_stride", stride))
        self.patch_embedding = None
        self.inverted_embedding = None
        self.cycle_queue = None
        self.cycle_input_projection = None
        self.cycle_activation = None
        def make_transformer_encoder(norm_layer):
            return Encoder(
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
                norm_layer=norm_layer,
            )

        if self.backbone == "patchtst":
            self.patch_embedding = PatchEmbedding(
                self.d_model, patch_len, stride, stride, cfg.dropout
            )
            self.encoder = make_transformer_encoder(
                nn.Sequential(
                    Transpose(1, 2),
                    nn.BatchNorm1d(self.d_model),
                    Transpose(1, 2),
                )
            )
            token_num = int((cfg.seq_len - patch_len) / stride + 2)
        elif self.backbone == "itransformer":
            self.inverted_embedding = DataEmbedding_inverted(
                cfg.seq_len,
                self.d_model,
                getattr(cfg, "embed", "timeF"),
                getattr(cfg, "freq", "h"),
                cfg.dropout,
            )
            self.encoder = make_transformer_encoder(nn.LayerNorm(self.d_model))
            token_num = 1
        else:
            # Official CycleNet MLP residual backbone: the first layer defines H,
            # while the shared forecast head is the second layer.  This is the
            # minimal CycleNet form that exposes a meaningful d_model feature.
            self.cycle_len = int(getattr(cfg, "cyclenet_cycle_len", 24))
            self.cycle_queue = RecurrentCycle(self.cycle_len, self.enc_in)
            self.cycle_input_projection = nn.Linear(self.seq_len, self.d_model)
            self.cycle_activation = nn.ReLU()
            self.encoder = None
            token_num = 1
        bottleneck = int(getattr(cfg, "predictive_env_bottleneck", 64))
        self.feature_variation = FeatureVariationAdapter(self.d_model, bottleneck)
        self.direct_variant_fusion = DirectGatedVariantFusion(
            self.d_model, self.variant_fusion_gate_type
        )
        self.film_variant_fusion = None
        if self.variant_fusion_mode in ("film", "film_decay_reg"):
            # Do not advance the initialization RNG of classifiers/decomposer;
            # direct_gated and FiLM therefore remain fair ablations.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(271828)
                self.film_variant_fusion = FilmVariantFusion(
                    self.d_model,
                    bottleneck,
                    float(getattr(cfg, "film_gamma_scale", 0.1)),
                    float(getattr(cfg, "film_beta_scale", 0.1)),
                )
        self.y_film_decay_fusion = None
        if self.variant_fusion_mode == "y_film_decay":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(161803)
                self.y_film_decay_fusion = YFilmDecayFusion(
                    self.d_model,
                    self.pred_len,
                    bottleneck,
                    float(getattr(cfg, "y_film_gamma_scale", 0.1)),
                    float(getattr(cfg, "y_film_beta_scale", 0.1)),
                    float(getattr(cfg, "y_film_decay_bias", -4.0)),
                )
        self.horizon_future_variant = None
        if self.variant_fusion_mode == "horizon_future_var":
            # Keep all established decomposer/classifier initializations fair.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(141421)
                self.horizon_future_variant = HorizonFutureVariant(
                    self.d_model,
                    self.pred_len,
                    int(getattr(cfg, "horizon_future_dim", 32)),
                    bottleneck,
                    float(getattr(cfg, "horizon_future_gamma_scale", 0.1)),
                    float(getattr(cfg, "horizon_future_beta_scale", 0.1)),
                    int(getattr(cfg, "future_patch_len", 16)),
                    (
                        int(getattr(cfg, "predictive_env_num", 0))
                        if bool(
                            getattr(
                                cfg,
                                "reliability_environment_disagreement",
                                False,
                            )
                        )
                        else 0
                    ),
                    str(
                        getattr(cfg, "horizon_reliability_gate", "on")
                    ).lower() == "on",
                )
        self.environment_classification = EnvironmentClassificationConstraint(
            self.d_model,
            int(getattr(cfg, "predictive_env_num", 6)),
            bottleneck,
            float(getattr(cfg, "predictive_env_grl_weight", 1.0)),
        )
        self.head_flatten = nn.Flatten(start_dim=-2)
        self.head_linear = nn.Linear(self.d_model * token_num, self.pred_len)
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

    def encode(self, x, cycle_index=None):
        mean = x.mean(1, keepdim=True).detach()
        centered = x - mean
        std = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        normalized = centered / std
        nvars = normalized.shape[-1]
        if self.backbone == "patchtst":
            embedded, nvars = self.patch_embedding(normalized.permute(0, 2, 1))
            encoded, _ = self.encoder(embedded)
            tokens = encoded.reshape(
                -1, nvars, encoded.shape[-2], encoded.shape[-1]
            )
        elif self.backbone == "itransformer":
            embedded = self.inverted_embedding(normalized, None)
            encoded, _ = self.encoder(embedded, attn_mask=None)
            tokens = encoded[:, :nvars, :].unsqueeze(2)
        else:
            if cycle_index is None:
                cycle_index = torch.zeros(
                    x.shape[0], dtype=torch.long, device=x.device
                )
            residual = normalized - self.cycle_queue(cycle_index, self.seq_len)
            hidden = self.cycle_activation(
                self.cycle_input_projection(residual.permute(0, 2, 1))
            )
            tokens = hidden.unsqueeze(2)
        return tokens, mean, std

    def normalized_cycle_forecast(self, cycle_index, batch_size, device):
        if self.backbone != "cyclenet":
            return None
        if cycle_index is None:
            cycle_index = torch.zeros(batch_size, dtype=torch.long, device=device)
        future_index = (cycle_index + self.seq_len) % self.cycle_len
        return self.cycle_queue(future_index, self.pred_len).permute(0, 2, 1)

    def forecast_head(self, tokens, cycle_index=None):
        # [B,C,P,D] -> [B,C,D,P] before flattening for every backbone.
        features = self.head_flatten(tokens.permute(0, 1, 3, 2))
        prediction = self.head_dropout(self.head_linear(features))
        cycle = self.normalized_cycle_forecast(
            cycle_index, tokens.shape[0], tokens.device
        )
        return prediction if cycle is None else prediction + cycle

    def deterministic_forecast_head(self, tokens, cycle_index=None):
        """Forecast without training-time dropout for effect attribution."""
        features = self.head_flatten(tokens.permute(0, 1, 3, 2))
        prediction = self.head_linear(features)
        cycle = self.normalized_cycle_forecast(
            cycle_index, tokens.shape[0], tokens.device
        )
        return prediction if cycle is None else prediction + cycle

    def frozen_h_head(self, hidden, cycle_index=None):
        features = self.head_flatten(hidden.permute(0, 1, 3, 2))
        prediction = self.h_forecast_head(features)
        cycle = self.normalized_cycle_forecast(
            cycle_index, hidden.shape[0], hidden.device
        )
        return prediction if cycle is None else prediction + cycle

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

    def conditional_predictions(
        self, z_inv, z_var, scale, shift, cycle_index=None
    ):
        paired, shuffled, permutation = (
            self.variant_conditional_predictor.paired_and_shuffled(z_inv, z_var)
        )
        cycle = self.normalized_cycle_forecast(
            cycle_index, z_inv.shape[0], z_inv.device
        )
        if cycle is not None:
            paired = paired + cycle.permute(0, 2, 1)
            shuffled = shuffled + cycle.permute(0, 2, 1)
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
            forward_output.get("cycle_index"),
        )
        (
            adapted,
            feature_gate,
            feature_delta,
            fusion_scale,
            film_gamma,
            film_beta,
        ) = self.fuse_tokens(
            z_inv, z_var, forward_output["hidden_tokens"].detach()
        )
        cycle_index = forward_output.get("cycle_index")
        (
            prediction,
            invariant_prediction,
            feature_gate,
            feature_delta,
            y_film_gamma,
            y_film_beta,
            decay_rho,
            horizon_decay,
            horizon_future_zvar,
            horizon_reliability,
            horizon_gamma,
            horizon_beta,
            horizon_current_zvar,
            horizon_future_change,
            horizon_future_anchor_indices,
            horizon_environment_corrections,
            horizon_environment_disagreement,
        ) = self.fuse_predictions(
            z_inv,
            z_var,
            adapted,
            feature_gate,
            feature_delta,
            cycle_index,
        )
        variation_ratio = self.variation_ratio(
            feature_delta, z_inv, invariant_prediction
        )
        if self.variant_fusion_mode == "film_decay_reg":
            effect_prediction = self.deterministic_forecast_head(
                adapted, cycle_index
            )
            effect_invariant = self.deterministic_forecast_head(
                z_inv, cycle_index
            )
        else:
            effect_prediction = prediction
            effect_invariant = invariant_prediction
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
            "film_gamma": film_gamma,
            "film_beta": film_beta,
            "y_film_gamma": y_film_gamma,
            "y_film_beta": y_film_beta,
            "decay_rho": decay_rho,
            "horizon_decay": horizon_decay,
            "horizon_future_zvar": horizon_future_zvar,
            "horizon_reliability": horizon_reliability,
            "horizon_gamma": horizon_gamma,
            "horizon_beta": horizon_beta,
            "horizon_current_zvar": horizon_current_zvar,
            "horizon_future_zvar_change": horizon_future_change,
            "horizon_future_anchor_indices": horizon_future_anchor_indices,
            "horizon_environment_corrections": (
                None
                if horizon_environment_corrections is None
                else horizon_environment_corrections.permute(0, 1, 3, 4, 2)
                * forward_output["input_scale"].unsqueeze(1).unsqueeze(2)
            ),
            "horizon_environment_disagreement": horizon_environment_disagreement,
            "normalized_prediction": prediction.permute(0, 2, 1),
            "normalized_invariant_prediction": invariant_prediction.permute(
                0, 2, 1
            ),
            "effect_normalized_prediction": effect_prediction.permute(0, 2, 1),
            "effect_normalized_invariant_prediction": effect_invariant.permute(
                0, 2, 1
            ),
            "variation_ratio": variation_ratio,
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
            film_gamma = film_beta = None
        elif self.variant_fusion_mode in ("film", "film_decay_reg"):
            (
                adapted,
                feature_gate,
                feature_delta,
                film_gamma,
                film_beta,
            ) = self.film_variant_fusion(z_inv, z_var)
        else:
            feature_gate = torch.zeros(
                *z_var.shape[:-1], 1, device=z_var.device, dtype=z_var.dtype
            )
            feature_delta = torch.zeros_like(z_var)
            adapted = z_inv
            film_gamma = film_beta = None
        if self.variant_fusion_mode == "legacy":
            film_gamma = film_beta = None
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
        return (
            adapted,
            feature_gate,
            feature_delta,
            fusion_scale,
            film_gamma,
            film_beta,
        )

    def fuse_predictions(
        self,
        z_inv,
        z_var,
        adapted,
        feature_gate,
        feature_delta,
        cycle_index=None,
    ):
        """Apply latent fusion or one of the independent Y-space paths."""
        invariant = self.forecast_head(z_inv, cycle_index)
        y_film_gamma = y_film_beta = decay_rho = horizon_decay = None
        horizon_future_zvar = horizon_reliability = None
        horizon_gamma = horizon_beta = horizon_current_zvar = None
        horizon_future_change = horizon_future_anchor_indices = None
        horizon_environment_corrections = None
        horizon_environment_disagreement = None
        if self.variant_fusion_mode == "y_film_decay":
            (
                full,
                feature_gate,
                feature_delta,
                y_film_gamma,
                y_film_beta,
                decay_rho,
                horizon_decay,
            ) = self.y_film_decay_fusion(invariant, z_var)
        elif self.variant_fusion_mode == "horizon_future_var":
            horizon_output = self.horizon_future_variant(invariant, z_inv, z_var)
            full = horizon_output["prediction"]
            feature_gate = horizon_output["reliability"]
            feature_delta = horizon_output["variation"]
            horizon_future_zvar = horizon_output["future_zvar"]
            horizon_reliability = horizon_output["reliability"]
            horizon_gamma = horizon_output["gamma"]
            horizon_beta = horizon_output["beta"]
            horizon_current_zvar = horizon_output["current_zvar"]
            horizon_future_change = horizon_output["future_zvar_change_norm"]
            horizon_future_anchor_indices = horizon_output[
                "future_zvar_anchor_indices"
            ]
            horizon_environment_corrections = horizon_output[
                "environment_corrections"
            ]
            horizon_environment_disagreement = horizon_output[
                "environment_disagreement"
            ]
        else:
            full = self.forecast_head(adapted, cycle_index)
        return (
            full,
            invariant,
            feature_gate,
            feature_delta,
            y_film_gamma,
            y_film_beta,
            decay_rho,
            horizon_decay,
            horizon_future_zvar,
            horizon_reliability,
            horizon_gamma,
            horizon_beta,
            horizon_current_zvar,
            horizon_future_change,
            horizon_future_anchor_indices,
            horizon_environment_corrections,
            horizon_environment_disagreement,
        )

    def variation_ratio(self, feature_delta, z_inv, invariant_prediction):
        numerator = feature_delta.flatten(1).norm(dim=1)
        denominator_source = (
            invariant_prediction
            if self.variant_fusion_mode in (
                "y_film_decay",
                "horizon_future_var",
            )
            else z_inv
        )
        denominator = denominator_source.flatten(1).norm(dim=1).clamp_min(1e-8)
        return numerator / denominator

    @torch.no_grad()
    def future_variant_target(self, future_x, cycle_index=None):
        """Encode a TRAIN-only future window as a stop-gradient teacher.

        Teacher modules are temporarily put in evaluation mode so dropout and
        BatchNorm buffers cannot make the auxiliary branch mutate the model.
        The exact same module instances and current parameters are used.
        """
        modules = tuple(
            module
            for module in (
                self.patch_embedding,
                self.inverted_embedding,
                self.encoder,
                self.cycle_queue,
                self.cycle_input_projection,
                self.cycle_activation,
                self.decomposer,
            )
            if module is not None
        )
        training_states = tuple(module.training for module in modules)
        try:
            for module in modules:
                module.eval()
            hidden, _, _ = self.encode(future_x, cycle_index)
            _, z_var, _ = self.decompose(hidden)
            return z_var.mean((1, 2)).detach()
        finally:
            for module, was_training in zip(modules, training_states):
                module.train(was_training)

    @torch.no_grad()
    def horizon_future_variant_targets(
        self, future_windows, future_cycle_indices=None
    ):
        """Encode TRAIN-only causal windows into channel-wise anchor targets.

        ``future_windows`` is ``[B,A,L,C]``.  Every anchor goes through the
        exact shared encoder and decomposer under stop-gradient.  No target
        data is needed by the normal forward/inference path.
        """
        if future_windows.ndim != 4:
            raise ValueError("future windows must be [B,A,L,C]")
        modules = tuple(
            module
            for module in (
                self.patch_embedding,
                self.inverted_embedding,
                self.encoder,
                self.cycle_queue,
                self.cycle_input_projection,
                self.cycle_activation,
                self.decomposer,
            )
            if module is not None
        )
        training_states = tuple(module.training for module in modules)
        targets = []
        try:
            for module in modules:
                module.eval()
            for anchor in range(future_windows.shape[1]):
                cycle_index = (
                    None
                    if future_cycle_indices is None
                    else future_cycle_indices[:, anchor]
                )
                hidden, _, _ = self.encode(
                    future_windows[:, anchor], cycle_index
                )
                _, z_var, _ = self.decompose(hidden)
                targets.append(z_var.mean(dim=2))
            return torch.stack(targets, dim=2).detach()
        finally:
            for module, was_training in zip(modules, training_states):
                module.train(was_training)

    def forward_components(self, x, cycle_index=None):
        hidden, mean, std = self.encode(x, cycle_index)
        z_inv, z_var, decomposition_gate = self.decompose(hidden)
        (
            adapted,
            feature_gate,
            feature_delta,
            fusion_scale,
            film_gamma,
            film_beta,
        ) = self.fuse_tokens(
            z_inv, z_var, hidden
        )
        (
            full,
            invariant,
            feature_gate,
            feature_delta,
            y_film_gamma,
            y_film_beta,
            decay_rho,
            horizon_decay,
            horizon_future_zvar,
            horizon_reliability,
            horizon_gamma,
            horizon_beta,
            horizon_current_zvar,
            horizon_future_change,
            horizon_future_anchor_indices,
            horizon_environment_corrections,
            horizon_environment_disagreement,
        ) = self.fuse_predictions(
            z_inv,
            z_var,
            adapted,
            feature_gate,
            feature_delta,
            cycle_index,
        )
        variation_ratio = self.variation_ratio(feature_delta, z_inv, invariant)
        if self.variant_fusion_mode == "film_decay_reg":
            effect_full = self.deterministic_forecast_head(adapted, cycle_index)
            effect_invariant = self.deterministic_forecast_head(
                z_inv, cycle_index
            )
        else:
            effect_full = full
            effect_invariant = invariant
        h_prediction = self.frozen_h_head(hidden, cycle_index)
        scale, shift = std[:, 0, None], mean[:, 0, None]
        conditional, conditional_shuffled, conditional_permutation = (
            self.conditional_predictions(
                z_inv, z_var, scale, shift, cycle_index
            )
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
            "film_gamma": film_gamma,
            "film_beta": film_beta,
            "y_film_gamma": y_film_gamma,
            "y_film_beta": y_film_beta,
            "decay_rho": decay_rho,
            "horizon_decay": horizon_decay,
            "horizon_future_zvar": horizon_future_zvar,
            "horizon_reliability": horizon_reliability,
            "horizon_gamma": horizon_gamma,
            "horizon_beta": horizon_beta,
            "horizon_current_zvar": horizon_current_zvar,
            "horizon_future_zvar_change": horizon_future_change,
            "horizon_future_anchor_indices": horizon_future_anchor_indices,
            "horizon_environment_corrections": (
                None
                if horizon_environment_corrections is None
                else horizon_environment_corrections.permute(0, 1, 3, 4, 2)
                * scale.unsqueeze(1).unsqueeze(2)
            ),
            "horizon_environment_disagreement": horizon_environment_disagreement,
            "decomposition_gate": decomposition_gate,
            "hidden_tokens": hidden,
            "final_tokens": adapted,
            "fusion_scale": fusion_scale,
            "normalized_prediction": full.permute(0, 2, 1),
            "normalized_invariant_prediction": invariant.permute(0, 2, 1),
            "effect_normalized_prediction": effect_full.permute(0, 2, 1),
            "effect_normalized_invariant_prediction": effect_invariant.permute(
                0, 2, 1
            ),
            "variation_ratio": variation_ratio,
            "input_scale": scale,
            "input_shift": shift,
            "cycle_index": cycle_index,
            "backbone": self.backbone,
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
    def identity_diagnostics(self, x, cycle_index=None):
        was_training = self.training
        self.eval()
        hidden, _, _ = self.encode(x, cycle_index)
        if self.decomposition_type in ("signed_gate", "complementary_gate"):
            z_inv, z_var, _ = self.decomposer(hidden)
        else:
            z_inv, z_var = self.decomposer(hidden)
        adapted, _, _ = self.feature_variation(z_inv, z_var, "forced")
        result = {"feature": float((adapted - z_inv).abs().mean())}
        self.train(was_training)
        return result
