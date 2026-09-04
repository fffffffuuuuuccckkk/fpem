"""PatchTST with legacy FPem and the opt-in FPem-PMG implementation."""

import os
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer
from models.PatchTST import FlattenHead, Transpose
from models.fpem import (
    AdaptiveVariantFutureMapper, FutureForecastCorrection, FutureMappingMemoryBuilder,
    HistoryMappingContextEncoder, PatternMappingFPem, PatternMappingGraphBuilder,
    StableFutureMappingMemory, StablePatternGraph,
)
from models.fpem.pattern_mapping_losses import invariant_consensus_loss, separation_loss


def _flag(configs: Any, name: str, default: int) -> bool:
    return bool(int(getattr(configs, name, default)))


def _percentile(values: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return values.new_zeros(())
    index = max(1, min(flat.numel(), int(round(fraction * flat.numel()))))
    return flat.kthvalue(index).values


def extract_raw_patches(
    x: torch.Tensor,
    patch_len: int,
    stride: int,
    padding: Optional[int] = None,
    eps: float = 1e-5,
) -> Dict[str, torch.Tensor]:
    """Extract PatchTST-aligned, content-only raw shapes from ``[B,L,C]``."""
    if x.ndim != 3:
        raise ValueError("raw patch input must be [B,L,C]")
    padding = int(stride if padding is None else padding)
    canonical = x.permute(0, 2, 1)
    padded = F.pad(canonical, (0, padding), mode="replicate")
    patches = padded.unfold(dimension=-1, size=int(patch_len), step=int(stride))
    patch_mean = patches.mean(dim=-1, keepdim=True)
    patch_std = patches.var(dim=-1, keepdim=True, unbiased=False).sqrt()
    normalized = (patches - patch_mean) / (patch_std + float(eps))
    if not bool(torch.isfinite(normalized).all()):
        raise RuntimeError("raw patch shape normalization produced non-finite values")
    return {
        "patches": patches,
        "normalized": normalized,
        "mean": patch_mean.squeeze(-1),
        "std": patch_std.squeeze(-1),
    }


class _PatchTSTEncoder(nn.Module):
    """PatchTST encoder whose native output is [B,C,D,P]."""

    def __init__(self, configs: Any, patch_len: int = 16, stride: int = 8) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, stride, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)),
        )

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("PatchTST input must be [B,L,C]")
        embedded, n_vars = self.patch_embedding(x.permute(0, 2, 1))
        encoded, _ = self.encoder(embedded)
        if embedded.shape != encoded.shape:
            raise RuntimeError("PatchEmbedding and Transformer hidden token shapes must match")
        if embedded.shape[0] != x.shape[0] * n_vars:
            raise RuntimeError("PatchEmbedding batch/channel flattening is inconsistent with n_vars")
        batch, patches, dimension = x.shape[0], embedded.shape[1], embedded.shape[2]
        embedding = embedded.reshape(batch, n_vars, patches, dimension)
        hidden = encoded.reshape(batch, n_vars, patches, dimension)
        if embedding.ndim != 4 or embedding.shape != hidden.shape:
            raise RuntimeError("canonical embedding/hidden must share [B,C,P,D]")
        return {"embedding": embedding, "hidden": hidden}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve the pre-existing native [B,C,D,P] return contract.
        return self.forward_features(x)["hidden"].permute(0, 1, 3, 2)


class Model(nn.Module):
    """Backward-compatible PatchTST-FPem with an independent PMG execution path."""

    def __init__(self, configs: Any, patch_len: int = 16, stride: int = 8) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.d_model = int(configs.d_model)
        self.patch_len = int(getattr(configs, "patch_len", patch_len))
        self.stride = stride
        self.fpem_pmg_enabled = _flag(configs, "fpem_pmg_enabled", 0)
        self.latest_fpem: Dict[str, torch.Tensor] = {}
        self.latest_diagnostics: Dict[str, torch.Tensor] = {}
        self._graph_metadata: Dict[str, Any] = {}
        self._graph_space_frozen = False
        if self.fpem_pmg_enabled:
            self._init_pmg(configs)
        else:
            self._init_legacy(configs)

    def _init_legacy(self, configs: Any) -> None:
        """The pre-existing FPem path is intentionally preserved."""
        self.lambda_inv = float(getattr(configs, "fpem_lambda_inv", 0.2))
        self.lambda_env = float(getattr(configs, "fpem_lambda_env", 0.2))
        self.lambda_sep = float(getattr(configs, "fpem_lambda_sep", 0.01))
        self.lambda_delta_sparse = float(getattr(configs, "fpem_lambda_delta_sparse", 0.0))
        self.use_gate = _flag(configs, "fpem_use_gate", 1)
        self.encoder_inv = _PatchTSTEncoder(configs, self.patch_len, self.stride)
        self.encoder_env = _PatchTSTEncoder(configs, self.patch_len, self.stride)
        head_nf = self.d_model * int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_inv = FlattenHead(self.enc_in, head_nf, self.pred_len, head_dropout=configs.dropout)
        self.head_env_delta = FlattenHead(self.enc_in, head_nf, self.pred_len, head_dropout=configs.dropout)
        self.gate = nn.Sequential(
            nn.Linear(self.d_model * 2 + 4, self.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.d_model, 1),
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.fpem_pmg_enabled and self.future_mapping_mode != "off" and self.history_mapping_encoder is None:
            input_weight = state_dict.get("history_mapping_encoder.input_projection.weight")
            output_weight = state_dict.get("adaptive_future_mapper.output.1.weight")
            future_queries = state_dict.get("adaptive_future_mapper.future_queries")
            if input_weight is not None and output_weight is not None and future_queries is not None:
                self._initialize_future_mapping_modules(
                    int(input_weight.shape[1]), int(output_weight.shape[0]), int(future_queries.shape[1])
                )
        return super().load_state_dict(state_dict, strict=strict)

    def _init_pmg(self, configs: Any) -> None:
        self.lambda_cons = float(getattr(configs, "fpem_pmg_lambda_cons", 0.1))
        self.lambda_sep = float(getattr(configs, "fpem_pmg_lambda_sep", getattr(configs, "fpem_lambda_sep", 0.01)))
        self.lambda_rel = float(getattr(configs, "fpem_pmg_lambda_rel", 0.1))
        self.lambda_future_map = float(getattr(configs, "fpem_pmg_lambda_future_map", 0.1))
        self.use_retrospective = _flag(configs, "fpem_pmg_use_retrospective_validity", 0)
        self.graph_build_mode = str(getattr(configs, "fpem_pmg_build_graph", "auto"))
        self.configured_graph_path = str(getattr(configs, "fpem_pmg_graph_path", "") or "")
        self.a0_checkpoint_path = str(getattr(configs, "fpem_pmg_a0_checkpoint", "") or "")
        self.representation_space = str(
            getattr(configs, "fpem_pmg_representation_space", "embedding")
        ).lower()
        if self.representation_space not in {"embedding", "hidden", "raw"}:
            raise ValueError("fpem_pmg_representation_space must be embedding, hidden, or raw")
        self.stage0_protocol = "full_forecast_early_stopping_v1"
        self._stage0_loaded_a0 = False
        self._representation_sanity_printed = False
        self._zinv_initial_sanity_printed = False
        self.freeze_patch_embedding_stage2 = str(
            getattr(configs, "fpem_pmg_freeze_patch_embedding_stage2", "auto")
        ).lower()
        if self.freeze_patch_embedding_stage2 not in {"auto", "true", "false"}:
            raise ValueError("freeze_patch_embedding_stage2 must be auto, true, or false")
        self._patch_embedding_frozen = False
        print("FPem-PMG representation space: {}".format(self.representation_space))
        self.cinv_mode = str(getattr(configs, "fpem_pmg_cinv_mode", "pattern_only")).lower()
        print("FPem-PMG c_inv mode: {}".format(self.cinv_mode))
        self.zinv_mode = str(
            getattr(configs, "fpem_pmg_zinv_mode", "stable_relation_correction")
        ).lower()
        print("FPem-PMG Z_inv mode: {}".format(self.zinv_mode))
        self.variant_input_mode = str(
            getattr(configs, "fpem_pmg_variant_input_mode", "latent_only")
        ).lower()
        if self.variant_input_mode.startswith("factorized_") and self.representation_space != "raw":
            raise ValueError("factorized raw variation requires representation_space=raw")
        print("FPem-PMG variant input mode: {}".format(self.variant_input_mode))
        print("FPem-PMG freeze PatchEmbedding Stage2: {}".format(self.freeze_patch_embedding_stage2))
        self.predictive_stability_mode = str(
            getattr(configs, "fpem_pmg_predictive_stability_mode", "train_global")
        ).lower()
        self.mapping_use_mode = str(getattr(configs, "fpem_pmg_mapping_use_mode", "full")).lower()
        self.fusion_mode = str(
            getattr(configs, "fpem_pmg_fusion_mode", "legacy_environment")
        ).lower()
        self.typed_fusion_components = str(
            getattr(configs, "fpem_pmg_typed_fusion_components", "full")
        ).lower()
        self.relation_mapping_mode = str(
            getattr(configs, "fpem_pmg_relation_mapping_mode", "legacy")
        ).lower()
        self.future_mapping_mode = str(
            getattr(configs, "fpem_pmg_future_mapping_mode", "off")
        ).lower()
        if self.future_mapping_mode not in {"off", "stable_only", "adaptive_unit", "adaptive_gated"}:
            raise ValueError("invalid future mapping mode: {}".format(self.future_mapping_mode))
        if self.fusion_mode not in {"invariant_only", "legacy_environment", "typed_raw"}:
            raise ValueError("fpem_pmg_fusion_mode must be invariant_only, legacy_environment, or typed_raw")
        if self.typed_fusion_components not in {"shape", "geometry", "full"}:
            raise ValueError("fpem_pmg_typed_fusion_components must be shape, geometry, or full")
        if self.fusion_mode == "typed_raw" and self.representation_space != "raw":
            raise ValueError("typed raw fusion requires raw representation space")
        print("FPem-PMG predictive stability mode: {}".format(self.predictive_stability_mode))
        print("FPem-PMG mapping use mode: {}".format(self.mapping_use_mode))
        print("FPem-PMG fusion mode: {}".format(self.fusion_mode))
        print("FPem-PMG typed fusion components: {}".format(self.typed_fusion_components))
        print("FPem-PMG relation mapping mode: {}".format(self.relation_mapping_mode))
        print("FPem-PMG history-to-future mode: {}".format(self.future_mapping_mode))
        self.register_buffer("zinv_initial_difference", torch.tensor(float("nan")))
        self._typed_initial_sanity_printed = False
        self.register_buffer("typed_initial_z_diff", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("typed_initial_shape_correction_norm", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("typed_initial_scale_log_mod_norm", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("typed_initial_shift_bias_norm", torch.tensor(float("nan")), persistent=False)
        self._relation_initial_sanity_printed = False
        self.register_buffer("relation_initial_z_diff", torch.tensor(float("nan")), persistent=False)
        self.future_pattern_graph = StablePatternGraph(self.patch_len)
        self.future_mapping_memory = StableFutureMappingMemory()
        self.history_mapping_encoder = None
        self.adaptive_future_mapper = None
        self.future_forecast_correction = None
        self._future_initial_sanity_printed = False
        self.register_buffer("future_initial_prediction_difference", torch.tensor(float("nan")), persistent=False)
        self.encoder_backbone = _PatchTSTEncoder(configs, self.patch_len, self.stride)
        patch_count = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_shared = FlattenHead(
            self.enc_in,
            self.d_model * patch_count,
            self.pred_len,
            head_dropout=configs.dropout,
        )
        graph_pattern_dim = self.patch_len if self.representation_space == "raw" else int(
            getattr(configs, "fpem_pmg_pattern_dim", 32)
        )
        self.pmg = PatternMappingFPem(
            hidden_dim=self.d_model,
            pattern_dim=graph_pattern_dim,
            env_dim=int(getattr(configs, "fpem_pmg_env_dim", 16)),
            use_pattern=_flag(configs, "fpem_pmg_use_pattern", 1),
            use_mapping=_flag(configs, "fpem_pmg_use_mapping", 1),
            use_mapping_delta=_flag(configs, "fpem_pmg_use_mapping_delta", 1),
            use_variant=_flag(configs, "fpem_pmg_use_variant", 1),
            use_env=_flag(configs, "fpem_pmg_use_env", 1),
            use_reliability=_flag(configs, "fpem_pmg_use_reliability", 1),
            cinv_mode=self.cinv_mode,
            zinv_mode=self.zinv_mode,
            variant_input_mode=self.variant_input_mode,
            mapping_use_mode=self.mapping_use_mode,
            fusion_mode=self.fusion_mode,
            typed_fusion_components=self.typed_fusion_components,
            relation_mapping_mode=self.relation_mapping_mode,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.fpem_pmg_enabled and self._graph_space_frozen:
            if self.representation_space == "hidden":
                # The complete contextual encoder defines hidden graph coordinates.
                if self._patch_embedding_frozen:
                    self.encoder_backbone.eval()
                else:
                    self.encoder_backbone.encoder.eval()
            if self._patch_embedding_frozen:
                self.encoder_backbone.patch_embedding.eval()
            self.pmg.projector.eval()
        return self

    def _initialize_future_mapping_modules(self, history_feature_dim: int,
                                           future_pattern_count: int,
                                           future_patch_count: int) -> None:
        if self.future_mapping_mode == "off":
            return
        compatible = (
            self.history_mapping_encoder is not None
            and self.adaptive_future_mapper.output[-1].out_features == future_pattern_count
            and self.adaptive_future_mapper.future_queries.shape[1] == future_patch_count
        )
        if compatible:
            return
        device = next(self.parameters()).device
        self.history_mapping_encoder = HistoryMappingContextEncoder(
            history_feature_dim, self.d_model
        ).to(device)
        self.adaptive_future_mapper = AdaptiveVariantFutureMapper(
            self.d_model, future_pattern_count, future_patch_count
        ).to(device)
        self.future_forecast_correction = FutureForecastCorrection(self.d_model).to(device)

    def _future_history_tokens(self, decomposition: Mapping[str, torch.Tensor]) -> torch.Tensor:
        scalars = torch.stack([
            decomposition["c_pat"], decomposition["mapping_novelty"],
            decomposition["raw_variation_scale_signed"], decomposition["raw_variation_shift_signed"],
            decomposition["raw_variation_u_scale"], decomposition["raw_variation_u_shift"],
        ], -1)
        return torch.cat([
            decomposition["query"], decomposition["mapping_deviation"], scalars
        ], -1)

    @staticmethod
    def _future_patch_reconstruction(patches: torch.Tensor, horizon: int,
                                     stride: int) -> torch.Tensor:
        """Overlap-add [B,C,Q,L] future patches back to [B,H,C]."""
        batch, channels, patch_count, patch_len = patches.shape
        reconstructed = patches.new_zeros(batch, channels, horizon)
        weight = patches.new_zeros(horizon)
        for patch_index in range(patch_count):
            start = patch_index * stride
            stop = min(horizon, start + patch_len)
            if start >= horizon:
                break
            reconstructed[..., start:stop] += patches[..., patch_index, :stop - start]
            weight[start:stop] += 1
        return (reconstructed / weight.clamp_min(1)).permute(0, 2, 1)

    def freeze_fpem_pmg_graph_space(self) -> None:
        if not self.fpem_pmg_enabled:
            return
        if self.representation_space == "hidden":
            for parameter in self.encoder_backbone.parameters():
                parameter.requires_grad_(False)
        elif self.representation_space == "embedding":
            for parameter in self.encoder_backbone.patch_embedding.parameters():
                parameter.requires_grad_(False)
            for parameter in self.encoder_backbone.encoder.parameters():
                parameter.requires_grad_(True)
        else:
            # Raw graph coordinates are parameter-free and cannot drift when
            # PatchEmbedding/Transformer continue Stage-2 fine-tuning.
            for parameter in self.encoder_backbone.parameters():
                parameter.requires_grad_(True)
        if self.freeze_patch_embedding_stage2 == "true":
            self._patch_embedding_frozen = True
        elif self.freeze_patch_embedding_stage2 == "false":
            self._patch_embedding_frozen = False
        else:
            self._patch_embedding_frozen = self.representation_space in {"embedding", "hidden"}
        for parameter in self.encoder_backbone.patch_embedding.parameters():
            parameter.requires_grad_(not self._patch_embedding_frozen)
        for parameter in self.pmg.projector.parameters():
            parameter.requires_grad_(False)
        # Reliability is fitted only from chronological cross-fit samples.
        for parameter in self.pmg.reliability_head.parameters():
            parameter.requires_grad_(False)
        self._graph_space_frozen = True
        if self.representation_space == "hidden":
            self.encoder_backbone.eval()
        if self._patch_embedding_frozen:
            self.encoder_backbone.patch_embedding.eval()
        self.pmg.projector.eval()

    def _select_graph_tokens(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.representation_space == "raw":
            raise RuntimeError("raw graph coordinates come from extract_raw_patches")
        return features[self.representation_space]

    def _raw_patch_data(self, x_enc: torch.Tensor) -> Dict[str, torch.Tensor]:
        return extract_raw_patches(x_enc, self.patch_len, self.stride, self.stride)

    def _decompose_features(
        self,
        hidden: torch.Tensor,
        encoder_features: Mapping[str, torch.Tensor],
        x_enc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.representation_space != "raw":
            return self.pmg.decompose(hidden, self._select_graph_tokens(encoder_features))
        raw = self._raw_patch_data(x_enc)
        query = raw["normalized"]
        if query.shape[:3] != encoder_features["embedding"].shape[:3] or query.shape[:3] != hidden.shape[:3]:
            raise RuntimeError("raw/E/H patch anchors do not match [B,C,P]")
        decomposition = self.pmg.decompose(
            hidden,
            graph_query=query,
            raw_patch_mean=raw["mean"],
            raw_patch_std=raw["std"],
        )
        stable_raw_patch = (
            decomposition["raw_stable_level"].unsqueeze(-1)
            + decomposition["raw_stable_scale"].unsqueeze(-1)
            * decomposition["pattern_reconstruction"]
        )
        decomposition.update(
            {
                "raw_patch_mean": raw["mean"],
                "raw_patch_std": raw["std"],
                "raw_reconstruction_error": (query - decomposition["pattern_reconstruction"]).square().mean(-1),
                "raw_deviation_norm": (query - decomposition["pattern_reconstruction"]).norm(dim=-1),
                "raw_stable_patch": stable_raw_patch,
                "raw_stable_reconstruction_norm": (
                    raw["patches"] - stable_raw_patch
                ).float().norm(dim=-1),
            }
        )
        return decomposition

    @staticmethod
    def _norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(1, keepdim=True).detach()
        centered = x - mean
        std = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return centered / std, mean, std

    @staticmethod
    def _repr(z: torch.Tensor) -> torch.Tensor:
        return z.mean(dim=(1, 3))

    def _forecast_head(self, canonical: torch.Tensor) -> torch.Tensor:
        if canonical.ndim != 4 or canonical.shape[-1] != self.d_model:
            raise ValueError("forecast head expects canonical [B,C,P,D]")
        return self.head_shared(canonical.permute(0, 1, 3, 2)).permute(0, 2, 1)

    @staticmethod
    def _denorm(prediction: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return prediction * std[:, 0, :].unsqueeze(1) + mean[:, 0, :].unsqueeze(1)

    def _legacy_forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_norm, mean, std = self._norm(x_enc)
        z_inv = self.encoder_inv(x_norm)
        z_env = self.encoder_env(x_norm)
        y_inv_n = self.head_inv(z_inv).permute(0, 2, 1)
        y_delta_n = self.head_env_delta(z_inv + z_env).permute(0, 2, 1)
        r_inv, r_env = self._repr(z_inv), self._repr(z_env)
        stats = torch.stack(
            [
                x_norm.mean(dim=(1, 2)),
                x_norm.std(dim=(1, 2), unbiased=False),
                x_norm[:, -1, :].mean(dim=1),
                (x_norm[:, -1, :] - x_norm[:, 0, :]).mean(dim=1),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(torch.cat([r_inv, r_env, stats], dim=-1))).view(-1, 1, 1)
        if not self.use_gate:
            gate = torch.ones_like(gate)
        y_env_n = y_inv_n + y_delta_n
        y_final_n = y_inv_n + gate * y_delta_n
        y_inv = self._denorm(y_inv_n, mean, std)
        y_env = self._denorm(y_env_n, mean, std)
        y_final = self._denorm(y_final_n, mean, std)
        self.latest_fpem = {
            "y_inv": y_inv,
            "y_env": y_env,
            "y_delta": y_env - y_inv,
            "gate": gate.detach(),
            "r_inv": r_inv,
            "r_env": r_env,
        }
        return y_final

    def _retrospective_gain(self, x_norm: torch.Tensor) -> torch.Tensor:
        batch = x_norm.shape[0]
        if not self.use_retrospective or self.seq_len <= self.patch_len:
            return x_norm.new_zeros(batch)
        horizon = min(self.pred_len, max(1, self.seq_len // 4), self.seq_len - self.patch_len)
        prefix_end = self.seq_len - horizon
        retrospective_input = x_norm.clone()
        retrospective_input[:, prefix_end:, :] = x_norm[:, prefix_end - 1 : prefix_end, :]
        with torch.no_grad():
            encoder_features = self.encoder_backbone.forward_features(retrospective_input)
            hidden = encoder_features["hidden"]
            decomposition = self._decompose_features(hidden, encoder_features, retrospective_input)
            prediction_inv = self._forecast_head(decomposition["z_inv"])
            z_env_candidate = decomposition["z_inv"] + decomposition["temporal_gate"].unsqueeze(-1) * (
                decomposition["z_env"] - decomposition["z_inv"]
            )
            prediction_env = self._forecast_head(z_env_candidate)
            observed_tail = x_norm[:, -horizon:, :]
            error_inv = (prediction_inv[:, :horizon, :] - observed_tail).square().mean(dim=(1, 2))
            error_env = (prediction_env[:, :horizon, :] - observed_tail).square().mean(dim=(1, 2))
        return ((error_inv - error_env) / error_inv.clamp_min(1e-6)).clamp(-1.0, 1.0)

    def _apply_future_mapping(self, decomposition: Dict[str, torch.Tensor],
                              base_prediction: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.future_mapping_mode == "off":
            return {"future_forecast": base_prediction}
        if self.history_mapping_encoder is None or not self.future_mapping_memory.ready:
            raise RuntimeError("history-to-future memory must be prepared before forecasting")
        tokens = self._future_history_tokens(decomposition)
        memory = self.future_mapping_memory.query(
            tokens, self.history_mapping_encoder, leave_nearest_out=self.training
        )
        state_relation_summary = torch.stack([
            decomposition["pattern_deviation"].norm(dim=-1).mean(2),
            decomposition["raw_variation_scale_signed"].abs().mean(2),
            decomposition["raw_variation_shift_signed"].abs().mean(2),
            decomposition["raw_variation_u_scale"].mean(2),
            decomposition["raw_variation_u_shift"].mean(2),
            decomposition["mapping_deviation"].norm(dim=-1).mean(2),
        ], -1)
        gate_mode = {
            "stable_only": "off", "adaptive_unit": "unit", "adaptive_gated": "learned"
        }[self.future_mapping_mode]
        adaptive = self.adaptive_future_mapper(
            memory["history_sequence"], memory["history_context"],
            memory["stable_future_logits"], state_relation_summary, gate_mode,
        )
        probability = torch.softmax(adaptive["final_future_logits"], -1)
        future_patches = torch.einsum(
            "bcqk,kl->bcql", probability,
            self.future_pattern_graph.means.to(probability.dtype),
        )
        future_context = self._future_patch_reconstruction(
            future_patches, self.pred_len, self.stride
        )
        correction = self.future_forecast_correction(
            base_prediction, future_context,
            memory["retrieval_confidence"], adaptive["future_variant_gate"],
        )
        result = dict(memory)
        result.update(adaptive)
        result.update(correction)
        result["future_pattern_probability"] = probability
        result["future_pattern_context"] = future_context
        return result

    def _pmg_forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_norm, mean, std = self._norm(x_enc)
        encoder_features = self.encoder_backbone.forward_features(x_norm)
        embedding = encoder_features["embedding"]
        hidden = encoder_features["hidden"]
        if not self._representation_sanity_printed:
            if embedding.shape != hidden.shape or embedding.ndim != 4:
                raise RuntimeError("embedding/hidden sanity check failed")
            cosine = F.cosine_similarity(embedding.float(), hidden.float(), dim=-1).mean()
            if torch.allclose(embedding, hidden):
                raise RuntimeError("PatchEmbedding and Transformer hidden unexpectedly match exactly")
            print(
                "FPem-PMG feature sanity: embedding={} hidden={} cosine(E,H)={:.6f}".format(
                    tuple(embedding.shape), tuple(hidden.shape), float(cosine.detach().cpu())
                )
            )
            self._representation_sanity_printed = True
        decomposition = self._decompose_features(hidden, encoder_features, x_enc)
        graph_tokens = decomposition["graph_tokens"]
        if self.zinv_mode == "stable_relation_correction" and not self._zinv_initial_sanity_printed:
            if not bool(torch.isfinite(self.zinv_initial_difference).item()):
                initial_difference = (
                    decomposition["z_inv"] - decomposition["invariant_base"]
                ).abs().mean().detach()
                self.zinv_initial_difference.copy_(initial_difference)
            initial_difference = self.zinv_initial_difference
            print(
                "FPem-PMG C2 initial mean(|Z_inv-H_I|): {:.10g}{}".format(
                    float(initial_difference.detach().cpu()),
                    " [warning: expected near zero]" if float(initial_difference.detach().cpu()) >= 1e-4 else "",
                )
            )
            self._zinv_initial_sanity_printed = True
        if self.fusion_mode == "typed_raw" and not self._typed_initial_sanity_printed:
            typed_initial = (
                ("typed_initial_z_diff", (decomposition["z_typed"] - decomposition["z_inv"]).abs().mean()),
                ("typed_initial_shape_correction_norm", decomposition["shape_correction"].float().norm(dim=-1).mean()),
                ("typed_initial_scale_log_mod_norm", decomposition["log_scale_mod"].float().norm(dim=-1).mean()),
                ("typed_initial_shift_bias_norm", decomposition["shift_bias"].float().norm(dim=-1).mean()),
            )
            for buffer_name, value in typed_initial:
                buffer = getattr(self, buffer_name)
                if not bool(torch.isfinite(buffer).item()):
                    buffer.copy_(value.detach())
            print(
                "FPem typed init: z_diff={:.10g} shape_correction_norm={:.10g} "
                "scale_log_mod_norm={:.10g} shift_bias_norm={:.10g}".format(
                    float(self.typed_initial_z_diff.cpu()),
                    float(self.typed_initial_shape_correction_norm.cpu()),
                    float(self.typed_initial_scale_log_mod_norm.cpu()),
                    float(self.typed_initial_shift_bias_norm.cpu()),
                )
            )
            self._typed_initial_sanity_printed = True
        if not self._relation_initial_sanity_printed:
            initial_relation_difference = (
                decomposition["z_relation"] - decomposition["z_typed"]
            ).abs().mean().detach()
            self.relation_initial_z_diff.copy_(initial_relation_difference)
            print("FPem relation init mean(|Z_relation-Z_typed|): {:.10g}".format(
                float(initial_relation_difference.cpu())
            ))
            self._relation_initial_sanity_printed = True

        # NO TEST-FUTURE INFORMATION BEYOND THIS POINT.
        # Reliability uses X-only novelty/activation. The retrospective API is
        # retained for experiments, but is not a calibrator input.
        retrospective_gain = self._retrospective_gain(x_norm)
        features = torch.stack(
            [
                decomposition["variation_activation"].mean(dim=(1, 2)),
                decomposition["pattern_novelty"].mean(dim=(1, 2)),
                decomposition["mapping_novelty"].mean(dim=(1, 2)),
            ],
            dim=-1,
        ).detach()
        reliability = self.pmg.reliability(features)
        z_fused = self.pmg.fuse(decomposition, reliability)
        z_env_candidate = decomposition["z_inv"] + decomposition["temporal_gate"].unsqueeze(-1) * (
            decomposition["z_env"] - decomposition["z_inv"]
        )
        y_base_normalized = self._forecast_head(z_fused)
        future_mapping = self._apply_future_mapping(decomposition, y_base_normalized)
        y_full = self._denorm(future_mapping["future_forecast"], mean, std)
        y_base = self._denorm(y_base_normalized, mean, std)
        y_inv = self._denorm(self._forecast_head(decomposition["z_inv"]), mean, std)
        y_env = self._denorm(self._forecast_head(z_env_candidate), mean, std)
        y_shape = self._denorm(self._forecast_head(decomposition["z_after_shape"]), mean, std)
        y_scale = self._denorm(self._forecast_head(decomposition["z_scale_only"]), mean, std)
        y_shift = self._denorm(self._forecast_head(decomposition["z_shift_only"]), mean, std)
        y_typed = self._denorm(self._forecast_head(decomposition["z_typed"]), mean, std)
        y_without_var_map = y_typed
        y_with_var_map = self._denorm(self._forecast_head(decomposition["z_relation"]), mean, std)
        decomposition.update(
            {
                "prediction": y_full,
                "prediction_before_future_mapping": y_base,
                "prediction_inv": y_inv,
                "prediction_env": y_env,
                "prediction_shape": y_shape,
                "prediction_scale": y_scale,
                "prediction_shift": y_shift,
                "prediction_typed": y_typed,
                "prediction_without_var_map": y_without_var_map,
                "prediction_with_var_map": y_with_var_map,
                "environment_reliability": reliability,
                "retrospective_gain": retrospective_gain,
                "reliability_features": features,
            }
        )
        decomposition.update(future_mapping)
        if self.future_mapping_mode != "off" and not self._future_initial_sanity_printed:
            initial_difference = (y_full - y_base).abs().mean().detach()
            self.future_initial_prediction_difference.copy_(initial_difference)
            print("FPem future init mean(|Y_future-Y_base|): {:.10g}".format(float(initial_difference.cpu())))
            self._future_initial_sanity_printed = True
        self.latest_fpem = decomposition
        self.latest_diagnostics = {
            "prediction": y_full,
            "prediction_inv": y_inv,
            "pattern_novelty": decomposition["pattern_novelty"],
            "mapping_novelty": decomposition["mapping_novelty"],
            "variation_activation": decomposition["variation_activation"],
            "environment": decomposition["environment"],
            "environment_reliability": reliability,
        }
        return y_full

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor) -> torch.Tensor:
        if self.fpem_pmg_enabled:
            return self._pmg_forecast(x_enc)
        return self._legacy_forecast(x_enc)

    def fpem_extra_loss(self, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.latest_fpem:
            return target.new_zeros(()), {}
        if not self.fpem_pmg_enabled:
            y_inv = self.latest_fpem["y_inv"][:, -target.shape[1] :, -target.shape[2] :]
            y_env = self.latest_fpem["y_env"][:, -target.shape[1] :, -target.shape[2] :]
            delta = self.latest_fpem["y_delta"][:, -target.shape[1] :, -target.shape[2] :]
            r_inv = self.latest_fpem["r_inv"]
            r_env = self.latest_fpem["r_env"]
            loss_inv = F.mse_loss(y_inv, target)
            loss_env = F.mse_loss(y_env, target)
            sep = F.cosine_similarity(r_inv, r_env, dim=-1).pow(2).mean()
            sparse = delta.abs().mean()
            loss = self.lambda_inv * loss_inv + self.lambda_env * loss_env + self.lambda_sep * sep + self.lambda_delta_sparse * sparse
            return loss, {
                "fpem_ts/loss_inv": float(loss_inv.detach().cpu()),
                "fpem_ts/loss_env": float(loss_env.detach().cpu()),
                "fpem_ts/loss_sep": float(sep.detach().cpu()),
                "fpem_ts/gate_mean": float(self.latest_fpem["gate"].mean().detach().cpu()),
            }

        values = self.latest_fpem
        y_full = values["prediction"][:, -target.shape[1] :, -target.shape[2] :]
        y_inv = values["prediction_inv"][:, -target.shape[1] :, -target.shape[2] :]
        y_env = values["prediction_env"][:, -target.shape[1] :, -target.shape[2] :]
        loss_full = F.mse_loss(y_full, target)
        loss_inv = F.mse_loss(y_inv, target)
        loss_cons = invariant_consensus_loss(values["z_inv"], values["consensus_context"], values["c_inv"])
        loss_sep = separation_loss(values["z_inv"], values["z_var"])
        sample_gain = (y_inv - target).square().mean(dim=(1, 2)) - (y_env - target).square().mean(dim=(1, 2))
        # The calibrator is deliberately not supervised on the final full-graph
        # training samples. It is fitted in chronological cross-fitting later.
        loss_rel = target.new_zeros(())
        future_map_loss = target.new_zeros(())
        stable_future_map_loss = target.new_zeros(())
        future_map_gain = target.new_zeros(target.shape[0])
        if self.future_mapping_mode != "off":
            target_future = extract_raw_patches(
                target, self.patch_len, self.stride, self.stride
            )["normalized"]
            with torch.no_grad():
                beta = self.future_pattern_graph.query(target_future)["responsibility"]
            stable_ce = -(beta * F.log_softmax(values["stable_future_logits"], -1)).sum(-1).mean(dim=(1, 2))
            full_ce = -(beta * F.log_softmax(values["final_future_logits"], -1)).sum(-1).mean(dim=(1, 2))
            stable_future_map_loss = stable_ce.mean()
            future_map_loss = full_ce.mean()
            future_map_gain = stable_ce - full_ce
        extra = (
            loss_inv + self.lambda_cons * loss_cons + self.lambda_sep * loss_sep
            + self.lambda_rel * loss_rel + self.lambda_future_map * future_map_loss
        )

        pattern_support = self.pmg.pattern_graph.window_support
        mapping_support = self.pmg.mapping_graph.window_support
        reliability = values["environment_reliability"]
        activation = values["variation_activation"]
        graph_tokens = values["graph_tokens"]
        query = values["query"]
        correction_norm = values["graph_correction"].float().norm(dim=-1)
        graph_gate = values["graph_gate"]
        context_norm = values["consensus_context"].float().norm(dim=-1)
        invariant_base_norm = values["invariant_base"].float().norm(dim=-1)
        zinv_norm = values["z_inv"].float().norm(dim=-1)
        correction_ratio = correction_norm / invariant_base_norm.clamp_min(1e-6)
        stable_delta_norm = values["stable_mapping_delta"].float().norm(dim=-1)
        mapping_deviation_norm = values["mapping_deviation"].float().norm(dim=-1)
        hidden_norm = values["hidden"].float().norm(dim=-1)
        zvar_norm = values["z_var"].float().norm(dim=-1)
        latent_variant_norm = values["variant_latent"].float().norm(dim=-1)
        raw_variation_correction_norm = values["raw_variation_correction"].float().norm(dim=-1)
        typed_shape_gate = values["shape_gate"].squeeze(-1)
        typed_scale_gate = values["scale_gate"].squeeze(-1)
        typed_shift_gate = values["shift_gate"].squeeze(-1)
        typed_shape_correction_norm = values["shape_correction"].float().norm(dim=-1)
        typed_shift_bias_norm = values["shift_bias"].float().norm(dim=-1)
        typed_z_inv_norm = values["z_inv"].float().norm(dim=-1)
        typed_z_shape_norm = values["z_after_shape"].float().norm(dim=-1)
        typed_z_scale_norm = values["z_after_scale"].float().norm(dim=-1)
        typed_z_final_norm = values["z_typed"].float().norm(dim=-1)
        projected_raw_norm = values["raw_deviation_projected"].float().norm(dim=-1)
        relation_correction_norm = values["variant_mapping_correction"].float().norm(dim=-1)
        relation_gate = values["variant_mapping_gate"].squeeze(-1)
        relation_sample_gain = (
            (values["prediction_without_var_map"][:, -target.shape[1]:, -target.shape[2]:] - target).square().mean(dim=(1, 2))
            - (values["prediction_with_var_map"][:, -target.shape[1]:, -target.shape[2]:] - target).square().mean(dim=(1, 2))
        )
        logs = {
            "loss/full": float(loss_full.detach().cpu()),
            "loss/inv": float(loss_inv.detach().cpu()),
            "loss/cons": float(loss_cons.detach().cpu()),
            "loss/sep": float(loss_sep.detach().cpu()),
            "loss/reliability": float(loss_rel.detach().cpu()),
            "future_mapping/ce_stable": float(stable_future_map_loss.detach().cpu()),
            "future_mapping/ce_full": float(future_map_loss.detach().cpu()),
            "future_mapping/gain_mean": float(future_map_gain.mean().detach().cpu()),
            "future_mapping/gain_positive_fraction": float((future_map_gain > 0).float().mean().detach().cpu()),
            "pattern/num_active": float(self.pmg.pattern_graph.num_active),
            "graph/num_patterns": float(self.pmg.pattern_graph.num_active),
            "pattern/mean_support": float(pattern_support.float().mean().cpu()) if pattern_support.numel() else 0.0,
            "pattern/mean_novelty": float(values["pattern_novelty"].mean().detach().cpu()),
            "pattern/null_rate": float(values["null_probability"].mean().detach().cpu()),
            "mapping/num_active_edges": float(self.pmg.mapping_graph.num_active),
            "graph/num_mappings": float(self.pmg.mapping_graph.num_active),
            "mapping/mean_support": float(mapping_support.float().mean().cpu()) if mapping_support.numel() else 0.0,
            "mapping/mean_novelty": float(values["mapping_novelty"].mean().detach().cpu()),
            "mapping/stable_delta_norm_mean": float(stable_delta_norm.mean().detach().cpu()),
            "mapping/stable_delta_norm_p90": float(_percentile(stable_delta_norm, 0.9).cpu()),
            "mapping/deviation_norm_mean": float(mapping_deviation_norm.mean().detach().cpu()),
            "mapping/deviation_norm_p90": float(_percentile(mapping_deviation_norm, 0.9).cpu()),
            "relation/variant_activation_mean": float(values["variant_mapping_activation"].mean().detach().cpu()),
            "relation/active_variant_edges_mean": float(values["active_variant_edges"].mean().detach().cpu()),
            "relation/p_var_activation_mean": float(values["variant_mapping_probability"].mean().detach().cpu()),
            "relation/gate_mean": float(relation_gate.mean().detach().cpu()),
            "relation/gate_std": float(relation_gate.std(unbiased=False).detach().cpu()),
            "relation/gate_p10": float(_percentile(relation_gate, 0.1).cpu()),
            "relation/gate_p50": float(_percentile(relation_gate, 0.5).cpu()),
            "relation/gate_p90": float(_percentile(relation_gate, 0.9).cpu()),
            "relation/correction_norm_mean": float(relation_correction_norm.mean().detach().cpu()),
            "relation/gain_mean": float(relation_sample_gain.mean().detach().cpu()),
            "relation/gain_positive_fraction": float((relation_sample_gain > 0).float().mean().detach().cpu()),
            "relation/initial_difference_from_typed": float(self.relation_initial_z_diff.detach().cpu()),
            "variant/activation_mean": float(activation.mean().detach().cpu()),
            "variant/activation_p90": float(_percentile(activation, 0.9).cpu()),
            "c_inv_mean": float(values["c_inv"].mean().detach().cpu()),
            "c_inv_p10": float(_percentile(values["c_inv"], 0.1).cpu()),
            "c_inv_p50": float(_percentile(values["c_inv"], 0.5).cpu()),
            "c_inv_p90": float(_percentile(values["c_inv"], 0.9).cpu()),
            "cinv/mean": float(values["c_inv"].mean().detach().cpu()),
            "cinv/p10": float(_percentile(values["c_inv"], 0.1).cpu()),
            "cinv/p50": float(_percentile(values["c_inv"], 0.5).cpu()),
            "cinv/p90": float(_percentile(values["c_inv"], 0.9).cpu()),
            "cinv/std": float(values["c_inv"].float().std(unbiased=False).detach().cpu()),
            "pattern/c_mean": float(values["c_pat"].mean().detach().cpu()),
            "pattern/c_shape_mean": float(values["c_shape"].mean().detach().cpu()),
            "pattern/c_rec_mean": float(values["c_rec"].mean().detach().cpu()),
            "pattern/c_pred_mean": float(values["c_pred"].mean().detach().cpu()),
            "mapping/c_mean": float(values["c_map"].mean().detach().cpu()),
            "graph/correction_norm_mean": float(correction_norm.mean().detach().cpu()),
            "graph/correction_norm_p90": float(_percentile(correction_norm, 0.9).cpu()),
            "graph/gate_mean": float(graph_gate.mean().detach().cpu()),
            "graph/gate_p10": float(_percentile(graph_gate, 0.1).cpu()),
            "graph/gate_p50": float(_percentile(graph_gate, 0.5).cpu()),
            "graph/gate_p90": float(_percentile(graph_gate, 0.9).cpu()),
            "graph/context_norm_mean": float(context_norm.mean().detach().cpu()),
            "hidden/invariant_base_norm_mean": float(invariant_base_norm.mean().detach().cpu()),
            "zinv/norm_mean": float(zinv_norm.mean().detach().cpu()),
            "hidden/norm_mean": float(hidden_norm.mean().detach().cpu()),
            "zvar/norm_mean": float(zvar_norm.mean().detach().cpu()),
            "zvar/norm_p90": float(_percentile(zvar_norm, 0.9).cpu()),
            "graph/correction_to_hidden_ratio": float(correction_ratio.mean().detach().cpu()),
            "zinv/initial_difference_from_hidden": float(self.zinv_initial_difference.detach().cpu())
            if bool(torch.isfinite(self.zinv_initial_difference).item()) else float("nan"),
            "graph_source/variance": float(graph_tokens.float().var(unbiased=False).detach().cpu()),
            "graph_source/norm": float(graph_tokens.float().norm(dim=-1).mean().detach().cpu()),
            "query/variance": float(query.float().var(unbiased=False).detach().cpu()),
            "query/norm": float(query.float().norm(dim=-1).mean().detach().cpu()),
            "env/norm": float(values["environment"].norm(dim=-1).mean().detach().cpu()),
            "env/reliability_mean": float(reliability.mean().detach().cpu()),
            "env/reliability_p10": float(_percentile(reliability, 0.1).cpu()),
            "env/reliability_p90": float(_percentile(reliability, 0.9).cpu()),
            "forecast/full_metric": float(loss_full.detach().cpu()),
            "forecast/inv_metric": float(loss_inv.detach().cpu()),
            "forecast/env_gain": float(sample_gain.mean().detach().cpu()),
            "graph/projector_used": float(self.representation_space != "raw"),
            "typed_init/z_diff": float(self.typed_initial_z_diff.detach().cpu()),
            "typed_init/shape_correction_norm": float(self.typed_initial_shape_correction_norm.detach().cpu()),
            "typed_init/scale_log_mod_norm": float(self.typed_initial_scale_log_mod_norm.detach().cpu()),
            "typed_init/shift_bias_norm": float(self.typed_initial_shift_bias_norm.detach().cpu()),
            "typed/shape_variation_norm": float(values["shape_variation_norm"].mean().detach().cpu()),
            "typed/shape_gate_mean": float(typed_shape_gate.mean().detach().cpu()),
            "typed/shape_gate_p50": float(_percentile(typed_shape_gate, 0.5).cpu()),
            "typed/shape_gate_p90": float(_percentile(typed_shape_gate, 0.9).cpu()),
            "typed/shape_correction_norm": float(typed_shape_correction_norm.mean().detach().cpu()),
            "typed/scale_signed_abs_mean": float(values["raw_variation_scale_signed"].abs().mean().detach().cpu()),
            "typed/u_scale_mean": float(values["raw_variation_u_scale"].mean().detach().cpu()),
            "typed/scale_gate_mean": float(typed_scale_gate.mean().detach().cpu()),
            "typed/scale_gate_p50": float(_percentile(typed_scale_gate, 0.5).cpu()),
            "typed/scale_gate_p90": float(_percentile(typed_scale_gate, 0.9).cpu()),
            "typed/log_scale_mod_mean": float(values["log_scale_mod"].mean().detach().cpu()),
            "typed/log_scale_mod_abs_mean": float(values["log_scale_mod"].abs().mean().detach().cpu()),
            "typed/scale_factor_mean": float(values["scale_factor"].mean().detach().cpu()),
            "typed/scale_factor_p10": float(_percentile(values["scale_factor"], 0.1).cpu()),
            "typed/scale_factor_p50": float(_percentile(values["scale_factor"], 0.5).cpu()),
            "typed/scale_factor_p90": float(_percentile(values["scale_factor"], 0.9).cpu()),
            "typed/shift_signed_abs_mean": float(values["raw_variation_shift_signed"].abs().mean().detach().cpu()),
            "typed/u_shift_mean": float(values["raw_variation_u_shift"].mean().detach().cpu()),
            "typed/shift_gate_mean": float(typed_shift_gate.mean().detach().cpu()),
            "typed/shift_gate_p50": float(_percentile(typed_shift_gate, 0.5).cpu()),
            "typed/shift_gate_p90": float(_percentile(typed_shift_gate, 0.9).cpu()),
            "typed/shift_bias_norm": float(typed_shift_bias_norm.mean().detach().cpu()),
            "typed/z_inv_norm": float(typed_z_inv_norm.mean().detach().cpu()),
            "typed/z_after_shape_norm": float(typed_z_shape_norm.mean().detach().cpu()),
            "typed/z_after_scale_norm": float(typed_z_scale_norm.mean().detach().cpu()),
            "typed/z_final_norm": float(typed_z_final_norm.mean().detach().cpu()),
            "typed/shape_correction_to_zinv_ratio": float(
                (typed_shape_correction_norm / typed_z_inv_norm.clamp_min(1e-6)).mean().detach().cpu()
            ),
            "typed/shift_bias_to_zinv_ratio": float(
                (typed_shift_bias_norm / typed_z_inv_norm.clamp_min(1e-6)).mean().detach().cpu()
            ),
        }
        if self.representation_space == "raw":
            raw_deviation = values["raw_deviation_norm"]
            raw_reconstruction_error = values["raw_reconstruction_error"]
            pattern_distance = values["pattern_best_distance"]
            logs.update(
                {
                    "raw/pattern_distance_mean": float(pattern_distance.mean().detach().cpu()),
                    "raw/pattern_distance_p90": float(_percentile(pattern_distance, 0.9).cpu()),
                    "raw/reconstruction_error_mean": float(raw_reconstruction_error.mean().detach().cpu()),
                    "raw/deviation_norm_mean": float(raw_deviation.mean().detach().cpu()),
                    "raw_deviation/norm_mean": float(raw_deviation.mean().detach().cpu()),
                    "raw_deviation/norm_p90": float(_percentile(raw_deviation, 0.9).cpu()),
                    "raw_deviation_projected/norm_mean": float(projected_raw_norm.mean().detach().cpu()),
                    "raw_deviation_projected/norm_p90": float(_percentile(projected_raw_norm, 0.9).cpu()),
                    "raw_deviation_projected/to_hidden_ratio": float(
                        (projected_raw_norm / hidden_norm.clamp_min(1e-6)).mean().detach().cpu()
                    ),
                    "raw_variation/shape_norm": float(values["raw_variation_shape"].float().norm(dim=-1).mean().detach().cpu()),
                    "raw_variation/shift_signed": float(values["raw_variation_shift_signed"].mean().detach().cpu()),
                    "raw_variation/scale_signed": float(values["raw_variation_scale_signed"].mean().detach().cpu()),
                    "raw_variation/u_shift": float(values["raw_variation_u_shift"].mean().detach().cpu()),
                    "raw_variation/u_scale": float(values["raw_variation_u_scale"].mean().detach().cpu()),
                    "raw_variation/a_shape": float(values["raw_variation_a_shape"].mean().detach().cpu()),
                    "raw_variation/a_geo": float(values["raw_variation_a_geo"].mean().detach().cpu()),
                    "raw_variation/a_var": float(values["raw_variation_a_var"].mean().detach().cpu()),
                    "raw_variation/stable_level": float(values["raw_stable_level"].mean().detach().cpu()),
                    "raw_variation/current_level": float(values["raw_patch_mean"].mean().detach().cpu()),
                    "raw_variation/stable_scale": float(values["raw_stable_scale"].mean().detach().cpu()),
                    "raw_variation/current_scale": float(values["raw_patch_std"].mean().detach().cpu()),
                    "raw_variation/raw_gate": float(values["raw_variation_gate"].mean().detach().cpu()),
                    "raw_variation/raw_correction_norm": float(raw_variation_correction_norm.mean().detach().cpu()),
                    "raw_variation/stable_reconstruction_norm": float(values["raw_stable_reconstruction_norm"].mean().detach().cpu()),
                    "variant/latent_norm": float(latent_variant_norm.mean().detach().cpu()),
                    "variant/raw_correction_norm": float(raw_variation_correction_norm.mean().detach().cpu()),
                    "variant/final_norm": float(zvar_norm.mean().detach().cpu()),
                }
            )
            node_ratio = self.pmg.pattern_graph.future_ratio.float()
            node_pred = self.pmg.pattern_graph.predictive_support.float()
            for prefix, node_values in (
                ("future_ratio", node_ratio),
                ("pattern_predictive_stability", node_pred),
            ):
                if node_values.numel():
                    logs.update({
                        prefix + "/min": float(node_values.min().cpu()),
                        prefix + "/p10": float(_percentile(node_values, 0.10).cpu()),
                        prefix + "/p25": float(_percentile(node_values, 0.25).cpu()),
                        prefix + "/p50": float(_percentile(node_values, 0.50).cpu()),
                        prefix + "/p75": float(_percentile(node_values, 0.75).cpu()),
                        prefix + "/p90": float(_percentile(node_values, 0.90).cpu()),
                        prefix + "/max": float(node_values.max().cpu()),
                        prefix + "/std": float(node_values.std(unbiased=False).cpu()),
                    })
        if self.future_mapping_mode != "off":
            future_gate = values["future_variant_gate"].squeeze(-1).squeeze(-1)
            logs.update({
                "future_mapping/retrieval_confidence": float(values["retrieval_confidence"].mean().detach().cpu()),
                "future_mapping/effective_memory_entries": float(values["effective_memory_entries"].mean().detach().cpu()),
                "future_mapping/history_memory_distance": float(values["history_memory_distance"].mean().detach().cpu()),
                "future_mapping/retrieved_p_inv": float(values["retrieved_p_inv"].mean().detach().cpu()),
                "future_mapping/variant_gate_mean": float(future_gate.mean().detach().cpu()),
                "future_mapping/variant_gate_std": float(future_gate.std(unbiased=False).detach().cpu()),
                "future_mapping/variant_gate_p10": float(_percentile(future_gate, .1).cpu()),
                "future_mapping/variant_gate_p50": float(_percentile(future_gate, .5).cpu()),
                "future_mapping/variant_gate_p90": float(_percentile(future_gate, .9).cpu()),
                "future_mapping/delta_logits_norm": float(values["variant_future_logits"].float().norm(dim=-1).mean().detach().cpu()),
                "future_mapping/stable_logits_norm": float(values["stable_future_logits"].float().norm(dim=-1).mean().detach().cpu()),
                "future_mapping/final_logits_norm": float(values["final_future_logits"].float().norm(dim=-1).mean().detach().cpu()),
                "future_mapping/delta_logits_sample_diversity": float(
                    values["variant_future_logits"].float().flatten(1).std(dim=0, unbiased=False).mean().detach().cpu()
                ),
                "future_mapping/future_pattern_entropy": float(
                    (-(values["future_pattern_probability"] * values["future_pattern_probability"].clamp_min(1e-6).log()).sum(-1)).mean().detach().cpu()
                ),
                "future_mapping/forecast_gate_mean": float(values["future_forecast_gate"].mean().detach().cpu()),
                "future_mapping/forecast_correction_norm": float(values["future_forecast_correction"].float().norm(dim=1).mean().detach().cpu()),
                "future_mapping/initial_prediction_difference": float(self.future_initial_prediction_difference.detach().cpu()),
            })
        return extra, logs

    @torch.no_grad()
    def fpem_pmg_graph_available(self, checkpoint_dir: str) -> bool:
        if not self.fpem_pmg_enabled or not self.pmg.use_pattern:
            return True
        if self.graph_build_mode.lower() in {"0", "false", "off", "none"}:
            return True
        graph_path = self.configured_graph_path or os.path.join(
            checkpoint_dir, "pattern_mapping_graph_{}.pt".format(self.representation_space)
        )
        if not os.path.exists(graph_path):
            return False
        artifact = torch.load(graph_path, map_location="cpu")
        metadata = artifact.get("metadata", {})
        return (
            "graph_space_state" in artifact
            and metadata.get("stage0_protocol") == self.stage0_protocol
        )

    def load_fpem_pmg_a0_checkpoint(self, checkpoint_path: str) -> int:
        """Initialize the matching PMG backbone/head from a PatchTST A0 checkpoint."""
        if not self.fpem_pmg_enabled:
            return 0
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, Mapping) and "model" in state:
            state = state["model"]
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
        backbone_state = self.encoder_backbone.state_dict()
        missing = [key for key in backbone_state if key not in state]
        incompatible = [
            key for key in backbone_state
            if key in state and tuple(state[key].shape) != tuple(backbone_state[key].shape)
        ]
        head_state = self.head_shared.state_dict()
        head_source = {key: "head." + key for key in head_state}
        missing += [source for key, source in head_source.items() if source not in state]
        incompatible += [
            source for key, source in head_source.items()
            if source in state and tuple(state[source].shape) != tuple(head_state[key].shape)
        ]
        if missing or incompatible:
            raise ValueError(
                "A0 checkpoint is not architecture-compatible; missing={}, incompatible={}".format(
                    missing, incompatible
                )
            )
        self.encoder_backbone.load_state_dict({key: state[key] for key in backbone_state})
        self.head_shared.load_state_dict({key: state[source] for key, source in head_source.items()})
        self._stage0_loaded_a0 = True
        return len(backbone_state) + len(head_state)

    def _graph_space_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "encoder_backbone": {
                key: value.detach().cpu() for key, value in self.encoder_backbone.state_dict().items()
            },
            "pattern_projector": {
                key: value.detach().cpu() for key, value in self.pmg.projector.state_dict().items()
            },
        }

    def _load_graph_space_state(self, artifact: Mapping[str, Any]) -> bool:
        state = artifact.get("graph_space_state")
        if not isinstance(state, Mapping):
            return False
        self.encoder_backbone.load_state_dict(state["encoder_backbone"])
        self.pmg.projector.load_state_dict(state["pattern_projector"])
        return True

    def fpem_pmg_warmup_forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Train backbone, projector and forecast head before graph construction."""
        if not self.fpem_pmg_enabled:
            raise RuntimeError("PMG warmup is only available on the PMG path")
        x_norm, mean, std = self._norm(x_enc)
        encoder_features = self.encoder_backbone.forward_features(x_norm)
        if self.representation_space == "raw":
            return self._denorm(self._forecast_head(encoder_features["hidden"]), mean, std)
        graph_tokens = self._select_graph_tokens(encoder_features)
        predictive_hidden = self.pmg.projector.predictive_hidden(graph_tokens)
        return self._denorm(self._forecast_head(predictive_hidden), mean, std)

    @torch.no_grad()
    def fpem_pmg_pattern_representation(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.representation_space == "raw":
            return self._raw_patch_data(x_enc)["normalized"]
        x_norm, _, _ = self._norm(x_enc)
        encoder_features = self.encoder_backbone.forward_features(x_norm)
        return self.pmg.projector(self._select_graph_tokens(encoder_features))

    @torch.no_grad()
    def fpem_pmg_future_representation(
        self, target: torch.Tensor, x_enc: torch.Tensor
    ) -> torch.Tensor:
        """Apply the graph's TRAIN-defined future transform to [B,H,C]."""
        future = target[:, -self.pred_len :, :].permute(0, 2, 1).float()
        if self.predictive_stability_mode == "per_future_znorm":
            return (future - future.mean(-1, keepdim=True)) / (
                future.var(-1, keepdim=True, unbiased=False).sqrt() + 1e-5
            )
        if self.predictive_stability_mode == "train_global":
            mean = torch.as_tensor(
                self._graph_metadata["future_train_mean"], device=future.device, dtype=future.dtype
            ).view(1, -1, 1)
            std = torch.as_tensor(
                self._graph_metadata["future_train_std"], device=future.device, dtype=future.dtype
            ).view(1, -1, 1)
            return (future - mean) / (std + 1e-5)
        last = x_enc[:, -1, :].unsqueeze(-1)
        past_std = x_enc.var(dim=1, unbiased=False).sqrt().unsqueeze(-1)
        return (future - last) / (past_std + 1e-5)

    @torch.no_grad()
    def fpem_future_target_responsibility(self, target: torch.Tensor) -> torch.Tensor:
        """Evaluation/training-target diagnostic only; never used by inference."""
        if self.future_mapping_mode == "off" or not bool(self.future_pattern_graph.ready.item()):
            return target.new_zeros(target.shape[0], target.shape[2], 0, 0)
        raw = extract_raw_patches(
            target[:, -self.pred_len:, :], self.patch_len, self.stride, self.stride
        )["normalized"]
        return self.future_pattern_graph.query(raw)["responsibility"]

    @torch.no_grad()
    def fpem_pmg_shuffled_mapping_forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Evaluation-only fixed-seed within-source edge-statistic shuffle."""
        graph = self.pmg.mapping_graph
        if graph.num_active < 2:
            return self._pmg_forecast(x_enc)
        fields = [
            "dst_index", "counts", "window_support", "delta_means", "delta_variances",
            "stability", "future_prototypes", "future_variance", "predictive_gain",
        ]
        original = {name: getattr(graph, name).clone() for name in fields}
        permutation = torch.arange(graph.num_active, device=graph.src_index.device)
        for source in torch.unique(graph.src_index):
            indices = torch.nonzero(graph.src_index == source, as_tuple=False).flatten()
            if indices.numel() > 1:
                permutation[indices] = indices.roll(1)
        try:
            for name in fields:
                value = original[name]
                if value.shape[0] == graph.num_active:
                    graph._buffers[name] = value[permutation]
            return self._pmg_forecast(x_enc)
        finally:
            for name, value in original.items():
                graph._buffers[name] = value

    @torch.no_grad()
    def prepare_fpem_future_mapping(
        self, train_loader: Any, device: torch.device, checkpoint_dir: str,
        force: bool = False,
    ) -> Optional[str]:
        """Build FuturePatternGraph and block memory from chronological TRAIN only."""
        if self.future_mapping_mode == "off":
            return None
        path = os.path.join(checkpoint_dir, "history_to_future_mapping.pt")
        if os.path.exists(path) and not force:
            artifact = torch.load(path, map_location="cpu")
            if artifact.get("metadata", {}).get("source_split") == "train":
                self.future_pattern_graph.load_statistics(artifact["future_pattern_graph"])
                self.future_mapping_memory.load_statistics(artifact["future_mapping_memory"])
                keys = artifact["future_mapping_memory"]["history_keys"]
                logits = artifact["future_mapping_memory"]["future_logits"]
                self._initialize_future_mapping_modules(keys.shape[-1], logits.shape[-1], logits.shape[-2])
                print("Loaded TRAIN-only history-to-future memory: {}".format(path))
                return path
        was_training = self.training
        self.eval()
        history_tokens, future_normalized, future_mean, future_std = [], [], [], []
        for batch in train_loader:
            batch_x = batch[0].float().to(device)
            batch_y = batch[1].float().to(device)[:, -self.pred_len:, :]
            x_norm, _, _ = self._norm(batch_x)
            encoder_features = self.encoder_backbone.forward_features(x_norm)
            decomposition = self._decompose_features(encoder_features["hidden"], encoder_features, batch_x)
            history_tokens.append(self._future_history_tokens(decomposition).cpu())
            raw_future = extract_raw_patches(batch_y, self.patch_len, self.stride, self.stride)
            future_normalized.append(raw_future["normalized"].cpu())
            future_mean.append(raw_future["mean"].cpu())
            future_std.append(raw_future["std"].cpu())
        history_all = torch.cat(history_tokens)
        future_all = torch.cat(future_normalized)
        future_mean_all = torch.cat(future_mean)
        future_std_all = torch.cat(future_std)
        future_builder = PatternMappingGraphBuilder(
            self.patch_len, seq_len=self.seq_len, representation_space="raw",
            patch_len=self.patch_len, stride=self.stride,
            predictive_stability_mode="per_future_znorm",
        )
        future_artifact = future_builder.build_from_embeddings(
            future_all, raw_patch_mean=future_mean_all, raw_patch_std=future_std_all,
            metadata={"source_split": "train", "graph_role": "future_pattern_graph"},
        )
        self.future_pattern_graph.load_statistics(future_artifact["pattern_graph"])
        future_responsibility = []
        for start in range(0, future_all.shape[0], 512):
            result = self.future_pattern_graph.query(future_all[start:start + 512].to(device))
            future_responsibility.append(result["responsibility"].cpu())
        beta = torch.cat(future_responsibility)
        memory = FutureMappingMemoryBuilder(self.seq_len).build(history_all, beta)
        artifact = {
            "future_pattern_graph": self.future_pattern_graph.export(),
            "future_mapping_memory": memory,
            "metadata": {
                "source_split": "train", "leakage_policy": "leave-nearest-anchor-block-out",
                "history_windows": int(history_all.shape[0]),
                "future_patterns": int(self.future_pattern_graph.num_active),
                "future_patches": int(future_all.shape[2]),
            },
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(artifact, path)
        self.future_mapping_memory.load_statistics(memory)
        self._initialize_future_mapping_modules(
            history_all.shape[-1], self.future_pattern_graph.num_active, future_all.shape[2]
        )
        if was_training:
            self.train()
        print("Built TRAIN-only FuturePatternGraph/Memory: {} patterns, {} entries".format(
            self.future_pattern_graph.num_active, memory["history_keys"].shape[0]
        ))
        return path

    @torch.no_grad()
    def prepare_fpem_pmg_graph(
        self,
        train_loader: Any,
        device: torch.device,
        checkpoint_dir: str,
        metadata: Optional[Mapping[str, Any]] = None,
        force: bool = False,
    ) -> Optional[str]:
        """Build/load graph from the explicitly supplied chronological TRAIN loader."""
        if not self.fpem_pmg_enabled or not self.pmg.use_pattern:
            return None
        graph_path = self.configured_graph_path or os.path.join(
            checkpoint_dir, "pattern_mapping_graph_{}.pt".format(self.representation_space)
        )
        expected = dict(metadata or {})
        expected["mapping_relation_decomposition"] = "train_only_diagonal_gaussian_mixture_v1"
        if os.path.exists(graph_path) and not force:
            artifact = torch.load(graph_path, map_location="cpu")
            saved = artifact.get("metadata", {})
            compatible = all(saved.get(key) == value for key, value in expected.items())
            has_graph_space = "graph_space_state" in artifact
            if compatible and has_graph_space:
                self._load_graph_space_state(artifact)
                self.pmg.load_graph_statistics(artifact)
                self._graph_metadata = dict(saved)
                self.freeze_fpem_pmg_graph_space()
                self.prepare_fpem_future_mapping(train_loader, device, checkpoint_dir, force=force)
                print("Loaded FPem-PMG graph: {}".format(graph_path))
                return graph_path
        if self.graph_build_mode.lower() in {"0", "false", "off", "none"}:
            return None

        was_training = self.training
        self.eval()
        embeddings = []
        future_targets = []
        history_last = []
        history_std = []
        raw_patch_means = []
        raw_patch_stds = []
        for batch in train_loader:
            batch_x, batch_y = batch[0].float().to(device), batch[1].float().to(device)
            if self.representation_space == "raw":
                raw = self._raw_patch_data(batch_x)
                embeddings.append(raw["normalized"].cpu())
                raw_patch_means.append(raw["mean"].cpu())
                raw_patch_stds.append(raw["std"].cpu())
            else:
                x_norm, _, _ = self._norm(batch_x)
                encoder_features = self.encoder_backbone.forward_features(x_norm)
                graph_tokens = self._select_graph_tokens(encoder_features)
                embeddings.append(self.pmg.projector(graph_tokens).cpu())
            future_targets.append(batch_y[:, -self.pred_len :, :].permute(0, 2, 1).cpu())
            history_last.append(batch_x[:, -1, :].cpu())
            history_std.append(batch_x.var(dim=1, unbiased=False).sqrt().cpu())
        if not embeddings:
            raise RuntimeError("cannot build FPem-PMG graph from an empty training loader")
        builder = PatternMappingGraphBuilder(
            self.pmg.pattern_dim,
            seq_len=self.seq_len,
            representation_space=self.representation_space,
            patch_len=self.patch_len,
            stride=self.stride,
            predictive_stability_mode=self.predictive_stability_mode,
        )
        artifact = builder.build_from_embeddings(
            torch.cat(embeddings), torch.cat(future_targets), metadata=expected,
            history_last=torch.cat(history_last), history_std=torch.cat(history_std),
            raw_patch_mean=torch.cat(raw_patch_means) if raw_patch_means else None,
            raw_patch_std=torch.cat(raw_patch_stds) if raw_patch_stds else None,
        )
        artifact["graph_space_state"] = self._graph_space_state()
        self.pmg.load_graph_statistics(artifact)
        self._graph_metadata = dict(artifact["metadata"])
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        torch.save(artifact, graph_path)
        if self.representation_space == "raw":
            pattern = artifact["pattern_graph"]
            mapping = artifact["mapping_graph"]
            raw_npz_path = os.path.join(checkpoint_dir, "raw_pattern_graph.npz")
            pattern_count, channel_count = pattern["stable_level_center"].shape
            np.savez_compressed(
                raw_npz_path,
                pattern_id=np.repeat(np.arange(pattern_count), channel_count),
                channel_id=np.tile(np.arange(channel_count), pattern_count),
                shape_pattern_id=np.arange(pattern_count),
                mean_shape=pattern["prototype_mean_shape"].numpy(),
                medoid_shape=pattern["prototype_medoid"].numpy(),
                shape_std=pattern["shape_std"].numpy(),
                support=pattern["support_count"].numpy(),
                anchor_support=pattern["anchor_support"].numpy(),
                predictive_stability=pattern["predictive_stability"].numpy(),
                future_ratio=pattern["future_ratio"].numpy(),
                overall_stability=pattern["overall_stability"].numpy(),
                pattern_distance_cdf=pattern["distance_cdf"].numpy(),
                medoid_window_index=pattern["medoid_window_index"].numpy(),
                medoid_variable_index=pattern["medoid_variable_index"].numpy(),
                medoid_patch_index=pattern["medoid_patch_index"].numpy(),
                medoid_absolute_time_index=pattern["medoid_absolute_time_index"].numpy(),
                src_pattern=mapping["src_index"].numpy(),
                dst_pattern=mapping["dst_index"].numpy(),
                mapping_support=mapping["window_support"].numpy(),
                mapping_count=mapping["counts"].numpy(),
                mapping_stability=mapping["stability"].numpy(),
                mapping_coverage=mapping["coverage"].numpy(),
                mapping_sample_entropy=mapping["sample_entropy"].numpy(),
                mapping_sample_concentration=mapping["sample_concentration"].numpy(),
                mapping_p_inv=mapping["p_inv"].numpy(),
                mapping_p_var=mapping["p_var"].numpy(),
                mean_raw_delta=mapping["delta_means"].numpy(),
                local_raw_delta=mapping["local_delta_means"].numpy(),
                delta_variance=mapping["delta_variances"].numpy(),
                edge_predictive_gain=mapping["predictive_gain"].numpy(),
                stable_level_center=pattern["stable_level_center"].numpy().reshape(-1),
                stable_level_mad=pattern["stable_level_mad"].numpy().reshape(-1),
                stable_scale_center=pattern["stable_log_scale_center"].exp().numpy().reshape(-1),
                stable_log_scale_center=pattern["stable_log_scale_center"].numpy().reshape(-1),
                stable_log_scale_mad=pattern["stable_log_scale_mad"].numpy().reshape(-1),
                geometry_support=pattern["geometry_support"].numpy().reshape(-1),
                shift_score_cdf=pattern["shift_score_cdf"].numpy(),
                scale_score_cdf=pattern["scale_score_cdf"].numpy(),
                source_split=np.asarray(["train"]),
                signal_scale=np.asarray(["pre_patch_normalization_not_physical_raw_units"]),
            )
            artifact["metadata"]["raw_interpretable_artifact"] = raw_npz_path
            torch.save(artifact, graph_path)
            print("Saved interpretable raw graph: {}".format(raw_npz_path))
        print(
            "Built FPem-PMG graph from TRAIN only: {} patterns, {} temporal edges".format(
                self.pmg.pattern_graph.num_active, self.pmg.mapping_graph.num_active
            )
        )
        self.freeze_fpem_pmg_graph_space()
        self.prepare_fpem_future_mapping(train_loader, device, checkpoint_dir, force=force)
        if was_training:
            self.train(True)
        return graph_path

    @torch.no_grad()
    def _crossfit_signals(
        self,
        x_enc: torch.Tensor,
        target: torch.Tensor,
        target_channel_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm, mean, std = self._norm(x_enc)
        encoder_features = self.encoder_backbone.forward_features(x_norm)
        hidden = encoder_features["hidden"]
        decomposition = self._decompose_features(hidden, encoder_features, x_enc)
        features = torch.stack(
            [
                decomposition["variation_activation"].mean(dim=(1, 2)),
                decomposition["pattern_novelty"].mean(dim=(1, 2)),
                decomposition["mapping_novelty"].mean(dim=(1, 2)),
            ],
            dim=-1,
        )
        z_env_candidate = decomposition["z_inv"] + decomposition["temporal_gate"].unsqueeze(-1) * (
            decomposition["z_env"] - decomposition["z_inv"]
        )
        y_inv = self._denorm(self._forecast_head(decomposition["z_inv"]), mean, std)
        y_env = self._denorm(self._forecast_head(z_env_candidate), mean, std)
        observed = target[:, -self.pred_len :, target_channel_start:]
        y_inv = y_inv[:, -self.pred_len :, target_channel_start:]
        y_env = y_env[:, -self.pred_len :, target_channel_start:]
        gain = (y_inv - observed).square().mean(dim=(1, 2)) - (
            y_env - observed
        ).square().mean(dim=(1, 2))
        return features, gain

    def calibrate_fpem_pmg_reliability(
        self,
        train_loader: Any,
        device: torch.device,
        checkpoint_dir: str,
        target_channel_start: int = 0,
    ) -> Optional[str]:
        """Fit the linear reliability head with TRAIN-only chronological cross-fitting."""
        if not (
            self.fpem_pmg_enabled
            and self.pmg.use_pattern
            and self.pmg.use_env
            and self.pmg.use_reliability
            and self.pmg.graph_ready
        ):
            return None
        was_training = self.training
        self.eval()
        embeddings = []
        future_targets = []
        input_windows = []
        observed_targets = []
        with torch.no_grad():
            for batch in train_loader:
                batch_x = batch[0].float().to(device)
                batch_y = batch[1].float().to(device)
                if self.representation_space == "raw":
                    embeddings.append(self._raw_patch_data(batch_x)["normalized"].cpu())
                else:
                    x_norm, _, _ = self._norm(batch_x)
                    encoder_features = self.encoder_backbone.forward_features(x_norm)
                    graph_tokens = self._select_graph_tokens(encoder_features)
                    embeddings.append(self.pmg.projector(graph_tokens).cpu())
                future_targets.append(batch_y[:, -self.pred_len :, :].permute(0, 2, 1).cpu())
                input_windows.append(batch_x.cpu())
                observed_targets.append(batch_y.cpu())
        if not embeddings:
            return None
        embeddings_all = torch.cat(embeddings)
        futures_all = torch.cat(future_targets)
        inputs_all = torch.cat(input_windows)
        targets_all = torch.cat(observed_targets)
        raw_all = self._raw_patch_data(inputs_all) if self.representation_space == "raw" else None
        window_count = embeddings_all.shape[0]
        if window_count < 5:
            return None

        final_graph = self.pmg.graph_artifact(self._graph_metadata)
        builder = PatternMappingGraphBuilder(
            self.pmg.pattern_dim,
            seq_len=self.seq_len,
            representation_space=self.representation_space,
            patch_len=self.patch_len,
            stride=self.stride,
            predictive_stability_mode=self.predictive_stability_mode,
        )
        boundaries = [
            int(window_count * 0.4),
            int(window_count * 0.6),
            int(window_count * 0.8),
            window_count,
        ]
        all_features = []
        all_gains = []
        split_records = []
        for split_index in range(3):
            prefix_end = max(1, boundaries[split_index])
            heldout_end = max(prefix_end + 1, boundaries[split_index + 1])
            heldout_end = min(window_count, heldout_end)
            artifact = builder.build_from_embeddings(
                embeddings_all[:prefix_end],
                futures_all[:prefix_end],
                metadata={
                    "source_split": "train-crossfit-prefix",
                    "prefix_end": prefix_end,
                },
                history_last=inputs_all[:prefix_end, -1, :],
                history_std=inputs_all[:prefix_end].var(dim=1, unbiased=False).sqrt(),
                raw_patch_mean=raw_all["mean"][:prefix_end] if raw_all is not None else None,
                raw_patch_std=raw_all["std"][:prefix_end] if raw_all is not None else None,
            )
            self.pmg.load_graph_statistics(artifact)
            for start in range(prefix_end, heldout_end, 256):
                stop = min(heldout_end, start + 256)
                features, gain = self._crossfit_signals(
                    inputs_all[start:stop].to(device),
                    targets_all[start:stop].to(device),
                    target_channel_start,
                )
                all_features.append(features.cpu())
                all_gains.append(gain.cpu())
            split_records.append(
                {
                    "graph_prefix": [0, prefix_end],
                    "sample_interval": [prefix_end, heldout_end],
                }
            )
        self.pmg.load_graph_statistics(final_graph)
        if not all_features:
            return None

        calibration_features = torch.cat(all_features).to(device)
        environment_gain = torch.cat(all_gains).to(device)
        reliability = self.pmg.reliability_head
        reliability.reset_gain_statistics()
        reliability.update_gain_statistics(environment_gain)
        calibration_target = reliability.target(environment_gain, update=False)
        with torch.no_grad():
            reliability.linear.weight.zero_()
            reliability.linear.bias.zero_()
        for parameter in reliability.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(reliability.parameters(), lr=0.05)
        for _ in range(200):
            optimizer.zero_grad()
            prediction = reliability(calibration_features)
            loss = reliability.loss(prediction, calibration_target)
            loss.backward()
            optimizer.step()
        for parameter in reliability.parameters():
            parameter.requires_grad_(False)

        calibration_path = os.path.join(checkpoint_dir, "reliability_crossfit.pt")
        torch.save(
            {
                "variation_strength": calibration_features[:, 0].detach().cpu(),
                "pattern_novelty": calibration_features[:, 1].detach().cpu(),
                "mapping_novelty": calibration_features[:, 2].detach().cpu(),
                "environment_gain": environment_gain.detach().cpu(),
                "calibration_target": calibration_target.detach().cpu(),
                "splits": split_records,
                "source_split": "train",
                "reliability_state": reliability.state_dict(),
            },
            calibration_path,
        )
        print(
            "Calibrated FPem-PMG reliability on {} chronological TRAIN cross-fit samples".format(
                calibration_features.shape[0]
            )
        )
        if was_training:
            self.train(True)
        return calibration_path

    def forward_with_diagnostics(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.fpem_pmg_enabled:
            prediction = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return {"prediction": prediction}
        self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dict(self.latest_diagnostics)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del mask
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len :, :]
        raise NotImplementedError("PatchTST_FPEM currently supports forecasting only")
