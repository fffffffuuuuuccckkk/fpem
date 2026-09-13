"""Neural and frozen-graph components for Pattern-Mapping Graph FPem.

All runtime queries are pure torch operations.  Graph construction is kept in
``pattern_graph_builder.py`` so no CPU/NumPy round-trip occurs in ``forward``.
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .pattern_mapping_reliability import (
    EnvironmentFusion,
    EnvironmentReliability,
    LatentEnvironmentEncoder,
    TypedVariationFusion,
)


def _empirical_novelty(values: Tensor, sorted_reference: Tensor) -> Tensor:
    """Map distance to [0,1], where larger means more novel.

    The lower half of the training distance distribution is treated as
    familiar.  The upper half is stretched to [0,1], making the training
    median a data-derived zero-novelty boundary without an OOD threshold.
    """
    if sorted_reference.numel() == 0:
        return torch.ones_like(values)
    reference = sorted_reference.float().contiguous()
    flat = values.float().contiguous().view(-1)
    rank = torch.searchsorted(reference, flat, right=True).to(flat.dtype)
    cdf = rank / float(reference.numel())
    return ((cdf - 0.5) * 2.0).clamp(0.0, 1.0).view_as(values).to(values.dtype)


def combine_invariant_evidence(c_pat: Tensor, c_map: Tensor, mode: str) -> Tensor:
    """Combine pattern/mapping evidence for the B0/B1/B2 ablation."""
    if c_pat.shape != c_map.shape:
        raise ValueError("c_pat and c_map must have the same shape")
    if mode == "product":
        combined = c_pat * c_map
    elif mode == "soft_mapping":
        combined = c_pat * (1.0 + c_map) * 0.5
    elif mode == "pattern_only":
        combined = c_pat
    else:
        raise ValueError("cinv mode must be product, soft_mapping, or pattern_only")
    return combined.clamp(0.0, 1.0)


class PatternProjector(nn.Module):
    """A shallow normalized projection from backbone tokens to pattern space."""

    def __init__(self, hidden_dim: int, pattern_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pattern_dim),
            nn.GELU(),
            nn.Linear(pattern_dim, pattern_dim),
        )
        # Stage-0 forecasting makes the graph coordinates predictive before
        # any offline graph statistics are collected.
        self.forecast_adapter = nn.Linear(pattern_dim, hidden_dim)

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 4:
            raise ValueError("canonical hidden representation must be [B,C,P,D]")
        return F.normalize(self.net(hidden), p=2.0, dim=-1, eps=1e-6)

    def predictive_hidden(self, hidden: Tensor) -> Tensor:
        return self.forecast_adapter(self.forward(hidden))


class _ResizableBufferModule(nn.Module):
    """Allow graph buffers with data-derived sizes to load from checkpoints."""

    _dynamic_buffers: Tuple[str, ...] = ()

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        prefix: str,
        local_metadata: Mapping[str, Any],
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        for name in self._dynamic_buffers:
            key = prefix + name
            if key in state_dict:
                incoming = state_dict[key]
                current = self._buffers[name]
                self._buffers[name] = torch.empty(
                    incoming.shape, dtype=incoming.dtype, device=current.device
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class StablePatternGraph(_ResizableBufferModule):
    """Frozen local Gaussian distributions for recurrent predictive patterns."""

    _dynamic_buffers = (
        "means",
        "variances",
        "counts",
        "window_support",
        "predictive_support",
        "stability",
        "distance_cdf",
        "future_ratio",
        "future_prototypes",
        "future_variance",
        "stable_level_center",
        "stable_level_mad",
        "stable_log_scale_center",
        "stable_log_scale_mad",
        "geometry_support",
        "shift_score_cdf",
        "scale_score_cdf",
    )

    def __init__(self, pattern_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.pattern_dim = int(pattern_dim)
        self.eps = float(eps)
        self.register_buffer("means", torch.empty(0, pattern_dim))
        self.register_buffer("variances", torch.empty(0, pattern_dim))
        self.register_buffer("counts", torch.empty(0))
        self.register_buffer("window_support", torch.empty(0))
        self.register_buffer("predictive_support", torch.empty(0))
        self.register_buffer("stability", torch.empty(0))
        self.register_buffer("distance_cdf", torch.empty(0))
        self.register_buffer("future_ratio", torch.empty(0))
        self.register_buffer("future_prototypes", torch.empty(0, 0))
        self.register_buffer("future_variance", torch.empty(0))
        self.register_buffer("stable_level_center", torch.empty(0, 0))
        self.register_buffer("stable_level_mad", torch.empty(0, 0))
        self.register_buffer("stable_log_scale_center", torch.empty(0, 0))
        self.register_buffer("stable_log_scale_mad", torch.empty(0, 0))
        self.register_buffer("geometry_support", torch.empty(0, 0))
        self.register_buffer("shift_score_cdf", torch.empty(0))
        self.register_buffer("scale_score_cdf", torch.empty(0))
        self.register_buffer("ready", torch.tensor(False, dtype=torch.bool))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        optional = {
            prefix + name for name in (
                "stable_level_center", "stable_level_mad",
                "stable_log_scale_center", "stable_log_scale_mad",
                "geometry_support", "shift_score_cdf", "scale_score_cdf",
            )
        }
        missing_keys[:] = [key for key in missing_keys if key not in optional]

    @property
    def num_active(self) -> int:
        return int(self.means.shape[0])

    @torch.no_grad()
    def load_statistics(self, statistics: Mapping[str, Tensor]) -> None:
        for name in self._dynamic_buffers:
            if name not in statistics:
                if name in {
                    "future_ratio", "future_prototypes", "future_variance",
                    "stable_level_center", "stable_level_mad",
                    "stable_log_scale_center", "stable_log_scale_mad",
                    "geometry_support", "shift_score_cdf", "scale_score_cdf",
                }:
                    empty_shape = (0, 0) if name in {
                        "future_prototypes", "stable_level_center", "stable_level_mad",
                        "stable_log_scale_center", "stable_log_scale_mad", "geometry_support",
                    } else (0,)
                    self._buffers[name] = torch.empty(
                        empty_shape, device=self.ready.device
                    )
                    continue
                raise KeyError("missing pattern graph statistic: {}".format(name))
            value = torch.as_tensor(statistics[name], device=self.ready.device)
            self._buffers[name] = value.clone()
        if self.means.ndim != 2 or self.means.shape[-1] != self.pattern_dim:
            raise ValueError("pattern means have incompatible feature dimension")
        self.ready.fill_(self.num_active > 0)

    @property
    def geometry_ready(self) -> bool:
        return bool(
            self.stable_level_center.ndim == 2
            and self.stable_level_center.shape[0] == self.num_active
            and self.stable_level_center.numel() > 0
            and self.shift_score_cdf.numel() > 0
            and self.scale_score_cdf.numel() > 0
        )

    def geometry(
        self,
        responsibility: Tensor,
        current_level: Tensor,
        current_scale: Tensor,
    ) -> Dict[str, Tensor]:
        """Reconstruct channel-conditioned stable geometry and its novelty."""
        if not self.geometry_ready:
            raise RuntimeError("factorized raw variation requires rebuilt raw graph")
        if responsibility.ndim != 4 or current_level.shape != responsibility.shape[:-1]:
            raise ValueError("responsibility/current_level must be [B,C,P,K]/[B,C,P]")
        if current_scale.shape != current_level.shape:
            raise ValueError("current_scale must match current_level")
        channels = responsibility.shape[1]
        if self.stable_level_center.shape[1] != channels:
            raise ValueError("raw geometry channel count does not match current input")
        dtype = responsibility.dtype
        level_center = self.stable_level_center.to(dtype=dtype)
        level_mad = self.stable_level_mad.to(dtype=dtype).clamp_min(self.eps)
        log_scale_center = self.stable_log_scale_center.to(dtype=dtype)
        log_scale_mad = self.stable_log_scale_mad.to(dtype=dtype).clamp_min(self.eps)
        stable_level = torch.einsum("bcpk,kc->bcp", responsibility, level_center)
        stable_log_scale = torch.einsum("bcpk,kc->bcp", responsibility, log_scale_center)
        stable_level_mad = torch.einsum("bcpk,kc->bcp", responsibility, level_mad).clamp_min(self.eps)
        stable_log_scale_mad = torch.einsum(
            "bcpk,kc->bcp", responsibility, log_scale_mad
        ).clamp_min(self.eps)
        stable_scale = stable_log_scale.exp().clamp_min(self.eps)
        current_log_scale = current_scale.clamp_min(self.eps).log()
        shift_signed = (current_level - stable_level) / stable_scale
        scale_signed = current_log_scale - stable_log_scale
        shift_score = (current_level - stable_level).abs() / stable_level_mad
        scale_score = scale_signed.abs() / stable_log_scale_mad
        return {
            "stable_level": stable_level,
            "stable_log_scale": stable_log_scale,
            "stable_scale": stable_scale,
            "stable_level_mad": stable_level_mad,
            "stable_log_scale_mad": stable_log_scale_mad,
            "shift_signed": shift_signed,
            "scale_signed": scale_signed,
            "shift_score": shift_score,
            "scale_score": scale_score,
            "u_shift": _empirical_novelty(shift_score, self.shift_score_cdf),
            "u_scale": _empirical_novelty(scale_score, self.scale_score_cdf),
        }

    def export(self) -> Dict[str, Tensor]:
        result = {name: getattr(self, name).detach().cpu() for name in self._dynamic_buffers}
        result["ready"] = self.ready.detach().cpu()
        return result

    def query(self, query: Tensor) -> Dict[str, Tensor]:
        """Return soft responsibilities, NULL probability, and stable context."""
        if query.ndim != 4 or query.shape[-1] != self.pattern_dim:
            raise ValueError("pattern query must have shape [B,C,P,pattern_dim]")
        prefix = query.shape[:-1]
        if not bool(self.ready.item()) or self.num_active == 0:
            return {
                "responsibility": query.new_zeros(*prefix, 0),
                "novelty": query.new_ones(prefix),
                "null_probability": query.new_ones(prefix),
                "context": query.new_zeros(query.shape),
                "stability_evidence": query.new_zeros(prefix),
                "shape_compatibility": query.new_zeros(prefix),
                "recurrence_confidence": query.new_zeros(prefix),
                "predictive_confidence": query.new_zeros(prefix),
                "best_distance": query.new_full(prefix, float("inf")),
            }

        means = self.means.to(dtype=query.dtype)
        variances = self.variances.to(dtype=query.dtype).clamp_min(self.eps)
        flat = query.reshape(-1, self.pattern_dim)
        inv_var = variances.reciprocal()
        # Expanded diagonal Mahalanobis formula avoids allocating [N,K,D].
        mahalanobis = (
            torch.matmul(flat.square(), inv_var.t())
            - 2.0 * torch.matmul(flat, (means * inv_var).t())
            + (means.square() * inv_var).sum(-1)[None, :]
        ).clamp_min(0.0) / float(self.pattern_dim)
        scores = -0.5 * mahalanobis + torch.log(self.stability.to(query.dtype).clamp_min(self.eps))[None, :]
        responsibility = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        best_distance = mahalanobis.min(dim=-1).values.view(prefix)
        novelty = _empirical_novelty(best_distance, self.distance_cdf)
        responsibility = responsibility.view(*prefix, self.num_active)
        context = torch.matmul(responsibility, means)
        stable = torch.matmul(responsibility, self.stability.to(query.dtype))
        compatibility = 1.0 - novelty
        recurrence = torch.log1p(self.window_support.to(query.dtype)) / torch.log1p(
            self.window_support.max().to(query.dtype)
        ).clamp_min(1.0)
        return {
            "responsibility": responsibility,
            "novelty": novelty,
            "null_probability": novelty,
            "context": context,
            "stability_evidence": (stable * compatibility).clamp(0.0, 1.0),
            "shape_compatibility": compatibility,
            "recurrence_confidence": torch.matmul(responsibility, recurrence).clamp(0.0, 1.0),
            "predictive_confidence": torch.matmul(
                responsibility, self.predictive_support.to(query.dtype)
            ).clamp(0.0, 1.0),
            "best_distance": best_distance,
        }


class StableMappingGraph(_ResizableBufferModule):
    """Sparse temporal edges with diagonal distributions over representation delta."""

    _dynamic_buffers = (
        "src_index",
        "dst_index",
        "counts",
        "window_support",
        "delta_means",
        "delta_variances",
        "stability",
        "distance_cdf",
        "future_prototypes",
        "future_variance",
        "predictive_gain",
        "coverage",
        "sample_entropy",
        "sample_concentration",
        "p_inv",
        "p_var",
        "local_delta_means",
        "mixture_feature_mean",
        "mixture_feature_std",
        "mixture_component_mean",
        "mixture_component_variance",
        "mixture_component_weight",
        "mixture_invariant_component",
    )

    def __init__(self, pattern_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.pattern_dim = int(pattern_dim)
        self.eps = float(eps)
        self.register_buffer("src_index", torch.empty(0, dtype=torch.long))
        self.register_buffer("dst_index", torch.empty(0, dtype=torch.long))
        self.register_buffer("counts", torch.empty(0))
        self.register_buffer("window_support", torch.empty(0))
        self.register_buffer("delta_means", torch.empty(0, pattern_dim))
        self.register_buffer("delta_variances", torch.empty(0, pattern_dim))
        self.register_buffer("stability", torch.empty(0))
        self.register_buffer("distance_cdf", torch.empty(0))
        self.register_buffer("future_prototypes", torch.empty(0, 0))
        self.register_buffer("future_variance", torch.empty(0))
        self.register_buffer("predictive_gain", torch.empty(0))
        self.register_buffer("coverage", torch.empty(0))
        self.register_buffer("sample_entropy", torch.empty(0))
        self.register_buffer("sample_concentration", torch.empty(0))
        self.register_buffer("p_inv", torch.empty(0))
        self.register_buffer("p_var", torch.empty(0))
        self.register_buffer("local_delta_means", torch.empty(0, pattern_dim))
        self.register_buffer("mixture_feature_mean", torch.empty(0))
        self.register_buffer("mixture_feature_std", torch.empty(0))
        self.register_buffer("mixture_component_mean", torch.empty(0, 0))
        self.register_buffer("mixture_component_variance", torch.empty(0, 0))
        self.register_buffer("mixture_component_weight", torch.empty(0))
        self.register_buffer("mixture_invariant_component", torch.empty(0, dtype=torch.long))
        self.register_buffer("ready", torch.tensor(False, dtype=torch.bool))

    @property
    def num_active(self) -> int:
        return int(self.src_index.numel())

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        optional = {
            prefix + name for name in (
                "coverage", "sample_entropy", "sample_concentration", "p_inv", "p_var",
                "local_delta_means", "mixture_feature_mean", "mixture_feature_std",
                "mixture_component_mean", "mixture_component_variance",
                "mixture_component_weight", "mixture_invariant_component",
            )
        }
        missing_keys[:] = [key for key in missing_keys if key not in optional]

    @torch.no_grad()
    def load_statistics(self, statistics: Mapping[str, Tensor]) -> None:
        for name in self._dynamic_buffers:
            if name not in statistics:
                if name in {
                    "future_prototypes", "future_variance", "predictive_gain",
                    "coverage", "sample_entropy", "sample_concentration",
                    "p_inv", "p_var", "local_delta_means",
                    "mixture_feature_mean", "mixture_feature_std",
                    "mixture_component_mean", "mixture_component_variance",
                    "mixture_component_weight", "mixture_invariant_component",
                }:
                    shape = (0, self.pattern_dim) if name == "local_delta_means" else (
                        (0, 0) if name in {"future_prototypes", "mixture_component_mean", "mixture_component_variance"}
                        else (0,)
                    )
                    dtype = torch.long if name == "mixture_invariant_component" else torch.float32
                    self._buffers[name] = torch.empty(shape, dtype=dtype, device=self.ready.device)
                    continue
                raise KeyError("missing mapping graph statistic: {}".format(name))
            value = torch.as_tensor(statistics[name], device=self.ready.device)
            self._buffers[name] = value.clone()
        self.ready.fill_(self.num_active > 0)

    def export(self) -> Dict[str, Tensor]:
        result = {name: getattr(self, name).detach().cpu() for name in self._dynamic_buffers}
        result["ready"] = self.ready.detach().cpu()
        return result

    def query(
        self,
        query: Tensor,
        responsibility: Tensor,
        pattern_means: Tensor,
        use_delta: bool = True,
    ) -> Dict[str, Tensor]:
        """Query same-variable transitions ``p-1 -> p`` against sparse edges."""
        if query.ndim != 4:
            raise ValueError("query must be [B,C,P,D]")
        batch, channels, patches, dim = query.shape
        novelty = query.new_zeros(batch, channels, patches)
        evidence = query.new_ones(batch, channels, patches)
        context = query.new_zeros(batch, channels, patches, dim)
        stable_delta = query.new_zeros(batch, channels, patches, dim)
        invariant_delta = query.new_zeros(batch, channels, patches, dim)
        variant_delta = query.new_zeros(batch, channels, patches, dim)
        invariant_activation = query.new_zeros(batch, channels, patches)
        variant_activation = query.new_zeros(batch, channels, patches)
        active_variant_edges = query.new_zeros(batch, channels, patches)
        variant_probability = query.new_zeros(batch, channels, patches)
        best_distance = query.new_zeros(batch, channels, patches)
        if patches < 2:
            return {"novelty": novelty, "stability_evidence": evidence, "context": context,
                    "stable_delta": stable_delta, "invariant_delta": invariant_delta,
                    "variant_delta": variant_delta, "invariant_activation": invariant_activation,
                    "variant_activation": variant_activation,
                    "variant_probability": variant_probability,
                    "active_variant_edges": active_variant_edges, "best_distance": best_distance}
        if not bool(self.ready.item()) or self.num_active == 0 or responsibility.shape[-1] == 0:
            novelty[..., 1:] = 1.0
            evidence[..., 1:] = 0.0
            return {"novelty": novelty, "stability_evidence": evidence, "context": context,
                    "stable_delta": stable_delta, "invariant_delta": invariant_delta,
                    "variant_delta": variant_delta, "invariant_activation": invariant_activation,
                    "variant_activation": variant_activation,
                    "variant_probability": variant_probability,
                    "active_variant_edges": active_variant_edges, "best_distance": best_distance}

        previous = responsibility[..., :-1, :].reshape(-1, responsibility.shape[-1])
        current = responsibility[..., 1:, :].reshape(-1, responsibility.shape[-1])
        src = self.src_index
        dst = self.dst_index
        delta = (query[..., 1:, :] - query[..., :-1, :]).reshape(-1, dim)
        transition_count = previous.shape[0]
        # Bound all N x E intermediates independently of data-derived graph size.
        chunk_size = max(1, 2_000_000 // max(1, self.num_active))
        compatible_parts = []
        stable_parts = []
        context_parts = []
        stable_delta_parts = []
        invariant_delta_parts = []
        variant_delta_parts = []
        invariant_activation_parts = []
        variant_activation_parts = []
        active_variant_edge_parts = []
        variant_probability_parts = []
        distance_parts = []
        edge_stability = self.stability.to(query.dtype)[None, :]
        dst_means = pattern_means.to(query.dtype)[dst]
        inv_var = self.delta_variances.to(query.dtype).clamp_min(self.eps).reciprocal()
        delta_mu = self.delta_means.to(query.dtype)
        local_delta_mu = (
            self.local_delta_means.to(query.dtype)
            if self.local_delta_means.shape == self.delta_means.shape
            else delta_mu
        )
        edge_p_inv = (
            self.p_inv.to(query.dtype) if self.p_inv.numel() == self.num_active
            else torch.ones(self.num_active, dtype=query.dtype, device=query.device)
        )[None, :]
        edge_p_var = (
            self.p_var.to(query.dtype) if self.p_var.numel() == self.num_active
            else torch.zeros(self.num_active, dtype=query.dtype, device=query.device)
        )[None, :]
        constant = (delta_mu.square() * inv_var).sum(-1)[None, :]
        for start in range(0, transition_count, chunk_size):
            stop = min(transition_count, start + chunk_size)
            edge_mass = previous[start:stop, src] * current[start:stop, dst]
            delta_chunk = delta[start:stop]
            if use_delta:
                distance = (
                    torch.matmul(delta_chunk.square(), inv_var.t())
                    - 2.0 * torch.matmul(delta_chunk, (delta_mu * inv_var).t())
                    + constant
                ).clamp_min(0.0) / float(dim)
                edge_compatibility = 1.0 - _empirical_novelty(distance, self.distance_cdf)
                weighted_distance = (edge_mass * distance).sum(-1) / edge_mass.sum(-1).clamp_min(self.eps)
            else:
                edge_compatibility = torch.ones_like(edge_mass)
                weighted_distance = torch.zeros(stop - start, dtype=query.dtype, device=query.device)
            compatible = (edge_mass * edge_compatibility).sum(-1).clamp(0.0, 1.0)
            context_weights = edge_mass * edge_compatibility * edge_stability
            stable = context_weights.sum(-1).clamp(0.0, 1.0)
            mapped = torch.matmul(context_weights, dst_means)
            mapped = mapped / context_weights.sum(-1, keepdim=True).clamp_min(self.eps)
            expected_delta = torch.matmul(context_weights, delta_mu)
            expected_delta = expected_delta / context_weights.sum(-1, keepdim=True).clamp_min(self.eps)
            invariant_weights = context_weights * edge_p_inv
            variant_weights = context_weights * edge_p_var
            posterior_variant_mass = (edge_mass * edge_p_var).sum(-1) / edge_mass.sum(-1).clamp_min(self.eps)
            invariant_weight_sum = invariant_weights.sum(-1, keepdim=True)
            variant_weight_sum = variant_weights.sum(-1, keepdim=True)
            expected_invariant_delta = torch.matmul(invariant_weights, delta_mu)
            expected_invariant_delta = expected_invariant_delta / invariant_weight_sum.clamp_min(self.eps)
            expected_variant_delta = torch.matmul(variant_weights, local_delta_mu)
            expected_variant_delta = expected_variant_delta / variant_weight_sum.clamp_min(self.eps)
            normalized_variant = variant_weights / variant_weight_sum.clamp_min(self.eps)
            effective_variant_edges = normalized_variant.square().sum(-1).clamp_min(self.eps).reciprocal()
            effective_variant_edges = torch.where(
                variant_weight_sum.squeeze(-1) > self.eps,
                effective_variant_edges,
                torch.zeros_like(effective_variant_edges),
            )
            compatible_parts.append(compatible)
            stable_parts.append(stable)
            context_parts.append(mapped)
            stable_delta_parts.append(expected_delta)
            invariant_delta_parts.append(expected_invariant_delta)
            variant_delta_parts.append(expected_variant_delta)
            invariant_activation_parts.append(invariant_weight_sum.squeeze(-1).clamp(0.0, 1.0))
            variant_activation_parts.append(variant_weight_sum.squeeze(-1).clamp(0.0, 1.0))
            active_variant_edge_parts.append(effective_variant_edges)
            variant_probability_parts.append(posterior_variant_mass.clamp(0.0, 1.0))
            distance_parts.append(weighted_distance)
        compatible_mass = torch.cat(compatible_parts)
        stable_mass = torch.cat(stable_parts)
        mapped_context = torch.cat(context_parts)
        expected_stable_delta = torch.cat(stable_delta_parts)
        expected_invariant_delta = torch.cat(invariant_delta_parts)
        expected_variant_delta = torch.cat(variant_delta_parts)
        invariant_activation_mass = torch.cat(invariant_activation_parts)
        variant_activation_mass = torch.cat(variant_activation_parts)
        effective_variant_edges = torch.cat(active_variant_edge_parts)
        posterior_variant_mass = torch.cat(variant_probability_parts)
        weighted_distance = torch.cat(distance_parts)

        transition_shape = (batch, channels, patches - 1)
        novelty = torch.cat(
            [query.new_zeros(batch, channels, 1), (1.0 - compatible_mass).view(transition_shape)], dim=-1
        )
        evidence = torch.cat(
            [query.new_ones(batch, channels, 1), stable_mass.view(transition_shape)], dim=-1
        )
        context = torch.cat(
            [query.new_zeros(batch, channels, 1, dim), mapped_context.view(batch, channels, patches - 1, dim)],
            dim=2,
        )
        stable_delta = torch.cat(
            [query.new_zeros(batch, channels, 1, dim),
             expected_stable_delta.view(batch, channels, patches - 1, dim)],
            dim=2,
        )
        invariant_delta = torch.cat(
            [query.new_zeros(batch, channels, 1, dim),
             expected_invariant_delta.view(batch, channels, patches - 1, dim)], dim=2
        )
        variant_delta = torch.cat(
            [query.new_zeros(batch, channels, 1, dim),
             expected_variant_delta.view(batch, channels, patches - 1, dim)], dim=2
        )
        invariant_activation = torch.cat(
            [query.new_zeros(batch, channels, 1),
             invariant_activation_mass.view(transition_shape)], dim=-1
        )
        variant_activation = torch.cat(
            [query.new_zeros(batch, channels, 1),
             variant_activation_mass.view(transition_shape)], dim=-1
        )
        active_variant_edges = torch.cat(
            [query.new_zeros(batch, channels, 1),
             effective_variant_edges.view(transition_shape)], dim=-1
        )
        variant_probability = torch.cat(
            [query.new_zeros(batch, channels, 1),
             posterior_variant_mass.view(transition_shape)], dim=-1
        )
        best_distance = torch.cat(
            [query.new_zeros(batch, channels, 1), weighted_distance.view(transition_shape)], dim=-1
        )
        return {"novelty": novelty, "stability_evidence": evidence, "context": context,
                "stable_delta": stable_delta, "invariant_delta": invariant_delta,
                "variant_delta": variant_delta, "invariant_activation": invariant_activation,
                "variant_activation": variant_activation,
                "variant_probability": variant_probability,
                "active_variant_edges": active_variant_edges, "best_distance": best_distance}


class PatternMappingRetriever(nn.Module):
    """Combine continuous pattern and temporal-mapping evidence."""

    def __init__(self, pattern_graph: StablePatternGraph, mapping_graph: StableMappingGraph) -> None:
        super().__init__()
        self.pattern_graph = pattern_graph
        self.mapping_graph = mapping_graph

    def forward(
        self,
        query: Tensor,
        use_mapping: bool,
        use_mapping_delta: bool,
        cinv_mode: str = "product",
    ) -> Dict[str, Tensor]:
        pattern = self.pattern_graph.query(query)
        if use_mapping:
            mapping = self.mapping_graph.query(
                query,
                pattern["responsibility"],
                self.pattern_graph.means,
                use_delta=use_mapping_delta,
            )
        else:
            mapping = {
                "novelty": query.new_zeros(query.shape[:-1]),
                "stability_evidence": query.new_ones(query.shape[:-1]),
                "context": query.new_zeros(query.shape),
                "stable_delta": query.new_zeros(query.shape),
                "invariant_delta": query.new_zeros(query.shape),
                "variant_delta": query.new_zeros(query.shape),
                "invariant_activation": query.new_zeros(query.shape[:-1]),
                "variant_activation": query.new_zeros(query.shape[:-1]),
                "variant_probability": query.new_zeros(query.shape[:-1]),
                "active_variant_edges": query.new_zeros(query.shape[:-1]),
                "best_distance": query.new_zeros(query.shape[:-1]),
            }
        c_pat = pattern["stability_evidence"]
        c_map = mapping["stability_evidence"]
        c_inv = combine_invariant_evidence(c_pat, c_map, cinv_mode)
        return {"pattern": pattern, "mapping": mapping, "c_pat": c_pat, "c_map": c_map, "c_inv": c_inv}


class InvariantPatternEncoder(nn.Module):
    """Build invariant content with an optional confidence-gated graph residual."""

    MODES = {"interpolate", "hidden_only", "stable_relation_correction"}

    def __init__(self, hidden_dim: int, pattern_dim: int,
                 zinv_mode: str = "stable_relation_correction") -> None:
        super().__init__()
        if zinv_mode not in self.MODES:
            raise ValueError("invalid zinv_mode: {}".format(zinv_mode))
        self.zinv_mode = zinv_mode
        self.context_projection = nn.Sequential(
            nn.LayerNorm(pattern_dim * 2),
            nn.Linear(pattern_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.current_stable_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.stable_delta_projection = nn.Sequential(
            nn.LayerNorm(pattern_dim),
            nn.Linear(pattern_dim, hidden_dim, bias=False),
        )
        self.graph_correction_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.graph_correction_encoder[-1].weight)
        nn.init.zeros_(self.graph_correction_encoder[-1].bias)
        nn.init.zeros_(self.graph_gate[-1].weight)
        nn.init.constant_(self.graph_gate[-1].bias, -2.0)

    def forward(
        self,
        hidden: Tensor,
        pattern_context: Tensor,
        mapping_context: Tensor,
        evidence: Tensor,
        stable_mapping_delta: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        graph_context = self.context_projection(torch.cat([pattern_context, mapping_context], dim=-1))
        current_stable_content = self.current_stable_encoder(hidden)
        weight = evidence.unsqueeze(-1).clamp(0.0, 1.0)
        if stable_mapping_delta is None:
            stable_mapping_delta = pattern_context.new_zeros(pattern_context.shape)
        if stable_mapping_delta.shape != pattern_context.shape:
            raise ValueError("stable_mapping_delta must match pattern context [B,C,P,d_p]")
        projected_delta = self.stable_delta_projection(stable_mapping_delta)
        correction_input = torch.cat(
            [current_stable_content, graph_context, graph_context - current_stable_content, projected_delta],
            dim=-1,
        )
        raw_correction = self.graph_correction_encoder(correction_input)
        learned_gate = torch.sigmoid(
            self.graph_gate(torch.cat([current_stable_content, graph_context], dim=-1))
        )
        graph_gate = weight * learned_gate
        graph_correction = graph_gate * raw_correction
        if self.zinv_mode == "interpolate":
            z_inv = weight * current_stable_content + (1.0 - weight) * graph_context
        elif self.zinv_mode == "hidden_only":
            z_inv = current_stable_content
        else:
            z_inv = current_stable_content + graph_correction
        return {
            "z_inv": z_inv,
            "graph_context": graph_context,
            "invariant_base": current_stable_content,
            "graph_correction": graph_correction,
            "graph_correction_raw": raw_correction,
            "graph_gate": graph_gate,
            "stable_mapping_delta_hidden": projected_delta,
        }


class VariantMappingCorrection(nn.Module):
    """Sample-conditioned relation drift correction; identity at initialization."""

    def __init__(self, hidden_dim: int, pattern_dim: int) -> None:
        super().__init__()
        self.delta_projection = nn.Sequential(
            nn.LayerNorm(pattern_dim), nn.Linear(pattern_dim, hidden_dim, bias=False)
        )
        self.correction_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.correction_encoder[-1].weight)
        nn.init.zeros_(self.correction_encoder[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, z_inv: Tensor, variant_delta: Tensor, activation: Tensor) -> Dict[str, Tensor]:
        projected = self.delta_projection(variant_delta)
        features = torch.cat([z_inv, projected], dim=-1)
        raw = self.correction_encoder(features)
        learned_gate = torch.sigmoid(self.gate(features))
        relation_gate = activation.unsqueeze(-1).clamp(0.0, 1.0) * learned_gate
        correction = relation_gate * raw
        return {
            "variant_mapping_delta_hidden": projected,
            "variant_mapping_correction_raw": raw,
            "variant_mapping_gate": relation_gate,
            "variant_mapping_correction": correction,
        }


class VariantPatternEncoder(nn.Module):
    """Independent variant encoder; deliberately not ``H - Z_inv``."""

    MODES = {
        "latent_only", "raw_deviation", "factorized_shape",
        "factorized_geometry", "factorized_full",
    }

    def __init__(
        self,
        hidden_dim: int,
        pattern_dim: int,
        input_mode: str = "latent_only",
    ) -> None:
        super().__init__()
        if input_mode not in self.MODES:
            raise ValueError("invalid variant input mode: {}".format(input_mode))
        self.input_mode = input_mode
        self.raw_deviation_projection = (
            nn.Linear(pattern_dim, hidden_dim) if input_mode == "raw_deviation" else None
        )
        latent_input_dim = hidden_dim * (3 if input_mode == "raw_deviation" else 2)
        self.net = nn.Sequential(
            nn.LayerNorm(latent_input_dim),
            nn.Linear(latent_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Instantiate the residual branch for latent_only as well.  It remains
        # unused there, but keeps all shared latent/environment initialization
        # identical across V0/Vshape/Vgeo/Vfull under the same seed.
        factorized_modules = input_mode != "raw_deviation"
        self.shape_projection = nn.Linear(pattern_dim, hidden_dim) if factorized_modules else None
        self.geometry_projection = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        ) if factorized_modules else None
        self.raw_variation_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        ) if factorized_modules else None
        self.raw_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        ) if factorized_modules else None
        if factorized_modules:
            nn.init.zeros_(self.raw_variation_encoder[-1].weight)
            nn.init.zeros_(self.raw_variation_encoder[-1].bias)
            nn.init.zeros_(self.raw_gate[-1].weight)
            nn.init.constant_(self.raw_gate[-1].bias, -2.0)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        if self.input_mode == "latent_only":
            optional_prefixes = tuple(
                prefix + name for name in (
                    "shape_projection.", "geometry_projection.",
                    "raw_variation_encoder.", "raw_gate.",
                )
            )
            missing_keys[:] = [
                key for key in missing_keys if not key.startswith(optional_prefixes)
            ]

    def forward(
        self,
        hidden: Tensor,
        z_inv_detached: Tensor,
        pattern_deviation: Tensor,
        geometry_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if self.input_mode == "raw_deviation":
            projected = self.raw_deviation_projection(pattern_deviation)
            encoded_input = torch.cat([hidden, z_inv_detached, projected], dim=-1)
        else:
            projected = hidden.new_zeros(hidden.shape)
            encoded_input = torch.cat([hidden, z_inv_detached], dim=-1)
        latent = self.net(encoded_input)
        raw_gate = hidden.new_zeros(*hidden.shape[:-1], 1)
        raw_correction = hidden.new_zeros(hidden.shape)
        raw_correction_raw = hidden.new_zeros(hidden.shape)
        shape_encoded = hidden.new_zeros(hidden.shape)
        geometry_encoded = hidden.new_zeros(hidden.shape)
        if self.input_mode.startswith("factorized_"):
            if geometry_features is None or geometry_features.shape != (*hidden.shape[:-1], 4):
                raise ValueError("factorized variation requires geometry features [B,C,P,4]")
            if self.input_mode in {"factorized_shape", "factorized_full"}:
                shape_encoded = self.shape_projection(pattern_deviation)
            if self.input_mode in {"factorized_geometry", "factorized_full"}:
                geometry_encoded = self.geometry_projection(geometry_features)
            raw_correction_raw = self.raw_variation_encoder(
                torch.cat([shape_encoded, geometry_encoded], dim=-1)
            )
            raw_gate = torch.sigmoid(self.raw_gate(torch.cat(
                [hidden, z_inv_detached, shape_encoded, geometry_encoded], dim=-1
            )))
            raw_correction = raw_gate * raw_correction_raw
        return {
            "candidate": latent + raw_correction,
            "latent": latent,
            "raw_deviation_projected": projected,
            "shape_encoded": shape_encoded,
            "geometry_encoded": geometry_encoded,
            "raw_gate": raw_gate,
            "raw_correction": raw_correction,
            "raw_correction_raw": raw_correction_raw,
        }


class PatternMappingFPem(nn.Module):
    """Full invariant/variant/environment decomposition over canonical tokens."""

    def __init__(
        self,
        hidden_dim: int,
        pattern_dim: int = 32,
        env_dim: int = 16,
        use_pattern: bool = True,
        use_mapping: bool = True,
        use_mapping_delta: bool = True,
        use_variant: bool = True,
        use_env: bool = True,
        use_reliability: bool = True,
        cinv_mode: str = "pattern_only",
        zinv_mode: str = "stable_relation_correction",
        variant_input_mode: str = "latent_only",
        mapping_use_mode: str = "full",
        fusion_mode: str = "legacy_environment",
        typed_fusion_components: str = "full",
        relation_mapping_mode: str = "legacy",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pattern_dim = int(pattern_dim)
        self.env_dim = int(env_dim)
        self.use_pattern = bool(use_pattern)
        self.use_mapping = bool(use_mapping)
        self.use_mapping_delta = bool(use_mapping_delta)
        self.use_variant = bool(use_variant)
        self.use_env = bool(use_env)
        self.use_reliability = bool(use_reliability)
        if cinv_mode not in {"product", "soft_mapping", "pattern_only"}:
            raise ValueError("invalid cinv_mode: {}".format(cinv_mode))
        self.cinv_mode = cinv_mode
        if zinv_mode not in InvariantPatternEncoder.MODES:
            raise ValueError("invalid zinv_mode: {}".format(zinv_mode))
        self.zinv_mode = zinv_mode
        self.variant_input_mode = variant_input_mode
        if mapping_use_mode not in {"none", "context_only", "delta_only", "full"}:
            raise ValueError("invalid mapping use mode: {}".format(mapping_use_mode))
        self.mapping_use_mode = mapping_use_mode
        if fusion_mode not in {"invariant_only", "legacy_environment", "typed_raw"}:
            raise ValueError("invalid fusion mode: {}".format(fusion_mode))
        self.fusion_mode = fusion_mode
        if typed_fusion_components not in TypedVariationFusion.COMPONENTS:
            raise ValueError("invalid typed fusion components: {}".format(typed_fusion_components))
        self.typed_fusion_components = typed_fusion_components
        relation_modes = {"legacy", "decomposed_no_variant", "decomposed_full", "variant_only"}
        if relation_mapping_mode not in relation_modes:
            raise ValueError("invalid relation mapping mode: {}".format(relation_mapping_mode))
        self.relation_mapping_mode = relation_mapping_mode
        self.projector = PatternProjector(hidden_dim, pattern_dim)
        self.pattern_graph = StablePatternGraph(pattern_dim)
        self.mapping_graph = StableMappingGraph(pattern_dim)
        self.retriever = PatternMappingRetriever(self.pattern_graph, self.mapping_graph)
        self.invariant_encoder = InvariantPatternEncoder(hidden_dim, pattern_dim, zinv_mode)
        self.variant_encoder = VariantPatternEncoder(hidden_dim, pattern_dim, variant_input_mode)
        self.environment_encoder = LatentEnvironmentEncoder(hidden_dim, env_dim)
        self.environment_fusion = EnvironmentFusion(hidden_dim, env_dim)
        self.reliability_head = EnvironmentReliability()
        self.typed_fusion = TypedVariationFusion(
            hidden_dim, pattern_dim, typed_fusion_components
        )
        self.variant_mapping_correction = VariantMappingCorrection(hidden_dim, pattern_dim)
        self.register_load_state_dict_post_hook(self._legacy_typed_state_compatibility)

    def _legacy_typed_state_compatibility(self, module, incompatible_keys) -> None:
        incompatible_keys.missing_keys[:] = [
            key for key in incompatible_keys.missing_keys
            if "variant_mapping_correction." not in key
            and not (
                self.fusion_mode != "typed_raw"
                and (".typed_fusion." in key or key.startswith("typed_fusion."))
            )
        ]

    @property
    def graph_ready(self) -> bool:
        return bool(self.pattern_graph.ready.item())

    def decompose(
        self,
        hidden: Tensor,
        graph_tokens: Optional[Tensor] = None,
        graph_query: Optional[Tensor] = None,
        raw_patch_mean: Optional[Tensor] = None,
        raw_patch_std: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if hidden.ndim != 4 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError("hidden must be canonical [B,C,P,D]")
        if graph_query is None:
            if graph_tokens is None:
                graph_tokens = hidden
            if graph_tokens.shape != hidden.shape:
                raise ValueError("graph_tokens and hidden must share canonical [B,C,P,D]")
            query = self.projector(graph_tokens)
        else:
            if graph_query.ndim != 4 or graph_query.shape[:-1] != hidden.shape[:-1]:
                raise ValueError("graph_query must share [B,C,P] with hidden")
            if graph_query.shape[-1] != self.pattern_dim:
                raise ValueError("graph_query has incompatible pattern dimension")
            query = graph_query
            graph_tokens = graph_query
        if self.use_pattern:
            retrieved = self.retriever(query, self.use_mapping, self.use_mapping_delta, self.cinv_mode)
            pattern_context = retrieved["pattern"]["context"]
            mapping_context = retrieved["mapping"]["context"]
            stable_mapping_delta = retrieved["mapping"]["stable_delta"]
            invariant_mapping_delta = retrieved["mapping"]["invariant_delta"]
            variant_mapping_delta = retrieved["mapping"]["variant_delta"]
            invariant_mapping_activation = retrieved["mapping"]["invariant_activation"]
            variant_mapping_activation = retrieved["mapping"]["variant_activation"]
            variant_mapping_probability = retrieved["mapping"]["variant_probability"]
            active_variant_edges = retrieved["mapping"]["active_variant_edges"]
            c_pat = retrieved["c_pat"]
            c_map = retrieved["c_map"]
            # C-series semantics are fixed to B2: mapping describes the
            # correction direction but never decides invariant activation.
            c_inv = c_pat
            pattern_novelty = retrieved["pattern"]["novelty"]
            mapping_novelty = retrieved["mapping"]["novelty"]
            null_probability = retrieved["pattern"]["null_probability"]
        else:
            pattern_context = torch.zeros_like(query)
            mapping_context = torch.zeros_like(query)
            stable_mapping_delta = torch.zeros_like(query)
            invariant_mapping_delta = torch.zeros_like(query)
            variant_mapping_delta = torch.zeros_like(query)
            c_inv = torch.ones_like(query[..., 0])
            invariant_mapping_activation = torch.zeros_like(c_inv)
            variant_mapping_activation = torch.zeros_like(c_inv)
            variant_mapping_probability = torch.zeros_like(c_inv)
            active_variant_edges = torch.zeros_like(c_inv)
            c_pat = c_inv
            c_map = c_inv
            pattern_novelty = torch.zeros_like(c_inv)
            mapping_novelty = torch.zeros_like(c_inv)
            null_probability = torch.zeros_like(c_inv)

        correction_mapping_context = (
            mapping_context if self.mapping_use_mode in {"context_only", "full"}
            else torch.zeros_like(mapping_context)
        )
        selected_invariant_delta = (
            stable_mapping_delta if self.relation_mapping_mode == "legacy"
            else invariant_mapping_delta
        )
        if self.relation_mapping_mode == "variant_only":
            selected_invariant_delta = torch.zeros_like(selected_invariant_delta)
        correction_stable_delta = (
            selected_invariant_delta if self.mapping_use_mode in {"delta_only", "full"}
            else torch.zeros_like(stable_mapping_delta)
        )
        invariant = self.invariant_encoder(
            hidden, pattern_context, correction_mapping_context, c_inv, correction_stable_delta
        )
        z_inv = invariant["z_inv"]
        consensus_context = invariant["graph_context"]
        actual_mapping_delta = torch.cat(
            [query.new_zeros(query.shape[0], query.shape[1], 1, query.shape[-1]),
             query[..., 1:, :] - query[..., :-1, :]],
            dim=2,
        )
        mapping_deviation = actual_mapping_delta - stable_mapping_delta
        responsibility = retrieved["pattern"]["responsibility"] if self.use_pattern else query.new_zeros(
            *query.shape[:-1], 0
        )
        pattern_future_prediction = query.new_zeros(*query.shape[:-1], 0)
        mapping_future_prediction = query.new_zeros(*query.shape[:-1], 0)
        if self.pattern_graph.future_prototypes.numel() and responsibility.shape[-1]:
            pattern_future_prediction = torch.matmul(
                responsibility, self.pattern_graph.future_prototypes.to(query.dtype)
            )
            mapping_future_prediction = pattern_future_prediction.clone()
            if self.mapping_graph.future_prototypes.numel() and query.shape[2] > 1:
                previous = responsibility[..., :-1, :]
                current = responsibility[..., 1:, :]
                edge_weights = (
                    previous[..., self.mapping_graph.src_index]
                    * current[..., self.mapping_graph.dst_index]
                )
                edge_prediction = torch.matmul(
                    edge_weights, self.mapping_graph.future_prototypes.to(query.dtype)
                ) / edge_weights.sum(-1, keepdim=True).clamp_min(1e-6)
                mapping_future_prediction[..., 1:, :] = edge_prediction
        a_shape = (1.0 - c_inv).clamp(0.0, 1.0)
        pattern_deviation = query - pattern_context
        geometry_features = hidden.new_zeros(*hidden.shape[:-1], 4)
        geometry = {
            "stable_level": hidden.new_zeros(hidden.shape[:-1]),
            "stable_scale": hidden.new_ones(hidden.shape[:-1]),
            "shift_signed": hidden.new_zeros(hidden.shape[:-1]),
            "scale_signed": hidden.new_zeros(hidden.shape[:-1]),
            "u_shift": hidden.new_zeros(hidden.shape[:-1]),
            "u_scale": hidden.new_zeros(hidden.shape[:-1]),
        }
        factorized_mode = self.variant_input_mode.startswith("factorized_")
        has_runtime_geometry = raw_patch_mean is not None and raw_patch_std is not None
        requires_geometry = factorized_mode or self.fusion_mode == "typed_raw"
        if requires_geometry and (not has_runtime_geometry or not self.pattern_graph.geometry_ready):
            if self.fusion_mode == "typed_raw":
                raise RuntimeError("typed raw fusion requires rebuilt raw graph")
            raise RuntimeError("factorized raw variation requires rebuilt raw graph")
        if has_runtime_geometry and self.pattern_graph.geometry_ready:
            geometry = self.pattern_graph.geometry(
                responsibility, raw_patch_mean, raw_patch_std
            )
            geometry_features = torch.stack(
                [geometry["shift_signed"], geometry["scale_signed"],
                 geometry["u_shift"], geometry["u_scale"]], dim=-1
            )
        a_geo = 1.0 - (1.0 - geometry["u_shift"]) * (1.0 - geometry["u_scale"])
        a_var = 1.0 - (1.0 - a_shape) * (1.0 - geometry["u_shift"]) * (
            1.0 - geometry["u_scale"]
        )
        if self.variant_input_mode == "factorized_geometry":
            activation = a_geo
        elif self.variant_input_mode == "factorized_full":
            activation = a_var
        else:
            activation = a_shape
        typed = self.typed_fusion(
            z_inv=z_inv,
            d_shape=pattern_deviation,
            d_scale=geometry["scale_signed"],
            d_shift=geometry["shift_signed"],
            a_shape=a_shape,
            u_scale=geometry["u_scale"],
            u_shift=geometry["u_shift"],
        )
        relation = self.variant_mapping_correction(
            z_inv, variant_mapping_delta, variant_mapping_activation
        )
        variant_relation_enabled = self.relation_mapping_mode in {"decomposed_full", "variant_only"}
        if not variant_relation_enabled:
            relation["variant_mapping_correction"] = torch.zeros_like(z_inv)
            relation["variant_mapping_gate"] = torch.zeros_like(relation["variant_mapping_gate"])
        relation["z_relation"] = typed["z_typed"] + relation["variant_mapping_correction"]
        if self.fusion_mode == "typed_raw":
            # Typed fusion preserves raw variation identities and bypasses the
            # unified Variant/Environment path completely.
            variant = {
                "candidate": torch.zeros_like(hidden),
                "latent": torch.zeros_like(hidden),
                "raw_deviation_projected": torch.zeros_like(hidden),
                "shape_encoded": torch.zeros_like(hidden),
                "geometry_encoded": torch.zeros_like(hidden),
                "raw_gate": hidden.new_zeros(*hidden.shape[:-1], 1),
                "raw_correction": torch.zeros_like(hidden),
                "raw_correction_raw": torch.zeros_like(hidden),
            }
            z_var_raw = torch.zeros_like(hidden)
            z_var = torch.zeros_like(hidden)
            activation = torch.zeros_like(a_shape)
            environment = hidden.new_zeros(hidden.shape[0], self.env_dim)
            z_env = z_inv
            temporal_gate = torch.zeros_like(a_shape)
        else:
            variant = self.variant_encoder(
                hidden, z_inv.detach(), pattern_deviation, geometry_features
            )
            z_var_raw = variant["candidate"]
            if self.use_variant:
                z_var = activation.unsqueeze(-1) * z_var_raw
            else:
                z_var = torch.zeros_like(z_var_raw)
                activation = torch.zeros_like(activation)
            environment = self.environment_encoder(z_var, activation)
            z_env, temporal_gate = self.environment_fusion.candidate(z_inv, environment)
        result = {
            "hidden": hidden,
            "graph_tokens": graph_tokens,
            "query": query,
            "z_inv": z_inv,
            "z_var_raw": z_var_raw,
            "z_var": z_var,
            "environment": environment,
            "z_env": z_env,
            "temporal_gate": temporal_gate,
            "consensus_context": consensus_context,
            "invariant_base": invariant["invariant_base"],
            "graph_correction": invariant["graph_correction"],
            "graph_correction_raw": invariant["graph_correction_raw"],
            "graph_gate": invariant["graph_gate"],
            "stable_mapping_delta": stable_mapping_delta,
            "invariant_mapping_delta": invariant_mapping_delta,
            "variant_mapping_delta": variant_mapping_delta,
            "invariant_mapping_activation": invariant_mapping_activation,
            "variant_mapping_activation": variant_mapping_activation,
            "variant_mapping_probability": variant_mapping_probability,
            "active_variant_edges": active_variant_edges,
            "stable_mapping_delta_hidden": invariant["stable_mapping_delta_hidden"],
            "actual_mapping_delta": actual_mapping_delta,
            "mapping_deviation": mapping_deviation,
            "pattern_future_prediction": pattern_future_prediction,
            "mapping_future_prediction": mapping_future_prediction,
            "pattern_reconstruction": pattern_context,
            "pattern_deviation": pattern_deviation,
            "raw_deviation_projected": variant["raw_deviation_projected"],
            "variant_latent": variant["latent"],
            "raw_variation_gate": variant["raw_gate"],
            "raw_variation_correction": variant["raw_correction"],
            "raw_variation_correction_raw": variant["raw_correction_raw"],
            "raw_variation_shape_encoded": variant["shape_encoded"],
            "raw_variation_geometry_encoded": variant["geometry_encoded"],
            "raw_variation_shape": pattern_deviation,
            "raw_variation_shift_signed": geometry["shift_signed"],
            "raw_variation_scale_signed": geometry["scale_signed"],
            "raw_variation_u_shift": geometry["u_shift"],
            "raw_variation_u_scale": geometry["u_scale"],
            "raw_variation_a_shape": a_shape,
            "raw_variation_a_geo": a_geo,
            "raw_variation_a_var": a_var,
            "raw_stable_level": geometry["stable_level"],
            "raw_stable_scale": geometry["stable_scale"],
            "c_shape": retrieved["pattern"]["shape_compatibility"]
            if self.use_pattern else torch.ones_like(c_inv),
            "c_rec": retrieved["pattern"]["recurrence_confidence"]
            if self.use_pattern else torch.ones_like(c_inv),
            "c_pred": retrieved["pattern"]["predictive_confidence"]
            if self.use_pattern else torch.ones_like(c_inv),
            "pattern_best_distance": retrieved["pattern"]["best_distance"]
            if self.use_pattern else torch.zeros_like(c_inv),
            "c_inv": c_inv,
            "c_pat": c_pat,
            "c_map": c_map,
            "variation_activation": activation,
            "pattern_novelty": pattern_novelty,
            "mapping_novelty": mapping_novelty,
            "null_probability": null_probability,
        }
        result.update(typed)
        result.update(relation)
        return result

    def reliability(self, features: Tensor) -> Tensor:
        if not self.use_env:
            return features.new_zeros(features.shape[0])
        if not self.use_reliability:
            return features.new_ones(features.shape[0])
        return self.reliability_head(features)

    def fuse(self, decomposition: Dict[str, Tensor], reliability: Tensor) -> Tensor:
        if self.fusion_mode == "invariant_only":
            return decomposition["z_inv"]
        if self.fusion_mode == "typed_raw":
            return decomposition["z_relation"]
        if not self.use_env:
            return decomposition["z_inv"]
        fused, _, _ = self.environment_fusion(
            decomposition["z_inv"], decomposition["environment"], reliability
        )
        return fused

    @torch.no_grad()
    def load_graph_statistics(self, artifact: Mapping[str, Any]) -> None:
        self.pattern_graph.load_statistics(artifact["pattern_graph"])
        self.mapping_graph.load_statistics(artifact["mapping_graph"])

    def graph_artifact(self, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return {
            "pattern_graph": self.pattern_graph.export(),
            "mapping_graph": self.mapping_graph.export(),
            "reliability_statistics": {
                "gain_count": self.reliability_head.gain_count.detach().cpu(),
                "gain_sum": self.reliability_head.gain_sum.detach().cpu(),
                "gain_sum_sq": self.reliability_head.gain_sum_sq.detach().cpu(),
            },
            "metadata": dict(metadata or {}),
        }
