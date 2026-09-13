"""Direct raw-pattern forecasting without hidden-state forecast decoding."""

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FuturePatternSelector(nn.Module):
    """Turn stable/variant future logits into invariant-pattern mixtures."""

    MAPPING_MODES = {"stable_only", "variant_unit", "variant_gated"}
    SELECTION_MODES = {"soft", "top1", "topk"}

    @staticmethod
    def select(logits: Tensor, selection_mode: str, topk: int,
               temperature: float, straight_through: bool,
               training: bool) -> Dict[str, Tensor]:
        if selection_mode not in FuturePatternSelector.SELECTION_MODES:
            raise ValueError("invalid pattern selection mode: {}".format(selection_mode))
        if temperature <= 0:
            raise ValueError("pattern temperature must be positive")
        pattern_count = logits.shape[-1]
        if topk < 1:
            raise ValueError("pattern topk must be at least one")
        scaled = logits / float(temperature)
        full_probability = torch.softmax(scaled, -1)
        if selection_mode == "soft":
            selected = full_probability
        elif selection_mode == "top1":
            hard = F.one_hot(full_probability.argmax(-1), pattern_count).to(full_probability.dtype)
            selected = (
                hard - full_probability.detach() + full_probability
                if training and straight_through else hard
            )
        else:
            keep = min(int(topk), pattern_count)
            indices = scaled.topk(keep, dim=-1).indices
            mask = torch.zeros_like(scaled, dtype=torch.bool).scatter(-1, indices, True)
            selected = torch.softmax(scaled.masked_fill(~mask, float("-inf")), -1)
        return {
            "raw_full_future_probability": full_probability,
            "raw_selected_probability": selected,
        }

    def forward(self, stable_logits: Tensor, variant_logits: Tensor,
                learned_gate: Tensor, mapping_mode: str,
                selection_mode: str = "soft", topk: int = 3,
                temperature: float = 1.0,
                straight_through: bool = True) -> Dict[str, Tensor]:
        if mapping_mode not in self.MAPPING_MODES:
            raise ValueError("invalid raw mapping mode: {}".format(mapping_mode))
        if mapping_mode == "stable_only":
            gate = torch.zeros_like(learned_gate)
        elif mapping_mode == "variant_unit":
            gate = torch.ones_like(learned_gate)
        else:
            gate = learned_gate
        final_logits = stable_logits + gate * variant_logits
        result = self.select(
            final_logits, selection_mode, topk, temperature,
            straight_through, self.training,
        )
        result.update({
            "raw_mapping_gate": gate,
            "raw_future_logits": final_logits,
            # Backward-compatible name now means the probability actually used
            # for waveform, level and scale reconstruction.
            "raw_future_probability": result["raw_selected_probability"],
        })
        return result


class VariantFutureMapping(nn.Module):
    """Semantic adapter exposing the sample-conditioned future-logit residual."""

    def forward(self, variant_logits: Tensor) -> Tensor:
        return variant_logits


class FuturePatternTransformer(nn.Module):
    """Per-future-patch Shape -> Scale -> Shift transformation."""

    MODES = {"off", "unit", "gated"}

    def __init__(self, hidden_dim: int, patch_len: int, future_patches: int,
                 decoder_context: str = "pattern_plus_hidden") -> None:
        super().__init__()
        if decoder_context not in {"pattern_only", "pattern_plus_hidden"}:
            raise ValueError("raw decoder context must be pattern_only or pattern_plus_hidden")
        self.decoder_context = decoder_context
        self.patch_len = int(patch_len)
        self.future_query = nn.Parameter(torch.randn(1, 1, future_patches, hidden_dim) * 0.02)
        self.pattern_projection = nn.Linear(patch_len, hidden_dim)
        context_dim = hidden_dim * (3 if decoder_context == "pattern_plus_hidden" else 2)
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim), nn.GELU()
        )
        self.shape_residual = nn.Linear(hidden_dim, patch_len)
        self.shape_amplitude = nn.Linear(hidden_dim, 1)
        self.scale_delta = nn.Linear(hidden_dim, 1)
        self.shift_delta = nn.Linear(hidden_dim, 1)
        self.shape_gate = nn.Linear(hidden_dim, 1)
        self.scale_gate = nn.Linear(hidden_dim, 1)
        self.shift_gate = nn.Linear(hidden_dim, 1)
        # The scalar amplitude is the zero-initialized last stage for shape.
        # Keeping its basis non-zero avoids a zero-times-zero gradient deadlock.
        for output in (self.shape_amplitude, self.scale_delta, self.shift_delta):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        for gate in (self.shape_gate, self.scale_gate, self.shift_gate):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.0)

    @staticmethod
    def _resolve_gate(logits: Tensor, mode: str) -> Tensor:
        if mode == "off":
            return torch.zeros_like(logits)
        if mode == "unit":
            return torch.ones_like(logits)
        if mode == "gated":
            return torch.sigmoid(logits)
        raise ValueError("variation mode must be off, unit, or gated")

    def forward(self, invariant_shape: Tensor, stable_level: Tensor, stable_scale: Tensor,
                history_context: Tensor, hidden_context: Tensor,
                shape_mode: str, scale_mode: str, shift_mode: str) -> Dict[str, Tensor]:
        batch, channels, future_patches, _ = invariant_shape.shape
        pattern_feature = self.pattern_projection(invariant_shape)
        history = history_context.unsqueeze(2).expand(-1, -1, future_patches, -1)
        parts = [pattern_feature, history]
        if self.decoder_context == "pattern_plus_hidden":
            parts.append(hidden_context.unsqueeze(2).expand_as(history))
        feature = self.context(torch.cat(parts, -1)) + self.future_query[:, :, :future_patches]

        shape_gate = self._resolve_gate(self.shape_gate(feature), shape_mode)
        raw_shape = self.shape_residual(feature)
        raw_shape = raw_shape - raw_shape.mean(-1, keepdim=True)
        unit_shape = raw_shape / raw_shape.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        shape_delta = torch.tanh(self.shape_amplitude(feature)) * unit_shape
        shape = invariant_shape + shape_gate * shape_delta

        scale_gate = self._resolve_gate(self.scale_gate(feature), scale_mode)
        log_scale_delta = torch.tanh(self.scale_delta(feature))
        scale_factor = torch.exp(scale_gate * log_scale_delta)
        scale = stable_scale.unsqueeze(-1) * scale_factor

        shift_gate = self._resolve_gate(self.shift_gate(feature), shift_mode)
        shift_delta = self.shift_delta(feature)
        shift = stable_level.unsqueeze(-1) + shift_gate * shift_delta
        invariant_signal = stable_level.unsqueeze(-1) + stable_scale.unsqueeze(-1) * invariant_shape
        after_shape_signal = stable_level.unsqueeze(-1) + stable_scale.unsqueeze(-1) * shape
        after_geometry_signal = shift + scale * invariant_shape
        final_signal = shift + scale * shape
        return {
            "raw_shape_gate": shape_gate, "raw_scale_gate": scale_gate,
            "raw_shift_gate": shift_gate, "raw_shape_delta": shape_delta,
            "raw_log_scale_delta": log_scale_delta, "raw_scale_factor": scale_factor,
            "raw_shift_delta": shift_delta, "raw_predicted_scale": scale.squeeze(-1),
            "raw_predicted_shift": shift.squeeze(-1),
            "raw_invariant_patches": invariant_signal,
            "raw_after_shape_patches": after_shape_signal,
            "raw_after_geometry_patches": after_geometry_signal,
            "raw_final_patches": final_signal,
        }


class RawPatchReconstructor(nn.Module):
    """Differentiable overlap-add with exact output length and overlap averaging."""

    def __init__(self, pred_len: int, stride: int) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.stride = int(stride)

    def forward(self, patches: Tensor) -> Tensor:
        if patches.ndim != 4:
            raise ValueError("predicted raw patches must be [B,C,Q,L]")
        batch, channels, count, patch_len = patches.shape
        output = patches.new_zeros(batch, channels, self.pred_len)
        weights = patches.new_zeros(self.pred_len)
        for index in range(count):
            start = index * self.stride
            stop = min(self.pred_len, start + patch_len)
            if start >= self.pred_len:
                break
            output[..., start:stop] += patches[..., index, :stop - start]
            weights[start:stop] += 1.0
        if bool((weights == 0).any()):
            raise RuntimeError("raw future patches do not cover pred_len")
        return (output / weights).permute(0, 2, 1)


class RawPatternForecaster(nn.Module):
    """Mapping selects patterns; typed transforms generate signal-space patches."""

    def __init__(self, hidden_dim: int, patch_len: int, future_patches: int,
                 pred_len: int, stride: int,
                 decoder_context: str = "pattern_plus_hidden") -> None:
        super().__init__()
        self.selector = FuturePatternSelector()
        self.variant_mapping = VariantFutureMapping()
        self.transformer = FuturePatternTransformer(
            hidden_dim, patch_len, future_patches, decoder_context
        )
        self.reconstructor = RawPatchReconstructor(pred_len, stride)

    @staticmethod
    def _mix_patterns(probability: Tensor, prototypes: Tensor,
                      stable_level_by_pattern: Tensor,
                      stable_scale_by_pattern: Tensor):
        invariant_shape = torch.einsum("bcqk,kl->bcql", probability, prototypes)
        level_table = stable_level_by_pattern.t()
        scale_table = stable_scale_by_pattern.t()
        stable_level = torch.einsum("bcqk,ck->bcq", probability, level_table)
        stable_scale = torch.einsum("bcqk,ck->bcq", probability, scale_table).clamp_min(1e-5)
        return invariant_shape, stable_level, stable_scale

    def forward(self, stable_logits: Tensor, variant_logits: Tensor,
                learned_mapping_gate: Tensor, prototypes: Tensor,
                stable_level_by_pattern: Tensor, stable_scale_by_pattern: Tensor,
                history_context: Tensor, hidden_context: Tensor,
                mapping_mode: str, shape_mode: str,
                scale_mode: str, shift_mode: str,
                selection_mode: str = "soft", pattern_topk: int = 3,
                temperature: float = 1.0,
                straight_through: bool = True) -> Dict[str, Tensor]:
        selection = self.selector(
            stable_logits, self.variant_mapping(variant_logits), learned_mapping_gate,
            mapping_mode, selection_mode, pattern_topk, temperature, straight_through,
        )
        probability = selection["raw_future_probability"]
        # Geometry tables are [K,C]. Query each channel independently.
        invariant_shape, stable_level, stable_scale = self._mix_patterns(
            probability, prototypes, stable_level_by_pattern, stable_scale_by_pattern
        )
        transformed = self.transformer(
            invariant_shape, stable_level, stable_scale,
            history_context, hidden_context, shape_mode, scale_mode, shift_mode,
        )
        result = dict(selection)
        result.update(transformed)
        for name in ("raw_invariant_patches", "raw_after_shape_patches",
                     "raw_after_geometry_patches", "raw_final_patches"):
            result[name.replace("patches", "forecast")] = self.reconstructor(result[name])
        stable_selection = self.selector(
            stable_logits, torch.zeros_like(variant_logits),
            torch.zeros_like(learned_mapping_gate), "stable_only",
            selection_mode, pattern_topk, temperature, straight_through,
        )
        stable_probability = stable_selection["raw_selected_probability"]
        stable_shape, stable_level, stable_scale = self._mix_patterns(
            stable_probability, prototypes, stable_level_by_pattern, stable_scale_by_pattern
        )
        stable_transformed = self.transformer(
            stable_shape, stable_level, stable_scale,
            history_context, hidden_context, shape_mode, scale_mode, shift_mode,
        )
        result["raw_stable_probability"] = stable_probability
        result["raw_without_variant_forecast"] = self.reconstructor(
            stable_transformed["raw_final_patches"]
        )
        return result

    def forecast_selected_probability(
        self, probability: Tensor, prototypes: Tensor,
        stable_level_by_pattern: Tensor, stable_scale_by_pattern: Tensor,
        history_context: Tensor, hidden_context: Tensor,
        shape_mode: str, scale_mode: str, shift_mode: str,
    ) -> Dict[str, Tensor]:
        """Diagnostic helper using an externally selected pattern mixture."""
        invariant_shape, stable_level, stable_scale = self._mix_patterns(
            probability, prototypes, stable_level_by_pattern, stable_scale_by_pattern
        )
        result = self.transformer(
            invariant_shape, stable_level, stable_scale,
            history_context, hidden_context, shape_mode, scale_mode, shift_mode,
        )
        for name in ("raw_invariant_patches", "raw_final_patches"):
            result[name.replace("patches", "forecast")] = self.reconstructor(result[name])
        return result
