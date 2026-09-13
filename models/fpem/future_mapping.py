"""TRAIN-only history-to-future pattern memory and adaptive future mapper."""

import math
from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _fit_two_component_mixture(features: Tensor, eps: float = 1e-6) -> Dict[str, Tensor]:
    """Deterministic diagonal GMM; the global component is named from learned centroids."""
    mean = features.mean(0)
    std = features.std(0, unbiased=False).clamp_min(eps)
    x = (features - mean) / std
    count = x.shape[0]
    if count < 2 or float(x.square().sum()) <= eps:
        posterior = x.new_zeros(count, 2)
        posterior[:, 0] = 1.0
        centers = torch.stack([x[0], x[0]])
        variances = torch.ones_like(centers)
        weights = x.new_tensor([1.0, 0.0])
        invariant = 0
    else:
        _, _, right = torch.linalg.svd(x, full_matrices=False)
        projection = x @ right[0]
        centers = torch.stack([x[projection.argmin()], x[projection.argmax()]])
        variances = torch.ones_like(centers)
        weights = x.new_full((2,), 0.5)
        for _ in range(50):
            logp = -0.5 * (((x[:, None] - centers[None]) ** 2 / variances[None]).sum(-1)
                            + variances.log().sum(-1)[None]) + weights.clamp_min(eps).log()[None]
            posterior = torch.softmax(logp, -1)
            mass = posterior.sum(0).clamp_min(eps)
            new_centers = posterior.t().matmul(x) / mass[:, None]
            difference = x[:, None] - new_centers[None]
            new_variances = (posterior[:, :, None] * difference.square()).sum(0) / mass[:, None]
            new_variances = new_variances.clamp_min(1e-4)
            new_weights = mass / float(count)
            converged = torch.max((new_centers - centers).abs()) < 1e-6
            centers, variances, weights = new_centers, new_variances, new_weights
            if bool(converged):
                break
        logp = -0.5 * (((x[:, None] - centers[None]) ** 2 / variances[None]).sum(-1)
                        + variances.log().sum(-1)[None]) + weights.clamp_min(eps).log()[None]
        posterior = torch.softmax(logp, -1)
        # coverage, log-support, entropy, consistency are all globality-positive.
        invariant = int(centers.mean(-1).argmax())
    return {
        "p_inv": posterior[:, invariant].clamp(0, 1),
        "feature_mean": mean, "feature_std": std,
        "component_mean": centers, "component_variance": variances,
        "component_weight": weights, "invariant_component": torch.tensor(invariant),
    }


class FutureMappingMemoryBuilder:
    """Aggregate chronological anchor blocks and learn soft stable-memory membership."""

    def __init__(self, seq_len: int, eps: float = 1e-6) -> None:
        self.seq_len = int(seq_len)
        self.eps = float(eps)

    @torch.no_grad()
    def build(self, history_tokens: Tensor, future_responsibility: Tensor) -> Dict[str, Tensor]:
        if history_tokens.ndim != 4 or future_responsibility.ndim != 4:
            raise ValueError("history/future tensors must be [N,C,P,F] and [N,C,Q,K]")
        if history_tokens.shape[:2] != future_responsibility.shape[:2]:
            raise ValueError("history and future must share TRAIN window/channel axes")
        windows, channels, patches, features = history_tokens.shape
        anchors = torch.div(torch.arange(windows), self.seq_len, rounding_mode="floor")
        anchor_count = int(anchors.max()) + 1
        block_keys, block_future, block_anchor = [], [], []
        for anchor in range(anchor_count):
            selected = anchors == anchor
            if not bool(selected.any()):
                continue
            for channel in range(channels):
                block_keys.append(history_tokens[selected, channel].mean(0))
                block_future.append(future_responsibility[selected, channel].mean(0))
                block_anchor.append(anchor)
        keys = torch.stack(block_keys).float()
        futures = torch.stack(block_future).float()
        block_anchor_tensor = torch.tensor(block_anchor, dtype=torch.long)
        flat = keys.flatten(1)
        flat = (flat - flat.mean(0)) / flat.std(0, unbiased=False).clamp_min(self.eps)
        cluster_count = max(2, min(flat.shape[0], int(math.ceil(math.sqrt(flat.shape[0])))))
        initial = torch.linspace(0, flat.shape[0] - 1, cluster_count).round().long()
        centers = flat[initial].clone()
        for _ in range(20):
            assignment = torch.cdist(flat, centers).argmin(-1)
            updated = centers.clone()
            for index in range(cluster_count):
                members = assignment == index
                if bool(members.any()):
                    updated[index] = flat[members].mean(0)
            if torch.allclose(updated, centers, atol=1e-5, rtol=0):
                centers = updated
                break
            centers = updated
        memory_keys, memory_future, support, coverage, entropy, consistency = [], [], [], [], [], []
        global_future_variance = futures.var(0, unbiased=False).mean().clamp_min(self.eps)
        for index in range(cluster_count):
            members = assignment == index
            member_count = int(members.sum())
            if member_count == 0:
                continue
            member_anchors = block_anchor_tensor[members]
            anchor_hist = torch.bincount(member_anchors, minlength=anchor_count).float()
            distribution = anchor_hist / anchor_hist.sum().clamp_min(1)
            ent = -(distribution * distribution.clamp_min(self.eps).log()).sum()
            ent = ent / math.log(float(anchor_count)) if anchor_count > 1 else ent.new_tensor(1.0)
            future_var = futures[members].var(0, unbiased=False).mean()
            memory_keys.append(keys[members].mean(0))
            memory_future.append(futures[members].mean(0).clamp_min(self.eps).log())
            support.append(float(member_count))
            coverage.append(float(torch.unique(member_anchors).numel()) / float(anchor_count))
            entropy.append(float(ent))
            consistency.append(float((1.0 + future_var / global_future_variance).reciprocal()))
        support_t = torch.tensor(support)
        coverage_t = torch.tensor(coverage)
        entropy_t = torch.tensor(entropy)
        consistency_t = torch.tensor(consistency)
        mixture_features = torch.stack([
            coverage_t, torch.log1p(support_t), entropy_t, consistency_t
        ], -1)
        mixture = _fit_two_component_mixture(mixture_features, self.eps)
        return {
            "history_keys": torch.stack(memory_keys),
            "future_logits": torch.stack(memory_future),
            "support": support_t, "coverage": coverage_t,
            "sample_entropy": entropy_t, "future_consistency": consistency_t,
            "p_inv": mixture["p_inv"],
            "mixture_feature_mean": mixture["feature_mean"],
            "mixture_feature_std": mixture["feature_std"],
            "mixture_component_mean": mixture["component_mean"],
            "mixture_component_variance": mixture["component_variance"],
            "mixture_component_weight": mixture["component_weight"],
            "mixture_invariant_component": mixture["invariant_component"],
            "source_split_code": torch.tensor(1),
        }


class HistoryMappingContextEncoder(nn.Module):
    """Encode every historical patch with self-attention and attention pooling."""

    def __init__(self, input_dim: int, hidden_dim: int, max_patches: int = 64) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, max_patches, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 2, 0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pool = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)

    def forward(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        if tokens.ndim != 4:
            raise ValueError("history mapping tokens must be [B,C,P,F]")
        batch, channels, patches, _ = tokens.shape
        if patches > self.position.shape[1]:
            raise ValueError("history patch count exceeds configured maximum")
        sequence = self.input_projection(tokens.reshape(batch * channels, patches, -1))
        sequence = self.encoder(sequence + self.position[:, :patches])
        query = self.pool_query.expand(batch * channels, -1, -1)
        pooled, _ = self.pool(query, sequence, sequence, need_weights=False)
        return sequence.view(batch, channels, patches, -1), pooled[:, 0].view(batch, channels, -1)


class StableFutureMappingMemory(nn.Module):
    """Frozen TRAIN memory queried in the current encoder coordinate system."""

    NAMES = ("history_keys", "future_logits", "support", "coverage", "sample_entropy",
             "future_consistency", "p_inv", "mixture_feature_mean", "mixture_feature_std",
             "mixture_component_mean", "mixture_component_variance", "mixture_component_weight",
             "mixture_invariant_component", "source_split_code")

    def __init__(self) -> None:
        super().__init__()
        for name in self.NAMES:
            dtype = torch.long if name in {"mixture_invariant_component", "source_split_code"} else torch.float32
            self.register_buffer(name, torch.empty(0, dtype=dtype))

    @property
    def ready(self) -> bool:
        return self.history_keys.numel() > 0

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs) -> None:
        for name in self.NAMES:
            key = prefix + name
            if key in state_dict:
                incoming = state_dict[key]
                self._buffers[name] = torch.empty_like(incoming, device=self.p_inv.device)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    @torch.no_grad()
    def load_statistics(self, statistics: Mapping[str, Tensor]) -> None:
        for name in self.NAMES:
            if name not in statistics:
                raise KeyError("missing future memory statistic: {}".format(name))
            self._buffers[name] = torch.as_tensor(statistics[name], device=self.p_inv.device).clone()

    def export(self) -> Dict[str, Tensor]:
        return {name: getattr(self, name).detach().cpu() for name in self.NAMES}

    def query(self, tokens: Tensor, encoder: HistoryMappingContextEncoder,
              leave_nearest_out: bool = False) -> Dict[str, Tensor]:
        if not self.ready:
            raise RuntimeError("stable future mapping memory is not ready")
        history_sequence, history_context = encoder(tokens)
        memory_tokens = self.history_keys.to(tokens.dtype).unsqueeze(1)
        _, memory_context = encoder(memory_tokens)
        memory_context = memory_context[:, 0]
        query = F.normalize(history_context.flatten(0, 1), dim=-1)
        keys = F.normalize(memory_context, dim=-1)
        similarity = query @ keys.t()
        logits = similarity + self.p_inv.to(tokens.dtype).clamp_min(1e-6).log()[None]
        excluded = torch.full((logits.shape[0],), -1, dtype=torch.long, device=logits.device)
        if leave_nearest_out and logits.shape[1] > 1:
            excluded = similarity.argmax(-1)
            logits = logits.scatter(1, excluded[:, None], float("-inf"))
        weights = torch.softmax(logits, -1)
        stable_logits = torch.einsum("nm,mqk->nqk", weights, self.future_logits.to(tokens.dtype))
        distance = 1.0 - (weights * similarity).sum(-1)
        result_shape = (*tokens.shape[:2], *stable_logits.shape[1:])
        return {
            "history_sequence": history_sequence,
            "history_context": history_context,
            "stable_future_logits": stable_logits.view(result_shape),
            "retrieval_confidence": weights.max(-1).values.view(tokens.shape[:2]),
            "effective_memory_entries": weights.square().sum(-1).clamp_min(1e-6).reciprocal().view(tokens.shape[:2]),
            "history_memory_distance": distance.view(tokens.shape[:2]),
            "retrieved_p_inv": (weights * self.p_inv.to(tokens.dtype)[None]).sum(-1).view(tokens.shape[:2]),
            "excluded_memory_index": excluded.view(tokens.shape[:2]),
        }


class AdaptiveVariantFutureMapper(nn.Module):
    """Future patch queries cross-attend to the complete history trajectory."""

    def __init__(self, hidden_dim: int, future_patterns: int, future_patches: int) -> None:
        super().__init__()
        self.future_queries = nn.Parameter(torch.randn(1, future_patches, hidden_dim) * 0.02)
        self.stable_projection = nn.Linear(future_patterns, hidden_dim)
        self.state_relation_projection = nn.Linear(6, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, future_patterns))
        self.gate = nn.Sequential(nn.LayerNorm(hidden_dim + 4), nn.Linear(hidden_dim + 4, hidden_dim),
                                  nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, history_sequence: Tensor, history_context: Tensor, stable_logits: Tensor,
                state_relation_summary: Tensor, gate_mode: str) -> Dict[str, Tensor]:
        batch, channels, patches, hidden = history_sequence.shape
        flat_history = history_sequence.flatten(0, 1)
        stable = stable_logits.flatten(0, 1)
        condition = self.state_relation_projection(state_relation_summary.flatten(0, 1)).unsqueeze(1)
        queries = self.future_queries.expand(batch * channels, -1, -1)
        queries = queries + self.stable_projection(stable) + condition
        decoded, _ = self.cross_attention(queries, flat_history, flat_history, need_weights=False)
        delta_logits = self.output(decoded).view_as(stable_logits)
        gate_features = torch.cat([history_context, state_relation_summary[..., :4]], -1)
        learned_gate = torch.sigmoid(self.gate(gate_features)).unsqueeze(-1)
        if gate_mode == "off":
            gate = torch.zeros_like(learned_gate)
        elif gate_mode == "unit":
            gate = torch.ones_like(learned_gate)
        elif gate_mode == "learned":
            gate = learned_gate
        else:
            raise ValueError("future variant gate mode must be off, unit, or learned")
        final_logits = stable_logits + gate * delta_logits
        return {"variant_future_logits": delta_logits, "future_variant_gate": gate,
                "final_future_logits": final_logits}


class FutureForecastCorrection(nn.Module):
    """Forecast-space residual initialized as an exact identity."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.correction = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.gate = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, base: Tensor, future_context: Tensor, confidence: Tensor,
                variant_gate: Tensor) -> Dict[str, Tensor]:
        raw = self.correction(torch.stack([base, future_context], -1)).squeeze(-1)
        summary = torch.stack([confidence, variant_gate.squeeze(-1).squeeze(-1),
                               future_context.std(1, unbiased=False)], -1)
        gate = torch.sigmoid(self.gate(summary)).transpose(1, 2)
        correction = gate * raw
        return {"future_forecast_gate": gate, "future_forecast_correction": correction,
                "future_forecast": base + correction}
