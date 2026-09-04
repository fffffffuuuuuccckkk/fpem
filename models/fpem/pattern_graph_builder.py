"""Leakage-safe offline builder for stable pattern and mapping graphs."""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def _even_sample_indices(length: int, maximum: int) -> Tensor:
    if length <= maximum:
        return torch.arange(length, dtype=torch.long)
    return torch.linspace(0, length - 1, steps=maximum).round().long().unique()


def _unique_anchor_support(assignments: Tensor, anchor_ids: Tensor, count: int) -> Tensor:
    support = torch.zeros(count, dtype=torch.float32)
    valid = assignments >= 0
    if not bool(valid.any()):
        return support
    anchor_count = int(anchor_ids.max().item()) + 1
    encoded = assignments[valid].long() * anchor_count + anchor_ids[valid].long()
    unique_encoded = torch.unique(encoded)
    nodes = torch.div(unique_encoded, anchor_count, rounding_mode="floor")
    support.index_add_(0, nodes, torch.ones_like(nodes, dtype=torch.float32))
    return support


class PatternMappingGraphBuilder:
    """Candidate -> stable graph construction using training windows only.

    Graph size is determined by a robust coverage radius estimated from the
    embeddings.  The constants below are memory safeguards/sample budgets, not
    requested pattern counts and do not determine the active K.
    """

    def __init__(
        self,
        pattern_dim: int,
        seq_len: int = 1,
        eps: float = 1e-5,
        representation_space: str = "embedding",
        patch_len: Optional[int] = None,
        stride: int = 1,
        predictive_stability_mode: str = "per_future_znorm",
    ) -> None:
        self.pattern_dim = int(pattern_dim)
        self.seq_len = max(1, int(seq_len))
        self.eps = float(eps)
        self.representation_space = str(representation_space).lower()
        self.raw_mode = self.representation_space == "raw"
        self.patch_len = int(patch_len or pattern_dim)
        self.stride = int(stride)
        if predictive_stability_mode not in {"per_future_znorm", "train_global", "relative_future"}:
            raise ValueError("invalid predictive_stability_mode: {}".format(predictive_stability_mode))
        self.predictive_stability_mode = predictive_stability_mode

    def _pairwise_distance(self, left: Tensor, right: Tensor) -> Tensor:
        distance = torch.cdist(left.float(), right.float())
        if self.raw_mode:
            distance = distance.square() / float(self.pattern_dim)
        return distance

    def _coverage_radius(self, points: Tensor) -> Tuple[float, Tensor]:
        sample = points[_even_sample_indices(points.shape[0], 2048)]
        if sample.shape[0] < 2:
            return 0.1, torch.tensor([0.1])
        distances = self._pairwise_distance(sample, sample)
        distances.fill_diagonal_(float("inf"))
        nearest = distances.min(dim=1).values
        median = nearest.median()
        mad = (nearest - median).abs().median()
        radius = (median + 1.4826 * mad).clamp_min(1e-3)
        return float(radius.item()), nearest.sort().values

    def _candidate_centers(self, points: Tensor, radius: float, maximum_candidates: int) -> Tensor:
        sample = points[_even_sample_indices(points.shape[0], 8192)].float()
        centers: List[Tensor] = []
        counts: List[int] = []
        for point in sample:
            if not centers:
                centers.append(point.clone())
                counts.append(1)
                continue
            stacked = torch.stack(centers)
            distances = (stacked - point[None, :]).square().mean(-1) if self.raw_mode else torch.norm(
                stacked - point[None, :], dim=-1
            )
            distance, index = distances.min(dim=0)
            node = int(index.item())
            if float(distance.item()) > radius:
                if len(centers) < maximum_candidates:
                    centers.append(point.clone())
                    counts.append(1)
            else:
                counts[node] += 1
                centers[node].add_((point - centers[node]) / float(counts[node]))
        centers_tensor = torch.stack(centers)
        if not self.raw_mode:
            centers_tensor = F.normalize(centers_tensor, p=2.0, dim=-1, eps=1e-6)
        return centers_tensor

    def _assign(self, points: Tensor, centers: Tensor, radius: Optional[float], chunk_size: int = 8192) -> Tuple[Tensor, Tensor]:
        assignments: List[Tensor] = []
        distances: List[Tensor] = []
        for start in range(0, points.shape[0], chunk_size):
            chunk = points[start : start + chunk_size].float()
            nearest_distance, nearest_index = self._pairwise_distance(chunk, centers).min(dim=-1)
            if radius is not None:
                nearest_index = nearest_index.masked_fill(nearest_distance > radius, -1)
            assignments.append(nearest_index.cpu())
            distances.append(nearest_distance.cpu())
        return torch.cat(assignments), torch.cat(distances)

    def _pattern_statistics(
        self,
        points: Tensor,
        anchor_ids: Tensor,
        future_values: Optional[Tensor],
        number_of_windows: int,
    ) -> Tuple[Dict[str, Tensor], Tensor, float]:
        radius, _ = self._coverage_radius(points)
        # Data-scale-derived compute budget, not a requested pattern count.
        # Candidate promotion and coverage still determine the final active K.
        candidate_budget = max(4, int(math.ceil(math.sqrt(number_of_windows))))
        candidates = self._candidate_centers(points, radius, candidate_budget)
        candidate_assignment, _ = self._assign(points, candidates, radius)
        candidate_support = _unique_anchor_support(candidate_assignment, anchor_ids, candidates.shape[0])
        stable_mask = candidate_support >= 3.0
        if not bool(stable_mask.any()):
            stable_mask[candidate_support.argmax()] = True
        initial_means = candidates[stable_mask]

        assignment, _ = self._assign(points, initial_means, radius)
        active = initial_means.shape[0]
        valid = assignment >= 0
        node = assignment[valid]
        selected = points[valid].float()
        counts = torch.zeros(active)
        sums = torch.zeros(active, self.pattern_dim)
        sums_sq = torch.zeros_like(sums)
        counts.index_add_(0, node, torch.ones_like(node, dtype=torch.float32))
        sums.index_add_(0, node, selected)
        sums_sq.index_add_(0, node, selected.square())
        means = sums / counts[:, None].clamp_min(1.0)
        variances = (sums_sq / counts[:, None].clamp_min(1.0) - means.square()).clamp_min(1e-4)
        if not self.raw_mode:
            means = F.normalize(means, p=2.0, dim=-1, eps=1e-6)
        window_support = _unique_anchor_support(assignment, anchor_ids, active)

        if future_values is None:
            predictive_support = torch.ones(active)
        else:
            values = future_values[valid].float()
            future_dim = values.shape[-1]
            target_sum = torch.zeros(active, future_dim)
            target_sum_sq = torch.zeros_like(target_sum)
            target_sum.index_add_(0, node, values)
            target_sum_sq.index_add_(0, node, values.square())
            target_mean = target_sum / counts[:, None].clamp_min(1.0)
            target_var = (
                target_sum_sq / counts[:, None].clamp_min(1.0) - target_mean.square()
            ).clamp_min(0.0).mean(-1)
            global_var = values.var(dim=0, unbiased=False).mean().clamp_min(1e-6)
            predictive_support = (1.0 + target_var / global_var).reciprocal().clamp(0.0, 1.0)
            future_ratio = target_var / global_var

        occurrence = torch.log1p(window_support) / torch.log1p(window_support.max()).clamp_min(1.0)
        stability = (occurrence * predictive_support).clamp(self.eps, 1.0)

        # Recompute normalized Mahalanobis best-match distances for the train CDF.
        distance_parts: List[Tensor] = []
        for start in range(0, points.shape[0], 4096):
            chunk = points[start : start + 4096].float()
            inv_var = variances.reciprocal()
            distance = (
                torch.matmul(chunk.square(), inv_var.t())
                - 2.0 * torch.matmul(chunk, (means * inv_var).t())
                + (means.square() * inv_var).sum(-1)[None, :]
            ).clamp_min(0.0) / float(self.pattern_dim)
            distance_parts.append(distance.min(dim=-1).values.cpu())
        best_distance = torch.cat(distance_parts)
        distance_cdf = best_distance[_even_sample_indices(best_distance.numel(), 16384)].sort().values
        statistics = {
            "means": means,
            "variances": variances,
            "counts": counts,
            "window_support": window_support,
            "predictive_support": predictive_support,
            "future_ratio": future_ratio if future_values is not None else torch.zeros(active),
            "stability": stability,
            "distance_cdf": distance_cdf,
        }
        if self.raw_mode:
            medoid_shapes = torch.zeros_like(means)
            medoid_window_index = torch.full((active,), -1, dtype=torch.long)
            medoid_variable_index = torch.full((active,), -1, dtype=torch.long)
            medoid_patch_index = torch.full((active,), -1, dtype=torch.long)
            medoid_absolute_time_index = torch.full((active,), -1, dtype=torch.long)
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            channels_patches = anchor_ids.numel() // number_of_windows
            # anchor_ids flattening is [window, variable, patch]. Infer the
            # patch count from the repeated temporal anchor layout.
            patches = channels_patches
            if future_values is not None:
                # future_values repeats once per [variable, patch], but does
                # not expose their separate sizes. They are supplied below as
                # explicit layout metadata during raw artifact enrichment.
                patches = int(getattr(self, "_raw_patch_count", patches))
            channels = max(1, channels_patches // max(1, patches))
            for pattern_index in range(active):
                member_mask = node == pattern_index
                member_points = selected[member_mask]
                member_indices = valid_indices[member_mask]
                if member_points.numel() == 0:
                    continue
                member_distance = (member_points - means[pattern_index]).square().mean(-1)
                flat_index = int(member_indices[member_distance.argmin()].item())
                window_index = flat_index // (channels * patches)
                remainder = flat_index % (channels * patches)
                variable_index = remainder // patches
                patch_index = remainder % patches
                medoid_shapes[pattern_index] = points[flat_index]
                medoid_window_index[pattern_index] = window_index
                medoid_variable_index[pattern_index] = variable_index
                medoid_patch_index[pattern_index] = patch_index
                medoid_absolute_time_index[pattern_index] = window_index + patch_index * self.stride
            statistics.update(
                {
                    "prototype_mean_shape": means.clone(),
                    "prototype_medoid": medoid_shapes,
                    "shape_std": variances.sqrt(),
                    "support_count": counts.clone(),
                    "anchor_support": window_support.clone(),
                    "predictive_stability": predictive_support.clone(),
                    "overall_stability": stability.clone(),
                    "medoid_window_index": medoid_window_index,
                    "medoid_variable_index": medoid_variable_index,
                    "medoid_patch_index": medoid_patch_index,
                    "medoid_absolute_time_index": medoid_absolute_time_index,
                }
            )
        return statistics, assignment, radius

    @staticmethod
    def _responsibility(points: Tensor, pattern: Mapping[str, Tensor]) -> Tensor:
        means = pattern["means"].float()
        variances = pattern["variances"].float().clamp_min(1e-5)
        points = points.float()
        inv_var = variances.reciprocal()
        distance = (
            torch.matmul(points.square(), inv_var.t())
            - 2.0 * torch.matmul(points, (means * inv_var).t())
            + (means.square() * inv_var).sum(-1)[None, :]
        ).clamp_min(0.0) / float(means.shape[-1])
        score = -0.5 * distance + torch.log(pattern["stability"].float().clamp_min(1e-5))[None, :]
        return torch.softmax(score, dim=-1)

    @staticmethod
    def _posterior_mass_prune(responsibility: Tensor, retained_mass: float = 0.99) -> Tensor:
        """Keep a variable number of states covering the posterior mass."""
        values, indices = responsibility.sort(dim=-1, descending=True)
        cumulative_before = values.cumsum(dim=-1) - values
        kept_values = values * (cumulative_before < retained_mass).to(values.dtype)
        return torch.zeros_like(responsibility).scatter(-1, indices, kept_values)

    def _mapping_statistics(
        self,
        embeddings: Tensor,
        pattern: Mapping[str, Tensor],
        anchor_ids: Tensor,
    ) -> Dict[str, Tensor]:
        windows, channels, patches, dim = embeddings.shape
        pattern_count = int(pattern["means"].shape[0])
        edge_count = pattern_count * pattern_count
        counts_all = torch.zeros(edge_count)
        sums_all = torch.zeros(edge_count, dim)
        sums_sq_all = torch.zeros_like(sums_all)
        anchor_count = int(anchor_ids.max().item()) + 1
        anchor_edge_mass = torch.zeros(anchor_count, edge_count)
        support_codes: List[Tensor] = []
        for start in range(0, windows, 32):
            stop = min(windows, start + 32)
            chunk = embeddings[start:stop].float()
            responsibilities = self._responsibility(chunk.reshape(-1, dim), pattern)
            responsibilities = responsibilities.view(chunk.shape[0], channels, patches, pattern_count)
            responsibilities = self._posterior_mass_prune(responsibilities)
            previous = responsibilities[..., :-1, :].reshape(-1, pattern_count)
            current = responsibilities[..., 1:, :].reshape(-1, pattern_count)
            deltas = (chunk[..., 1:, :] - chunk[..., :-1, :]).reshape(-1, dim)
            transition_anchors = anchor_ids[start:stop, None, None].expand(
                stop - start, channels, patches - 1
            ).reshape(-1)
            # Bound the dense posterior outer product. Retained entries are
            # accumulated sparsely; there is no fixed top-k approximation.
            transition_chunk = max(1, 500_000 // max(1, edge_count))
            for offset in range(0, previous.shape[0], transition_chunk):
                end = min(previous.shape[0], offset + transition_chunk)
                weights = previous[offset:end, :, None] * current[offset:end, None, :]
                locations = torch.nonzero(weights > 0.0, as_tuple=False)
                if locations.numel() == 0:
                    continue
                row, src, dst = locations.unbind(-1)
                edge_ids = src * pattern_count + dst
                edge_weights = weights[row, src, dst]
                delta_values = deltas[offset:end][row]
                counts_all.index_add_(0, edge_ids, edge_weights)
                sums_all.index_add_(0, edge_ids, edge_weights[:, None] * delta_values)
                sums_sq_all.index_add_(0, edge_ids, edge_weights[:, None] * delta_values.square())
                anchors = transition_anchors[offset:end][row]
                anchor_edge_mass.view(-1).index_add_(
                    0, anchors * edge_count + edge_ids, edge_weights
                )
                support_codes.append(torch.unique(anchors * edge_count + edge_ids))

        if support_codes:
            unique_support = torch.unique(torch.cat(support_codes))
            supported_edges = unique_support.remainder(edge_count)
            window_support_all = torch.zeros(edge_count)
            window_support_all.index_add_(0, supported_edges, torch.ones_like(supported_edges, dtype=torch.float32))
        else:
            window_support_all = torch.zeros(edge_count)
        active_mask = window_support_all >= 2.0
        edge_ids = torch.nonzero(active_mask, as_tuple=False).flatten()
        if edge_ids.numel() == 0:
            empty = torch.empty(0)
            return {
                "src_index": torch.empty(0, dtype=torch.long),
                "dst_index": torch.empty(0, dtype=torch.long),
                "counts": empty,
                "window_support": empty,
                "delta_means": torch.empty(0, dim),
                "delta_variances": torch.empty(0, dim),
                "stability": empty,
                "distance_cdf": empty,
                "coverage": empty,
                "sample_entropy": empty,
                "sample_concentration": empty,
            }

        counts = counts_all[edge_ids]
        window_support = window_support_all[edge_ids]
        sums = sums_all[edge_ids]
        sums_sq = sums_sq_all[edge_ids]
        delta_means = sums / counts[:, None].clamp_min(self.eps)
        delta_variances = (sums_sq / counts[:, None].clamp_min(self.eps) - delta_means.square()).clamp_min(1e-4)
        global_delta_var = (embeddings[..., 1:, :] - embeddings[..., :-1, :]).var(dim=(0, 1, 2), unbiased=False).mean().clamp_min(1e-6)
        consistency = (1.0 + delta_variances.mean(-1) / global_delta_var).reciprocal()
        recurrence = torch.log1p(window_support) / torch.log1p(window_support.max()).clamp_min(1.0)
        stability = (recurrence * consistency).clamp(self.eps, 1.0)
        coverage = window_support / float(max(1, anchor_count))
        active_anchor_mass = anchor_edge_mass[:, edge_ids]
        occurrence_distribution = active_anchor_mass / active_anchor_mass.sum(
            dim=0, keepdim=True
        ).clamp_min(self.eps)
        sample_entropy = -(
            occurrence_distribution
            * occurrence_distribution.clamp_min(self.eps).log()
        ).sum(dim=0)
        if anchor_count > 1:
            sample_entropy = sample_entropy / math.log(float(anchor_count))
        else:
            sample_entropy = torch.ones_like(sample_entropy)
        sample_entropy = sample_entropy.clamp(0.0, 1.0)
        sample_concentration = 1.0 - sample_entropy

        src_index = torch.div(edge_ids, pattern_count, rounding_mode="floor")
        dst_index = edge_ids.remainder(pattern_count)
        edge_lookup = {int(edge_id): index for index, edge_id in enumerate(edge_ids.tolist())}
        sampled = embeddings.reshape(-1, patches, dim)[_even_sample_indices(windows * channels, 4096)]
        resp = self._responsibility(sampled.reshape(-1, dim), pattern).view(sampled.shape[0], patches, pattern_count)
        hard = resp.argmax(-1)
        observed_distances: List[Tensor] = []
        for row in range(sampled.shape[0]):
            for patch in range(patches - 1):
                edge_id = int(hard[row, patch].item()) * pattern_count + int(hard[row, patch + 1].item())
                if edge_id not in edge_lookup:
                    continue
                index = edge_lookup[edge_id]
                delta = sampled[row, patch + 1] - sampled[row, patch]
                distance = ((delta - delta_means[index]).square() / delta_variances[index]).mean()
                observed_distances.append(distance)
        distance_cdf = torch.stack(observed_distances).sort().values if observed_distances else torch.empty(0)
        if distance_cdf.numel() > 16384:
            distance_cdf = distance_cdf[_even_sample_indices(distance_cdf.numel(), 16384)].sort().values
        return {
            "src_index": src_index,
            "dst_index": dst_index,
            "counts": counts,
            "window_support": window_support,
            "delta_means": delta_means,
            "delta_variances": delta_variances,
            "stability": stability,
            "distance_cdf": distance_cdf,
            "coverage": coverage,
            "sample_entropy": sample_entropy,
            "sample_concentration": sample_concentration,
        }

    def _attach_mapping_mixture(self, mapping: Dict[str, Tensor]) -> None:
        """Fit a deterministic TRAIN-only two-component diagonal Gaussian mixture."""
        edge_count = int(mapping["src_index"].numel())
        if edge_count == 0:
            empty = torch.empty(0)
            mapping.update({
                "p_inv": empty, "p_var": empty,
                "local_delta_means": torch.empty(0, self.pattern_dim),
            })
            return
        predictive_gain = mapping.get("predictive_gain", torch.zeros(edge_count)).float()
        features = torch.stack(
            [
                mapping["coverage"].float(),
                torch.log1p(mapping["counts"].float()),
                mapping["sample_entropy"].float(),
                mapping["stability"].float(),
                predictive_gain,
            ],
            dim=-1,
        )
        feature_mean = features.mean(dim=0)
        feature_std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
        standardized = (features - feature_mean) / feature_std
        if edge_count == 1 or float(standardized.square().sum().item()) <= self.eps:
            posterior = torch.zeros(edge_count, 2)
            posterior[:, 0] = 1.0
            component_mean = torch.stack([standardized[0], standardized[0]])
            component_variance = torch.ones_like(component_mean)
            component_weight = torch.tensor([1.0, 0.0])
            invariant_component = 0
        else:
            _, _, right = torch.linalg.svd(standardized, full_matrices=False)
            projection = standardized @ right[0]
            component_mean = torch.stack(
                [standardized[projection.argmin()], standardized[projection.argmax()]]
            )
            component_variance = torch.ones(2, standardized.shape[1])
            component_weight = torch.full((2,), 0.5)
            for _ in range(50):
                log_probability = -0.5 * (
                    ((standardized[:, None, :] - component_mean[None, :, :]).square()
                     / component_variance[None, :, :]).sum(-1)
                    + component_variance.log().sum(-1)[None, :]
                ) + component_weight.clamp_min(self.eps).log()[None, :]
                posterior = torch.softmax(log_probability, dim=-1)
                mass = posterior.sum(dim=0).clamp_min(self.eps)
                new_weight = mass / float(edge_count)
                new_mean = posterior.t().matmul(standardized) / mass[:, None]
                difference = standardized[:, None, :] - new_mean[None, :, :]
                new_variance = (
                    posterior[:, :, None] * difference.square()
                ).sum(dim=0) / mass[:, None]
                new_variance = new_variance.clamp_min(1e-4)
                if torch.max((new_mean - component_mean).abs()) < 1e-6:
                    component_mean, component_variance, component_weight = (
                        new_mean, new_variance, new_weight
                    )
                    break
                component_mean, component_variance, component_weight = (
                    new_mean, new_variance, new_weight
                )
            log_probability = -0.5 * (
                ((standardized[:, None, :] - component_mean[None, :, :]).square()
                 / component_variance[None, :, :]).sum(-1)
                + component_variance.log().sum(-1)[None, :]
            ) + component_weight.clamp_min(self.eps).log()[None, :]
            posterior = torch.softmax(log_probability, dim=-1)
            # Component names are resolved from their learned TRAIN centroids:
            # global relations have higher cross-block coverage and entropy.
            globality = component_mean[:, 0] + component_mean[:, 2]
            invariant_component = int(globality.argmax().item())
        mapping["p_inv"] = posterior[:, invariant_component].clamp(0.0, 1.0)
        mapping["p_var"] = (1.0 - mapping["p_inv"]).clamp(0.0, 1.0)
        # First version shares the learned edge delta.  This explicit buffer is
        # the future extension point for subgroup/local delta estimates.
        mapping["local_delta_means"] = mapping["delta_means"].clone()
        mapping["mixture_feature_mean"] = feature_mean
        mapping["mixture_feature_std"] = feature_std
        mapping["mixture_component_mean"] = component_mean
        mapping["mixture_component_variance"] = component_variance
        mapping["mixture_component_weight"] = component_weight
        mapping["mixture_invariant_component"] = torch.tensor(invariant_component)

    def _attach_future_prototypes(
        self,
        embeddings: Tensor,
        pattern: Dict[str, Tensor],
        mapping: Dict[str, Tensor],
        future_representation: Tensor,
    ) -> None:
        """Attach TRAIN-only soft pattern/edge future prototypes and dispersion."""
        windows, channels, patches, dim = embeddings.shape
        horizon = future_representation.shape[-1]
        pattern_count = pattern["means"].shape[0]
        pattern_weight = torch.zeros(pattern_count)
        pattern_sum = torch.zeros(pattern_count, horizon)
        pattern_sum_sq = torch.zeros_like(pattern_sum)
        edge_count = mapping["src_index"].numel()
        edge_weight = torch.zeros(edge_count)
        edge_sum = torch.zeros(edge_count, horizon)
        edge_sum_sq = torch.zeros_like(edge_sum)
        for start in range(0, windows, 16):
            stop = min(windows, start + 16)
            chunk = embeddings[start:stop].float()
            responsibilities = self._responsibility(chunk.reshape(-1, dim), pattern).view(
                stop - start, channels, patches, pattern_count
            )
            responsibilities = self._posterior_mass_prune(responsibilities)
            future = future_representation[start:stop].float()
            pattern_future = future[:, :, None, :].expand(-1, -1, patches, -1).reshape(-1, horizon)
            flat_resp = responsibilities.reshape(-1, pattern_count)
            pattern_weight += flat_resp.sum(0)
            pattern_sum += flat_resp.t().matmul(pattern_future)
            pattern_sum_sq += flat_resp.t().matmul(pattern_future.square())
            if edge_count == 0 or patches < 2:
                continue
            previous = responsibilities[..., :-1, :].reshape(-1, pattern_count)
            current = responsibilities[..., 1:, :].reshape(-1, pattern_count)
            transition_future = future[:, :, None, :].expand(
                -1, -1, patches - 1, -1
            ).reshape(-1, horizon)
            transition_chunk = max(1, 2_000_000 // max(1, edge_count))
            src = mapping["src_index"]
            dst = mapping["dst_index"]
            for offset in range(0, previous.shape[0], transition_chunk):
                end = min(previous.shape[0], offset + transition_chunk)
                weights = previous[offset:end, src] * current[offset:end, dst]
                targets = transition_future[offset:end]
                edge_weight += weights.sum(0)
                edge_sum += weights.t().matmul(targets)
                edge_sum_sq += weights.t().matmul(targets.square())
        pattern_prototype = pattern_sum / pattern_weight[:, None].clamp_min(self.eps)
        pattern_variance = (
            pattern_sum_sq / pattern_weight[:, None].clamp_min(self.eps) - pattern_prototype.square()
        ).clamp_min(0.0).mean(-1)
        edge_prototype = edge_sum / edge_weight[:, None].clamp_min(self.eps)
        edge_variance = (
            edge_sum_sq / edge_weight[:, None].clamp_min(self.eps) - edge_prototype.square()
        ).clamp_min(0.0).mean(-1)
        if edge_count:
            destination_variance = pattern_variance[mapping["dst_index"]]
            predictive_gain = (destination_variance - edge_variance) / destination_variance.clamp_min(self.eps)
        else:
            predictive_gain = torch.empty(0)
        pattern["future_prototypes"] = pattern_prototype
        pattern["future_variance"] = pattern_variance
        mapping["future_prototypes"] = edge_prototype
        mapping["future_variance"] = edge_variance
        mapping["predictive_gain"] = predictive_gain

    def _attach_raw_geometry_statistics(
        self,
        embeddings: Tensor,
        pattern: Dict[str, Tensor],
        assignment: Tensor,
        raw_patch_mean: Tensor,
        raw_patch_std: Tensor,
    ) -> None:
        """Attach TRAIN-only robust geometry for each shared shape node × channel.

        Pattern discovery remains entirely in normalized-shape space.  Level
        and scale are attached only after nodes have been fixed, so they cannot
        influence candidate growth, distance, or assignment.
        """
        windows, channels, patches, _ = embeddings.shape
        expected = (windows, channels, patches)
        if tuple(raw_patch_mean.shape) != expected or tuple(raw_patch_std.shape) != expected:
            raise ValueError("raw_patch_mean/raw_patch_std must be [W,C,P]")
        raw_patch_mean = raw_patch_mean.detach().cpu().float()
        raw_log_scale = raw_patch_std.detach().cpu().float().clamp_min(self.eps).log()
        assignment_3d = assignment.view(windows, channels, patches)
        pattern_count = int(pattern["means"].shape[0])

        level_center = torch.empty(pattern_count, channels)
        level_mad = torch.empty_like(level_center)
        log_scale_center = torch.empty_like(level_center)
        log_scale_mad = torch.empty_like(level_center)
        geometry_support = torch.zeros_like(level_center)
        for channel in range(channels):
            channel_level = raw_patch_mean[:, channel, :].reshape(-1)
            channel_log_scale = raw_log_scale[:, channel, :].reshape(-1)
            global_level = channel_level.median()
            global_level_mad = (channel_level - global_level).abs().median().clamp_min(self.eps)
            global_log_scale = channel_log_scale.median()
            global_log_scale_mad = (
                channel_log_scale - global_log_scale
            ).abs().median().clamp_min(self.eps)
            channel_assignment = assignment_3d[:, channel, :].reshape(-1)
            for node in range(pattern_count):
                selected = channel_assignment == node
                count = int(selected.sum().item())
                geometry_support[node, channel] = float(count)
                if count >= 3:
                    node_level = channel_level[selected]
                    node_log_scale = channel_log_scale[selected]
                    node_level_center = node_level.median()
                    node_log_scale_center = node_log_scale.median()
                    node_level_mad = (node_level - node_level_center).abs().median()
                    node_log_scale_mad = (
                        node_log_scale - node_log_scale_center
                    ).abs().median()
                    level_center[node, channel] = node_level_center
                    level_mad[node, channel] = node_level_mad.clamp_min(self.eps)
                    log_scale_center[node, channel] = node_log_scale_center
                    log_scale_mad[node, channel] = node_log_scale_mad.clamp_min(self.eps)
                else:
                    # A shared shape node may be absent in one variable.  Use
                    # that variable's TRAIN-global robust geometry, never a
                    # different channel and never validation/test statistics.
                    level_center[node, channel] = global_level
                    level_mad[node, channel] = global_level_mad
                    log_scale_center[node, channel] = global_log_scale
                    log_scale_mad[node, channel] = global_log_scale_mad

        pattern.update(
            {
                "stable_level_center": level_center,
                "stable_level_mad": level_mad,
                "stable_log_scale_center": log_scale_center,
                "stable_log_scale_mad": log_scale_mad,
                "geometry_support": geometry_support,
            }
        )

        # Build the two empirical references with the exact soft-responsibility
        # runtime formula.  These are pre-patch-normalization signal-scale
        # statistics; the data loader may already have standardized the data.
        flat_points = embeddings.reshape(-1, self.pattern_dim)
        flat_level = raw_patch_mean.reshape(-1)
        flat_log_scale = raw_log_scale.reshape(-1)
        channel_ids = torch.arange(channels)[None, :, None].expand(
            windows, channels, patches
        ).reshape(-1)
        shift_scores: List[Tensor] = []
        scale_scores: List[Tensor] = []
        for start in range(0, flat_points.shape[0], 8192):
            stop = min(flat_points.shape[0], start + 8192)
            responsibility = self._responsibility(flat_points[start:stop], pattern)
            channel = channel_ids[start:stop]
            centers_level = level_center[:, channel].t()
            centers_log_scale = log_scale_center[:, channel].t()
            spreads_level = level_mad[:, channel].t()
            spreads_log_scale = log_scale_mad[:, channel].t()
            stable_level = (responsibility * centers_level).sum(-1)
            stable_log_scale = (responsibility * centers_log_scale).sum(-1)
            stable_level_mad = (responsibility * spreads_level).sum(-1).clamp_min(self.eps)
            stable_log_scale_mad = (
                responsibility * spreads_log_scale
            ).sum(-1).clamp_min(self.eps)
            shift_scores.append(
                ((flat_level[start:stop] - stable_level).abs() / stable_level_mad).cpu()
            )
            scale_scores.append(
                ((flat_log_scale[start:stop] - stable_log_scale).abs() / stable_log_scale_mad).cpu()
            )
        shift_score = torch.cat(shift_scores)
        scale_score = torch.cat(scale_scores)
        pattern["shift_score_cdf"] = shift_score[
            _even_sample_indices(shift_score.numel(), 16384)
        ].sort().values
        pattern["scale_score_cdf"] = scale_score[
            _even_sample_indices(scale_score.numel(), 16384)
        ].sort().values

    def build_from_embeddings(
        self,
        embeddings: Tensor,
        future_targets: Optional[Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        history_last: Optional[Tensor] = None,
        history_std: Optional[Tensor] = None,
        raw_patch_mean: Optional[Tensor] = None,
        raw_patch_std: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Build an artifact from chronological TRAIN embeddings only.

        Args:
            embeddings: Canonical projected tokens ``[W,C,P,d_p]``.
            future_targets: Optional full training horizons ``[W,C,H]`` used
                only for node-level predictive consistency.
        """
        if embeddings.ndim != 4 or embeddings.shape[-1] != self.pattern_dim:
            raise ValueError("embeddings must be [W,C,P,pattern_dim]")
        embeddings = embeddings.detach().cpu().float()
        if not self.raw_mode:
            embeddings = F.normalize(embeddings, dim=-1, eps=1e-6)
        windows, channels, patches, _ = embeddings.shape
        self._raw_patch_count = patches
        points = embeddings.reshape(-1, self.pattern_dim)
        window_anchor_ids = torch.div(torch.arange(windows), self.seq_len, rounding_mode="floor")
        anchor_ids = window_anchor_ids[:, None, None].expand(windows, channels, patches).reshape(-1)
        future_values = None
        if future_targets is not None:
            if future_targets.ndim != 3 or future_targets.shape[:2] != (windows, channels):
                raise ValueError("future_targets must have shape [W,C,H]")
            future_representation = future_targets.detach().cpu().float()
            if self.raw_mode and self.predictive_stability_mode == "per_future_znorm":
                future_mean = future_representation.mean(dim=-1, keepdim=True)
                future_std = future_representation.var(
                    dim=-1, keepdim=True, unbiased=False
                ).add(self.eps).sqrt()
                future_representation = (future_representation - future_mean) / future_std
            elif self.raw_mode and self.predictive_stability_mode == "train_global":
                train_mean = future_representation.mean(dim=(0, 2), keepdim=True)
                train_std = future_representation.var(
                    dim=(0, 2), keepdim=True, unbiased=False
                ).sqrt()
                future_representation = (future_representation - train_mean) / (train_std + self.eps)
                artifact_future_mean = train_mean.view(channels)
                artifact_future_std = train_std.view(channels)
            elif self.raw_mode and self.predictive_stability_mode == "relative_future":
                if history_last is None or history_std is None:
                    raise ValueError("relative_future requires TRAIN history_last and history_std")
                if history_last.shape != (windows, channels) or history_std.shape != (windows, channels):
                    raise ValueError("history_last/history_std must be [W,C]")
                future_representation = (
                    future_representation - history_last.detach().cpu().float().unsqueeze(-1)
                ) / (history_std.detach().cpu().float().unsqueeze(-1) + self.eps)
            else:
                future_representation = F.normalize(future_representation, dim=-1, eps=1e-6)
            future_values = future_representation[:, :, None, :].expand(
                windows, channels, patches, future_representation.shape[-1]
            ).reshape(-1, future_representation.shape[-1])

        pattern, assignment, coverage_radius = self._pattern_statistics(
            points, anchor_ids, future_values, windows
        )
        if self.raw_mode:
            if raw_patch_mean is None or raw_patch_std is None:
                raise ValueError("raw graph construction requires raw_patch_mean/raw_patch_std")
            self._attach_raw_geometry_statistics(
                embeddings, pattern, assignment, raw_patch_mean, raw_patch_std
            )
        mapping = self._mapping_statistics(embeddings, pattern, window_anchor_ids)
        if future_targets is not None:
            self._attach_future_prototypes(
                embeddings, pattern, mapping, future_representation
            )
        self._attach_mapping_mixture(mapping)
        artifact_metadata = dict(metadata or {})
        artifact_metadata.update(
            {
                "feature_dimension": self.pattern_dim,
                "number_of_train_windows": windows,
                "number_of_anchor_groups": int(window_anchor_ids.max().item()) + 1,
                "anchor_group_size": self.seq_len,
                "number_of_active_nodes": int(pattern["means"].shape[0]),
                "number_of_active_temporal_edges": int(mapping["src_index"].numel()),
                "coverage_radius": coverage_radius,
                "creation_date": datetime.now(timezone.utc).isoformat(),
                "source_split": "train",
                "predictive_target_representation": "normalized_full_horizon",
                "mapping_responsibility": "soft_posterior_mass_0.99",
                "representation_space": self.representation_space,
                "pattern_distance": "mean_squared_shape" if self.raw_mode else "euclidean_growth",
                "pattern_assignment": "content_addressed_only",
                "projector_used": not self.raw_mode,
                "predictive_stability_mode": self.predictive_stability_mode,
                "mapping_relation_decomposition": "train_only_diagonal_gaussian_mixture_v1",
                "mapping_mixture_features": "coverage,log_support,sample_entropy,stability,predictive_gain",
                "raw_geometry_statistics": "pattern_channel_train_robust_v1" if self.raw_mode else "none",
                "raw_signal_scale": "pre_patch_normalization_not_physical_raw_units",
            }
        )
        if self.raw_mode and self.predictive_stability_mode == "train_global":
            artifact_metadata["future_train_mean"] = artifact_future_mean
            artifact_metadata["future_train_std"] = artifact_future_std
        return {"pattern_graph": pattern, "mapping_graph": mapping, "metadata": artifact_metadata}
