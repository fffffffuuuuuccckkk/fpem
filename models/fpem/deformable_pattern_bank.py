"""TRAIN-only canonical pattern bank with explicit low-rank shape deformation."""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .future_mapping import _fit_two_component_mixture
from .pattern_graph_builder import PatternMappingGraphBuilder, _even_sample_indices


def _normalize_shape(values: Tensor, eps: float = 1e-5) -> Tensor:
    centered = values - values.mean(-1, keepdim=True)
    return centered / centered.var(-1, keepdim=True, unbiased=False).sqrt().clamp_min(eps)


class DeformablePatternBank:
    """Immutable query object for ``P_k + U_k c`` reconstruction."""

    def __init__(self, statistics: Dict[str, Tensor], coefficient_ridge: float = 0.05):
        self.statistics = statistics
        self.patterns = statistics["canonical_patterns"].float()
        self.shape_basis = statistics["shape_basis"].float()
        self.p_inv = statistics["pattern_p_inv"].float()
        self.coefficient_ridge = float(coefficient_ridge)

    @property
    def rank(self) -> int:
        return int(self.shape_basis.shape[-1])

    def _distance(self, points: Tensor, chunk_size: int = 4096) -> Tuple[Tensor, Tensor]:
        distances, coefficients = [], []
        for start in range(0, points.shape[0], chunk_size):
            point = points[start:start + chunk_size].float()
            residual = point[:, None, :] - self.patterns[None, :, :]
            if self.rank:
                coefficient = torch.einsum("nkl,klr->nkr", residual, self.shape_basis)
                coefficient = coefficient / (1.0 + self.coefficient_ridge)
                reconstructed = torch.einsum("nkr,klr->nkl", coefficient, self.shape_basis)
                error = (residual - reconstructed).square().mean(-1)
                error = error + self.coefficient_ridge * coefficient.square().mean(-1)
            else:
                coefficient = point.new_zeros(point.shape[0], self.patterns.shape[0], 0)
                error = residual.square().mean(-1)
            distances.append(error.cpu())
            coefficients.append(coefficient.cpu())
        return torch.cat(distances), torch.cat(coefficients)

    def _best(self, points: Tensor, chunk_size: int = 4096) -> Tuple[Tensor, Tensor, Tensor]:
        errors, indices, coefficients = [], [], []
        for start in range(0, points.shape[0], chunk_size):
            point = points[start:start + chunk_size].float()
            residual = point[:, None, :] - self.patterns[None, :, :]
            if self.rank:
                all_coefficients = torch.einsum("nkl,klr->nkr", residual, self.shape_basis)
                all_coefficients = all_coefficients / (1.0 + self.coefficient_ridge)
                reconstructed = torch.einsum("nkr,klr->nkl", all_coefficients, self.shape_basis)
                distance = (residual - reconstructed).square().mean(-1)
                distance = distance + self.coefficient_ridge * all_coefficients.square().mean(-1)
            else:
                all_coefficients = point.new_zeros(point.shape[0], self.patterns.shape[0], 0)
                distance = residual.square().mean(-1)
            error, index = distance.min(-1)
            row = torch.arange(point.shape[0])
            errors.append(error.cpu())
            indices.append(index.cpu())
            coefficients.append(all_coefficients[row, index].cpu())
        return torch.cat(errors), torch.cat(indices), torch.cat(coefficients)

    def query(self, normalized_patches: Tensor,
              include_responsibility: bool = True) -> Dict[str, Tensor]:
        """Explain ``[..., L]`` as canonical identity plus structured shape."""
        prefix = normalized_patches.shape[:-1]
        flat = normalized_patches.reshape(-1, normalized_patches.shape[-1]).float().cpu()
        best_error, canonical_id, coefficient = self._best(flat)
        canonical = self.patterns[canonical_id]
        if self.rank:
            deformation = torch.einsum("nr,nlr->nl", coefficient, self.shape_basis[canonical_id])
        else:
            deformation = torch.zeros_like(canonical)
        reconstruction = canonical + deformation
        result = {
            "canonical_pattern_id": canonical_id.view(prefix),
            "shape_coefficients": coefficient.view(*prefix, self.rank),
            "canonical_reconstruction": canonical.view(*prefix, canonical.shape[-1]),
            "shape_reconstruction": reconstruction.view(*prefix, reconstruction.shape[-1]),
            "reconstruction_error": (flat - reconstruction).square().mean(-1).view(prefix),
            "pattern_p_inv": self.p_inv[canonical_id].view(prefix),
        }
        if include_responsibility:
            canonical_distance = []
            for start in range(0, flat.shape[0], 8192):
                canonical_distance.append(
                    torch.cdist(flat[start:start + 8192], self.patterns).square().div(self.patterns.shape[-1])
                )
            responsibility = torch.softmax(-torch.cat(canonical_distance), -1)
            result["responsibility"] = responsibility.view(*prefix, self.patterns.shape[0])
        return result

    def reconstruct_signal(self, raw_patches: Tensor) -> Dict[str, Tensor]:
        mean = raw_patches.mean(-1, keepdim=True)
        scale = raw_patches.var(-1, keepdim=True, unbiased=False).sqrt().clamp_min(1e-5)
        normalized = (raw_patches - mean) / scale
        result = self.query(normalized)
        result["patch_mean"] = mean.squeeze(-1)
        result["patch_scale"] = scale.squeeze(-1)
        result["signal_reconstruction"] = mean + scale * result["shape_reconstruction"]
        return result


class DeformablePatternBankBuilder:
    """Radius initialization followed by alternating canonical/PCA refinement."""

    def __init__(self, patch_len: int, seq_len: int, shape_basis_rank: int = 2,
                 coefficient_ridge: float = 0.05, refinement_steps: int = 4,
                 invariant_consolidation: bool = False, local_merge: bool = False):
        if shape_basis_rank not in {0, 1, 2, 4}:
            raise ValueError("shape_basis_rank must be one of 0, 1, 2, 4")
        if shape_basis_rank >= patch_len - 1:
            raise ValueError("shape basis rank must be smaller than patch_len - 1")
        self.patch_len = int(patch_len)
        self.seq_len = max(1, int(seq_len))
        self.rank = int(shape_basis_rank)
        self.coefficient_ridge = float(coefficient_ridge)
        self.refinement_steps = int(refinement_steps)
        self.invariant_consolidation = bool(invariant_consolidation)
        self.local_merge = bool(local_merge)
        self.legacy_initializer = PatternMappingGraphBuilder(
            patch_len, seq_len=seq_len, representation_space="raw",
            patch_len=patch_len,
        )

    @staticmethod
    def _anchor_statistics(assignment: Tensor, anchor_ids: Tensor,
                           pattern_count: int) -> Tuple[Tensor, Tensor, Tensor]:
        anchor_count = int(anchor_ids.max()) + 1
        coverage, entropy, concentration = [], [], []
        for index in range(pattern_count):
            counts = torch.bincount(anchor_ids[assignment == index], minlength=anchor_count).float()
            present = counts > 0
            coverage.append(present.float().mean())
            probability = counts / counts.sum().clamp_min(1)
            value = -(probability * probability.clamp_min(1e-8).log()).sum()
            entropy.append(value / math.log(anchor_count) if anchor_count > 1 else value.new_tensor(1.0))
            concentration.append(probability.max())
        return torch.stack(coverage), torch.stack(entropy), torch.stack(concentration)

    @staticmethod
    def _orthogonal_shape_basis(residual: Tensor, canonical: Tensor, rank: int) -> Tensor:
        if rank == 0:
            return residual.new_zeros(residual.shape[-1], 0)
        residual = residual - residual.mean(-1, keepdim=True)
        canonical_direction = canonical / canonical.norm().clamp_min(1e-6)
        residual = residual - (residual @ canonical_direction)[:, None] * canonical_direction[None]
        if residual.shape[0] < 2 or float(residual.square().sum()) < 1e-8:
            candidates = torch.eye(residual.shape[-1])
        else:
            _, _, candidates = torch.linalg.svd(residual, full_matrices=False)
        vectors = []
        for candidate in candidates:
            vector = candidate - candidate.mean()
            vector = vector - torch.dot(vector, canonical_direction) * canonical_direction
            for previous in vectors:
                vector = vector - torch.dot(vector, previous) * previous
            norm = vector.norm()
            if float(norm) > 1e-6:
                vectors.append(vector / norm)
            if len(vectors) == rank:
                break
        if len(vectors) < rank:
            raise RuntimeError("could not form requested orthogonal shape basis")
        return torch.stack(vectors, -1)

    def _refine(self, points: Tensor, initial: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        centers = _normalize_shape(initial.float())
        bases = centers.new_zeros(centers.shape[0], self.patch_len, self.rank)
        fitting = points[_even_sample_indices(points.shape[0], min(points.shape[0], 100000))]
        for _ in range(self.refinement_steps):
            bank = DeformablePatternBank({
                "canonical_patterns": centers, "shape_basis": bases,
                "pattern_p_inv": torch.ones(centers.shape[0]),
            }, self.coefficient_ridge)
            assignment = bank.query(fitting, include_responsibility=False)["canonical_pattern_id"].reshape(-1)
            updated_centers, updated_bases = [], []
            keep = []
            for index in range(centers.shape[0]):
                members = fitting[assignment == index]
                if members.shape[0] < max(2, self.rank + 1):
                    continue
                canonical = _normalize_shape(members.mean(0, keepdim=True))[0]
                basis = self._orthogonal_shape_basis(members - canonical, canonical, self.rank)
                if self.rank:
                    coefficient = (members - canonical) @ basis / (1.0 + self.coefficient_ridge)
                    canonical = _normalize_shape((members - coefficient @ basis.t()).mean(0, keepdim=True))[0]
                    basis = self._orthogonal_shape_basis(members - canonical, canonical, self.rank)
                updated_centers.append(canonical)
                updated_bases.append(basis)
                keep.append(index)
            centers = torch.stack(updated_centers)
            bases = torch.stack(updated_bases)
        bank = DeformablePatternBank({
            "canonical_patterns": centers, "shape_basis": bases,
            "pattern_p_inv": torch.ones(centers.shape[0]),
        }, self.coefficient_ridge)
        result = bank.query(points, include_responsibility=False)
        return centers, bases, result["canonical_pattern_id"].reshape(-1), result["reconstruction_error"].reshape(-1)

    def fit(self, normalized_patches: Tensor, future_values: Optional[Tensor] = None,
            source_split: str = "train") -> Dict[str, Tensor]:
        if source_split != "train":
            raise ValueError("deformable pattern bank may only be fitted from TRAIN")
        if normalized_patches.ndim != 4 or normalized_patches.shape[-1] != self.patch_len:
            raise ValueError("normalized_patches must be [W,C,P,patch_len]")
        windows, channels, patches, _ = normalized_patches.shape
        points = _normalize_shape(normalized_patches.detach().cpu().float()).reshape(-1, self.patch_len)
        window_anchor = torch.div(torch.arange(windows), self.seq_len, rounding_mode="floor")
        anchor_ids = window_anchor[:, None, None].expand(windows, channels, patches).reshape(-1)
        radius, _ = self.legacy_initializer._coverage_radius(points)
        budget = max(4, int(math.ceil(math.sqrt(windows))))
        provisional = self.legacy_initializer._candidate_centers(points, radius, budget)
        centers, bases, assignment, reconstruction_error = self._refine(points, provisional)
        count = torch.bincount(assignment, minlength=centers.shape[0]).float()
        coverage, occurrence_entropy, concentration = self._anchor_statistics(
            assignment, anchor_ids, centers.shape[0]
        )
        if future_values is None:
            predictive = torch.ones_like(coverage)
        else:
            if future_values.shape[:2] != (windows, channels):
                raise ValueError("future_values must be [W,C,H]")
            repeated = future_values.detach().cpu().float()[:, :, None, :].expand(
                windows, channels, patches, future_values.shape[-1]
            ).reshape(-1, future_values.shape[-1])
            repeated = _normalize_shape(repeated)
            global_variance = repeated.var(0, unbiased=False).mean().clamp_min(1e-6)
            predictive_values = []
            for index in range(centers.shape[0]):
                local = repeated[assignment == index]
                variance = local.var(0, unbiased=False).mean() if local.shape[0] else global_variance
                predictive_values.append((1.0 + variance / global_variance).reciprocal())
            predictive = torch.stack(predictive_values)
        mixture = _fit_two_component_mixture(
            torch.stack([coverage, occurrence_entropy, predictive], -1)
        )
        p_inv = mixture["p_inv"]
        p_local = 1.0 - p_inv
        invariant_component = mixture["invariant_component"]
        posterior = torch.stack([p_inv, p_local], -1)
        invariant_mask = posterior.argmax(-1) == 0
        if not bool(invariant_mask.any()):
            invariant_mask[p_inv.argmax()] = True

        local_indices = torch.nonzero(~invariant_mask, as_tuple=False).flatten()
        invariant_indices = torch.nonzero(invariant_mask, as_tuple=False).flatten()
        local_to_canonical = torch.full((centers.shape[0],), -1, dtype=torch.long)
        local_coefficients = torch.zeros(centers.shape[0], self.rank)
        local_merge_error = torch.full((centers.shape[0],), float("nan"))
        local_explained = torch.zeros(centers.shape[0], dtype=torch.bool)
        if local_indices.numel() and invariant_indices.numel():
            invariant_bank = DeformablePatternBank({
                "canonical_patterns": centers[invariant_indices],
                "shape_basis": bases[invariant_indices],
                "pattern_p_inv": p_inv[invariant_indices],
            }, self.coefficient_ridge)
            distance, coefficients = invariant_bank._distance(centers[local_indices])
            error, match = distance.min(-1)
            rows = torch.arange(local_indices.numel())
            local_to_canonical[local_indices] = invariant_indices[match]
            local_merge_error[local_indices] = error
            if self.rank:
                local_coefficients[local_indices] = coefficients[rows, match]
            if local_indices.numel() >= 2 and float(error.std(unbiased=False)) > 1e-8:
                error_mixture = _fit_two_component_mixture((-error).unsqueeze(-1))
                local_explained[local_indices] = error_mixture["p_inv"] >= error_mixture["p_inv"].new_tensor(0.5)
            else:
                local_explained[local_indices] = True

        final_indices = invariant_indices if self.invariant_consolidation else torch.arange(centers.shape[0])
        statistics = {
            "provisional_patterns": provisional,
            "canonical_patterns_all": centers,
            "shape_basis_all": bases,
            "canonical_patterns": centers[final_indices],
            "shape_basis": bases[final_indices],
            "support_count": count,
            "coverage": coverage,
            "occurrence_entropy": occurrence_entropy,
            "concentration": concentration,
            "predictive_consistency": predictive,
            "pattern_p_inv_all": p_inv,
            "pattern_p_local_all": p_local,
            "pattern_p_inv": p_inv[final_indices],
            "pattern_p_local": p_local[final_indices],
            "invariant_indices": invariant_indices,
            "final_source_indices": final_indices,
            "train_assignment": assignment,
            "train_reconstruction_error": reconstruction_error,
            "local_to_canonical_index": local_to_canonical,
            "local_shape_coefficients": local_coefficients,
            "local_merge_error": local_merge_error,
            "local_explained": local_explained,
            "coverage_radius": torch.tensor(radius),
            "number_of_anchors": torch.tensor(int(window_anchor.max()) + 1),
            "source_split_code": torch.tensor(1),
            "shape_basis_rank": torch.tensor(self.rank),
            "mixture_component_mean": mixture["component_mean"],
            "mixture_invariant_component": invariant_component,
        }
        final_bank = DeformablePatternBank(statistics, self.coefficient_ridge)
        final_query = final_bank.query(points, include_responsibility=False)
        statistics["final_train_reconstruction_error"] = final_query["reconstruction_error"].reshape(-1)
        return statistics
