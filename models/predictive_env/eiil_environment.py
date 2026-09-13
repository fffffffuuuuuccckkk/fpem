"""Model-agnostic EIIL-style predictive-conflict environment inference."""
from __future__ import annotations
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _corr(left, right):
    if left.std() < 1e-12 or right.std() < 1e-12: return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


class PredictiveConflictEnvironment:
    def __init__(self, sample_count, env_num=6, steps=50, lr=.1, balance_weight=10.0,
                 entropy_weight=.1, seed=2021, diagnostic_samples=1024,
                 matching="overlap", matching_q_weight=1.0,
                 matching_gradient_weight=1.0):
        self.sample_count, self.env_num = int(sample_count), int(env_num)
        self.steps, self.lr = int(steps), float(lr)
        self.balance_weight, self.entropy_weight = float(balance_weight), float(entropy_weight)
        generator = torch.Generator().manual_seed(seed)
        self.logits = .01 * torch.randn(sample_count, env_num, generator=generator)
        self.q = self.logits.softmax(-1)
        self.diagnostic_samples = int(diagnostic_samples)
        if matching not in ("overlap", "gradient_aware"):
            raise ValueError("matching must be overlap or gradient_aware")
        self.matching = matching
        self.matching_q_weight = float(matching_q_weight)
        self.matching_gradient_weight = float(matching_gradient_weight)
        self.previous_probe_gradients = None
        self.update_index = 0

    def _align_to_previous(self, previous, current, current_gradients):
        """Hungarian-match new environment columns to the previous identities."""
        overlap = previous.t() @ current
        before = torch.diagonal(overlap).sum() / self.sample_count
        matching_used = self.matching
        if self.matching == "gradient_aware" and self.previous_probe_gradients is not None:
            old_mass = previous.sum(0)
            new_mass = current.sum(0)
            normalized_overlap = overlap / torch.sqrt(
                old_mass[:, None] * new_mass[None, :]
            ).clamp_min(1e-8)
            gradient_distance = torch.cdist(
                self.previous_probe_gradients, current_gradients
            )
            cost = (
                self.matching_q_weight * (1.0 - normalized_overlap)
                + self.matching_gradient_weight * gradient_distance
            )
        else:
            # First gradient-aware update has no old gradients, so it safely
            # falls back to the required overlap-only matching.
            cost = -overlap
            if self.matching == "gradient_aware":
                matching_used = "overlap_first_update"
        old_index, new_index = linear_sum_assignment(cost.detach().cpu().numpy())
        permutation = torch.empty(self.env_num, dtype=torch.long)
        permutation[torch.as_tensor(old_index)] = torch.as_tensor(new_index)
        after = overlap[torch.arange(self.env_num), permutation].sum() / self.sample_count
        return (
            current[:, permutation],
            current_gradients[permutation],
            permutation,
            float(before),
            float(after),
            matching_used,
        )

    @staticmethod
    def _environment_gradients(q, probe_per_sample):
        mass = q.sum(0).clamp_min(1e-6)
        return torch.einsum("ne,ng->eg", q, probe_per_sample) / mass[:, None]

    @staticmethod
    def _gradient_conflict(q, probe_per_sample):
        gradients = PredictiveConflictEnvironment._environment_gradients(
            q, probe_per_sample
        )
        if gradients.shape[0] < 2:
            return gradients.new_zeros(())
        return torch.pdist(gradients).square().mean()

    def _gradient_distance_diagnostics(self, q, probe_per_sample):
        """Soft within/between sample-gradient distances on a bounded subset."""
        count = min(self.sample_count, self.diagnostic_samples)
        selected = torch.linspace(0, self.sample_count - 1, count).long()
        q_selected = q[selected]
        gradient_selected = probe_per_sample[selected]
        distance = torch.cdist(gradient_selected, gradient_selected)
        same_environment = q_selected @ q_selected.t()
        mask = ~torch.eye(count, dtype=torch.bool)
        distance = distance[mask]
        same_environment = same_environment[mask]
        different_environment = 1.0 - same_environment
        within = (same_environment * distance).sum() / same_environment.sum().clamp_min(1e-8)
        between = (different_environment * distance).sum() / different_environment.sum().clamp_min(1e-8)
        separation = between / within.clamp_min(1e-8)
        return float(within), float(between), float(separation)

    def _random_partition_diagnostics(self, q, probe_per_sample, repeats=100):
        """Permutation null preserving K, exact masses, and assignment softness."""
        generator = torch.Generator().manual_seed(7919 + self.update_index)
        values = []
        for _ in range(repeats):
            permutation = torch.randperm(self.sample_count, generator=generator)
            values.append(
                self._gradient_conflict(q[permutation], probe_per_sample)
            )
        values = torch.stack(values)
        ours = self._gradient_conflict(q, probe_per_sample)
        mean = values.mean()
        std = values.std(unbiased=False)
        z_score = (ours - mean) / std.clamp_min(1e-12)
        return float(ours), float(mean), float(std), float(z_score)

    def _probe_gradients(self, prediction, target):
        error = prediction - target
        dims = tuple(range(1, prediction.ndim))
        grad_scale = 2.0 * (error * prediction).mean(dims)
        grad_bias = 2.0 * error.mean(dims)
        return torch.stack([grad_scale, grad_bias], -1)

    def update(self, prediction, target, sample_id, source_split="train"):
        if source_split.lower() != "train": raise ValueError("EIIL environment inference is TRAIN-only")
        sample_id = torch.as_tensor(sample_id).long()
        if not torch.equal(sample_id, torch.arange(len(sample_id))):
            raise ValueError("EIIL requires chronological complete TRAIN sample IDs")
        pred, true = torch.as_tensor(prediction).float(), torch.as_tensor(target).float()
        probe_per_sample = self._probe_gradients(pred, true)
        sample_loss = (pred - true).square().flatten(1).mean(1)
        logits = self.logits.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=self.lr)
        for _ in range(self.steps):
            q = logits.softmax(-1); mass = q.sum(0).clamp_min(1e-6)
            gradients = self._environment_gradients(q, probe_per_sample)
            conflict = gradients.var(0, unbiased=False).sum()
            balance = (q.mean(0) - 1.0 / self.env_num).square().sum()
            negative_entropy = (q * q.clamp_min(1e-8).log()).sum(1).mean()
            objective = -conflict + self.balance_weight * balance + self.entropy_weight * negative_entropy
            optimizer.zero_grad(); objective.backward(); optimizer.step()
        previous = self.q
        unaligned_logits = logits.detach().cpu()
        unaligned_q = unaligned_logits.softmax(-1)
        unaligned_gradients = self._environment_gradients(
            unaligned_q, probe_per_sample
        )
        (
            self.q,
            aligned_gradients,
            permutation,
            overlap_before,
            overlap_after,
            matching_used,
        ) = self._align_to_previous(previous, unaligned_q, unaligned_gradients)
        self.logits = unaligned_logits[:, permutation]
        mass = self.q.mean(0)
        gradients = self._environment_gradients(self.q, probe_per_sample)
        if not torch.allclose(gradients, aligned_gradients, atol=1e-6, rtol=1e-5):
            raise RuntimeError("Hungarian environment alignment corrupted gradients")
        risks = torch.einsum("ne,n->e", self.q, sample_loss) / self.q.sum(0).clamp_min(1e-6)
        disagreement = torch.pdist(gradients).square().mean() if self.env_num > 1 else torch.tensor(0.)
        change = (self.q - previous).norm(dim=1).mean()
        n = min(self.sample_count, self.diagnostic_samples)
        ids = torch.linspace(0, self.sample_count - 1, n).long()
        mask = ~torch.eye(n, dtype=torch.bool)
        old_sim = (previous[ids] @ previous[ids].t())[mask].numpy()
        new_sim = (self.q[ids] @ self.q[ids].t())[mask].numpy()
        within, between, separation = self._gradient_distance_diagnostics(
            self.q, probe_per_sample
        )
        conflict, random_mean, random_std, random_z = self._random_partition_diagnostics(
            self.q, probe_per_sample, repeats=100
        )
        similarity_correlation = _corr(old_sim, new_sim)
        record = {
            "update": self.update_index, "source_split": "TRAIN only",
            "mean_q_change": float(change), "environment_mass": mass.tolist(),
            "min_environment_mass": float(mass.min()),
            "environment_risk": risks.tolist(), "probe_gradients": gradients.tolist(),
            "gradient_disagreement": float(disagreement),
            "D_within_grad": within,
            "D_between_grad": between,
            "gradient_separation": separation,
            "random_partition_repeats": 100,
            "conflict_ours": conflict,
            "random_partition_conflict_mean": random_mean,
            "random_partition_conflict_std": random_std,
            "random_partition_z_score": random_z,
            "environment_similarity_correlation": similarity_correlation,
            "hungarian_permutation": permutation.tolist(),
            "alignment_overlap_before": overlap_before,
            "alignment_overlap_after": overlap_after,
            "environment_matching_requested": self.matching,
            "environment_matching_used": matching_used,
        }
        self.previous_probe_gradients = gradients.detach().clone()
        self.update_index += 1
        return self.q.numpy(), record
