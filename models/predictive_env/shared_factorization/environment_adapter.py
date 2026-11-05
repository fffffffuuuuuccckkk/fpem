"""Soft, sample-indexed interface to the existing predictive-conflict EIIL q."""

import torch


class EnvironmentPartitionAdapter:
    def __init__(self, sample_count: int, env_num: int, mode="predictive", seed=2021):
        if mode not in {"predictive", "shuffled", "random", "uniform", "no_env"}:
            raise ValueError(mode)
        self.sample_count, self.env_num = int(sample_count), int(env_num)
        self.mode, self.seed = mode, int(seed)
        self.stage = -1
        self.q = None

    def refresh_after_environment_update(self, discovered_q):
        discovered_q = torch.as_tensor(discovered_q, dtype=torch.float32).detach().cpu()
        if discovered_q.shape != (self.sample_count, self.env_num):
            raise ValueError(f"q shape {discovered_q.shape} != {(self.sample_count, self.env_num)}")
        if not torch.allclose(discovered_q.sum(-1), torch.ones(self.sample_count), atol=1e-4):
            raise ValueError("environment assignments must sum to one")
        self.stage += 1
        generator = torch.Generator().manual_seed(self.seed + 104729 * self.stage)
        if self.mode == "shuffled":
            self.q = discovered_q[torch.randperm(self.sample_count, generator=generator)]
        elif self.mode == "random":
            self.q = torch.softmax(torch.randn(discovered_q.shape, generator=generator), -1)
        elif self.mode == "uniform":
            self.q = torch.full_like(discovered_q, 1.0 / self.env_num)
        elif self.mode == "no_env":
            self.q = torch.ones(self.sample_count, 1)
        else:
            self.q = discovered_q
        return self.q

    def get_train_assignments(self):
        if self.q is None:
            raise RuntimeError("environment assignments have not been refreshed")
        return self.q

    def get_assignments(self, sample_indices):
        ids = torch.as_tensor(sample_indices, dtype=torch.long).cpu()
        if ids.numel() and (ids.min() < 0 or ids.max() >= self.sample_count):
            raise IndexError("TRAIN sample index out of range")
        return self.get_train_assignments()[ids]

    def get_environment_counts(self):
        return self.get_train_assignments().sum(0)

    def get_environment_statistics(self, hidden, sample_indices):
        pooled = hidden.detach().float().mean((1, 2)).cpu()
        q = self.get_assignments(sample_indices)
        mass = q.sum(0).clamp_min(1e-8)
        means = q.T @ pooled / mass[:, None]
        centered = pooled[:, None, :] - means[None, :, :]
        cov = torch.einsum("nk,nkd,nke->kde", q, centered, centered) / mass[:, None, None]
        return means, cov
