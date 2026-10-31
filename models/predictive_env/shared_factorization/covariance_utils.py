"""CPU/float64 streaming moments; no sample-by-sample N x N matrix."""

from dataclasses import dataclass

import torch


def symmetric(matrix):
    return (matrix + matrix.mT) * 0.5


@dataclass
class HiddenStatistics:
    mean: torch.Tensor
    covariance: torch.Tensor
    environment_means: torch.Tensor
    environment_covariances: torch.Tensor
    environment_mass: torch.Tensor
    sample_rows: torch.Tensor
    count: int


class StreamingHiddenStatistics:
    def __init__(self, width, env_num, max_rows=4096):
        self.width, self.env_num = int(width), int(env_num)
        self.max_rows = int(max_rows)
        self.mass = torch.zeros(env_num, dtype=torch.float64)
        self.sums = torch.zeros(env_num, width, dtype=torch.float64)
        self.outer = torch.zeros(env_num, width, width, dtype=torch.float64)
        self.count = 0
        self.rows = []
        self.rows_seen = 0
        self._generator = torch.Generator().manual_seed(2021)

    def add(self, hidden, q):
        # Exactly one observation per TRAIN sample, regardless of C or P.
        pooled = hidden.detach().float().mean((1, 2)).to("cpu", torch.float64)
        q = torch.as_tensor(q, dtype=torch.float64, device="cpu")
        if q.shape != (pooled.shape[0], self.env_num):
            raise ValueError("hidden/q sample alignment mismatch")
        self.mass += q.sum(0)
        self.sums += q.T @ pooled
        self.outer += torch.einsum("nk,nd,ne->kde", q, pooled, pooled)
        self.count += pooled.shape[0]
        if self.max_rows:
            # Bounded reservoir across the entire chronology, not only its prefix.
            for row in pooled:
                self.rows_seen += 1
                if len(self.rows) < self.max_rows:
                    self.rows.append(row.clone())
                else:
                    slot = int(torch.randint(self.rows_seen, (1,), generator=self._generator))
                    if slot < self.max_rows:
                        self.rows[slot] = row.clone()

    def finish(self):
        if self.count == 0:
            raise ValueError("no TRAIN hidden summaries")
        means = self.sums / self.mass.clamp_min(1e-8)[:, None]
        cov = self.outer / self.mass.clamp_min(1e-8)[:, None, None]
        cov = symmetric(cov - torch.einsum("kd,ke->kde", means, means))
        total_mass = self.mass.sum()
        mean = self.sums.sum(0) / total_mass
        total_cov = symmetric(self.outer.sum(0) / total_mass - torch.outer(mean, mean))
        sampled = torch.stack(self.rows) if self.rows else torch.empty(0, self.width, dtype=torch.float64)
        return HiddenStatistics(mean, total_cov, means, cov, self.mass.clone(),
                                sampled, self.count)


def soft_projectors(stats):
    """Stable C(C+lambda I)^-1 with one TRAIN-derived ridge scale."""
    covariance = symmetric(stats.covariance)
    eig = torch.linalg.eigvalsh(covariance).clamp_min(0)
    positive = eig[eig > eig.max().clamp_min(1e-12) * 1e-7]
    ridge = (positive.median() if positive.numel() else eig.new_tensor(1.0)) * 0.1
    ridge = ridge.clamp_min(1e-8)
    projectors = []
    for cov in stats.environment_covariances:
        values, vectors = torch.linalg.eigh(symmetric(cov))
        response = values.clamp_min(0) / (values.clamp_min(0) + ridge)
        projectors.append(symmetric((vectors * response) @ vectors.T))
    return torch.stack(projectors), float(ridge)


def spectral_energy_statistics(model, loader, device, basis, q):
    """Second TRAIN-only pass: per-sample directional energy and soft env energy."""
    width = basis.shape[0]
    sums = torch.zeros(width, dtype=torch.float64)
    squares = torch.zeros(width, dtype=torch.float64)
    env_sums = torch.zeros(q.shape[1], width, dtype=torch.float64)
    env_mass = torch.zeros(q.shape[1], dtype=torch.float64)
    count = 0
    basis = basis.to(device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for x, _, ids, cycle in loader:
            if ids[0] >= len(q):
                break  # bounded smoke subset; full TRAIN fit sees every ID
            hidden, _, _ = model.encode(x.float().to(device), cycle.to(device))
            energy = (hidden @ basis).square().mean((1, 2)).double().cpu()
            keep = ids.long() < len(q)
            energy = energy[keep]
            weights = q[ids.long()[keep]].double()
            sums += energy.sum(0)
            squares += energy.square().sum(0)
            env_sums += weights.T @ energy
            env_mass += weights.sum(0)
            count += energy.shape[0]
    prevalence = (sums.square() / (count * squares + 1e-12)).clamp(0, 1)
    return prevalence, env_sums / env_mass.clamp_min(1e-8)[:, None], sums / count
