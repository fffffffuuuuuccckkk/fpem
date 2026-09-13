"""Environment-aware multi-scale latent pattern bank."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def _two_component_soft(values: torch.Tensor) -> torch.Tensor:
    """Data-driven two-normal mixture posterior for the high component."""
    x = values.float().reshape(-1)
    if x.numel() < 2 or float(x.std()) < 1e-7:
        return torch.full_like(x, 0.5)
    means = torch.quantile(x, torch.tensor([0.25, 0.75], device=x.device))
    variances = torch.full((2,), x.var().clamp_min(1e-4), device=x.device)
    weights = torch.full((2,), 0.5, device=x.device)
    for _ in range(30):
        logp = -0.5 * ((x[:, None] - means) ** 2 / variances + variances.log()) + weights.log()
        resp = logp.softmax(1)
        mass = resp.sum(0).clamp_min(1e-6)
        weights = mass / x.numel()
        means = (resp * x[:, None]).sum(0) / mass
        variances = (resp * (x[:, None] - means) ** 2).sum(0) / mass
        variances = variances.clamp_min(1e-4)
    high = int(means.argmax())
    return resp[:, high]


class EnvironmentPatternBank(nn.Module):
    def __init__(self, d_model: int, pattern_count: int = 32, scales=(1, 2, 3, 4), temperature=0.2):
        super().__init__()
        self.d_model = int(d_model)
        self.pattern_count = int(pattern_count)
        self.scales = tuple(int(s) for s in scales)
        self.temperature = float(temperature)
        self.salience = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        # A neutral start makes TRAIN bank statistics deterministic; the scorer
        # remains learnable once Pattern adapters are trained.
        nn.init.zeros_(self.salience[-1].weight)
        nn.init.zeros_(self.salience[-1].bias)
        self.register_buffer("prototypes", torch.zeros(pattern_count, d_model))
        self.register_buffer("p_inv", torch.full((pattern_count,), 0.5))
        self.register_buffer("env_occurrence", torch.empty(pattern_count, 0))
        self.register_buffer("fitted", torch.tensor(False))

    def segments(self, hidden: torch.Tensor):
        # hidden [B,C,T,D]. Each segment remains attached to its start/scale.
        parts, owners = [], []
        b, c, t, d = hidden.shape
        for scale in self.scales:
            if scale > t:
                continue
            unfolded = hidden.unfold(2, scale, 1).permute(0, 1, 2, 4, 3)
            pooled = unfolded.mean(3)
            parts.append(pooled.reshape(-1, d))
            owners.append((scale, pooled.shape[2]))
        return torch.cat(parts, 0), owners

    def responsibilities(self, embeddings: torch.Tensor) -> torch.Tensor:
        z = F.normalize(embeddings, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        return ((z @ p.t()) / self.temperature).softmax(-1)

    def sample_presence(self, hidden: torch.Tensor) -> torch.Tensor:
        """Soft probability that each Pattern occurs at least once per sample."""
        b, c, t, d = hidden.shape
        pieces, salience_logits = [], []
        for scale in self.scales:
            if scale > t:
                continue
            pooled = hidden.unfold(2, scale, 1).permute(0, 1, 2, 4, 3).mean(3)
            resp = self.responsibilities(pooled.reshape(-1, d)).reshape(b, -1, self.pattern_count)
            pieces.append(resp)
            salience_logits.append(self.salience(pooled).reshape(b, -1, 1))
        responsibility = torch.cat(pieces, dim=1)
        # The existing segment salience is normalized within each sample, so a
        # long/multivariate sample does not receive larger support merely for
        # containing more segment slots. There is no threshold or new stride.
        salience = torch.cat(salience_logits, dim=1).softmax(1)
        responsibility = (salience * responsibility).double()
        log_absent = torch.log1p(-responsibility.clamp(max=1.0 - 1e-12)).sum(1)
        return (1.0 - torch.exp(log_absent)).float()

    @torch.no_grad()
    def fit(self, hidden_batches, environment_batches, max_segments=200000, iterations=20):
        if not hidden_batches:
            raise ValueError("TRAIN-only hidden batches are required")
        all_z, all_env = [], []
        for hidden, env in zip(hidden_batches, environment_batches):
            batch_z, batch_env = [], []
            b, c, t, d = hidden.shape
            for scale in self.scales:
                if scale > t:
                    continue
                pooled = hidden.unfold(2, scale, 1).permute(0, 1, 2, 4, 3).mean(3)
                batch_z.append(pooled.reshape(-1, d))
                batch_env.append(env[:, None, None].expand(b, c, pooled.shape[2]).reshape(-1))
            z = torch.cat(batch_z)
            env_expanded = torch.cat(batch_env)
            # Bound memory before global concatenation.
            per_batch_cap = max(1024, max_segments // max(len(hidden_batches), 1) * 2)
            if z.shape[0] > per_batch_cap:
                ids = torch.linspace(0, z.shape[0] - 1, per_batch_cap).long()
                z, env_expanded = z[ids], env_expanded[ids]
            all_z.append(z.cpu())
            all_env.append(env_expanded.cpu())
        z = F.normalize(torch.cat(all_z), dim=-1)
        env = torch.cat(all_env)
        if z.shape[0] > max_segments:
            ids = torch.linspace(0, z.shape[0] - 1, max_segments).long()
            z, env = z[ids], env[ids]
        # Deterministic spherical farthest-point initialization and soft
        # spherical k-means. Fit and query now use the same cosine geometry.
        proto = [z[0]]
        distance = torch.full((z.shape[0],), float("inf"))
        for _ in range(1, self.pattern_count):
            distance = torch.minimum(distance, 1.0 - z @ proto[-1])
            proto.append(z[int(distance.argmax())])
        proto = F.normalize(torch.stack(proto), dim=-1)
        for _ in range(iterations):
            resp = ((z @ proto.t()) / max(self.temperature, 1e-4)).softmax(1)
            proto = (resp.t() @ z) / resp.sum(0)[:, None].clamp_min(1e-6)
            proto = F.normalize(proto, dim=-1)

        # Install the fitted coordinates before any responsibility query. This
        # ordering is essential: occurrence must never see the old zero bank.
        self.prototypes.copy_(proto.to(self.prototypes.device))

        # Each TRAIN sample contributes one soft-presence value per Pattern,
        # regardless of how many channels/scales/segments it contains:
        # o_ik = 1 - product_j(1-r_ijk), c_ke = mean_{i in e}(o_ik).
        env_num = max(int(batch_env.max()) for batch_env in environment_batches) + 1
        occurrence_sum = torch.zeros(self.pattern_count, env_num)
        sample_count = torch.zeros(env_num)
        for hidden, batch_environment in zip(hidden_batches, environment_batches):
            presence = self.sample_presence(hidden.to(self.prototypes.device)).cpu()
            for e in range(env_num):
                mask = batch_environment == e
                occurrence_sum[:, e] += presence[mask].sum(0)
                sample_count[e] += mask.sum()
        occurrence = occurrence_sum / sample_count[None].clamp_min(1.0)
        normalized = occurrence / occurrence.sum(1, keepdim=True).clamp_min(1e-6)
        entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(1) / math.log(max(env_num, 2))
        coverage = (occurrence / occurrence.max(1, keepdim=True).values.clamp_min(1e-6)).mean(1)
        score = 0.5 * (entropy + coverage)
        if float(score.std()) < 1e-7:
            p_inv = _two_component_soft(score)
        else:
            standardized_score = (score - score.mean()) / score.std().clamp_min(1e-7)
            p_inv = _two_component_soft(standardized_score)
        self.p_inv.copy_(p_inv.to(self.p_inv.device))
        self.env_occurrence = occurrence.to(self.prototypes.device)
        self.fitted.fill_(True)
        return {"entropy": entropy, "coverage": coverage, "score": score}

    def query(self, hidden: torch.Tensor):
        if not bool(self.fitted):
            # Identity-safe pre-bank behavior.
            zero = torch.zeros_like(hidden)
            return zero, zero, torch.empty(*hidden.shape[:-1], self.pattern_count, device=hidden.device)
        b, c, t, d = hidden.shape
        token_mass = torch.zeros(b, c, t, self.pattern_count, device=hidden.device)
        token_weight = torch.zeros(b, c, t, 1, device=hidden.device)
        for scale in self.scales:
            if scale > t:
                continue
            pooled = hidden.unfold(2, scale, 1).permute(0, 1, 2, 4, 3).mean(3)
            resp = self.responsibilities(pooled.reshape(-1, d)).reshape(b, c, pooled.shape[2], -1)
            salience = torch.sigmoid(self.salience(pooled))
            for offset in range(scale):
                token_mass[:, :, offset:offset + pooled.shape[2]] += salience * resp
                token_weight[:, :, offset:offset + pooled.shape[2]] += salience
        token_resp = token_mass / token_weight.clamp_min(1e-6)
        inv_w = token_resp * self.p_inv
        var_w = token_resp * (1.0 - self.p_inv)
        inv_w = inv_w / inv_w.sum(-1, keepdim=True).clamp_min(1e-6)
        var_w = var_w / var_w.sum(-1, keepdim=True).clamp_min(1e-6)
        inv = inv_w @ self.prototypes
        var = var_w @ self.prototypes
        return inv, var, token_resp
