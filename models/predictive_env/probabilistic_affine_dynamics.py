"""Invariant-anchored conditional flow over joint future affine trajectories."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .affine_target_projection import (
    decode_affine, interpolate_patches, project_affine_targets,
)
from .stochastic_innovation import StochasticInnovation


class AffineCenterHead(nn.Module):
    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model + 3), nn.Linear(d_model + 3, hidden),
            nn.GELU(), nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, future_zvar, local_stats):
        return self.net(torch.cat((future_zvar, local_stats), dim=-1))


class TemporalFiLMBlock(nn.Module):
    def __init__(self, hidden, condition_dim):
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.norm = nn.GroupNorm(8, hidden)
        self.film = nn.Linear(condition_dim, hidden * 2)

    def forward(self, x, condition):
        scale, shift = self.film(condition).transpose(1, 2).chunk(2, dim=1)
        hidden = self.norm(self.conv(x))
        return x + F.gelu(hidden * (1 + 0.1 * torch.tanh(scale)) + shift)


class JointAffineVelocity(nn.Module):
    """Temporal conv couples neighboring future patches in each trajectory."""

    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.input = nn.Conv1d(2, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([
            TemporalFiLMBlock(hidden, d_model + 4),
            TemporalFiLMBlock(hidden, d_model + 4),
        ])
        self.output = nn.Conv1d(hidden, 2, 3, padding=1)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, theta, time, future_zvar, local_stats):
        # theta [B,S,C,M,2]; all S trajectories share the same causal condition.
        batch, samples, channels, patches, _ = theta.shape
        condition = torch.cat((future_zvar, local_stats), dim=-1)
        condition = condition[:, None].expand(-1, samples, -1, -1, -1)
        if not torch.is_tensor(time):
            time = theta.new_tensor(time)
        time = time.to(device=theta.device, dtype=theta.dtype)
        if time.ndim == 0:
            time = time.expand(batch, samples, channels, patches, 1)
        else:
            time = time.expand(batch, samples, channels, patches, 1)
        condition = torch.cat((condition, time), dim=-1)
        condition = condition.reshape(batch * samples * channels, patches, -1)
        value = theta.reshape(batch * samples * channels, patches, 2).transpose(1, 2)
        value = self.input(value)
        for block in self.blocks:
            value = block(value, condition)
        return self.output(value).transpose(1, 2).reshape(batch, samples, channels, patches, 2)


class ProbabilisticAffineDynamics(nn.Module):
    def __init__(self, d_model, pred_len, patch_len=16, hidden=64,
                 sigma0=1.0, ridge=1e-3, target_clip=8.0):
        super().__init__()
        self.pred_len = int(pred_len)
        self.patch_len = int(patch_len)
        self.patch_count = math.ceil(self.pred_len / self.patch_len)
        self.sigma0 = float(sigma0)
        self.ridge = float(ridge)
        self.target_clip = float(target_clip)
        if self.sigma0 <= 0 or self.ridge <= 0 or self.target_clip <= 0:
            raise ValueError("sigma0/ridge/target_clip must be positive")
        self.center_head = AffineCenterHead(d_model, hidden)
        self.velocity = JointAffineVelocity(d_model, hidden)
        self.innovation = StochasticInnovation(d_model, hidden)
        self.gaussian_head = nn.Sequential(
            nn.LayerNorm(d_model + 3), nn.Linear(d_model + 3, hidden),
            nn.GELU(), nn.Linear(hidden, 4),
        )
        self.register_buffer("theta_scale", torch.ones(2))

    @torch.no_grad()
    def set_theta_scale(self, scale):
        scale = torch.as_tensor(scale, device=self.theta_scale.device,
                                dtype=self.theta_scale.dtype)
        if scale.shape != (2,) or not bool(torch.isfinite(scale).all()):
            raise ValueError("affine robust scale must be a finite pair")
        self.theta_scale.copy_(scale.clamp_min(0.01))

    def local_statistics(self, y_inv, input_scale):
        # Only predicted Yinv and historical X scale are observed at inference.
        inv = y_inv.detach().transpose(1, 2)
        scale = input_scale.detach().transpose(1, 2).clamp_min(1e-4)
        stats = []
        for start in range(0, self.pred_len, self.patch_len):
            patch = inv[..., start:min(start + self.patch_len, self.pred_len)]
            local = torch.stack((
                (patch.mean(-1) / scale[..., 0]).clamp(-20, 20),
                patch.std(-1, unbiased=False) / scale[..., 0],
                (patch[..., -1] - patch[..., 0]) / scale[..., 0],
            ), dim=-1)
            stats.append(local)
        return torch.stack(stats, dim=2)

    def condition(self, future_zvar, y_inv, input_scale, unconditional=False):
        if future_zvar.shape[2] != self.patch_count:
            raise ValueError("future Zvar patch count does not match prediction horizon")
        future = torch.zeros_like(future_zvar) if unconditional else future_zvar
        stats = self.local_statistics(y_inv, input_scale)
        center = self.center_head(future, stats)
        center_patch = torch.stack([
            y_inv.detach().transpose(1, 2)[..., start:min(start + self.patch_len, self.pred_len)].mean(-1)
            for start in range(0, self.pred_len, self.patch_len)
        ], dim=2)
        return {"future": future, "stats": stats, "mu_normalized": center,
                "center_patch": center_patch}

    def supervised_losses(self, condition, y_true, y_inv, input_scale,
                          use_innovation=True, gaussian_baseline=False):
        target, target_center = project_affine_targets(
            y_true, y_inv.detach(), input_scale.detach(), self.patch_len, self.ridge
        )
        raw = target / self.theta_scale.view(1, 1, 1, 2)
        clip_fraction = (raw.abs() > self.target_clip).float().mean().detach()
        normalized = raw.clamp(-self.target_clip, self.target_clip).detach()
        mu = condition["mu_normalized"]
        center_loss = F.smooth_l1_loss(mu, normalized)
        smooth_loss = (
            (mu[:, :, 1:] - mu[:, :, :-1]).square().mean()
            if self.patch_count > 1 else mu.new_zeros(())
        )
        fm_loss = mu.new_zeros(())
        gaussian_loss = mu.new_zeros(())
        if gaussian_baseline:
            raw_gaussian = self.gaussian_head(torch.cat(
                (condition["future"], condition["stats"]), dim=-1
            ))
            location, log_sigma = raw_gaussian.chunk(2, dim=-1)
            sigma = log_sigma.clamp(-5, 3).exp()
            gaussian_loss = (log_sigma.clamp(-5, 3) +
                             0.5 * ((normalized - location) / sigma).square()).mean()
        else:
            theta0 = mu.detach() + self.sigma0 * torch.randn_like(mu)
            tau = torch.rand((*mu.shape[:2], 1, 1), device=mu.device, dtype=mu.dtype)
            theta_tau = (1 - tau) * theta0 + tau * normalized
            velocity = self.velocity(theta_tau.unsqueeze(1), tau.unsqueeze(1),
                                     condition["future"], condition["stats"])[:, 0]
            fm_loss = F.mse_loss(velocity, (normalized - theta0).detach())
        innovation_loss = mu.new_zeros(())
        innovation_stats = {}
        if use_innovation:
            innovation_loss, innovation_stats = self.innovation.nll(
                condition["future"], y_true, y_inv.detach(), input_scale,
                target, target_center, self.patch_len,
            )
        return {
            "center": center_loss, "flow": fm_loss, "gaussian": gaussian_loss,
            "innovation": innovation_loss, "smooth": smooth_loss,
            "target_clip_fraction": clip_fraction,
            "target_gamma_abs_mean": target[..., 0].detach().abs().mean(),
            "target_beta_abs_mean": target[..., 1].detach().abs().mean(),
            **innovation_stats,
        }

    @torch.no_grad()
    def sample(self, condition, y_inv, input_scale, num_samples=100,
               steps=12, method="flow", use_innovation=True,
               chunk_size=10, generator=None):
        if num_samples < 1 or steps < 1 or chunk_size < 1:
            raise ValueError("num_samples/steps/chunk_size must be positive")
        batch, channels, patches, _ = condition["mu_normalized"].shape
        chunks = []
        gaussian = None
        if method == "gaussian":
            raw = self.gaussian_head(torch.cat(
                (condition["future"], condition["stats"]), dim=-1
            ))
            loc, log_sigma = raw.chunk(2, dim=-1)
            gaussian = (loc, log_sigma.clamp(-5, 3).exp())
        for start in range(0, num_samples, chunk_size):
            count = min(chunk_size, num_samples - start)
            if method == "center":
                theta = condition["mu_normalized"][:, None].expand(-1, count, -1, -1, -1)
            else:
                noise = torch.randn(
                    batch, count, channels, patches, 2,
                    device=y_inv.device, dtype=y_inv.dtype, generator=generator,
                )
                if method == "gaussian":
                    theta = gaussian[0][:, None] + gaussian[1][:, None] * noise
                elif method == "flow":
                    theta = condition["mu_normalized"][:, None] + self.sigma0 * noise
                    for step in range(steps):
                        theta = theta + self.velocity(
                            theta, (step + 0.5) / steps,
                            condition["future"], condition["stats"],
                        ) / steps
                else:
                    raise ValueError(f"unknown affine sampling method: {method}")
            chunks.append(theta * self.theta_scale.view(1, 1, 1, 1, 2))
        physical = torch.cat(chunks, dim=1)
        innovation = (
            self.innovation.sample(condition["future"], self.pred_len,
                                   self.patch_len, num_samples, generator)
            if use_innovation else None
        )
        paths = decode_affine(
            y_inv, input_scale, physical, condition["center_patch"],
            self.patch_len, innovation,
        )
        return {
            "y_inv": y_inv,
            "y_samples": paths,
            "y_mean": paths.mean(dim=1),
            "y_median": paths.median(dim=1).values,
            "lower_95": torch.quantile(paths, 0.025, dim=1),
            "upper_95": torch.quantile(paths, 0.975, dim=1),
            "gamma_samples": physical[..., 0],
            "beta_samples": physical[..., 1],
            "innovation_samples": innovation,
        }
