"""Small autoregressive innovation over affine-unexplained residuals."""

import torch
import torch.nn.functional as F
from torch import nn

from .affine_target_projection import decode_affine, interpolate_patches


class StochasticInnovation(nn.Module):
    def __init__(self, d_model, hidden=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias[:] = torch.tensor([0.0, -2.0])

    def parameters_for_patches(self, future_zvar):
        raw = self.head(future_zvar)
        rho = 0.98 * torch.tanh(raw[..., 0])
        sigma = F.softplus(raw[..., 1]).clamp(0.01, 5.0)
        return rho, sigma

    def parameters_for_horizon(self, future_zvar, pred_len, patch_len):
        rho, sigma = self.parameters_for_patches(future_zvar)
        rho = interpolate_patches(rho.unsqueeze(-1), pred_len, patch_len)[..., 0]
        sigma = interpolate_patches(sigma.unsqueeze(-1), pred_len, patch_len)[..., 0]
        return rho, sigma

    def nll(self, future_zvar, y_true, y_inv, input_scale,
            target_theta, center_patch, patch_len):
        horizon = y_true.shape[1]
        rho, sigma = self.parameters_for_horizon(future_zvar, horizon, patch_len)
        oracle = decode_affine(y_inv, input_scale, target_theta, center_patch, patch_len)[:, 0]
        scale = input_scale.detach().transpose(1, 2).clamp_min(1e-4)
        residual = ((y_true.detach() - oracle.detach()).transpose(1, 2) / scale)
        previous = torch.cat((torch.zeros_like(residual[..., :1]), residual[..., :-1]), dim=-1)
        innovation = residual - rho * previous
        nll = (sigma.log() + 0.5 * (innovation / sigma).square()).mean()
        diagnostics = {
            "innovation_sigma_mean": sigma.detach().mean(),
            "innovation_rho_abs_mean": rho.detach().abs().mean(),
            "innovation_predicted_variance": sigma.detach().square().mean(),
            "innovation_residual_variance": residual.detach().var(unbiased=False),
        }
        diagnostics["innovation_variance_ratio"] = (
            diagnostics["innovation_predicted_variance"] /
            diagnostics["innovation_residual_variance"].clamp_min(1e-8)
        )
        return nll, diagnostics

    @torch.no_grad()
    def sample(self, future_zvar, pred_len, patch_len, num_samples, generator=None):
        rho, sigma = self.parameters_for_horizon(future_zvar, pred_len, patch_len)
        batch, channels, horizon = rho.shape
        noise = torch.randn(
            batch, num_samples, channels, horizon,
            device=rho.device, dtype=rho.dtype, generator=generator,
        )
        previous = torch.zeros(batch, num_samples, channels, device=rho.device, dtype=rho.dtype)
        paths = []
        for time in range(horizon):
            previous = rho[:, None, :, time] * previous + sigma[:, None, :, time] * noise[..., time]
            paths.append(previous)
        return torch.stack(paths, dim=-1)
