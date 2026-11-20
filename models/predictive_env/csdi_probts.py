"""Forecasting adaptation of ProbTS-main's CSDI baseline.

Architecture, mask convention, quadratic beta schedule and DDPM reverse step
follow ProbTS-main/probts/model/forecaster/prob_forecaster/csdi.py. The only
interface adaptation is accepting the existing TSL [B,L,C] train/val/test
windows directly instead of GluonTS ``batch_data``. No FPEM/Zvar is used.
"""

import numpy as np
import torch
from torch import nn

from .csdi_probts_layers import diff_CSDI


class ProbTSCSDI(nn.Module):
    def __init__(self, channels, context_length, prediction_length,
                 hidden_channels=64, emb_time_dim=128, emb_feature_dim=16,
                 diffusion_embedding_dim=128, num_steps=50, num_heads=8,
                 n_layers=4, beta_start=1e-4, beta_end=0.5,
                 schedule="quad"):
        super().__init__()
        self.channels = int(channels)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.num_steps = int(num_steps)
        self.emb_time_dim = int(emb_time_dim)
        self.feature_embedding = nn.Embedding(channels, emb_feature_dim)
        self.diffmodel = diff_CSDI(
            hidden_channels, diffusion_embedding_dim,
            emb_time_dim + emb_feature_dim + 1,
            num_steps, num_heads, n_layers, inputdim=2, linear=False,
        )
        if schedule == "quad":
            beta = np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                               num_steps) ** 2
        elif schedule == "linear":
            beta = np.linspace(beta_start, beta_end, num_steps)
        else:
            raise ValueError(f"Unknown CSDI beta schedule: {schedule}")
        beta = torch.as_tensor(beta, dtype=torch.float32)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_hat", 1 - beta)
        self.register_buffer("alpha", torch.cumprod(1 - beta, dim=0))

    def _prepare(self, x, y=None):
        if x.ndim != 3 or x.shape[1:] != (self.context_length, self.channels):
            raise ValueError("past input must be [B,context_length,channels]")
        if y is not None and y.shape != (
                x.shape[0], self.prediction_length, self.channels):
            raise ValueError("future target must be [B,prediction_length,channels]")
        future = (y if y is not None else
                  x.new_zeros(x.shape[0], self.prediction_length,
                              self.channels))
        observed = torch.cat((x, future), dim=1).transpose(1, 2)
        mask = observed.new_zeros(observed.shape)
        mask[..., :self.context_length] = 1
        return observed, mask

    def _side_info(self, batch, device, dtype):
        length = self.context_length + self.prediction_length
        positions = torch.arange(length, device=device, dtype=dtype)
        frequencies = 1 / torch.pow(
            torch.tensor(10000.0, device=device, dtype=dtype),
            torch.arange(0, self.emb_time_dim, 2, device=device,
                         dtype=dtype) / self.emb_time_dim,
        )
        angles = positions[:, None] * frequencies[None]
        time = torch.empty(length, self.emb_time_dim, device=device,
                           dtype=dtype)
        time[:, 0::2] = torch.sin(angles)
        time[:, 1::2] = torch.cos(angles)
        time = time.T[None, :, None, :].expand(batch, -1, self.channels, -1)
        feature = self.feature_embedding(
            torch.arange(self.channels, device=device)
        ).to(dtype).T[None, :, :, None].expand(batch, -1, -1, length)
        mask = time.new_zeros(batch, 1, self.channels, length)
        mask[..., :self.context_length] = 1
        return torch.cat((time, feature, mask), dim=1)

    @staticmethod
    def _diffusion_input(noisy, observed, mask):
        return torch.cat(((mask * observed).unsqueeze(1),
                          ((1 - mask) * noisy).unsqueeze(1)), dim=1)

    def training_loss(self, x, y):
        observed, mask = self._prepare(x, y)
        batch = x.shape[0]
        timestep = torch.randint(self.num_steps, (batch,), device=x.device)
        current_alpha = self.alpha[timestep].view(batch, 1, 1)
        noise = torch.randn_like(observed)
        noisy = current_alpha.sqrt() * observed + (1-current_alpha).sqrt() * noise
        predicted = self.diffmodel(
            self._diffusion_input(noisy, observed, mask),
            self._side_info(batch, x.device, x.dtype), timestep,
        )
        target_mask = 1 - mask
        return (((noise - predicted) * target_mask).square().sum()
                / target_mask.sum().clamp_min(1))

    @torch.no_grad()
    def sample(self, x, num_samples=100, sample_chunk=4,
               generator=None):
        observed, mask = self._prepare(x)
        batch = x.shape[0]
        results = []
        for start in range(0, num_samples, sample_chunk):
            count = min(sample_chunk, num_samples - start)
            obs = observed.repeat_interleave(count, dim=0)
            cond = mask.repeat_interleave(count, dim=0)
            side = self._side_info(batch, x.device, x.dtype).repeat_interleave(
                count, dim=0)
            current = torch.randn(obs.shape, device=x.device, dtype=x.dtype,
                                  generator=generator)
            for t in range(self.num_steps - 1, -1, -1):
                timestep = torch.full((batch * count,), t, device=x.device,
                                      dtype=torch.long)
                predicted = self.diffmodel(
                    self._diffusion_input(current, obs, cond), side, timestep
                )
                coeff1 = self.alpha_hat[t].rsqrt()
                coeff2 = self.beta[t] / (1 - self.alpha[t]).sqrt()
                current = coeff1 * (current - coeff2 * predicted)
                if t > 0:
                    sigma = ((1 - self.alpha[t-1]) /
                             (1 - self.alpha[t]) * self.beta[t]).sqrt()
                    current = current + sigma * torch.randn(
                        current.shape, device=x.device, dtype=x.dtype,
                        generator=generator,
                    )
            future = current[..., self.context_length:].transpose(1, 2)
            results.append(future.reshape(batch, count,
                                          self.prediction_length, self.channels))
        return torch.cat(results, dim=1)
