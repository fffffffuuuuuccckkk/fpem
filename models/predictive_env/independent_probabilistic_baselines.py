"""Project adapters for TimeGrad, TSFlow and TSDiff core objectives.

These share the FPEM CSV split and metric protocol, but do not share an FPEM
encoder. They are transparent, compact PyTorch adaptations rather than
bit-for-bit copies of the upstream GluonTS/Lightning implementations.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def history_scale(x, minimum=0.0):
    loc = x.mean(dim=1, keepdim=True)
    scale = x.var(dim=1, keepdim=True, unbiased=False).add(1e-5).sqrt()
    return loc, scale.clamp_min(minimum)


def draw(shape, device, generator=None):
    return torch.randn(shape, device=device, generator=generator)


def diffusion_schedule(steps, start=1e-4, end=0.02):
    beta = torch.linspace(start, end, steps)
    alpha = 1.0 - beta
    alpha_bar = alpha.cumprod(0)
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
    posterior = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar).clamp_min(1e-8)
    return beta, alpha, alpha_bar, posterior


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(8, hidden)
        self.conv = nn.Conv1d(hidden, 2 * hidden, 3, dilation=dilation,
                              padding=dilation)
        self.output = nn.Conv1d(hidden, hidden, 1)

    def forward(self, x):
        gate, content = self.conv(F.silu(self.norm(x))).chunk(2, dim=1)
        return (x + self.output(torch.sigmoid(gate) * torch.tanh(content))) / math.sqrt(2)


class TemporalNetwork(nn.Module):
    """Dilated temporal residual network; sequence positions interact."""

    def __init__(self, input_dim, output_dim, hidden=64, blocks=4):
        super().__init__()
        self.input = nn.Conv1d(input_dim, hidden, 1)
        self.blocks = nn.Sequential(*(ResidualTemporalBlock(hidden, 2 ** (i % 3))
                                      for i in range(blocks)))
        self.output = nn.Sequential(nn.GroupNorm(8, hidden), nn.SiLU(),
                                    nn.Conv1d(hidden, output_dim, 1))

    def forward(self, value):
        return self.output(self.blocks(self.input(value.transpose(1, 2)))).transpose(1, 2)


class TimeGradAdapter(nn.Module):
    """Autoregressive GRU conditioner with per-step DDPM noise prediction."""

    def __init__(self, channels, seq_len, pred_len, hidden=64, steps=20):
        super().__init__()
        self.channels, self.seq_len, self.pred_len, self.steps = channels, seq_len, pred_len, steps
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.step_embed = nn.Embedding(steps, hidden)
        self.denoiser = nn.Sequential(nn.Linear(channels + 2 * hidden, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden), nn.SiLU(),
                                      nn.Linear(hidden, channels))
        beta, alpha, alpha_bar, posterior = diffusion_schedule(steps)
        for name, value in (('beta', beta), ('alpha', alpha), ('alpha_bar', alpha_bar),
                            ('posterior', posterior)):
            self.register_buffer(name, value)

    def _predict_noise(self, noisy, condition, time):
        time_feature = self.step_embed(time)
        return self.denoiser(torch.cat([noisy, condition, time_feature], dim=-1))

    def training_loss(self, past, future):
        loc, scale = history_scale(past)
        past_n, future_n = (past - loc) / scale, (future - loc) / scale
        teacher_input = torch.cat([past_n, future_n[:, :-1]], dim=1)
        hidden, _ = self.gru(teacher_input)
        condition = hidden[:, self.seq_len - 1:]
        time = torch.randint(self.steps, (past.shape[0], self.pred_len), device=past.device)
        noise = torch.randn_like(future_n)
        a = self.alpha_bar[time].unsqueeze(-1)
        noisy = a.sqrt() * future_n + (1 - a).sqrt() * noise
        prediction = self._predict_noise(noisy, condition, time)
        return F.mse_loss(prediction, noise)

    @torch.no_grad()
    def sample(self, past, num_samples, sample_chunk=10, generator=None):
        loc, scale = history_scale(past)
        past_n = (past - loc) / scale
        output = []
        for first in range(0, num_samples, sample_chunk):
            count = min(sample_chunk, num_samples - first)
            repeated = past_n.repeat_interleave(count, dim=0)
            _, state = self.gru(repeated)
            predicted = []
            for _ in range(self.pred_len):
                current = draw((repeated.shape[0], self.channels), past.device, generator)
                for step in reversed(range(self.steps)):
                    time = torch.full((repeated.shape[0],), step, device=past.device, dtype=torch.long)
                    eps = self._predict_noise(current, state[-1], time)
                    beta, alpha, alpha_bar = self.beta[step], self.alpha[step], self.alpha_bar[step]
                    current = (current - beta / (1 - alpha_bar).sqrt() * eps) / alpha.sqrt()
                    if step:
                        current += self.posterior[step].sqrt() * draw(current.shape, past.device, generator)
                predicted.append(current)
                _, state = self.gru(current.unsqueeze(1), state)
            values = torch.stack(predicted, dim=1).reshape(past.shape[0], count, self.pred_len, self.channels)
            output.append(values * scale[:, None] + loc[:, None])
        return torch.cat(output, dim=1)


class GaussianProcessPrior(nn.Module):
    """Fixed RBF conditional GP prior used by the TSFlow adapter."""

    def __init__(self, context, horizon, length_scale=24.0, jitter=0.1):
        super().__init__()
        past_time = torch.arange(context).float()
        future_time = torch.arange(context, context + horizon).float()
        kernel = lambda a, b: torch.exp(-0.5 * ((a[:, None] - b[None, :]) / length_scale).square())
        pp = kernel(past_time, past_time) + jitter * torch.eye(context)
        fp = kernel(future_time, past_time)
        ff = kernel(future_time, future_time)
        weights = torch.linalg.solve(pp, fp.T).T
        covariance = ff - weights @ fp.T + 1e-4 * torch.eye(horizon)
        self.register_buffer('weights', weights)
        self.register_buffer('chol', torch.linalg.cholesky(covariance))

    def forward(self, past, generator=None):
        mean = torch.einsum('hl,blc->bhc', self.weights, past)
        noise = draw(mean.shape, past.device, generator)
        sample = mean + torch.einsum('hl,blc->bhc', self.chol, noise)
        return sample, mean


class TSFlowAdapter(nn.Module):
    """Conditional flow matching from a GP-regression prior to future series."""

    def __init__(self, channels, seq_len, pred_len, hidden=64, steps=16):
        super().__init__()
        self.channels, self.pred_len, self.steps = channels, pred_len, steps
        self.prior = GaussianProcessPrior(seq_len, pred_len)
        self.encoder = nn.GRU(channels, hidden, batch_first=True)
        self.context = nn.Linear(hidden, hidden)
        self.velocity = TemporalNetwork(2 * channels + hidden + 1, channels, hidden)

    def _velocity(self, state, gp_mean, context, time):
        batch, horizon, _ = state.shape
        if not torch.is_tensor(time):
            time = state.new_full((batch, 1, 1), time)
        time = time.expand(batch, horizon, 1)
        condition = self.context(context).unsqueeze(1).expand(batch, horizon, -1)
        return self.velocity(torch.cat([state, gp_mean, condition, time], dim=-1))

    def training_loss(self, past, future):
        # TSFlow's upstream StdScaler uses minimum_scale=1. A tiny local
        # variance must not magnify a nearly constant channel by 100x.
        loc, scale = history_scale(past, minimum=1.0)
        past_n, future_n = (past - loc) / scale, (future - loc) / scale
        x0, gp_mean = self.prior(past_n)
        _, context = self.encoder(past_n)
        time = torch.rand((past.shape[0], 1, 1), device=past.device)
        sigma_max, sigma_min = 1.0, 0.001
        noise = torch.randn_like(x0)
        psi = (1 - time) * x0 + time * future_n + ((1 - time) * sigma_max + time * sigma_min) * noise
        target = future_n - x0 + (sigma_min - sigma_max) * noise
        return F.mse_loss(self._velocity(psi, gp_mean, context[-1], time), target)

    @torch.no_grad()
    def sample(self, past, num_samples, sample_chunk=10, generator=None):
        loc, scale = history_scale(past, minimum=1.0)
        past_n = (past - loc) / scale
        output = []
        for first in range(0, num_samples, sample_chunk):
            count = min(sample_chunk, num_samples - first)
            repeated = past_n.repeat_interleave(count, dim=0)
            state, gp_mean = self.prior(repeated, generator)
            state = state + draw(state.shape, past.device, generator)
            _, context = self.encoder(repeated)
            for step in range(self.steps):
                state = state + self._velocity(state, gp_mean, context[-1], step / self.steps) / self.steps
            values = state.reshape(past.shape[0], count, self.pred_len, self.channels)
            output.append(values * scale[:, None] + loc[:, None])
        return torch.cat(output, dim=1)


class TSDiffAdapter(nn.Module):
    """Unconditional DDPM with observation self-guidance at inference."""

    def __init__(self, channels, seq_len, pred_len, hidden=64, steps=50, guidance=1.0):
        super().__init__()
        self.channels, self.seq_len, self.pred_len, self.steps = channels, seq_len, pred_len, steps
        self.guidance = guidance
        self.time_embed = nn.Embedding(steps, hidden)
        self.denoiser = TemporalNetwork(channels + hidden, channels, hidden)
        beta, alpha, alpha_bar, posterior = diffusion_schedule(steps)
        for name, value in (('beta', beta), ('alpha', alpha), ('alpha_bar', alpha_bar),
                            ('posterior', posterior)):
            self.register_buffer(name, value)

    def _predict_noise(self, noisy, time):
        emb = self.time_embed(time).unsqueeze(1).expand(-1, noisy.shape[1], -1)
        return self.denoiser(torch.cat([noisy, emb], dim=-1))

    def training_loss(self, past, future):
        loc, scale = history_scale(past)
        series = (torch.cat([past, future], dim=1) - loc) / scale
        time = torch.randint(self.steps, (past.shape[0],), device=past.device)
        noise = torch.randn_like(series)
        a = self.alpha_bar[time, None, None]
        noisy = a.sqrt() * series + (1 - a).sqrt() * noise
        return F.mse_loss(self._predict_noise(noisy, time), noise)

    def sample(self, past, num_samples, sample_chunk=10, generator=None):
        loc, scale = history_scale(past)
        output = []
        for first in range(0, num_samples, sample_chunk):
            count = min(sample_chunk, num_samples - first)
            observed = ((past - loc) / scale).repeat_interleave(count, dim=0)
            current = draw((observed.shape[0], self.seq_len + self.pred_len, self.channels),
                           past.device, generator)
            for step in reversed(range(self.steps)):
                time = torch.full((current.shape[0],), step, device=past.device, dtype=torch.long)
                with torch.no_grad():
                    eps = self._predict_noise(current, time)
                    mean = (current - self.beta[step] / (1 - self.alpha_bar[step]).sqrt() * eps) / self.alpha[step].sqrt()
                    if step:
                        mean = mean + self.posterior[step].sqrt() * draw(mean.shape, past.device, generator)
                # TSDiff observation self-guidance: score of the denoised
                # history's discrepancy, not conditioning during training.
                if self.guidance:
                    with torch.enable_grad():
                        guided = mean.detach().requires_grad_(True)
                        guided_eps = self._predict_noise(guided, time)
                        x0 = (guided - (1 - self.alpha_bar[step]).sqrt() * guided_eps) / self.alpha_bar[step].sqrt()
                        energy = (x0[:, :self.seq_len] - observed).square().mean()
                        score = -torch.autograd.grad(energy, guided)[0]
                    mean = mean + self.guidance * self.posterior[step].clamp_min(1e-4) * score.detach()
                current = mean.detach()
            values = current[:, self.seq_len:].reshape(past.shape[0], count, self.pred_len, self.channels)
            output.append(values * scale[:, None] + loc[:, None])
        return torch.cat(output, dim=1)


METHODS = {'timegrad': TimeGradAdapter, 'tsflow': TSFlowAdapter, 'tsdiff': TSDiffAdapter}
