"""Plug-in conditional probability heads sharing one FPEM encoder pass.

All train targets are detached from Yinv. Sampling never accepts Y_true.
The legacy full IAPAD head is deliberately kept as a separate comparator.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

from .affine_target_projection import decode_affine, project_affine_targets
from .probabilistic_affine_dynamics import (
    JointAffineVelocity, ProbabilisticAffineDynamics, TemporalFiLMBlock,
)


HEAD_NAMES = (
    "affine_deterministic", "affine_gaussian", "affine_mdn", "affine_flow",
    "residual_gaussian", "residual_lowrank", "residual_flow", "legacy_full",
)


def patchify(values, patch_len):
    """[B,C,L] -> padded [B,C,M,P] and a valid-point mask."""
    length = values.shape[-1]
    padded = math.ceil(length / patch_len) * patch_len
    result = F.pad(values, (0, padded - length)).reshape(
        *values.shape[:2], padded // patch_len, patch_len
    )
    mask = values.new_zeros(padded)
    mask[:length] = 1
    return result, mask.reshape(1, 1, padded // patch_len, patch_len)


def _mlp(dim, output, hidden=64):
    net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden),
                        nn.GELU(), nn.Linear(hidden, output))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class FutureConditionEncoder(nn.Module):
    """A common Future-Zvar/Yinv patch condition for all non-legacy heads."""

    def __init__(self, dim, patch_len=16, mode="full_shape"):
        super().__init__()
        if mode not in ("full_shape", "three_stats"):
            raise ValueError(f"unknown condition mode: {mode}")
        self.mode = mode
        self.patch_len = patch_len
        shape_dim = patch_len + 1 if mode == "full_shape" else 3
        self.shape_encoder = _mlp(shape_dim, dim, max(32, dim // 2))
        self.combine = nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(),
        )

    def forward(self, future_zvar, y_inv, input_scale):
        prediction = y_inv.detach().transpose(1, 2)
        scale = input_scale.detach().transpose(1, 2).clamp_min(1e-4)
        patches, valid = patchify(prediction, self.patch_len)
        counts = valid.sum(-1).clamp_min(1)
        mean = (patches * valid).sum(-1, keepdim=True) / counts[..., None]
        if self.mode == "full_shape":
            centered_shape = ((patches - mean) / scale[..., None]) * valid
            level = mean[..., 0] / scale[..., 0][..., None]
            shape = torch.cat((centered_shape, level[..., None]), dim=-1)
        else:
            variance = (((patches - mean) * valid).square().sum(-1)
                        / counts).sqrt()
            last_index = (counts.long() - 1)[..., None].expand(
                *patches.shape[:-1], 1
            )
            slope = patches.gather(-1, last_index)[..., 0] - patches[..., 0]
            divisor = scale[..., 0][..., None]
            shape = torch.stack((mean[..., 0] / divisor,
                                 variance / divisor,
                                 slope / divisor), dim=-1)
        if future_zvar.shape[:3] != shape.shape[:3]:
            raise ValueError("Future-Zvar and Yinv patch shapes disagree")
        return self.combine(torch.cat((future_zvar, self.shape_encoder(shape)), -1))


class ProbabilityHead(nn.Module):
    is_affine = False

    def __init__(self, dim, pred_len, patch_len=16):
        super().__init__()
        self.dim, self.pred_len, self.patch_len = dim, pred_len, patch_len
        self.patches = math.ceil(pred_len / patch_len)
        self.register_buffer("theta_scale", torch.ones(2))

    def set_theta_scale(self, scale):
        self.theta_scale.copy_(torch.as_tensor(scale, device=self.theta_scale.device,
                                               dtype=self.theta_scale.dtype))

    def affine_target(self, y_inv, y_true, scale):
        target, center = project_affine_targets(
            y_true, y_inv.detach(), scale.detach(), self.patch_len
        )
        return (target / self.theta_scale.view(1, 1, 1, 2)).clamp(-8, 8).detach(), center

    def residual_target(self, y_inv, y_true, scale):
        return ((y_true - y_inv.detach()) / scale.detach()).transpose(1, 2)

    def decode_affine(self, theta, y_inv, scale):
        _, center = project_affine_targets(
            y_inv.detach(), y_inv.detach(), scale.detach(), self.patch_len
        )
        theta = theta * self.theta_scale.view(1, 1, 1, 1, 2)
        return decode_affine(y_inv, scale, theta, center, self.patch_len), theta

    def decode_residual(self, residual, y_inv, scale):
        residual = residual[..., :self.pred_len]
        return y_inv[:, None] + scale[:, None] * residual.permute(0, 1, 3, 2)

    def training_loss(self, condition, y_inv, y_true, scale):
        raise NotImplementedError

    def sample(self, condition, y_inv, scale, num_samples, generator,
               chunk_size=10, steps=12):
        raise NotImplementedError


class DeterministicAffineHead(ProbabilityHead):
    is_affine = True

    def __init__(self, dim, pred_len, patch_len=16):
        super().__init__(dim, pred_len, patch_len)
        self.net = _mlp(dim, 2)

    def training_loss(self, condition, y_inv, y_true, scale):
        target, _ = self.affine_target(y_inv, y_true, scale)
        loss = F.smooth_l1_loss(self.net(condition), target)
        return loss, {"center_huber": loss.detach()}

    def sample(self, condition, y_inv, scale, num_samples, generator,
               chunk_size=10, steps=12):
        theta = self.net(condition)[:, None].expand(-1, num_samples, -1, -1, -1)
        paths, physical = self.decode_affine(theta, y_inv, scale)
        return {"y_samples": paths, "gamma_samples": physical[..., 0],
                "beta_samples": physical[..., 1], "residual_samples": None}


class GaussianAffineHead(ProbabilityHead):
    is_affine = True

    def __init__(self, dim, pred_len, patch_len=16):
        super().__init__(dim, pred_len, patch_len)
        self.net = _mlp(dim, 4)

    def distribution(self, condition):
        mu, log_sigma = self.net(condition).chunk(2, -1)
        return mu, log_sigma.clamp(-6, 3)

    def training_loss(self, condition, y_inv, y_true, scale):
        target, _ = self.affine_target(y_inv, y_true, scale)
        mu, log_sigma = self.distribution(condition)
        nll = (log_sigma + 0.5 * ((target - mu) / log_sigma.exp()).square()
               + 0.5 * math.log(2 * math.pi)).mean()
        return nll, {"gaussian_nll": nll.detach(),
                     "center_huber": F.smooth_l1_loss(mu, target).detach()}

    def sample(self, condition, y_inv, scale, num_samples, generator,
               chunk_size=10, steps=12):
        mu, log_sigma = self.distribution(condition)
        noise = torch.randn((*mu.shape[:1], num_samples, *mu.shape[1:]),
                            device=mu.device, dtype=mu.dtype, generator=generator)
        paths, theta = self.decode_affine(
            mu[:, None] + log_sigma.exp()[:, None] * noise, y_inv, scale
        )
        return {"y_samples": paths, "gamma_samples": theta[..., 0],
                "beta_samples": theta[..., 1], "residual_samples": None}


class MixtureAffineHead(ProbabilityHead):
    is_affine = True

    def __init__(self, dim, pred_len, patch_len=16, components=3):
        super().__init__(dim, pred_len, patch_len)
        self.components = components
        self.net = _mlp(dim, components * 5)

    def distribution(self, condition):
        values = self.net(condition).reshape(*condition.shape[:-1], self.components, 5)
        return (values[..., 0].log_softmax(-1), values[..., 1:3],
                values[..., 3:5].clamp(-6, 3))

    def training_loss(self, condition, y_inv, y_true, scale):
        target, _ = self.affine_target(y_inv, y_true, scale)
        log_pi, mu, log_sigma = self.distribution(condition)
        log_component = (-0.5 * ((target[..., None, :] - mu) /
                                 log_sigma.exp()).square() - log_sigma
                         - 0.5 * math.log(2 * math.pi)).sum(-1)
        nll = -torch.logsumexp(log_pi + log_component, dim=-1).mean()
        return nll, {"mixture_nll": nll.detach()}

    def sample(self, condition, y_inv, scale, num_samples, generator,
               chunk_size=10, steps=12):
        log_pi, mu, log_sigma = self.distribution(condition)
        b, c, m, k = log_pi.shape
        indices = torch.multinomial(log_pi.exp().reshape(-1, k), num_samples,
                                    replacement=True, generator=generator)
        indices = indices.reshape(b, c, m, num_samples).permute(0, 3, 1, 2)
        mu = mu[:, None].expand(-1, num_samples, -1, -1, -1, -1)
        sigma = log_sigma.exp()[:, None].expand_as(mu)
        select = indices[..., None, None].expand(-1, -1, -1, -1, 1, 2)
        center = mu.gather(4, select).squeeze(4)
        deviation = sigma.gather(4, select).squeeze(4)
        noise = torch.randn(center.shape, device=center.device,
                            dtype=center.dtype, generator=generator)
        paths, theta = self.decode_affine(center + deviation * noise, y_inv, scale)
        return {"y_samples": paths, "gamma_samples": theta[..., 0],
                "beta_samples": theta[..., 1], "residual_samples": None}


class FlowAffineHead(ProbabilityHead):
    is_affine = True

    def __init__(self, dim, pred_len, patch_len=16):
        super().__init__(dim, pred_len, patch_len)
        self.center = _mlp(dim, 2)
        self.velocity = JointAffineVelocity(dim)

    def _velocity(self, theta, time, condition):
        zeros = condition.new_zeros(*condition.shape[:-1], 3)
        return self.velocity(theta, time, condition, zeros)

    def training_loss(self, condition, y_inv, y_true, scale):
        target, _ = self.affine_target(y_inv, y_true, scale)
        center = self.center(condition)
        center_loss = F.smooth_l1_loss(center, target)
        theta0 = center.detach() + torch.randn_like(center)
        tau = torch.rand((center.shape[0], center.shape[1], 1, 1),
                         device=center.device, dtype=center.dtype)
        theta_t = (1 - tau) * theta0 + tau * target
        velocity = self._velocity(theta_t[:, None], tau[:, None], condition)[:, 0]
        fm = F.mse_loss(velocity, (target - theta0).detach())
        return center_loss + fm, {"center_huber": center_loss.detach(),
                                  "flow_matching": fm.detach()}

    def sample(self, condition, y_inv, scale, num_samples, generator,
               chunk_size=10, steps=12):
        center = self.center(condition)
        paths, thetas = [], []
        for start in range(0, num_samples, chunk_size):
            n = min(chunk_size, num_samples - start)
            noise = torch.randn((center.shape[0], n, *center.shape[1:]),
                                device=center.device, dtype=center.dtype,
                                generator=generator)
            theta = center[:, None] + noise
            for step in range(steps):
                theta = theta + self._velocity(theta, (step + 0.5) / steps,
                                               condition) / steps
            decoded, physical = self.decode_affine(theta, y_inv, scale)
            paths.append(decoded)
            thetas.append(physical)
        joined = torch.cat(thetas, 1)
        return {"y_samples": torch.cat(paths, 1),
                "gamma_samples": joined[..., 0], "beta_samples": joined[..., 1],
                "residual_samples": None}


class IndependentGaussianResidualHead(ProbabilityHead):
    def __init__(self, dim, pred_len, patch_len=16):
        super().__init__(dim, pred_len, patch_len)
        self.net = _mlp(dim, 2 * patch_len)

    def distribution(self, condition):
        mu, log_sigma = self.net(condition).chunk(2, -1)
        return mu, log_sigma.clamp(-6, 3)

    def training_loss(self, condition, y_inv, y_true, scale):
        target, mask = patchify(self.residual_target(y_inv, y_true, scale),
                                self.patch_len)
        mu, log_sigma = self.distribution(condition)
        nll = log_sigma + 0.5 * ((target - mu) / log_sigma.exp()).square()
        loss = (nll * mask).sum() / (mask.sum() * target.shape[0] * target.shape[1])
        return loss, {"independent_gaussian_nll": loss.detach()}

    def sample(self, condition, y_inv, scale, num_samples, generator,
               chunk_size=10, steps=12):
        mu, log_sigma = self.distribution(condition)
        noise = torch.randn((mu.shape[0], num_samples, *mu.shape[1:]),
                            device=mu.device, dtype=mu.dtype, generator=generator)
        residual = mu[:, None] + log_sigma.exp()[:, None] * noise
        residual = residual.flatten(-2)
        return {"y_samples": self.decode_residual(residual, y_inv, scale),
                "gamma_samples": None, "beta_samples": None,
                "residual_samples": residual[..., :self.pred_len]}


class LowRankGaussianResidualHead(ProbabilityHead):
    def __init__(self, dim, pred_len, patch_len=16, rank=4):
        super().__init__(dim, pred_len, patch_len)
        self.rank = rank
        self.net = _mlp(dim, patch_len * (2 + rank))

    def distribution(self, condition):
        raw = self.net(condition)
        p, r = self.patch_len, self.rank
        mu = raw[..., :p]
        log_sigma = raw[..., p:2 * p].clamp(-6, 3)
        factors = raw[..., 2 * p:].reshape(*raw.shape[:-1], p, r) * 0.1
        return mu, log_sigma, factors

    def training_loss(self, condition, y_inv, y_true, scale):
        target, _ = patchify(self.residual_target(y_inv, y_true, scale),
                             self.patch_len)
        mu, log_sigma, factors = self.distribution(condition)
        def group_nll(patch_slice, width):
            diff = (target[:, :, patch_slice, :width]
                    - mu[:, :, patch_slice, :width])
            log_s = log_sigma[:, :, patch_slice, :width]
            u = factors[:, :, patch_slice, :width, :]
            inv_diag = torch.exp(-2 * log_s)
            weighted_u = u * inv_diag[..., None]
            eye = torch.eye(self.rank, device=u.device, dtype=u.dtype)
            inner = eye + torch.einsum("bcmpr,bcmps->bcmrs", u, weighted_u)
            chol = torch.linalg.cholesky(inner)
            projected = torch.einsum("bcmpr,bcmp->bcmr", weighted_u, diff)
            correction = torch.cholesky_solve(projected[..., None], chol)[..., 0]
            quad = (diff.square() * inv_diag).sum(-1) - (
                projected * correction).sum(-1)
            logdet = 2 * log_s.sum(-1) + 2 * torch.log(
                torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
            return 0.5 * (quad + logdet + width * math.log(2 * math.pi))

        full_count = self.pred_len // self.patch_len
        parts = []
        if full_count:
            parts.append(group_nll(slice(0, full_count), self.patch_len))
        if self.pred_len % self.patch_len:
            parts.append(group_nll(slice(full_count, full_count + 1),
                                   self.pred_len % self.patch_len))
        loss = torch.cat(parts, dim=2).mean()
        return loss, {"lowrank_gaussian_nll": loss.detach()}

    def sample(self, condition, y_inv, scale, num_samples, generator,
               chunk_size=10, steps=12):
        mu, log_sigma, u = self.distribution(condition)
        shape = (mu.shape[0], num_samples, *mu.shape[1:])
        diagonal = torch.randn(shape, device=mu.device, dtype=mu.dtype,
                               generator=generator)
        rank_noise = torch.randn((*shape[:-1], self.rank), device=mu.device,
                                 dtype=mu.dtype, generator=generator)
        correlated = (u[:, None] * rank_noise[..., None, :]).sum(-1)
        residual = (mu[:, None] + log_sigma.exp()[:, None] * diagonal
                    + correlated).flatten(-2)
        return {"y_samples": self.decode_residual(residual, y_inv, scale),
                "gamma_samples": None, "beta_samples": None,
                "residual_samples": residual[..., :self.pred_len]}


class JointResidualVelocity(nn.Module):
    """Temporal Conv/FiLM couples all future residual points and patches."""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.input = nn.Conv1d(1, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([
            TemporalFiLMBlock(hidden, dim + 1),
            TemporalFiLMBlock(hidden, dim + 1),
        ])
        self.output = nn.Conv1d(hidden, 1, 3, padding=1)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, residual, time, condition, patch_len):
        b, s, c, length = residual.shape
        local = condition.repeat_interleave(patch_len, 2)[..., :length, :]
        local = local[:, None].expand(-1, s, -1, -1, -1)
        time = torch.as_tensor(time, device=residual.device, dtype=residual.dtype)
        time = time.expand(b, s, c, length, 1)
        local = torch.cat((local, time), dim=-1).reshape(b * s * c, length, -1)
        value = self.input(residual.reshape(b * s * c, 1, length))
        for block in self.blocks:
            value = block(value, local)
        return self.output(value).reshape(b, s, c, length)


class FlowResidualHead(ProbabilityHead):
    def __init__(self, dim, pred_len, patch_len=16):
        super().__init__(dim, pred_len, patch_len)
        self.velocity = JointResidualVelocity(dim)

    def training_loss(self, condition, y_inv, y_true, scale):
        target = self.residual_target(y_inv, y_true, scale)
        initial = torch.randn_like(target)
        tau = torch.rand((target.shape[0], 1, 1), device=target.device,
                         dtype=target.dtype)
        mixed = (1 - tau) * initial + tau * target
        velocity = self.velocity(mixed[:, None], tau[:, None, None], condition,
                                 self.patch_len)[:, 0]
        loss = F.mse_loss(velocity, (target - initial).detach())
        return loss, {"residual_flow_matching": loss.detach()}

    def sample(self, condition, y_inv, scale, num_samples, generator,
               chunk_size=10, steps=12):
        b, c = condition.shape[:2]
        all_paths, all_residuals = [], []
        for start in range(0, num_samples, chunk_size):
            n = min(chunk_size, num_samples - start)
            residual = torch.randn((b, n, c, self.pred_len), device=y_inv.device,
                                   dtype=y_inv.dtype, generator=generator)
            for step in range(steps):
                residual = residual + self.velocity(
                    residual, (step + 0.5) / steps, condition,
                    self.patch_len,
                ) / steps
            all_paths.append(self.decode_residual(residual, y_inv, scale))
            all_residuals.append(residual)
        return {"y_samples": torch.cat(all_paths, 1),
                "gamma_samples": None, "beta_samples": None,
                "residual_samples": torch.cat(all_residuals, 1)}


class LegacyFullHead(ProbabilityHead):
    is_affine = True

    def __init__(self, dim, pred_len, patch_len=16):
        super().__init__(dim, pred_len, patch_len)
        self.dynamics = ProbabilisticAffineDynamics(dim, pred_len, patch_len)

    def set_theta_scale(self, scale):
        super().set_theta_scale(scale)
        self.dynamics.set_theta_scale(scale)

    def training_loss(self, condition, y_inv, y_true, scale):
        future = condition
        old_condition = self.dynamics.condition(future, y_inv, scale)
        aux = self.dynamics.supervised_losses(
            old_condition, y_true, y_inv, scale, use_innovation=True,
        )
        loss = (0.1 * aux["center"] + 0.1 * aux["flow"]
                + 0.05 * aux["innovation"] + 0.001 * aux["smooth"])
        return loss, {key: value.detach() for key, value in aux.items()}

    def sample(self, condition, y_inv, scale, num_samples, generator,
               chunk_size=10, steps=12):
        old_condition = self.dynamics.condition(condition, y_inv, scale)
        result = self.dynamics.sample(old_condition, y_inv, scale, num_samples,
                                      steps, "flow", True, chunk_size, generator)
        result["residual_samples"] = None
        return result


class ProbabilityHeadBank(nn.Module):
    """One condition encoder and a ModuleDict of independently named heads."""

    def __init__(self, dim, pred_len, patch_len=16, condition_mode="full_shape",
                 heads=HEAD_NAMES):
        super().__init__()
        self.condition_encoder = FutureConditionEncoder(dim, patch_len,
                                                          condition_mode)
        classes = {
            "affine_deterministic": DeterministicAffineHead,
            "affine_gaussian": GaussianAffineHead,
            "affine_mdn": MixtureAffineHead,
            "affine_flow": FlowAffineHead,
            "residual_gaussian": IndependentGaussianResidualHead,
            "residual_lowrank": LowRankGaussianResidualHead,
            "residual_flow": FlowResidualHead,
            "legacy_full": LegacyFullHead,
        }
        if len(set(heads)) != len(heads) or set(heads) - set(classes):
            raise ValueError("probability heads must be distinct supported names")
        self.heads = nn.ModuleDict({name: classes[name](dim, pred_len, patch_len)
                                    for name in heads})

    def set_theta_scale(self, scale):
        for head in self.heads.values():
            head.set_theta_scale(scale)

    def condition(self, future, y_inv, scale):
        return self.condition_encoder(future, y_inv, scale)

    def training_losses(self, condition, future, y_inv, y_true, scale):
        output = {}
        for name, head in self.heads.items():
            signal = future if name == "legacy_full" else condition
            output[name] = head.training_loss(signal, y_inv, y_true, scale)
        return output

    def sample_head(self, name, condition, future, y_inv, scale, num_samples,
                    generator, chunk_size=10, steps=12):
        head = self.heads[name]
        signal = future if name == "legacy_full" else condition
        result = head.sample(signal, y_inv, scale, num_samples, generator,
                             chunk_size, steps)
        samples = result["y_samples"]
        result.update({
            "y_inv": y_inv, "y_mean": samples.mean(1),
            "y_median": samples.median(1).values,
            "lower_95": torch.quantile(samples, 0.025, dim=1),
            "upper_95": torch.quantile(samples, 0.975, dim=1),
        })
        return result
