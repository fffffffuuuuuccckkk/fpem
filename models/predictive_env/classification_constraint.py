"""Soft environment classification with adversarial invariant features."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, value, weight):
        ctx.weight = weight
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.weight * gradient, None


def gradient_reverse(value, weight=1.0):
    return _GradientReversal.apply(value, float(weight))


def soft_cross_entropy(logits, target):
    target = target.detach()
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


class EnvironmentClassificationConstraint(nn.Module):
    """Predict q from Z_var and adversarially remove q from Z_inv."""

    def __init__(self, d_model, env_num, bottleneck=64, grl_weight=1.0):
        super().__init__()
        hidden = min(int(bottleneck), int(d_model))
        self.var_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, env_num),
        )
        self.inv_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, env_num),
        )
        self.grl_weight = float(grl_weight)

    def logits(self, z_inv, z_var, reverse_invariant=True):
        invariant_input = (
            gradient_reverse(z_inv, self.grl_weight)
            if reverse_invariant
            else z_inv
        )
        return self.inv_classifier(invariant_input), self.var_classifier(z_var)

    def forward(self, z_inv, z_var, target_q):
        target_q = target_q.detach()
        inv_logits, var_logits = self.logits(z_inv, z_var, reverse_invariant=True)
        inv_loss = soft_cross_entropy(inv_logits, target_q)
        var_loss = soft_cross_entropy(var_logits, target_q)
        return inv_loss + var_loss, {
            "inv_env_soft_ce": inv_loss,
            "var_env_soft_ce": var_loss,
            "inv_env_logits": inv_logits,
            "var_env_logits": var_logits,
        }


@torch.no_grad()
def classification_diagnostics(inv_logits, var_logits, target_q):
    target_q = target_q.detach()
    inv_probability = inv_logits.softmax(-1)
    var_probability = var_logits.softmax(-1)
    target_label = target_q.argmax(-1)
    return {
        "inv_env_soft_ce": soft_cross_entropy(inv_logits, target_q),
        "var_env_soft_ce": soft_cross_entropy(var_logits, target_q),
        "inv_env_accuracy_argmax": (inv_logits.argmax(-1) == target_label).float().mean(),
        "var_env_accuracy_argmax": (var_logits.argmax(-1) == target_label).float().mean(),
        "inv_env_entropy": -(
            inv_probability * inv_probability.clamp_min(1e-8).log()
        ).sum(-1).mean(),
        "var_env_entropy": -(
            var_probability * var_probability.clamp_min(1e-8).log()
        ).sum(-1).mean(),
    }
