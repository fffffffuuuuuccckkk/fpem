"""Two-loss Adam updates for S4, inspired by AdaTask (not a reproduction).

Each loss owns its own moments and step counter. Directions are computed at
the same parameter value and applied in one update. The original S4 pilots
use Adam with zero weight decay, so that is the default here as well.
"""

from __future__ import annotations

import math
import torch


class LosswiseAdam:
    def __init__(self, groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.groups = list(groups)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}
        seen = set()
        for group in self.groups:
            for p in group["params"]:
                if id(p) in seen:
                    raise ValueError("parameter belongs to multiple loss-wise groups")
                seen.add(id(p))

    @staticmethod
    def _norm(tensors):
        return math.sqrt(sum(float(t.detach().float().square().sum()) for t in tensors))

    @torch.no_grad()
    def step(self, prediction_grads, auxiliary_grads, prediction_multiplier=1.0,
             auxiliary_multiplier=1.0):
        if len(prediction_grads) != len(auxiliary_grads):
            raise ValueError("gradient lists differ in length")
        params = [p for group in self.groups for p in group["params"]]
        if len(params) != len(prediction_grads):
            raise ValueError("gradient list does not match parameters")
        norms = {"prediction_update_norm": 0.0, "auxiliary_update_norm": 0.0,
                 "parameter_update_norm": 0.0}
        index = 0
        for group in self.groups:
            base_lr = group["lr"]
            for p in group["params"]:
                gp, ga = prediction_grads[index], auxiliary_grads[index]
                index += 1
                if gp is None and ga is None:
                    continue
                state = self.state.setdefault(p, {})
                updates = []
                for key, grad, lr in (("prediction", gp, base_lr * prediction_multiplier),
                                      ("auxiliary", ga, base_lr * auxiliary_multiplier)):
                    if grad is None:
                        continue
                    if not torch.isfinite(grad).all():
                        raise FloatingPointError("nonfinite loss-wise gradient")
                    branch = state.setdefault(key, {"step": 0,
                                                     "m": torch.zeros_like(p),
                                                     "v": torch.zeros_like(p)})
                    branch["step"] += 1
                    beta1, beta2 = self.betas
                    branch["m"].mul_(beta1).add_(grad, alpha=1 - beta1)
                    branch["v"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m = branch["m"] / (1 - beta1 ** branch["step"])
                    v = branch["v"] / (1 - beta2 ** branch["step"])
                    update = lr * m / (v.sqrt() + self.eps)
                    updates.append(update)
                    norms[key + "_update_norm"] += float(update.float().square().sum())
                total = sum(updates)
                # Decoupled decay once, not once per loss. Baseline has wd=0.
                if self.weight_decay:
                    total = total + base_lr * self.weight_decay * p
                p.add_(-total)
                norms["parameter_update_norm"] += float(total.float().square().sum())
        return {key: math.sqrt(value) for key, value in norms.items()}

    def state_dict(self):
        params = [p for group in self.groups for p in group["params"]]
        return {"state": [self.state.get(p, {}) for p in params],
                "groups": [{"name": g["name"], "lr": g["lr"]} for g in self.groups],
                "betas": self.betas, "eps": self.eps, "weight_decay": self.weight_decay}
