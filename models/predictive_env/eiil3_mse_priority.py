"""MSE-first optimization of an A0-initialized linear forecasting head.

The Soft-IRM term is the existing analytic full-head gradient penalty. Only
its gradient is gated/projected; the prediction gradient is never modified.
"""
import torch

from models.predictive_env.eiil3_modules import soft_risks_and_full_head_penalty


def priority_step(head, optimizer, design, truth, ratio, q, mode, lam=1e-4,
                  rho=None, eps=1e-12):
    if mode not in {"V0", "V1", "V2", "V3", "V4", "V5"}:
        raise ValueError(mode)
    risks, penalty, _, prediction = soft_risks_and_full_head_penalty(
        head, design, truth, ratio, q)
    mse = (prediction - truth).square().mean()
    params = tuple(head.parameters())
    optimizer.zero_grad(set_to_none=True)
    gm = torch.autograd.grad(mse, params, retain_graph=True)
    gs = torch.autograd.grad(penalty, params, retain_graph=True)
    with torch.no_grad():
        dot = sum((a * b).sum() for a, b in zip(gm, gs))
        m2 = sum(a.square().sum() for a in gm)
        s2 = sum(a.square().sum() for a in gs)
        cosine = dot / (m2.sqrt() * s2.sqrt() + eps)
        conflict = bool(dot.item() < 0)
        if mode == "V2" and conflict:
            auxiliary = tuple(torch.zeros_like(g) for g in gs)
        elif mode in {"V3", "V4"} and conflict and m2.item() > eps:
            auxiliary = tuple(g - (dot / (m2 + eps)) * m for m, g in zip(gm, gs))
        else:
            auxiliary = tuple(g.clone() for g in gs)
        if mode in {"V2", "V3", "V4"} and m2.item() <= eps:
            auxiliary = tuple(torch.zeros_like(g) for g in gs)
        aux2 = sum(g.square().sum() for g in auxiliary)
        scale = 1.0
        if mode == "V4":
            if rho is None or rho <= 0:
                raise ValueError("V4 requires a positive rho")
            scale = min(1.0, float((rho * m2.sqrt() / (lam * aux2.sqrt() + eps)).item()))
        effective_aux_ratio = float((lam * scale * aux2.sqrt() / (m2.sqrt() + eps)).item())
    if mode in {"V0", "V1", "V5"}:
        objective = (risks.mean() if mode == "V0" else mse)
        if mode != "V5":
            objective = objective + lam * penalty
        objective.backward()
    else:
        for parameter, main, auxiliary_gradient in zip(params, gm, auxiliary):
            parameter.grad = main.detach() + lam * scale * auxiliary_gradient.detach()
    optimizer.step()
    return {"mse": float(mse.detach()), "env_risk": float(risks.mean().detach()),
            "penalty": float(penalty.detach()), "gradient_cosine": float(cosine),
            "gradient_conflict": conflict, "main_norm": float(m2.sqrt()),
            "aux_norm": float(s2.sqrt()), "effective_aux_ratio": effective_aux_ratio}
