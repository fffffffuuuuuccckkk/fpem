import torch


def environment_risk_consistency(per_sample_loss, q):
    mass = q.sum(0).clamp_min(1e-6)
    risks = (q * per_sample_loss[:, None]).sum(0) / mass
    return risks.var(unbiased=False), risks
