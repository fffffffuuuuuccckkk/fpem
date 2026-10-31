"""Split-conformal simultaneous trajectory bands and Gaussian CRPS diagnostic.

Exchangeability is required for distribution-free target coverage; arbitrary
future environment drift does not preserve that guarantee.
"""
from __future__ import annotations

import math
import torch


class ConformalCalibrator:
    def __init__(self, alpha: float = .05):
        self.alpha = alpha
        self.quantile = None

    @torch.no_grad()
    def fit(self, prediction: torch.Tensor, truth: torch.Tensor,
            scale: torch.Tensor):
        if prediction.shape != truth.shape or scale.shape != prediction.shape[:1]+prediction.shape[2:]:
            raise ValueError("prediction [N,T,C], scale [N,C] expected")
        score = ((truth-prediction).abs()/scale[:,None].clamp_min(1e-8)).amax(dim=1).reshape(-1)
        n = score.numel()
        rank = min(n, math.ceil((n+1)*(1-self.alpha)))
        self.quantile = float(score.kthvalue(rank).values)
        return {"calibration_entries": n, "simultaneous_q": self.quantile,
                "alpha": self.alpha}

    @torch.no_grad()
    def interval(self, prediction, scale):
        if self.quantile is None:
            raise RuntimeError("fit on disjoint calibration data first")
        half = self.quantile*scale[:,None]
        return prediction-half,prediction+half


@torch.no_grad()
def interval_metrics(prediction, truth, scale, lower, upper,
                     alpha: float = .05) -> dict:
    error = truth-prediction
    std = scale[:,None].clamp_min(1e-6)
    z = error/std
    phi = torch.exp(-.5*z.square()) / math.sqrt(2*math.pi)
    cdf = .5*(1.+torch.erf(z/math.sqrt(2.)))
    crps = std*(z*(2*cdf-1)+2*phi-1/math.sqrt(math.pi))
    inside = (truth>=lower)&(truth<=upper)
    trajectory = inside.all(dim=1)
    width = upper-lower
    interval_score = width + 2/alpha*(lower-truth).clamp_min(0) + 2/alpha*(truth-upper).clamp_min(0)
    # Weighted interval score using one central 95% interval and its median.
    wis = (.5*error.abs() + alpha/2*interval_score) / 1.5
    return {"mse":float(error.square().mean()), "mae":float(error.abs().mean()),
            "gaussian_crps":float(crps.mean()),
            "picp95_marginal":float(inside.float().mean()),
            "picp95_trajectory":float(trajectory.float().mean()),
            "mpiw95":float(width.mean()), "interval_score95":float(interval_score.mean()),
            "wis_95_single_interval":float(wis.mean()),
            "scale_mean":float(scale.mean())}
