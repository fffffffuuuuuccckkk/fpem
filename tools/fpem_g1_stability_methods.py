"""Small TRAIN-only G1 estimators.  No method reads a query target in fit/predict.

The registry deliberately distinguishes implemented estimators from mathematical
duplicates and methods that need a separate, unimplemented protocol.  In
particular, these estimators are *candidates for stable shared prediction*, not
proofs of environment invariance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


EPS = 1e-8
IMPLEMENTED = {
    "RIDGE", "ZERO", "U01", "U02", "U03", "U04", "U05", "U06",
    "U07", "U10", "U11", "U12", "U16", "U17", "U18", "U19",
    "U20", "U21", "U23", "U24", "U25", "U26", "U27", "U28",
    "U29", "U30", "U31", "U35", "U37", "U38", "G07", "G08",
    "G09", "G12",
}
DUPLICATES = {
    "U09": "lower-tail gain is the same empirical tail set as U06 CVaR(-gain)",
    "U13": "the selected L2 Wasserstein bound reduces to a reparameterized Ridge penalty",
    "U14": "entropy-constrained adversarial weights equal the KL-DRO dual U11",
    "U15": "entropic negative-gain adversary has the same dual weighting as U06",
    "U22": "random-subset agreement with a linear map reduces to bootstrap shrinkage U18",
    "U32": "isotropic key-noise consistency is exactly an L2 coefficient penalty",
    "U33": "selected anisotropic quadratic perturbation is the U16 covariance penalty",
    "U36": "the linear G1 Jacobian Frobenius penalty is ordinary Ridge",
}
DEFERRED = {
    "U08": "nonconvex gain-variance objective needs a separate convergence/stability audit",
    "U34": "label-preserving raw X/A0 augmentation cannot be formed from cached keys alone",
    "U39": "SAM is only relevant after selecting a nonclosed candidate",
    "U40": "OOF safe shrinkage is implemented as a selection layer in the runner",
    "U41": "block-bootstrap lower-confidence shrinkage is implemented in the runner",
    "U42": "horizon-wise OOF shrinkage is implemented in the runner",
    "U43": "history-only disagreement gate is evaluated in U19",
    "U44": "stable-subspace plus safe scale is evaluated by the combination runner",
    "U45": "OOF convex stacking is evaluated by the combination runner",
    "G01": "chronological proxy labels are constructed for G07-G12",
    "G02": "history-key proxy labels are constructed for G07-G12",
    "G03": "dynamics proxy labels are constructed for G07-G12",
    "G04": "TRAIN residual-pattern proxy labels are constructed for G07-G12",
    "G05": "cross-fitted EIIL assignment requires a separate validated implementation",
    "G06": "soft adversarial assignment overlaps U11; distinct multi-group objective deferred",
    "G10": "group gradient agreement needs explicit nontriviality audit",
    "G11": "linear fixed-feature IRM penalty may be trivial; no invariant claim made",
    "G13": "forward-transfer agreement is measured in runner OOF analysis",
    "G14": "two-view stability needs valid G01 and G02 winners first",
}


def registry_rows() -> list[dict[str, Any]]:
    rows = []
    for name in ["ZERO", "RIDGE"] + [f"U{i:02d}" for i in range(1, 46)] + [f"G{i:02d}" for i in range(1, 15)]:
        if name in IMPLEMENTED:
            status, reason = "implemented", ""
        elif name in DUPLICATES:
            status, reason = "mathematical_duplicate", DUPLICATES[name]
        else:
            status, reason = "deferred", DEFERRED.get(name, "requires prerequisite pilot evidence")
        rows.append({"method_id": name, "family": "LATENT-GROUP" if name.startswith("G") else "STRICT-GROUP-FREE",
                     "status": status, "reason": reason})
    return rows


def design(keys: torch.Tensor) -> torch.Tensor:
    x = keys.reshape(-1, keys.shape[-1]).double()
    return torch.cat([x, x.new_ones(len(x), 1)], dim=1)


def _weighted_ridge(x, y, ridge, weights=None, penalty=None):
    if weights is None:
        weights = x.new_ones(len(x))
    weights = weights.double().clamp_min(0)
    weights = weights / weights.mean().clamp_min(EPS)
    d = x.shape[1]
    reg = torch.eye(d, dtype=x.dtype) * float(ridge) if penalty is None else penalty.double() * float(ridge)
    reg[-1, -1] = float(ridge) * .01
    gram = x.T @ (weights[:, None] * x) / len(x)
    rhs = x.T @ (weights[:, None] * y) / len(x)
    return torch.linalg.solve(gram + reg + 1e-9 * torch.eye(d, dtype=x.dtype), rhs)


def _sample_loss(pred, target, n, c):
    return (pred - target).square().reshape(n, c, -1).mean((1, 2))


def _gain(pred, target, n, c):
    return (target.square() - (target - pred).square()).reshape(n, c, -1).mean((1, 2))


def _kmeans(x, count, seed=2021, steps=12):
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(x), generator=gen)[:count]
    centers = x[indices].clone()
    for _ in range(steps):
        labels = torch.cdist(x, centers).argmin(1)
        for k in range(count):
            mask = labels == k
            if mask.any():
                centers[k] = x[mask].mean(0)
    return torch.cdist(x, centers).argmin(1)


def proxy_groups(keys, residual, metadata, kind, count=4):
    n = len(keys)
    if kind == "time":
        starts = metadata["starts"].double()
        ranks = starts.argsort().argsort()
        return (ranks * count // n).clamp_max(count-1).long()
    if kind == "key":
        return _kmeans(keys.double().mean(1), count)
    if kind == "dynamics":
        k = keys.double()
        features = torch.stack([k.std((1, 2)), (k[..., 1:] - k[..., :-1]).abs().mean((1, 2)),
                                (k[..., 1:] * k[..., :-1]).mean((1, 2))], dim=1)
        return _kmeans(features, count)
    if kind == "residual":
        r = residual.double()
        features = torch.stack([r.mean((1, 2)), r.std((1, 2)), r.abs().mean((1, 2))], dim=1)
        return _kmeans(features, count)
    raise ValueError(kind)


@dataclass
class G1Correction:
    weights: torch.Tensor | None
    method: str
    extras: dict[str, Any]

    def predict(self, query_keys: torch.Tensor) -> torch.Tensor:
        b, c = query_keys.shape[:2]
        if self.method == "ZERO":
            return query_keys.new_zeros(b, c, self.extras["horizon"])
        x = design(query_keys)
        if self.method == "U31":
            projection = self.extras["projection"].double()
            phase = self.extras["phase"].double()
            z = math.sqrt(2. / projection.shape[1]) * torch.cos(x[:, :-1] @ projection + phase)
            x = torch.cat([z, z.new_ones(len(z), 1)], dim=1)
        pred = x @ self.weights.double()
        if self.method in {"U19"}:
            members = self.extras["members"]
            stack = torch.stack([x @ w.double() for w in members])
            spread = stack.std(0, unbiased=False).mean(1)
            scale = float(self.extras["spread_scale"])
            pred = pred * (1. / (1. + spread / max(scale, EPS)))[:, None]
        if self.method in {"U20"}:
            basis = self.extras["stable_basis"].double()
            pred = (pred @ basis) @ basis.T
        return pred.reshape(b, c, -1).float()

    def export(self):
        return {"method": self.method, "weights": self.weights, "extras": self.extras}

    @classmethod
    def load(cls, state):
        return cls(state["weights"], state["method"], state["extras"])


def _bootstrap(x, y, ridge, n, c, repeats=8, seed=2021, fraction=1.):
    gen = torch.Generator().manual_seed(seed)
    members = []
    for _ in range(repeats):
        rows = torch.randint(n, (max(8, int(n*fraction)),), generator=gen) if fraction == 1. else torch.randperm(n, generator=gen)[:max(8, int(n*fraction))]
        flat = (rows[:, None]*c + torch.arange(c)[None]).reshape(-1)
        members.append(_weighted_ridge(x[flat], y[flat], ridge))
    return members


def _iterative_weights(method, x, y, n, c, ridge, config, groups=None):
    w = _weighted_ridge(x, y, ridge)
    gamma = float(config.get("gamma", .2))
    temperature = float(config.get("temperature", .1))
    for _ in range(int(config.get("irls_steps", 5))):
        p = x @ w
        loss = _sample_loss(p, y, n, c)
        gain = _gain(p, y, n, c)
        if method == "U01":
            radius = loss.clamp_min(EPS).sqrt()
            delta = float(config.get("delta", .5))
            weight = (delta / radius).clamp(max=1.)
        elif method == "U02":
            delta = float(config.get("delta", .5))
            weight = (1 + loss / (delta*delta)).rsqrt()
        elif method == "U04":
            radius = loss.sqrt()
            threshold = radius.quantile(.8).clamp_min(EPS)
            weight = (1 - (radius / threshold).square()).clamp_min(.05).square()
        elif method == "U05":
            tail = loss >= loss.quantile(.8)
            weight = 1 + gamma * tail.double() / .2
        elif method == "U06":
            tail = gain <= gain.quantile(.2)
            weight = 1 + gamma * tail.double() / .2
        elif method == "U07":
            weight = 1 + gamma * torch.sigmoid(-gain / temperature) / temperature
        elif method == "U11":
            weight = torch.softmax(loss / temperature, 0) * n
            weight = weight.clamp(max=10.)
        elif method == "U12":
            z = (loss - loss.mean()) / loss.std().clamp_min(EPS)
            weight = (1 + gamma*z).clamp(.05, 5.)
        elif method == "U21":
            # Ridge leverage plus residual magnitude approximates LOO influence.
            gram = x.T @ x / len(x) + ridge*torch.eye(x.shape[1], dtype=x.dtype)
            leverage = (x * torch.linalg.solve(gram, x.T).T).sum(1).reshape(n, c).mean(1) / len(x)
            influence = leverage * loss.sqrt()
            weight = (1 + influence / influence.median().clamp_min(EPS)).reciprocal()
        elif method == "U37":
            # Per-sample gradient agreement with the mean gradient, not an environment mask.
            grad_rows = torch.einsum("nd,nt->ndt", x, p-y).reshape(n, c, x.shape[1], y.shape[1]).mean(1)
            mean_grad = grad_rows.mean(0)
            cosine = (grad_rows * mean_grad).sum((1, 2)) / (grad_rows.norm(dim=(1, 2))*mean_grad.norm()+EPS)
            weight = (.5 + .5*cosine).clamp_min(.05)
        elif method == "U38":
            weight = (loss.sqrt() / loss.sqrt().median().clamp_min(EPS)).clamp(max=2.).reciprocal()
        elif method.startswith("G"):
            if groups is None:
                raise ValueError("proxy group labels required")
            distinct = groups.unique()
            gl = torch.stack([loss[groups == g].mean() for g in distinct])
            if method == "G07":
                base = y.square().reshape(n, c, -1).mean((1, 2))
                excess = torch.stack([(loss-base)[groups == g].mean() for g in distinct])
                chosen = excess.argmax()
                multiplier = torch.ones_like(gl)
                multiplier[chosen] += gamma
            elif method == "G08":
                multiplier = torch.softmax(gl / temperature, 0) * len(gl)
            elif method == "G09":
                multiplier = (1 + gamma*(gl-gl.mean())/gl.std().clamp_min(EPS)).clamp(.1, 5.)
            else:
                raise ValueError(method)
            weight = torch.ones(n, dtype=x.dtype)
            for i, g in enumerate(distinct):
                weight[groups == g] = multiplier[i] / max(1, int((groups == g).sum()))
            weight *= n / weight.sum()
        else:
            raise ValueError(method)
        w = _weighted_ridge(x, y, ridge, weight.repeat_interleave(c))
    return w


def fit(method: str, train_keys: torch.Tensor, train_residuals: torch.Tensor,
        metadata: dict, config: dict | None = None) -> G1Correction:
    """Fit a history-only correction.  Query labels are never an argument."""
    config = dict(config or {})
    n, c, key_size = train_keys.shape
    if key_size != 64 or train_residuals.shape[:2] != (n, c):
        raise ValueError("keys [N,C,64] and residuals [N,C,T] required")
    ridge = float(config.get("ridge", .1))
    x, y = design(train_keys), train_residuals.reshape(n*c, -1).double()
    horizon = y.shape[1]
    if method == "ZERO":
        return G1Correction(None, method, {"horizon": horizon})
    if method not in IMPLEMENTED:
        raise NotImplementedError(method)
    extra = {"horizon": horizon}
    if method in {"RIDGE", "U32"}:
        w = _weighted_ridge(x, y, ridge)
    elif method == "U03":
        q = float(config.get("winsor_quantile", .975))
        bound = y.abs().quantile(q, dim=0).clamp_min(EPS)
        w = _weighted_ridge(x, y.clamp(-bound, bound), ridge)
    elif method == "U10":
        scale = train_residuals.double().square().mean((1, 2)).sqrt().clamp_min(.05)
        weights = (scale.median()/scale).clamp(.2, 5.)
        w = _weighted_ridge(x, y, ridge, weights.repeat_interleave(c))
    elif method in {"U01", "U02", "U04", "U05", "U06", "U07", "U11", "U12", "U21", "U37", "U38"}:
        w = _iterative_weights(method, x, y, n, c, ridge, config)
    elif method in {"G07", "G08", "G09"}:
        labels = proxy_groups(train_keys, train_residuals, metadata,
                              config.get("proxy", "time"), int(config.get("num_groups", 4)))
        w = _iterative_weights(method, x, y, n, c, ridge, config, labels)
        extra["proxy"] = config.get("proxy", "time")
    elif method == "G12":
        labels = proxy_groups(train_keys, train_residuals, metadata,
                              config.get("proxy", "time"), int(config.get("num_groups", 4)))
        members = []
        for label in labels.unique():
            keep = labels != label
            flat = (keep[:, None].expand(-1, c)).reshape(-1)
            members.append(_weighted_ridge(x[flat], y[flat], ridge))
        w = torch.stack(members).mean(0)
        extra["proxy"] = config.get("proxy", "time")
    elif method in {"U17", "U18", "U19", "U20"}:
        repeats = int(config.get("repeats", 8))
        members = _bootstrap(x, y, ridge, n, c, repeats, int(config.get("seed", 2021)),
                             .7 if method == "U18" else 1.)
        stack = torch.stack(members)
        w = stack.mean(0)
        if method == "U18":
            signal = w.abs() / stack.std(0, unbiased=False).clamp_min(EPS)
            w = w * (signal / (1 + signal))
        elif method == "U19":
            spread = torch.stack([x @ m for m in members]).std(0, unbiased=False).mean(1)
            extra.update({"members": [m.float() for m in members],
                          "spread_scale": float(spread.median().clamp_min(EPS))})
        elif method == "U20":
            pred = torch.stack([x @ m for m in members]).reshape(-1, horizon)
            rank = min(int(config.get("rank", 8)), horizon, pred.shape[0])
            # CPU SVD of [M*N*C,T] is costly; use the smaller T*T Gram.
            gram = pred.T @ pred / len(pred)
            _, basis = torch.linalg.eigh(gram)
            extra["stable_basis"] = basis[:, -rank:].float()
    elif method == "U23":
        # Empirical-Bayes evidence iteration for a shared isotropic prior.
        w = _weighted_ridge(x, y, ridge)
        for _ in range(5):
            pred = x @ w
            noise = (y-pred).square().mean().clamp_min(EPS)
            prior = w[:-1].square().mean().clamp_min(EPS)
            ridge = float((noise / prior).clamp(1e-5, 1e4)) / len(x)
            w = _weighted_ridge(x, y, ridge)
        extra["evidence_ridge"] = ridge
    elif method == "U16":
        variance = x[:, :-1].var(0, unbiased=False).clamp_min(.01)
        penalty = torch.diag(torch.cat([1/variance, variance.new_ones(1)]))
        w = _weighted_ridge(x, y, ridge, penalty=penalty)
    elif method in {"U24", "U25"}:
        # Proximal gradient with the same quadratic normal equations as Ridge.
        gram = x.T @ x / len(x)
        rhs = x.T @ y / len(x)
        w = _weighted_ridge(x, y, ridge)
        step = 1. / (float(torch.linalg.eigvalsh(gram).max()) + ridge + EPS)
        sparse = float(config.get("sparsity", .01))
        for _ in range(80):
            w = w - step*(gram@w-rhs+ridge*w)
            if method == "U24":
                w[:-1] = w[:-1].sign() * (w[:-1].abs()-step*sparse).clamp_min(0)
            else:
                norms = w[:-1].norm(dim=1, keepdim=True).clamp_min(EPS)
                w[:-1] = w[:-1]*(1-step*sparse/norms).clamp_min(0)
    elif method in {"U26", "U27", "U30"}:
        w = _weighted_ridge(x, y, ridge)
        u, s, vh = torch.linalg.svd(w[:-1], full_matrices=False)
        if method == "U27":
            s = (s-float(config.get("nuclear", .05))*s.max()).clamp_min(0)
        elif method == "U30":
            s = s * s.square()/(s.square()+float(config.get("spectral", .1))*s.max().square()+EPS)
        else:
            rank = min(int(config.get("rank", 8)), len(s))
            s[rank:] = 0
        w[:-1] = (u*s)@vh
    elif method in {"U28", "U29"}:
        w = _weighted_ridge(x, y, ridge)
        freq = torch.fft.rfft(w[:-1], dim=-1)
        frequencies = torch.arange(freq.shape[-1], dtype=x.dtype) / horizon
        if method == "U28":
            order = int(config.get("order", 1))
            penalty = (4*torch.sin(math.pi*frequencies).square())**order
        else:
            penalty = frequencies.square() if config.get("frequency", "low") == "low" else 1/(frequencies.square()+.01)
        strength = float(config.get("smooth", .2))
        freq = freq/(1+strength*penalty[None])
        w[:-1] = torch.fft.irfft(freq, n=horizon, dim=-1)
        extra["note"] = "spectral post-fit shrinkage, not exact finite-boundary Sobolev optimization"
    elif method == "U31":
        features = int(config.get("features", 128))
        gen = torch.Generator().manual_seed(int(config.get("seed", 2021)))
        projection = torch.randn(64, features, generator=gen, dtype=x.dtype)/float(config.get("bandwidth", 2.))
        phase = torch.rand(features, generator=gen, dtype=x.dtype)*2*math.pi
        z = math.sqrt(2./features)*torch.cos(x[:, :-1]@projection+phase)
        z = torch.cat([z, z.new_ones(len(z), 1)], dim=1)
        w = _weighted_ridge(z, y, ridge)
        extra.update({"projection": projection.float(), "phase": phase.float()})
    elif method == "U35":
        # Sparse history-key kNN graph on TRAIN only, with a common linear map.
        w = _weighted_ridge(x, y, ridge)
        cap = min(n, int(config.get("graph_rows", 1024)))
        gen = torch.Generator().manual_seed(int(config.get("seed", 2021)))
        rows = torch.randperm(n, generator=gen)[:cap]
        pooled = train_keys[rows].double().mean(1)
        distances = torch.cdist(pooled, pooled)
        distances.fill_diagonal_(float("inf"))
        neighbors = distances.topk(k=min(4, cap-1), largest=False).indices
        a = design(train_keys[rows]).reshape(cap, c, -1).mean(1)
        difference = (a[:, None]-a[neighbors]).reshape(-1, a.shape[-1])
        graph = difference.T@difference/len(difference)
        strength = float(config.get("graph_strength", .1))
        penalty = torch.eye(x.shape[1], dtype=x.dtype)*ridge
        penalty[-1,-1] = ridge*.01
        w = torch.linalg.solve(x.T@x/len(x)+penalty+strength*graph,
                               x.T@y/len(x))
    else:
        raise NotImplementedError(method)
    if not torch.isfinite(w).all():
        raise FloatingPointError(f"nonfinite {method} weights")
    return G1Correction(w.float(), method, extra)
