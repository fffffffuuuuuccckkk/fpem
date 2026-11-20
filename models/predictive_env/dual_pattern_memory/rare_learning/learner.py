"""Joint rare-pattern / full-future candidate learning for DPPM pilots.

The frozen Backbone and stable DPPM bank are deliberately outside this module.
Every inference input is a history key; future labels only enter ``fit``.
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn
from torch.nn import functional as F


class _RareNet(nn.Module):
    def __init__(self, method: str, horizon: int, patterns: int, rank: int,
                 reference: torch.Tensor, seed: int):
        super().__init__()
        self.method, self.patterns, self.horizon = method, patterns, horizon
        self.rank = rank
        self.register_buffer("reference", reference.float().clone())  # [65,T]
        generator = torch.Generator().manual_seed(seed)
        self.centers = nn.Parameter(torch.randn(patterns, 64, generator=generator) * .15)
        self.null_logit = nn.Parameter(torch.tensor(-1.0))
        self.route = nn.Linear(64, patterns, bias=False)
        nn.init.zeros_(self.route.weight)
        self.event_shapelets = nn.Parameter(torch.randn(patterns, 8, generator=generator) * .1)
        self.transition_centers = nn.Parameter(torch.randn(patterns, 16, generator=generator) * .1)
        self.output_basis = nn.Parameter(torch.randn(horizon, rank, generator=generator) / math.sqrt(horizon))
        self.expert = nn.Parameter(torch.zeros(patterns, 65, rank))
        self.full_expert = nn.Parameter(torch.zeros(patterns, 65, horizon)) if method in ("R1", "R2") else None
        self.temperature = .5

    @staticmethod
    def _transition(key):
        # Sixteen ordered changes of four-point local mean: genuine temporal order.
        state = key.reshape(*key.shape[:-1], 16, 4).mean(-1)
        return torch.cat((state[..., 1:] - state[..., :-1], state[..., -1:]-state[..., :1]), -1)

    def _scores(self, flat):
        if self.method == "R4":
            patches = flat.unfold(-1, 8, 4)
            # A pattern may fire anywhere; record location via route features.
            distance = (patches[:, :, None] - self.event_shapelets[None, None]).square().mean(-1)
            return -distance.min(1).values / self.temperature + self.route(flat)
        if self.method == "R7":
            transition = self._transition(flat)
            distance = (transition[:, None] - self.transition_centers[None]).square().mean(-1)
            return -distance / self.temperature + self.route(flat)
        distance = (flat[:, None] - self.centers[None]).square().mean(-1)
        return -distance / self.temperature + self.route(flat)

    def forward(self, key, null_offset: float = 0.):
        b, c, d = key.shape
        if d != 64:
            raise ValueError("rare learner requires normalized history key [B,C,64]")
        flat = key.reshape(b*c, d)
        design = torch.cat((flat, flat.new_ones(len(flat), 1)), -1)
        score = self._scores(flat)
        logits = torch.cat((score, (self.null_logit+null_offset).expand(len(flat), 1)), -1)
        # Sparsemax yields exact zeros and an explicit NULL state.
        shifted = logits - logits.max(-1, keepdim=True).values
        ordered = shifted.sort(-1, descending=True).values
        cssv = ordered.cumsum(-1) - 1
        k = torch.arange(1, ordered.shape[-1]+1, device=ordered.device)
        support = ordered > cssv/k
        kmax = support.sum(-1).clamp_min(1)
        tau = cssv.gather(-1, (kmax-1)[:, None]).squeeze(-1)/kmax
        probability = (shifted-tau[:, None]).clamp_min(0)
        weight = probability[:, :self.patterns]
        shared = design @ self.reference
        if self.full_expert is not None:
            delta = torch.einsum("nd,kdt->nkt", design, self.full_expert)
        else:
            latent = torch.einsum("nd,kdr->nkr", design, self.expert)
            delta = latent @ self.output_basis.T
        candidate = shared[:, None] + delta
        return {"weight": weight.reshape(b,c,self.patterns),
                "null_weight": probability[:, -1].reshape(b,c),
                "candidate": candidate.reshape(b,c,self.patterns,self.horizon),
                "prediction": (weight[:, :, None]*candidate).sum(1).reshape(b,c,self.horizon),
                "score": score.reshape(b,c,self.patterns)}


class RarePatternLearner:
    """Common fit/query/candidate/export API for R1–R7.

    R0 is the immutable original PatternBank and is evaluated by the runner.
    """
    def __init__(self, method: str, patterns: int = 8, rank: int = 8,
                 seed: int = 2021, epochs: int = 10, lr: float = 1e-3,
                 lambda_sparse: float = .002, freeze_router: bool = False,
                 optimizer_config: dict | None = None):
        if method not in {f"R{i}" for i in range(1, 8)}:
            raise ValueError("R0 uses the original PatternBank")
        self.method, self.patterns, self.rank = method, patterns, rank
        self.seed, self.epochs, self.lr = seed, epochs, lr
        self.lambda_sparse, self.freeze_router = lambda_sparse, freeze_router
        self.optimizer_config = dict(optimizer_config or {})
        self.net = None
        self.stats = {}

    def fit(self, train_x, train_y, context):
        """Train on historical keys and normalized future; no query future access.

        context: reference_map [65,T], starts [N], span, optional teacher_gradient
        and optional selection-free precomputed stable forecast for diagnostics.
        """
        if train_x.ndim != 3 or train_x.shape[-1] != 64 or train_y.shape[:2] != train_x.shape[:2]:
            raise ValueError("expected history [N,C,64] and future [N,C,T]")
        ref = context["reference_map"]
        self.net = _RareNet(self.method, train_y.shape[-1], self.patterns,
                            self.rank, ref, self.seed)
        device = torch.device(context.get("device", "cpu"))
        self.net.to(device)
        cfg = self.optimizer_config
        if cfg:
            head_lr = float(cfg.get("head_lr", self.lr))
            groups = [
                {"params": [self.net.expert], "lr": head_lr, "base_lr": head_lr, "role": "mapping"},
                {"params": [self.net.centers], "lr": head_lr*float(cfg.get("dictionary_ratio", 1.)),
                 "base_lr": head_lr*float(cfg.get("dictionary_ratio", 1.)), "role": "dictionary"},
                {"params": [self.net.output_basis], "lr": head_lr*float(cfg.get("basis_ratio", 1.)),
                 "base_lr": head_lr*float(cfg.get("basis_ratio", 1.)), "role": "basis"},
                {"params": [self.net.route.weight, self.net.null_logit],
                 "lr": head_lr*float(cfg.get("route_ratio", 1.)),
                 "base_lr": head_lr*float(cfg.get("route_ratio", 1.)), "role": "routing"},
            ]
            kind = cfg.get("optimizer", "adam")
            if kind not in {"adam", "adamw", "adamw_lbfgs"}:
                raise ValueError(f"unsupported optimizer: {kind}")
            optimizer_type = torch.optim.Adam if kind == "adam" else torch.optim.AdamW
            optim = optimizer_type(groups, weight_decay=float(cfg.get("weight_decay", 0.)))
            with torch.no_grad():
                self.net.output_basis.mul_(float(cfg.get("basis_init_scale", 1.)))
        else:
            optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        x, y = train_x.float(), train_y.float()
        starts = context["starts"].long()
        event = torch.div(starts, int(context["span"]), rounding_mode="floor")
        # One anchor per independent event per epoch; overlap does not multiply evidence.
        event_rows = torch.stack([(event == e).nonzero().flatten()[0] for e in event.unique(sorted=True)])
        source = event_rows if len(event_rows) >= 16 else torch.arange(len(x))
        # History-only initialization ensures at least some non-NULL gradient
        # before the sparse routing and the future maps co-adapt.
        flat_seed = x[source].reshape(-1,64)
        pick = torch.linspace(0,max(0,len(flat_seed)-1),self.patterns).long()
        with torch.no_grad():
            self.net.centers.copy_(flat_seed[pick].to(device))
            if self.method == "R4":
                windows = flat_seed[pick].unfold(-1,8,4)
                self.net.event_shapelets.copy_(windows[:,windows.shape[1]//2].to(device))
            if self.method == "R7":
                self.net.transition_centers.copy_(self.net._transition(flat_seed[pick].to(device)))
        if self.freeze_router:
            for name, parameter in self.net.named_parameters():
                if name in {"centers", "null_logit", "route.weight",
                            "event_shapelets", "transition_centers"}:
                    parameter.requires_grad_(False)
        generator = torch.Generator().manual_seed(self.seed)
        trace, started = [], time.monotonic()
        best_metric, best_state, stale = float("inf"), None, 0
        ema_state = None
        for epoch in range(self.epochs):
            if cfg:
                schedule = cfg.get("schedule", "constant")
                progress = epoch / max(1, self.epochs-1)
                if schedule == "constant" or schedule == "plateau":
                    factor = 1.
                elif schedule == "cosine":
                    factor = .1 + .9 * .5 * (1+math.cos(math.pi*progress))
                elif schedule == "warmup_cosine":
                    warm = max(1, self.epochs//5)
                    factor = (epoch+1)/warm if epoch < warm else .1+.9*.5*(1+math.cos(
                        math.pi*(epoch-warm)/max(1,self.epochs-1-warm)))
                else:
                    raise ValueError(f"unsupported schedule: {schedule}")
                order = cfg.get("order", "S0")
                for group in optim.param_groups:
                    role = group["role"]
                    active = True
                    if order == "S1" and epoch < max(1,self.epochs//5):
                        active = role == "mapping"
                    elif order == "S2" and epoch < max(1,self.epochs//2):
                        active = role != "basis"
                    elif order == "S3":
                        active = (role in {"mapping","basis"}) if epoch%2 == 0 else role in {"dictionary","routing"}
                    group["lr"] = group["base_lr"] * factor * (1. if active else 0.)
                    if order == "S2" and epoch >= max(1,self.epochs//2) and role == "basis":
                        group["lr"] *= .1
            shuffled = source[torch.randperm(len(source), generator=generator)]
            epoch_grad = {"mapping":0.,"dictionary":0.,"basis":0.,"routing":0.}
            steps = 0
            for rows in shuffled.split(64):
                xb, yb = x[rows].to(device), y[rows].to(device)
                output = self.net(xb)
                w = output["weight"]
                candidate = output["candidate"]
                null = output["null_weight"]
                shared = torch.cat((xb, xb.new_ones(*xb.shape[:2], 1)), -1) @ self.net.reference
                mixture = (w[..., None]*candidate).sum(2) + null[..., None]*shared
                forecast = F.mse_loss(mixture, yb)
                # Every expert receives truthful future supervision only through its
                # history-only activation; there is no Y-routed inference shortcut.
                expert = (w[..., None]*(candidate-yb[:, :, None]).square()).mean()
                sparse = w.sum(-1).mean() + .1*(w > 0).float().sum(-1).mean()
                distinct = self.net.expert.square().mean()
                if self.net.full_expert is not None:
                    distinct = self.net.full_expert.square().mean()
                loss = forecast + .2*expert + self.lambda_sparse*sparse
                if self.method == "R2" and not self.freeze_router:
                    quality = -(candidate-yb[:, :, None]).square().mean(-1).detach()
                    teacher = quality.softmax(-1)
                    loss += .05*F.kl_div(output["score"].log_softmax(-1), teacher, reduction="batchmean") / xb.shape[1]
                if self.method == "R5" and not self.freeze_router:
                    teacher = context.get("teacher_gradient")
                    if teacher is not None:
                        tg = teacher[rows].to(device)
                        loss += .05*F.mse_loss(self.net._scores(xb.reshape(-1,64)).reshape_as(tg), tg)
                if self.method == "R6":
                    loss += .1*distinct
                elif self.method == "R3":
                    loss += .005*distinct
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if cfg:
                    for group in optim.param_groups:
                        values=[p.grad.detach().square().sum() for p in group["params"] if p.grad is not None]
                        if values:
                            epoch_grad[group["role"]] += float(torch.stack(values).sum().sqrt())
                    steps += 1
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2.)
                optim.step()
                if cfg.get("ema_decay"):
                    decay=float(cfg["ema_decay"])
                    if ema_state is None:
                        ema_state={name:p.detach().clone() for name,p in self.net.named_parameters()}
                    else:
                        for name,p in self.net.named_parameters():
                            ema_state[name].mul_(decay).add_(p.detach(),alpha=1-decay)
            trace.append({"epoch": epoch+1, "loss": float(loss.detach()),
                          "forecast": float(forecast.detach()),
                          "grad_norm": {name:value/max(1,steps) for name,value in epoch_grad.items()}})
            validation = context.get("validation") if cfg else None
            if validation is not None:
                with torch.no_grad():
                    vx,vy=validation
                    vout=self.net(vx.to(device))
                    vw,vnull=vout["weight"],vout["null_weight"]
                    vdesign=torch.cat((vx.to(device),vx.new_ones(*vx.shape[:2],1).to(device)),-1)
                    vshared=vdesign @ self.net.reference
                    vpred=(vw[...,None]*vout["candidate"]).sum(2)+vnull[...,None]*vshared
                    metric=float(F.mse_loss(vpred,vy.to(device)))
                trace[-1]["validation_candidate_loss"]=metric
                if metric < best_metric-1e-8:
                    best_metric,stale=metric,0
                    best_state={name:p.detach().clone() for name,p in self.net.named_parameters()}
                else:
                    stale+=1
                if cfg.get("schedule")=="plateau" and stale>=2:
                    for group in optim.param_groups:
                        group["base_lr"] *= .5
                    stale=0
                if cfg.get("early_stop_patience") and stale>=int(cfg["early_stop_patience"]):
                    break
        if cfg.get("optimizer")=="adamw_lbfgs":
            head=[self.net.expert,self.net.output_basis]
            solver=torch.optim.LBFGS(head,lr=.25,max_iter=8,line_search_fn="strong_wolfe")
            xb,yb=x[source].to(device),y[source].to(device)
            def closure():
                solver.zero_grad(set_to_none=True)
                out=self.net(xb)
                shared=torch.cat((xb,xb.new_ones(*xb.shape[:2],1)),-1) @ self.net.reference
                pred=(out["weight"][...,None]*out["candidate"]).sum(2)+out["null_weight"][...,None]*shared
                objective=F.mse_loss(pred,yb)+.2*(out["weight"][...,None]*(out["candidate"]-yb[:,:,None]).square()).mean()
                objective.backward()
                return objective
            solver.step(closure)
        if cfg.get("restore_best") and best_state is not None:
            with torch.no_grad():
                for name,p in self.net.named_parameters():
                    p.copy_(best_state[name])
        if ema_state is not None:
            with torch.no_grad():
                for name,p in self.net.named_parameters():
                    p.copy_(ema_state[name])
        self.net.cpu().eval()
        self.stats = {"train_seconds": time.monotonic()-started, "loss_trace": trace,
                      "train_events": int(event.unique().numel()),
                      "parameters": sum(p.numel() for p in self.net.parameters())}
        return self

    @torch.no_grad()
    def query(self, history_x, null_offset: float = 0.):
        if self.net is None:
            raise RuntimeError("fit before query")
        return self.net(history_x.float().cpu(), null_offset)

    def predict_candidates(self, history_x):
        return self.query(history_x)["candidate"]

    def export(self):
        if self.net is None:
            raise RuntimeError("fit before export")
        return {"method": self.method, "state_dict": self.net.state_dict(),
                "patterns": self.patterns, "rank": self.rank,
                "lambda_sparse": self.lambda_sparse, "freeze_router": self.freeze_router,
                "optimizer_config":self.optimizer_config,
                "statistics": self.stats}

    @classmethod
    def from_export(cls, saved):
        obj = cls(saved["method"], saved["patterns"], saved["rank"],
                  lambda_sparse=saved.get("lambda_sparse", .002),
                  freeze_router=saved.get("freeze_router", False))
        ref = saved["state_dict"]["reference"]
        obj.net = _RareNet(obj.method, ref.shape[-1], obj.patterns,
                           obj.rank, ref, obj.seed)
        obj.net.load_state_dict(saved["state_dict"])
        obj.net.eval()
        obj.stats = saved.get("statistics", {})
        return obj
