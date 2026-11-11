"""A0-preserving unified Head for [H_query,H_1,...,H_K]."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DirectConcatHead(nn.Module):
    def __init__(self, a0_head: nn.Linear, tokens: int, d_model: int,
                 pred_len: int, retrieved: int):
        super().__init__()
        if a0_head.in_features != tokens*d_model or a0_head.out_features != pred_len:
            raise ValueError("A0 flattened Head shape is incompatible")
        self.tokens, self.d_model, self.pred_len, self.retrieved = (
            tokens, d_model, pred_len, retrieved)
        self.head = nn.Linear((retrieved+1)*tokens*d_model, pred_len)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[:, :tokens*d_model].copy_(a0_head.weight)
            self.head.bias.copy_(a0_head.bias)
        # Existing A0 mapping and bias remain fixed; only new retrieval columns
        # learn. This is still one unified linear Head over concatenated H.
        self.head.bias.requires_grad_(False)
        self.head.weight.register_hook(self._mask_reference_gradient)

    def _mask_reference_gradient(self, grad):
        result = grad.clone()
        result[:, :self.tokens*self.d_model] = 0
        return result

    def forward(self, model, current: torch.Tensor, retrieved_hidden: torch.Tensor,
                cycle=None) -> torch.Tensor:
        b,c,p,d = current.shape
        if (p,d) != (self.tokens,self.d_model):
            raise ValueError("Backbone hidden layout differs from A0 Head")
        if retrieved_hidden.shape != (b,c,self.retrieved,p,d):
            raise ValueError("retrieved H must preserve every independent sample")
        # [B,C,K+1,P,D] -> [B,C,P,(K+1)D] -> [B,C,(K+1)D,P]
        cat = torch.cat((current[:,:,None], retrieved_hidden), dim=2)
        cat = cat.permute(0,1,3,2,4).reshape(b,c,p,(self.retrieved+1)*d)
        flat = model.head_flatten(cat.permute(0,1,3,2))
        normalized = self.head(flat)
        cycle_value = model.normalized_cycle_forecast(cycle,b,current.device)
        return normalized if cycle_value is None else normalized + cycle_value


class DirectFutureResidualHead(nn.Module):
    def __init__(self, pred_len: int, retrieved: int):
        super().__init__()
        self.pred_len, self.retrieved = pred_len, retrieved
        self.head = nn.Linear((retrieved+1)*pred_len, pred_len)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, point_normalized: torch.Tensor,
                future_normalized: torch.Tensor) -> torch.Tensor:
        if future_normalized.shape != (*point_normalized.shape[:2],
                                       self.retrieved, self.pred_len):
            raise ValueError("future values must be B,C,K,T")
        # Preserve the unmodified TRAIN future in the memory bank; only bound
        # the numerical input of this linear Head on constant-history windows.
        values = torch.cat((point_normalized.clamp(-20,20),
                            future_normalized.clamp(-20,20).flatten(start_dim=-2)), dim=-1)
        return self.head(values)


class DirectConcatPredictor(nn.Module):
    """Separate H and future paths with one final point forecast."""
    def __init__(self, a0_head, tokens, d_model, pred_len, retrieved):
        super().__init__()
        self.hidden_head = DirectConcatHead(a0_head,tokens,d_model,pred_len,retrieved)
        self.residual_head = DirectFutureResidualHead(pred_len,retrieved)

    def forward(self, model, current, memory, mean, std, cycle=None,
                future_mode="none", future_memory=None):
        point_norm = self.hidden_head(model,current,memory["hidden"],cycle)
        values = memory if future_memory is None else future_memory
        if future_mode == "none":
            delta = torch.zeros_like(point_norm)
        else:
            future = values["residual"] if future_mode == "residual" else values["future"]
            if future_mode == "shuffled":
                # Shift valid histories only; no future outside the causal pool.
                mask = values["valid"]
                count = mask.sum(-1,keepdim=True)
                slot = torch.arange(future.shape[2],device=future.device)[None,None]
                shifted = torch.remainder(slot+1,count.clamp_min(1))
                future = future.gather(2,shifted[...,None].expand_as(future)) * mask[...,None]
            delta = self.residual_head(point_norm, future)
        full_norm = point_norm + delta
        return (full_norm.permute(0,2,1)*std+mean,
                point_norm.permute(0,2,1)*std+mean,
                delta.permute(0,2,1)*std)
