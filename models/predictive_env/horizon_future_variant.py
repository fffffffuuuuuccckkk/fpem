"""Future-patch variant prediction and output-space modulation."""

import math

import torch
from torch import nn


class HorizonFutureVariant(nn.Module):
    """Predict one ``Z_var`` per future patch and modulate that output patch.

    ``future_patch_len`` partitions the prediction interval independently of
    the forecasting backbone. The only high-dimensional future tensor is
    ``[B,C,K,D]``, where ``K=ceil(H/future_patch_len)``. The modulation head
    still emits a distinct scale/shift for every point inside each patch.
    Zero initialization makes the initial full forecast exactly invariant.
    """

    def __init__(
        self,
        d_model,
        pred_len,
        horizon_dim=32,
        bottleneck=64,
        gamma_scale=0.1,
        beta_scale=0.1,
        future_patch_len=16,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.pred_len = int(pred_len)
        self.horizon_dim = int(horizon_dim)
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.future_patch_len = int(future_patch_len)
        if self.pred_len <= 0 or self.horizon_dim <= 0:
            raise ValueError("pred_len and horizon_dim must be positive")
        if self.future_patch_len <= 0:
            raise ValueError("future_patch_len must be positive")
        if self.gamma_scale <= 0 or self.beta_scale <= 0:
            raise ValueError("future gamma/beta scales must be positive")
        self.future_patch_count = math.ceil(
            self.pred_len / self.future_patch_len
        )

        self.horizon_encoder = nn.Sequential(
            nn.Linear(1, self.horizon_dim),
            nn.GELU(),
            nn.Linear(self.horizon_dim, self.horizon_dim),
        )
        self.future_var_net = nn.Sequential(
            nn.LayerNorm(self.d_model + self.horizon_dim),
            nn.Linear(self.d_model + self.horizon_dim, int(bottleneck)),
            nn.GELU(),
            nn.Linear(int(bottleneck), self.d_model),
        )
        reliability_dim = (
            3 * self.d_model
            + self.horizon_dim
            + 2 * self.future_patch_len
        )
        self.reliability_net = nn.Sequential(
            nn.LayerNorm(reliability_dim),
            nn.Linear(reliability_dim, int(bottleneck)),
            nn.GELU(),
            nn.Linear(int(bottleneck), 1),
        )
        self.modulation_generator = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, int(bottleneck)),
            nn.GELU(),
            nn.Linear(int(bottleneck), 2 * self.future_patch_len),
        )
        nn.init.zeros_(self.future_var_net[-1].weight)
        nn.init.zeros_(self.future_var_net[-1].bias)
        nn.init.zeros_(self.modulation_generator[-1].weight)
        nn.init.zeros_(self.modulation_generator[-1].bias)
        nn.init.zeros_(self.reliability_net[-1].weight)
        nn.init.zeros_(self.reliability_net[-1].bias)

        starts = torch.arange(self.future_patch_count) * self.future_patch_len
        ends = torch.minimum(
            starts + self.future_patch_len,
            torch.tensor(self.pred_len),
        )
        # Integer, one-based locations are used only for causal TRAIN teachers.
        # Embeddings use the exact (possibly half-step) patch centers.
        centers = torch.div(starts + ends - 1, 2, rounding_mode="floor") + 1
        coordinates = ((starts + ends + 1).float() / 2.0) / self.pred_len
        self.register_buffer("patch_center_indices", centers.long())
        self.register_buffer("patch_center_coordinates", coordinates[:, None])

    def horizon_coordinates(self, device, dtype):
        return self.patch_center_coordinates.to(device=device, dtype=dtype)

    def forward(self, invariant_prediction, z_inv, z_var):
        if invariant_prediction.ndim != 3:
            raise ValueError("invariant prediction must be [B,C,H]")
        if invariant_prediction.shape[-1] != self.pred_len:
            raise ValueError("invariant prediction horizon does not match pred_len")
        if z_inv.ndim != 4 or z_var.ndim != 4 or z_inv.shape != z_var.shape:
            raise ValueError("z_inv and z_var must align as [B,C,P,D]")
        if z_inv.shape[-1] != self.d_model:
            raise ValueError("representation dimension does not match d_model")

        batch, channels = z_var.shape[:2]
        z_inv_pool = z_inv.mean(dim=2)
        z_var_pool = z_var.mean(dim=2)
        encoded = self.horizon_encoder(
            self.horizon_coordinates(z_var.device, z_var.dtype)
        )
        horizon = encoded.view(
            1, 1, self.future_patch_count, self.horizon_dim
        ).expand(batch, channels, -1, -1)
        current = z_var_pool.unsqueeze(2).expand(
            -1, -1, self.future_patch_count, -1
        )
        delta = self.future_var_net(torch.cat((current, horizon), dim=-1))
        future = current + delta

        modulation = self.modulation_generator(future)
        gamma_raw, beta_raw = modulation.chunk(2, dim=-1)
        gamma_patch = self.gamma_scale * torch.tanh(gamma_raw)
        beta_patch = self.beta_scale * torch.tanh(beta_raw)

        # One reliability value per sample and future patch. Inputs are
        # detached so L_r can only update this lightweight head.
        reliability_horizon = encoded.unsqueeze(0).expand(batch, -1, -1)
        reliability_input = torch.cat(
            (
                z_inv_pool.mean(dim=1).unsqueeze(1).expand(
                    -1, self.future_patch_count, -1
                ),
                z_var_pool.mean(dim=1).unsqueeze(1).expand(
                    -1, self.future_patch_count, -1
                ),
                future.mean(dim=1),
                reliability_horizon,
                gamma_patch.mean(dim=1),
                beta_patch.mean(dim=1),
            ),
            dim=-1,
        ).detach()
        reliability = torch.sigmoid(
            self.reliability_net(reliability_input)
        ).squeeze(-1)

        full_patches = []
        gamma_patches = []
        beta_patches = []
        variation_patches = []
        for patch_index in range(self.future_patch_count):
            start = patch_index * self.future_patch_len
            width = min(self.future_patch_len, self.pred_len - start)
            # Variant refinement cannot update or deliberately degrade the
            # invariant forecasting anchor through the full-forecast path.
            invariant_patch = invariant_prediction[
                ..., start : start + width
            ].detach()
            gamma = gamma_patch[:, :, patch_index, :width]
            beta = beta_patch[:, :, patch_index, :width]
            raw_variation = gamma * invariant_patch + beta
            gated_variation = (
                reliability[:, patch_index, None, None] * raw_variation
            )
            patch_variation = raw_variation if self.training else gated_variation
            full_patches.append(invariant_patch + patch_variation)
            gamma_patches.append(gamma)
            beta_patches.append(beta)
            variation_patches.append(patch_variation)

        return {
            "prediction": torch.cat(full_patches, dim=-1),
            "future_zvar": future,
            "reliability": reliability,
            "gamma": torch.cat(gamma_patches, dim=-1),
            "beta": torch.cat(beta_patches, dim=-1),
            "variation": torch.cat(variation_patches, dim=-1),
            "current_zvar": z_var_pool,
            "future_zvar_change_norm": delta.norm(dim=-1),
            "future_zvar_anchor_indices": self.patch_center_indices - 1,
            "future_patch_count": self.future_patch_count,
            "future_patch_len": self.future_patch_len,
        }
