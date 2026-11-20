from types import SimpleNamespace

import pytest
import torch

from models.PatchTST_PredictiveEnvIV import Model
from models.predictive_env.probability_heads import ProbabilityHeadBank


@pytest.mark.parametrize("horizon", [24, 48])
def test_modern_tcn_joint_probability_heads(horizon):
    config = SimpleNamespace(
        seq_len=96, pred_len=horizon, d_model=16, d_ff=16,
        n_heads=2, e_layers=1, dropout=0.0, factor=3,
        activation="gelu", enc_in=3,
        predictive_env_backbone="moderntcn", predictive_env_ablation="A2",
        decomposition_type="complementary_gate",
        representation_constraint="classification", predictive_env_num=3,
        variant_fusion_mode="prob_affine_flow", future_patch_len=16,
        predictive_env_bottleneck=16, horizon_future_dim=8,
        modern_tcn_patch_size=8, modern_tcn_patch_stride=4,
        modern_tcn_num_stages=1, modern_tcn_ffn_ratio=1,
        modern_tcn_large_size=51, modern_tcn_small_size=5,
        modern_tcn_downsample_ratio=2, modern_tcn_head_dropout=0.0,
    )
    model = Model(config)
    model.probability_detach_zvar_input = True
    bank = ProbabilityHeadBank(16, horizon, 16, "full_shape",
                               ("affine_gaussian", "residual_flow"))
    x, y = torch.randn(2, 96, 3), torch.randn(2, horizon, 3)
    output = model.forward_components(x)
    assert output["invariant_prediction"].shape == y.shape
    future = output["probabilistic_condition"]["future"]
    assert future.shape == (2, 3, (horizon + 15) // 16, 16)
    condition = bank.condition(future, output["invariant_prediction"],
                               output["input_scale"])
    losses = bank.training_losses(condition, future,
                                  output["invariant_prediction"], y,
                                  output["input_scale"])
    total = sum(item[0] for item in losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert model.horizon_future_variant.future_var_net[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.head_linear.parameters())
    with torch.no_grad():
        sampled = bank.sample_head("affine_gaussian", condition, future,
                                   output["invariant_prediction"],
                                   output["input_scale"], 3,
                                   torch.Generator().manual_seed(2021), 2, 2)
    assert sampled["y_samples"].shape == (2, 3, horizon, 3)
    assert torch.isfinite(sampled["y_samples"]).all()
