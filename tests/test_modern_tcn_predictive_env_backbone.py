from types import SimpleNamespace

import pytest
import torch

from models.PatchTST_PredictiveEnvIV import Model


def config(stages=1, **overrides):
    values = dict(
        seq_len=96, pred_len=48, d_model=16, n_heads=2, e_layers=1,
        d_ff=16, dropout=0.0, factor=3, activation="gelu", enc_in=3,
        predictive_env_backbone="moderntcn", predictive_env_ablation="A0",
        modern_tcn_patch_size=8, modern_tcn_patch_stride=4,
        modern_tcn_num_stages=stages, modern_tcn_ffn_ratio=1,
        modern_tcn_large_size=51, modern_tcn_small_size=5,
        modern_tcn_downsample_ratio=2, modern_tcn_head_dropout=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("stages,tokens", [(1, 24), (3, 6)])
def test_modern_tcn_shared_forecast_contract(stages, tokens):
    model = Model(config(stages))
    output = model.forward_components(torch.randn(2, 96, 3))
    assert output["hidden_tokens"].shape == (2, 3, tokens, 16)
    assert output["prediction"].shape == (2, 48, 3)
    assert output["backbone"] == "moderntcn"
    output["prediction"].square().mean().backward()
    assert model.modern_tcn_backbone.downsample_layers[0][0].weight.grad is not None


def test_teacher_does_not_update_modern_tcn_batchnorm():
    model = Model(config())
    model.train()
    bn = model.modern_tcn_backbone.downsample_layers[0][1]
    before = bn.running_mean.clone()
    model.future_variant_target(torch.randn(2, 96, 3))
    assert torch.equal(before, bn.running_mean)
    assert model.training and bn.training
