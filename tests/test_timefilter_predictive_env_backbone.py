from types import SimpleNamespace

import torch

from models.PatchTST_PredictiveEnvIV import Model, timefilter_region_masks


def config(**overrides):
    values = dict(
        seq_len=96,
        pred_len=48,
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_ff=64,
        dropout=0.0,
        factor=3,
        activation="gelu",
        enc_in=3,
        predictive_env_backbone="timefilter",
        predictive_env_ablation="A0",
        timefilter_patch_len=4,
        timefilter_alpha=0.1,
        timefilter_top_p=0.5,
        timefilter_pos=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_timefilter_uses_shared_token_and_forecast_contract():
    model = Model(config())
    output = model.forward_components(torch.randn(2, 96, 3))
    assert output["hidden_tokens"].shape == (2, 3, 24, 32)
    assert output["prediction"].shape == (2, 48, 3)
    assert output["backbone"] == "timefilter"
    assert torch.isfinite(output["backbone_aux_loss"])


def test_timefilter_region_masks_partition_all_relations():
    masks = timefilter_region_masks(3, 4)
    assert masks.shape == (12, 3, 12)
    assert torch.equal(masks.sum(dim=1), torch.ones(12, 12))
    # Upstream assigns the diagonal to the same-variable temporal region;
    # MaskMoE then explicitly adds its identity path as well.
    assert torch.equal(masks[:, 1, :].diagonal(), torch.ones(12))


def test_timefilter_rejects_non_divisible_patch_length():
    try:
        Model(config(timefilter_patch_len=5))
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("non-divisible TimeFilter patch length was accepted")
