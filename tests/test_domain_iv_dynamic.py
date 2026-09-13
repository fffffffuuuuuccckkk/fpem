from types import SimpleNamespace
import numpy as np
import pytest
import torch

from models.PatchTST_DomainIV_Dynamic import Model, VARIATIONS
from models.domain_iv.dynamic_environment import DynamicEnvironmentDiscovery


def config(mode):
    return SimpleNamespace(seq_len=32, pred_len=8, d_model=16, n_heads=2, e_layers=1,
        d_ff=32, dropout=0., factor=1, activation="gelu", patch_len=8,
        domain_iv_stride=4, domain_iv_variation=mode, domain_iv_bottleneck=8, domain_iv_rank=2)


def test_all_variation_modes_forward_backward():
    for mode in VARIATIONS:
        model = Model(config(mode)); result = model.forward_components(torch.randn(4, 32, 3))
        assert result["prediction"].shape == (4, 8, 3)
        assert result["feature_gate"].shape[-1] == 1
        result["prediction"].square().mean().backward()


def test_feature_and_mapping_are_identity_initialized():
    model = Model(config("V5"))
    values = model.identity_diagnostics(torch.randn(3, 32, 3))
    assert values["feature_identity_difference"] < 1e-8
    assert values["mapping_identity_difference"] < 1e-8


def test_dynamic_environment_updates_and_is_permutation_invariant():
    rng = np.random.RandomState(3)
    rep = np.r_[rng.randn(20, 5) - 2, rng.randn(20, 5) + 2].astype("float32")
    discovery = DynamicEnvironmentDiscovery(2, ema_beta=.5, seed=2)
    q0, first = discovery.update(rep, "train")
    q1, second = discovery.update(rep + .1 * rng.randn(*rep.shape), "train")
    assert np.isfinite(second["mean_q_change"])
    assert np.allclose(q0 @ q0.T, q0[:, ::-1] @ q0[:, ::-1].T, atol=1e-6)
    assert first["source_split"] == "TRAIN only"
    assert q1.shape == (40, 2)


def test_dynamic_environment_rejects_non_train_update():
    with pytest.raises(ValueError, match="TRAIN"):
        DynamicEnvironmentDiscovery(2).update(np.random.randn(10, 3), "test")
