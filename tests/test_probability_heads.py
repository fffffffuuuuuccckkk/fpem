from types import SimpleNamespace

import pytest
import torch

from models.PatchTST_PredictiveEnvIV import Model
from models.predictive_env.probability_heads import (
    HEAD_NAMES, FutureConditionEncoder, ProbabilityHeadBank,
)
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.fpem_probabilistic_training import probability_loss_weight
from tools.run_predictive_env_iv_patchtst import gradient_relative_maturity


def config(horizon, channels):
    return SimpleNamespace(
        seq_len=96, pred_len=horizon, d_model=16, n_heads=2, e_layers=1,
        d_ff=32, dropout=0.0, factor=3, activation="gelu", enc_in=channels,
        predictive_env_backbone="patchtst", predictive_env_ablation="A2",
        decomposition_type="complementary_gate",
        representation_constraint="classification", predictive_env_num=3,
        variant_fusion_mode="prob_affine_flow", future_patch_len=16,
        predictive_env_bottleneck=16, horizon_future_dim=8,
    )


@pytest.mark.parametrize("horizon", [96, 95, 336, 720])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("condition_mode", ["full_shape", "three_stats"])
def test_all_heads_shapes_finite_losses_and_seeded_diverse_samples(
        horizon, channels, condition_mode):
    torch.manual_seed(7)
    model = Model(config(horizon, channels)).eval()
    bank = ProbabilityHeadBank(16, horizon, 16, condition_mode)
    x = torch.randn(2, 96, channels)
    y = torch.randn(2, horizon, channels)
    output = model.forward_components(x)
    y_inv = output["invariant_prediction"]
    future = output["probabilistic_condition"]["future"]
    scale = output["input_scale"]
    condition = bank.condition(future, y_inv, scale)
    assert condition.shape == (2, channels, (horizon + 15) // 16, 16)
    for name, (loss, _) in bank.training_losses(condition, future, y_inv, y,
                                                scale).items():
        assert torch.isfinite(loss), name
    with torch.no_grad():
        for name in HEAD_NAMES:
            kwargs = dict(num_samples=3, chunk_size=2, steps=2)
            first = bank.sample_head(
                name, condition, future, y_inv, scale,
                generator=torch.Generator().manual_seed(37), **kwargs,
            )
            assert first["y_samples"].shape == (2, 3, horizon, channels)
            assert torch.isfinite(first["y_samples"]).all(), name
            assert torch.equal(first["y_mean"], first["y_samples"].mean(1))
            if name != "affine_deterministic":
                assert not torch.equal(first["y_samples"][:, 0],
                                       first["y_samples"][:, 1]), name
            if name.startswith("affine_") or name == "legacy_full":
                assert first["gamma_samples"].shape[:3] == (2, 3, channels)
            else:
                assert first["residual_samples"].shape == (
                    2, 3, channels, horizon
                )
            metrics = probabilistic_metrics(first["y_samples"], y, y_inv)
            assert torch.isfinite(metrics["CRPS"]), name


def test_shared_head_gradient_reaches_future_predictor_not_zvar_or_yinv_head():
    model = Model(config(96, 3))
    model.probability_detach_zvar_input = True
    bank = ProbabilityHeadBank(16, 96, heads=("affine_flow", "residual_flow"))
    x, y = torch.randn(2, 96, 3), torch.randn(2, 96, 3)
    output = model.forward_components(x)
    output["z_var_tokens"].retain_grad()
    future = output["probabilistic_condition"]["future"]
    condition = bank.condition(future, output["invariant_prediction"],
                               output["input_scale"])
    losses = bank.training_losses(condition, future,
                                  output["invariant_prediction"], y,
                                  output["input_scale"])
    objective = sum(value[0] for value in losses.values())
    head_grad = torch.autograd.grad(
        objective, tuple(model.head_linear.parameters()),
        retain_graph=True, allow_unused=True,
    )
    assert all(gradient is None for gradient in head_grad)
    objective.backward()
    future_grad = model.horizon_future_variant.future_var_net[-1].weight.grad
    assert future_grad is not None and torch.isfinite(future_grad).all()
    assert future_grad.abs().sum() > 0
    assert output["z_var_tokens"].grad is None


def test_shared_bank_gradient_maturity_and_parameter_scope():
    model = Model(config(96, 3))
    bank = ProbabilityHeadBank(16, 96, heads=("affine_gaussian", "residual_lowrank"))
    x, y = torch.randn(2, 96, 3), torch.randn(2, 96, 3)
    output = model.forward_components(x)
    condition = bank.condition(output["probabilistic_condition"]["future"],
                               output["invariant_prediction"],
                               output["input_scale"])
    losses = bank.training_losses(
        condition, output["probabilistic_condition"]["future"],
        output["invariant_prediction"], y, output["input_scale"],
    )
    inv_loss = (output["invariant_prediction"] - y).square().mean()
    prob_loss = torch.stack([value[0] for value in losses.values()]).mean()
    args = SimpleNamespace(prob_loss_weight_mode="gradient_relative_maturity",
                           prob_loss_min_weight=0.05, prob_ramp_epochs=2)
    bank_parameters = tuple(bank.parameters())
    assert not {id(p) for p in bank_parameters}.intersection(
        id(p) for p in model.environment_classification.parameters()
    )
    weight, diagnostics = probability_loss_weight(
        inv_loss, prob_loss, tuple(model.head_linear.parameters()),
        bank_parameters, args, 0, gradient_relative_maturity,
    )
    assert not weight.requires_grad
    assert 0.05 <= float(weight) <= 1.0
    assert all(torch.isfinite(value) for value in diagnostics.values())
