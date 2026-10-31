from types import SimpleNamespace

import pytest
import torch

from models.PatchTST_PredictiveEnvIV import Model
from models.predictive_env.affine_target_projection import (
    decode_affine, project_affine_targets,
)
from tools.evaluate_fpem_probabilistic import probabilistic_metrics
from tools.fpem_probabilistic_training import (
    active_probability_maturity_parameters, probability_loss_weight,
)
from tools.run_predictive_env_iv_patchtst import gradient_relative_maturity


def config(backbone="patchtst", horizon=96, channels=3, mode="prob_affine_flow"):
    return SimpleNamespace(
        seq_len=96, pred_len=horizon, d_model=16, n_heads=2, e_layers=1,
        d_ff=32, dropout=0.0, factor=3, activation="gelu", enc_in=channels,
        predictive_env_backbone=backbone, predictive_env_ablation="A2",
        decomposition_type="complementary_gate",
        representation_constraint="classification", predictive_env_num=3,
        variant_fusion_mode=mode, future_patch_len=16,
        predictive_env_bottleneck=16, horizon_future_dim=8,
    )


@pytest.mark.parametrize("backbone", ["patchtst", "itransformer"])
@pytest.mark.parametrize("horizon", [96, 336, 720])
@pytest.mark.parametrize("channels", [1, 3])
def test_model_contract_and_anchor(backbone, horizon, channels):
    model = Model(config(backbone, horizon, channels)).eval()
    x = torch.randn(2, 96, channels)
    with torch.no_grad():
        output = model.forward_components(x)
        assert output["prediction"].shape == (2, horizon, channels)
        assert torch.equal(output["prediction"], output["invariant_prediction"])
        paths = model.probabilistic_forecast(
            x, num_samples=3, steps=2, chunk_size=2,
            generator=torch.Generator().manual_seed(5),
        )
        assert paths["y_samples"].shape == (2, 3, horizon, channels)
        assert paths["gamma_samples"].shape == (2, 3, channels, (horizon + 15) // 16)
        assert torch.isfinite(paths["y_samples"]).all()


def test_projection_partial_patch_and_exact_zero_recovery():
    inv = torch.randn(2, 95, 3)
    scale = torch.rand(2, 1, 3) + 0.5
    theta, center = project_affine_targets(inv + 0.1, inv, scale, 16)
    assert theta.shape == (2, 3, 6, 2)
    zero = decode_affine(inv, scale, torch.zeros_like(theta), center, 16)[:, 0]
    assert torch.equal(zero, inv)


def test_non_divisible_model_horizon_and_counterfactual_conditions():
    model = Model(config(horizon=95)).eval()
    x = torch.randn(2, 96, 3)
    with torch.no_grad():
        for ablation in ("deterministic_affine_center", "affine_flow",
                         "shuffled_zvar", "unconditional_flow",
                         "gaussian_baseline"):
            generated = model.probabilistic_forecast(
                x, num_samples=3, steps=2, ablation=ablation,
                generator=torch.Generator().manual_seed(11),
            )
            assert generated["y_samples"].shape == (2, 3, 95, 3)
            assert torch.isfinite(generated["y_samples"]).all()


def test_sampling_is_seeded_and_noncollapsed():
    model = Model(config()).eval()
    x = torch.randn(2, 96, 3)
    with torch.no_grad():
        first = model.probabilistic_forecast(
            x, num_samples=4, steps=2,
            generator=torch.Generator().manual_seed(7),
        )["y_samples"]
        second = model.probabilistic_forecast(
            x, num_samples=4, steps=2,
            generator=torch.Generator().manual_seed(7),
        )["y_samples"]
    assert torch.equal(first, second)
    assert not torch.equal(first[:, 0], first[:, 1])


def test_flow_reaches_future_zvar_not_invariant_head():
    model = Model(config())
    x = torch.randn(2, 96, 3)
    y = torch.randn(2, 96, 3)
    output = model.forward_components(x)
    dynamics = model.probabilistic_affine_dynamics
    losses = dynamics.supervised_losses(
        output["probabilistic_condition"], y,
        output["invariant_prediction"], output["input_scale"],
    )
    aux = losses["center"] + losses["flow"] + losses["innovation"]
    head_grad = torch.autograd.grad(
        aux, tuple(model.head_linear.parameters()), retain_graph=True,
        allow_unused=True,
    )
    assert all(gradient is None for gradient in head_grad)
    aux.backward()
    future_grad = model.horizon_future_variant.future_var_net[-1].weight.grad
    assert future_grad is not None and torch.isfinite(future_grad).all()
    assert future_grad.abs().sum() > 0
    assert all(torch.isfinite(value).all() for value in losses.values()
               if torch.is_tensor(value))


def test_old_mode_and_probability_scores():
    old = Model(config(mode="horizon_future_var")).eval()
    with torch.no_grad():
        output = old.forward_components(torch.randn(2, 96, 3))
    assert output["probabilistic_condition"] is None
    target = torch.zeros(2, 96, 3)
    samples = torch.randn(2, 5, 96, 3)
    metrics = probabilistic_metrics(samples, target, target)
    assert torch.isfinite(metrics["CRPS"])
    assert torch.isfinite(metrics["quantile_CRPS_normalized"])


@pytest.mark.parametrize("ablation,innovation,expected", [
    ("full", 1, ("center_head", "velocity", "innovation")),
    ("full", 0, ("center_head", "velocity")),
    ("affine_flow", 1, ("center_head", "velocity")),
    ("shuffled_zvar", 1, ("center_head", "velocity")),
    ("unconditional_flow", 1, ("center_head", "velocity")),
    ("deterministic_affine_center", 1, ("center_head",)),
    ("gaussian_baseline", 1, ("center_head", "gaussian_head")),
    ("invariant_only", 1, ()),
])
def test_maturity_parameter_subset_matches_active_ablation(
        ablation, innovation, expected):
    model = Model(config())
    args = SimpleNamespace(
        lambda_prob_mu=0.1, lambda_prob_fm=0.1,
        lambda_prob_innovation=0.05, lambda_prob_smooth=0.001,
        use_stochastic_innovation=innovation,
    )
    dynamics = model.probabilistic_affine_dynamics
    selected = active_probability_maturity_parameters(dynamics, args, ablation)
    expected_ids = {id(parameter) for name in expected
                    for parameter in getattr(dynamics, name).parameters()}
    assert {id(parameter) for parameter in selected} == expected_ids
    assert not expected_ids.intersection(
        id(parameter) for parameter in model.environment_classification.parameters()
    )
    args.lambda_prob_mu = args.lambda_prob_fm = 0.0
    args.lambda_prob_innovation = args.lambda_prob_smooth = 0.0
    assert active_probability_maturity_parameters(dynamics, args, ablation) == ()


def test_gradient_maturity_is_detached_and_legacy_ramp_replays():
    invariant_parameter = torch.nn.Parameter(torch.tensor(1.0))
    probability_parameter = torch.nn.Parameter(torch.tensor(1.0))
    inv_loss = (2.0 * invariant_parameter).square()
    prob_loss = probability_parameter.square()
    args = SimpleNamespace(
        prob_loss_weight_mode="gradient_relative_maturity",
        prob_loss_min_weight=0.05, prob_ramp_epochs=2,
    )
    weight, diagnostics = probability_loss_weight(
        inv_loss, prob_loss, (invariant_parameter,),
        (probability_parameter,), args, 0, gradient_relative_maturity,
    )
    assert not weight.requires_grad
    assert float(diagnostics["G_inv"]) == pytest.approx(8.0)
    assert float(diagnostics["G_prob"]) == pytest.approx(2.0)
    assert float(diagnostics["maturity"]) == pytest.approx(0.2)
    assert float(weight) == pytest.approx(0.24)
    (inv_loss + weight * prob_loss).backward()
    assert float(invariant_parameter.grad) == pytest.approx(8.0)
    assert float(probability_parameter.grad) == pytest.approx(0.48)

    args.prob_loss_weight_mode = "fixed_epoch_ramp"
    old_weight, old_diagnostics = probability_loss_weight(
        inv_loss, prob_loss, (), (), args, 0, None,
    )
    assert float(old_weight) == pytest.approx(0.6)
    assert old_diagnostics["prob_loss_ramp"] == pytest.approx(0.6)
