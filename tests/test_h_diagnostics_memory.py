import torch

from tools.run_predictive_env_iv_patchtst import (
    _bounded_flatten,
    _compact_prediction,
    h_stability_diagnostics,
)


def test_bounded_flatten_is_deterministic_and_bounded():
    values = torch.arange(10_000, dtype=torch.float32)
    first = _bounded_flatten(values, max_points=127)
    second = _bounded_flatten(values, max_points=127)
    assert first.numel() <= 127
    assert torch.equal(first, second)


def test_h_stability_accepts_compact_reference_cache():
    prediction = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)
    target = prediction + 1.0
    reference = _compact_prediction(prediction, width=5)
    norm = torch.tensor([2.0, 4.0])
    metrics = h_stability_diagnostics(
        prediction,
        target,
        torch.tensor([0, 1]),
        norm,
        reference,
        norm.clone(),
    )
    assert metrics["h_prediction_MSE"] == 1.0
    assert metrics["h_prediction_MAE"] == 1.0
    assert metrics["h_prediction_drift_from_pretrained"] == 0.0
    assert abs(metrics["h_prediction_corr_with_pretrained"] - 1.0) < 1e-12
    assert metrics["H_norm_ratio_to_pretrained"] == 1.0
