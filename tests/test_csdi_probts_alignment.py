"""CSDI and ProbTS scoring regression tests without GluonTS dependency."""

import numpy as np
import pytest
import torch

from models.predictive_env.csdi_probts import ProbTSCSDI
from tools.evaluate_fpem_probabilistic import probts_quantile_crps


def reference_probts_evaluator_crps(samples, target, quantiles_num):
    # Direct scalar translation of ProbTS-main/probts/utils/evaluator.py:
    # np.quantile -> quantile_loss -> wQuantileLoss -> mean q -> mean sample.
    sample_scores = []
    for i in range(len(target)):
        scores = []
        for q in np.arange(1, quantiles_num) / quantiles_num:
            forecast = np.quantile(samples[i], q, axis=0)
            quantile_loss = 2 * np.abs(
                (forecast - target[i]) * ((target[i] <= forecast) - q)
            ).sum()
            scores.append(quantile_loss / np.abs(target[i]).sum())
        sample_scores.append(np.mean(scores))
    return np.asarray(sample_scores)


@pytest.mark.parametrize("quantiles_num", [10, 20])
def test_probts_crps_matches_reference_per_sample_and_variable_sum(quantiles_num):
    np.random.seed(11)
    targets = np.random.normal(size=(3, 7, 2)).astype(np.float64)
    targets[0] *= 100  # makes global normalization observably wrong
    samples = np.random.normal(size=(3, 13, 7, 2)).astype(np.float64)
    observed = probts_quantile_crps(
        torch.from_numpy(samples), torch.from_numpy(targets), quantiles_num,
    ).numpy()
    expected = reference_probts_evaluator_crps(samples, targets, quantiles_num)
    assert np.allclose(observed, expected, rtol=1e-12, atol=1e-12)
    observed_sum = probts_quantile_crps(
        torch.from_numpy(samples.sum(-1, keepdims=True)),
        torch.from_numpy(targets.sum(-1, keepdims=True)), quantiles_num,
    ).numpy()
    expected_sum = reference_probts_evaluator_crps(
        samples.sum(-1, keepdims=True),
        targets.sum(-1, keepdims=True), quantiles_num,
    )
    assert np.allclose(observed_sum, expected_sum, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("horizon,channels", [(15, 1), (17, 3), (96, 7)])
def test_csdi_loss_sampling_shape_grad_and_seed(horizon, channels):
    torch.manual_seed(3)
    model = ProbTSCSDI(
        channels, context_length=8, prediction_length=horizon,
        hidden_channels=16, emb_time_dim=16, emb_feature_dim=4,
        diffusion_embedding_dim=16, num_steps=4, num_heads=2, n_layers=1,
    )
    x = torch.randn(2, 8, channels)
    y = torch.randn(2, horizon, channels)
    loss = model.training_loss(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())
    model.eval()
    first = model.sample(x, 3, 2, torch.Generator().manual_seed(99))
    again = model.sample(x, 3, 2, torch.Generator().manual_seed(99))
    assert first.shape == (2, 3, horizon, channels)
    assert torch.equal(first, again)
    assert not torch.equal(first[:, 0], first[:, 1])


def test_csdi_sampling_does_not_accept_future_target():
    model = ProbTSCSDI(
        1, 8, 15, hidden_channels=16, emb_time_dim=16,
        emb_feature_dim=4, diffusion_embedding_dim=16,
        num_steps=2, num_heads=2, n_layers=1,
    )
    assert "y" not in model.sample.__code__.co_varnames[
        :model.sample.__code__.co_argcount]
    with pytest.raises(ValueError):
        model.training_loss(torch.randn(2, 7, 1), torch.randn(2, 15, 1))
