import numpy as np
import torch

from tools.run_fpem_f03_safe import analytic_alpha, forecast, interpolation, largest_safe_alpha, mse_mae


def test_analytic_alpha_and_hard_protection():
    y0 = np.array([[2.0], [2.0]], np.float32)
    y1 = np.array([[0.0], [0.0]], np.float32)
    truth = np.array([[1.0], [1.0]], np.float32)
    alpha = analytic_alpha(y0, y1, truth)
    assert abs(alpha - .5) < 1e-8
    assert mse_mae(y0 + alpha * (y1 - y0), truth)["mse"] == 0
    safe = largest_safe_alpha(y0, y1, truth, 1.0, 0.0)
    assert safe == 1.0  # Full update ties A0, and is therefore admissible.


def test_harmful_head_falls_back_and_linear_interpolation_exact():
    record = {"design": np.array([[1., 2.], [3., 4.]], np.float32),
              "ratio": np.ones((2, 1), np.float32)}
    a0 = {"weight": torch.tensor([[1., 0.]]), "bias": torch.tensor([0.])}
    bad = {"weight": torch.tensor([[2., 0.]]), "bias": torch.tensor([0.])}
    y0, y1 = forecast(record, a0), forecast(record, bad)
    assert analytic_alpha(y0, y1, y0) == 0
    assert largest_safe_alpha(y0, y1, y0, 1., 0.) < 1e-5
    alpha = .37
    blended = forecast(record, interpolation(a0, bad, alpha))
    assert np.allclose(blended, y0 + alpha * (y1 - y0), atol=1e-6)
