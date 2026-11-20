from types import SimpleNamespace

import numpy as np
import torch

from tools.fpem_probabilistic_training import probability_loss_weight
from tools.run_fpem_head_comparison import probability_future_condition
from tools.run_predictive_env_iv_patchtst import controlled_environment_assignment


def test_zinv_only_drops_future_zvar_and_its_gradient():
    future = torch.randn(2, 3, 4, 8, requires_grad=True)
    condition = probability_future_condition(future, "zinv_only")
    assert condition.shape == future.shape
    assert torch.count_nonzero(condition) == 0
    assert not condition.requires_grad
    assert probability_future_condition(future, "future_zvar") is future


def test_constant_probability_weight_is_one_and_detached():
    inv = torch.tensor(2.0, requires_grad=True)
    prob = torch.tensor(3.0, requires_grad=True)
    weight, logs = probability_loss_weight(
        inv, prob, (), (),
        SimpleNamespace(prob_loss_weight_mode="constant"), 0, None,
    )
    assert weight.item() == 1.0
    assert not weight.requires_grad
    assert logs["prob_loss_constant"] == 1.0


def test_environment_shuffle_is_deterministic_and_mass_preserving():
    assignment = np.arange(30, dtype=np.float32).reshape(10, 3)
    assignment /= assignment.sum(1, keepdims=True)
    shuffled = controlled_environment_assignment(assignment, "shuffled", 2021, 0)
    again = controlled_environment_assignment(assignment, "shuffled", 2021, 0)
    assert np.array_equal(shuffled, again)
    assert not np.array_equal(shuffled, assignment)
    assert np.allclose(shuffled.sum(0), assignment.sum(0))
