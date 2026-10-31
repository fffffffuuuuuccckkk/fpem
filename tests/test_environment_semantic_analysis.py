import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "tools/analyze_predictive_environment_semantics.py"
spec = importlib.util.spec_from_file_location("semantic", MODULE)
semantic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(semantic)


def test_rolling_window_alignment_and_regimes():
    values = np.arange(10, dtype=float)
    starts = np.arange(4)
    assert np.allclose(semantic.rolling_mean(values, starts, 3), [1, 2, 3, 4])
    labels, thresholds = semantic.regimes(np.arange(9, dtype=float))
    assert list(labels).count("Low") > 0 and list(labels).count("High") > 0
    assert thresholds[0] < thresholds[1]


def test_nmi_and_composition_orientation():
    labels = np.array(["a", "a", "b", "b", "c", "c"])
    hard = np.array([0, 0, 1, 1, 2, 2])
    table, _ = semantic.contingency(labels, hard, ["a", "b", "c"])
    assert np.array_equal(table, np.eye(3, dtype=int) * 2)
    assert np.isclose(semantic.nmi_from_tables(table)[0], 1.0)
    p_env_given_group = table / table.sum(1, keepdims=True)
    p_group_given_env = table / table.sum(0, keepdims=True)
    assert np.allclose(p_env_given_group.sum(1), 1)
    assert np.allclose(p_group_given_env.sum(0), 1)


def test_permutation_is_deterministic():
    table = np.array([[20, 2, 1], [2, 20, 1], [1, 2, 20]])
    first, _ = semantic.permutation_nmi(table, 40, 2021)
    second, _ = semantic.permutation_nmi(table, 40, 2021)
    assert np.array_equal(first, second)
