import numpy as np

from tools.run_fpem_prototype_quality import (
    frequency_features, trend_seasonal_features, phase_features, phase_route,
    pair_evidence, independent_events,
)


def test_features_are_fixed_width_and_finite():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5, 64)).astype(np.float32)
    for value in (frequency_features(x), trend_seasonal_features(x),
                  phase_features(np.linspace(0, 1, 5))):
        assert value.shape == (5, 64)
        assert np.isfinite(value).all()


def test_phase_hint_is_weak_and_ambiguity_sensitive():
    distance = np.array([[.3, .31], [.1, 5.0]], dtype=np.float32)
    adjusted, weight = phase_route(distance, np.array([0., 0.]),
                                    np.array([[1., 0.], [-1., 0.]]))
    assert adjusted.shape == distance.shape
    assert weight[0] > weight[1]
    assert np.max(weight) <= .15


def test_future_lift_missing_without_nonoverlapping_support():
    block = {"raw": np.zeros((30, 64), np.float32),
             "y": np.zeros((30, 12), np.float32),
             "channels": np.zeros(30, int), "ids": np.arange(30)}
    result, frame = pair_evidence(block, np.zeros(30, int), span=96)
    assert np.isnan(result["lift"])
    assert frame is None
    assert independent_events(np.arange(30), 96) == 1
