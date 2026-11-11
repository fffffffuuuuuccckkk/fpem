import numpy as np

from models.predictive_env.relation_prototypes import (
    relation_fingerprint, local_operators, operator_behaviors, train_router,
    route, teacher_quality,
)


def test_relation_fingerprint_uses_history_scale_and_marks_weak_phase():
    rng = np.random.default_rng(3)
    history = rng.normal(size=(20, 96)).astype(np.float32)
    future = rng.normal(size=(20, 192)).astype(np.float32)
    feature, valid = relation_fingerprint(history, future, period=24, phase_floor=100.)
    assert feature.shape == (20, 11)
    assert np.isfinite(feature).all()
    assert not valid.any()
    assert np.all(feature[:, 3:5] == 0)


def test_local_operator_support_and_reference_mask():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(40, 4)).astype(np.float32)
    v = z @ rng.normal(size=(4, 4)).astype(np.float32)
    ids = np.arange(40) * 100
    operators = local_operators(z, v, ids, span=96, anchor_count=12,
                                neighbor_pool=40, max_neighbors=10)
    assert len(operators) == 12
    assert all(op["support"] >= 6 and op["reliable"] for op in operators)
    response, support = operator_behaviors(operators, z[:8])
    assert response.shape == (12, 8, 4)
    assert support.shape == (12, 8)


def test_student_route_is_history_only_and_normalized():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(80, 64)).astype(np.float32)
    q = np.eye(8, dtype=np.float32)[np.arange(80) % 8]
    network, _, params = train_router(x, q, epochs=2)
    first = route(network, x)
    changed_future = rng.normal(size=(80, 720))  # not supplied to route
    second = route(network, x)
    assert params > 0
    assert np.allclose(first, second)
    assert np.allclose(first.sum(axis=1), 1)
    assert np.isfinite(teacher_quality(q, first)["soft_kl"])
