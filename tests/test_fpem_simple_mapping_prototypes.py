import numpy as np

from tools.fpem_simple_mapping_descriptors import NAMES, DIMS, combinations, descriptors, robust_groups, combine


def test_complete_unique_registry():
    groups=combinations()
    assert len(groups)==255
    assert len(set(groups))==255
    assert [sum(len(group)==k for group in groups) for k in range(1,9)]==[8,28,56,70,56,28,8,1]


def test_eight_descriptors_finite_and_group_balanced():
    rng=np.random.default_rng(2022)
    history=rng.normal(size=(32,96)).astype(np.float32)
    future=rng.normal(size=(32,48)).astype(np.float32)
    values,meta=descriptors(history,future,24)
    assert tuple(values)==NAMES
    assert [values[name].shape[1] for name in NAMES]==list(DIMS)
    assert all(np.isfinite(value).all() for value in values.values())
    fitted=robust_groups(values,NAMES)
    mixed=combine(values,fitted,NAMES)
    assert mixed.shape==(32,sum(DIMS)) and np.isfinite(mixed).all()
    assert meta["seasonal_valid"]


def test_constant_history_and_invalid_period_have_safe_fallback():
    history=np.ones((16,32),np.float32)
    future=np.ones((16,9),np.float32)*2
    values,meta=descriptors(history,future,200)
    assert not meta["seasonal_valid"]
    assert all(np.isfinite(value).all() for value in values.values())
    fitted=robust_groups(values,NAMES)
    mixed=combine(values,fitted,("M1","M4","M8"))
    assert np.isfinite(mixed).all()
