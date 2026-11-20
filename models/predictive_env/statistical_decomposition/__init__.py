"""H-space statistical factorization; no EIIL labels or forecast bypass."""

from .factorizers import StatisticalDecomposer, make_statistical_decomposer

__all__ = ["StatisticalDecomposer", "make_statistical_decomposer"]
