"""Permutation-agnostic soft environment assignments."""
from models.eivpm.environment_decomposer import FoilEnvironmentProvider


class DynamicEnvironmentProvider(FoilEnvironmentProvider):
    """Reuse the repository FOIL history partition while retaining soft q.

    Downstream domain constraints consume only q_i @ q_j, so simultaneous
    permutation of every environment coordinate leaves them unchanged.
    """
