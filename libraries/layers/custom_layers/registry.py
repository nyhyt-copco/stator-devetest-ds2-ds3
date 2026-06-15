"""Registry of layer classes."""

from .config.compute_a_matrix_coefficients import ComputeAMatrixCoefficientsConfig
from .config.random_rotation import RandomRotationConfig

# Registry of layer classes
layer_registry = [
    RandomRotationConfig,
    ComputeAMatrixCoefficientsConfig,
]