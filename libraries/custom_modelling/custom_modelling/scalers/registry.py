"""Registry of scaler classes."""

from .config.example_min_max import ExampleMinMaxScalerConfig

# Registry of scaler classes
scaler_registry = [
    ExampleMinMaxScalerConfig,
]
