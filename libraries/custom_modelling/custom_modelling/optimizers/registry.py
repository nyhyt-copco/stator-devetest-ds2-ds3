"""Registry of optimizer classes."""

from .config.example_adam import ExampleAdamOptimizerConfig

# Registry of optimizer classes
optimizer_registry = [
    ExampleAdamOptimizerConfig,
]
