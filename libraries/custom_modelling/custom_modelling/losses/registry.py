"""Registry of loss classes."""

from .config.huber import ExampleHuberLossConfig

# Registry of loss classes
loss_registry = [
    ExampleHuberLossConfig,
]
