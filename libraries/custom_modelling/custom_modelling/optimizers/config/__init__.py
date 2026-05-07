"""Configuration objects for optimizers."""

from ..registry import optimizer_registry


def register_optimizer_configs():
    """Register optimizer configs."""
    for optimizer_class in optimizer_registry:
        optimizer_class.add_to_registry()
