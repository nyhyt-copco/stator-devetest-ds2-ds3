"""Defines configuration class for the base layer class and core implementations."""

from ..registry import layer_registry


def register_layer_configs():
    """Register layer configs."""
    for layer_class in layer_registry:
        layer_class.add_to_registry()
