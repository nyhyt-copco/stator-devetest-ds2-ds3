"""Configuration objects for models."""

from ..registry import model_registry


def register_model_configs():
    """Register model configs."""
    for model_class in model_registry:
        model_class.add_to_registry()
