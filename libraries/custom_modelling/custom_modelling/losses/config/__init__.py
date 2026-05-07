"""PyTorch config representing loss functions."""

from ..registry import loss_registry


def register_loss_configs():
    """Register loss configs."""
    for config_class in loss_registry:
        config_class.add_to_registry()
