"""Configuration objects for scalers."""

from ..registry import scaler_registry


def register_scaler_configs():
    """Register scaler configs."""
    for scaler_class in scaler_registry:
        scaler_class.add_to_registry()
