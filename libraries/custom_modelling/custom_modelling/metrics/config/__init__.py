"""Configuration objects for metrics."""

from ..registry import metric_registry


def register_metric_configs():
    """Register metric configs."""
    for metric_class in metric_registry:
        metric_class.add_to_registry()
