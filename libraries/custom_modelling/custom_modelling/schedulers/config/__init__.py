"""Configuration objects for schedulers."""

from ..registry import scheduler_registry


def register_scheduler_configs():
    """Register scheduler configs."""
    for scheduler_class in scheduler_registry:
        scheduler_class.add_to_registry()
