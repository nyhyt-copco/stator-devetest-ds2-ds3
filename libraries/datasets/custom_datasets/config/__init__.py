"""Defines configuration classes for custom dataset types."""

from ..registry import dataset_registry


def register_dataset_configs():
    """Register dataset configs."""
    for dataset_class in dataset_registry:
        dataset_class.add_to_registry()
