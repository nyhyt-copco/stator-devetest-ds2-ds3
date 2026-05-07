"""Defines configuration for splitting samples into several datasets."""

from ..registry import split_registry


def register_split_configs():
    """Register split configs."""
    for split_class in split_registry:
        split_class.add_to_registry()
