"""Defines configuration class for the base artifact class and core implementations."""

from ..registry import artifact_registry


def register_artifact_configs():
    """Register artifact configs."""
    for artifact_class in artifact_registry:
        artifact_class.add_to_registry()
