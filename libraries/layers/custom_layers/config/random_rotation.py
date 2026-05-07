"""Defines config for layer to convert pv.PolyData with cell data to point data."""

from typing import Literal

from pydantic import Field
from samples.layers.config.base import LayerConfig

from ..core.random_rotation import RandomRotation


class RandomRotationConfig(LayerConfig[RandomRotation]):
    """Config for creating the RandomRotation Layer."""

    type: Literal["example_random_rotation"] = "example_random_rotation"
    input_datablock: str = Field(min_length=1)
    rotation_axes: Literal["x", "y", "z", "xy", "xz", "yz", "xyz"] = "xyz"
    angle_range_degrees: tuple[float, float] = (0.0, 360.0)
    seed: int | None = None

    def initialize(self) -> RandomRotation:
        """Initialize RandomRotation Layer class."""
        return RandomRotation(
            input_datablock=self.input_datablock,
            rotation_axes=self.rotation_axes,
            angle_range_degrees=self.angle_range_degrees,
            seed=self.seed,
            tags=self.tags,
        )
