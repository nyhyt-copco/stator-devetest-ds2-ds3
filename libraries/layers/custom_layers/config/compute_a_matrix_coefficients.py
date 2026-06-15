"""Config for the ComputeAMatrixCoefficients layer."""

from typing import Literal

from pydantic import Field
from samples.layers.config.base import LayerConfig

from ..core.compute_a_matrix_coefficients import ComputeAMatrixCoefficients


class ComputeAMatrixCoefficientsConfig(LayerConfig[ComputeAMatrixCoefficients]):
    """Config for creating the ComputeAMatrixCoefficients layer."""

    type: Literal["compute_a_matrix_coefficients_layer"] = (
        "compute_a_matrix_coefficients_layer"
    )
    input_datablock: str = Field(min_length=1)
    n_components: int = 16

    def initialize(self) -> ComputeAMatrixCoefficients:
        """Initialize ComputeAMatrixCoefficients layer."""
        return ComputeAMatrixCoefficients(
            input_datablock=self.input_datablock,
            n_components=self.n_components,
            tags=self.tags,
        )
