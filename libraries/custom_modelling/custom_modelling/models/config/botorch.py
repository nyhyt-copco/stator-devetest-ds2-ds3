"""Defines BoTorch model configuration class."""

from typing import Literal

from pxc.modelling.models.base import ModelConfig
from pxs.experimental.models.tabular.botorch import (
    BoTorchConfig,
    BoTorchModel,
    BoTorchTrainingConfig,
)


class ExampleBoTorchConfig(ModelConfig[BoTorchModel]):
    """Model configuration for BoTorch."""

    model_type: Literal["example_botorch"] = "example_botorch"

    config: BoTorchConfig
    training: BoTorchTrainingConfig | None = None

    def initialize(self, *, device: str = "cpu") -> BoTorchModel:
        """Initialize BoTorch model."""
        return BoTorchModel(self.config, device=device)
