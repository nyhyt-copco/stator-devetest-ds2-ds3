"""Adam Optimizer Config class definition."""

from typing import Literal

import torch
from pydantic import Field

from pxs.optimizers.pytorch.config.base import BaseOptimizerConfig
from pxs.typing.pytorch import ModelParametersType


class ExampleAdamOptimizerConfig(BaseOptimizerConfig):
    """Config representing a PyTorch adam optimizer."""

    optimizer_type: Literal["example_adam"] = "example_adam"

    learning_rate: float = Field(0.001, ge=0)
    weight_decay: float = Field(0.0, ge=0)

    def initialize(
        self,
        *,
        model_params: ModelParametersType,
    ) -> torch.optim.Adam:
        """Initialize optimizer.

        Args:
            model_params: Model weights

        Returns:
            Initialized optimizer
        """
        return torch.optim.Adam(
            params=model_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
