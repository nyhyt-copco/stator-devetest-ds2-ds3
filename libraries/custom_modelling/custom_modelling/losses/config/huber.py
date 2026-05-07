"""Example Huber Loss Config class definition."""

from typing import Any, Literal

import torch

from pxs.losses.pytorch.config.base import BaseLossConfig


class ExampleHuberLossConfig(BaseLossConfig):
    """Config representing an example Huber PyTorch loss function."""

    loss_type: Literal["example_huber"] = "example_huber"

    def initialize(self, **_: Any) -> torch.nn.HuberLoss:
        """Initialise loss function using config.

        Returns:
            Initialized loss object.
        """
        return torch.nn.HuberLoss(reduction=self.reduction)
