"""OneCycleLR Config class definition."""

from typing import Literal

import torch
from pydantic import Field

from pxs.schedulers.pytorch.config.base import BaseSchedulerConfig
from pxs.schedulers.pytorch.config.one_cycle import OneCycleSchedulerConfig
from ..core.power_decay_one_cycle import ExamplePowerDecayOneCycleLR


class ExamplePowerDecayOneCycleSchedulerConfig(BaseSchedulerConfig):
    """Config representing PowerDecay OneCycleLR scheduler."""

    scheduler_type: Literal["example_power_decay_one_cycle"] = "example_power_decay_one_cycle"

    power: float = Field(0.5, gt=0)
    one_cycle_n_epochs_fraction: float = Field(0.125, gt=0)
    one_cycle_config: OneCycleSchedulerConfig = OneCycleSchedulerConfig()

    def initialize(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> ExamplePowerDecayOneCycleLR:
        """Initialize scheduler.

        Args:
            optimizer: optimizer
            total_steps: total number of steps for scheduler (number of epochs * steps per epoch)

        Returns:
            Initialized scheduler
        """
        one_cycle_total_steps = int(total_steps * self.one_cycle_n_epochs_fraction)
        return ExamplePowerDecayOneCycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_config.max_learning_rate,
            power=self.power,
            one_cycle_total_steps=one_cycle_total_steps,
            pct_start=self.one_cycle_config.fraction_spent_in_first_phase,
            div_factor=self.one_cycle_config.max_to_initial_learning_rate_ratio,
            final_div_factor=self.one_cycle_config.max_to_final_learning_rate_ratio,
        )
